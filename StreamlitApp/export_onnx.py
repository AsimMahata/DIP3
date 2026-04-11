# -*- coding: utf-8 -*-
"""
Export the custom MultiTaskAbuseDetector to ONNX format.

The model has a shared XLM-RoBERTa-Large backbone with two output heads:
  - offensive_logits: (B, 1) -- raw logit, apply sigmoid for probability
  - language_logits:  (B, 3) -- raw logits, apply softmax for probabilities

Usage:
    python export_onnx.py
"""

import os
import sys
import io
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# -- Config --
MODEL_PATH = "./model"
ONNX_SAVE_DIR = "./onnx_model"
ONNX_FILE = os.path.join(ONNX_SAVE_DIR, "model.onnx")
MAX_LEN = 128
NUM_LANGUAGES = 3


# -- Model Definition (same as training notebook) --
class MultiTaskAbuseDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_PATH)
        hidden = self.backbone.config.hidden_size  # 1024
        self.dropout = nn.Dropout(0.1)
        self.offensive_head = nn.Linear(hidden, 1)
        self.language_head = nn.Linear(hidden, NUM_LANGUAGES)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        offensive_logits = self.offensive_head(cls)
        language_logits = self.language_head(cls)
        return offensive_logits, language_logits


def main():
    print("=" * 60)
    print("  ONNX Export -- MultiTaskAbuseDetector")
    print("=" * 60)

    # 1. Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("   Tokenizer loaded [OK]")

    # 2. Build model and load weights
    print("\n2. Building model and loading weights...")
    model = MultiTaskAbuseDetector()

    heads_path = os.path.join(MODEL_PATH, "heads.pt")
    if os.path.exists(heads_path):
        heads = torch.load(heads_path, map_location="cpu", weights_only=True)
        model.offensive_head.load_state_dict(heads["offensive_head"])
        model.language_head.load_state_dict(heads["language_head"])
        model.dropout.load_state_dict(heads["dropout"])
        print("   Custom heads loaded from heads.pt [OK]")
    else:
        print("   WARNING: heads.pt not found! Exporting with random heads.")

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model built -- {n_params:,} parameters")

    # 3. Create dummy inputs
    print("\n3. Creating dummy inputs...")
    dummy_text = "This is a test sentence for ONNX export."
    dummy_inputs = tokenizer(
        dummy_text, return_tensors="pt",
        max_length=MAX_LEN, truncation=True, padding="max_length"
    )
    input_ids = dummy_inputs["input_ids"]
    attention_mask = dummy_inputs["attention_mask"]
    print(f"   input_ids shape: {input_ids.shape}")
    print(f"   attention_mask shape: {attention_mask.shape}")

    # 4. Verify PyTorch inference
    print("\n4. Verifying PyTorch inference...")
    with torch.no_grad():
        off_logits, lang_logits = model(input_ids, attention_mask)
    off_prob = torch.sigmoid(off_logits).item()
    lang_probs = torch.softmax(lang_logits, dim=-1).numpy()[0]
    print(f"   Offensive prob: {off_prob:.4f}")
    print(f"   Language probs: {dict(zip(['en', 'hi', 'bn'], [f'{p:.4f}' for p in lang_probs]))}")

    # 5. Export to ONNX
    print("\n5. Exporting to ONNX...")
    os.makedirs(ONNX_SAVE_DIR, exist_ok=True)

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        ONNX_FILE,
        input_names=["input_ids", "attention_mask"],
        output_names=["offensive_logits", "language_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "offensive_logits": {0: "batch_size"},
            "language_logits": {0: "batch_size"},
        },
        opset_version=18,
        do_constant_folding=True,
    )
    file_size_mb = os.path.getsize(ONNX_FILE) / (1024 * 1024)
    print(f"   ONNX model saved: {ONNX_FILE} ({file_size_mb:.1f} MB)")

    # 6. Verify ONNX model
    print("\n6. Verifying ONNX model...")
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(ONNX_FILE, providers=["CPUExecutionProvider"])

        ort_inputs = {
            "input_ids": input_ids.numpy().astype(np.int64),
            "attention_mask": attention_mask.numpy().astype(np.int64),
        }
        ort_outputs = session.run(None, ort_inputs)

        onnx_off_logit = ort_outputs[0][0, 0]
        onnx_off_prob = 1.0 / (1.0 + np.exp(-onnx_off_logit))
        onnx_lang_logits = ort_outputs[1][0]
        onnx_lang_exp = np.exp(onnx_lang_logits - np.max(onnx_lang_logits))
        onnx_lang_probs = onnx_lang_exp / onnx_lang_exp.sum()

        print(f"   ONNX offensive prob: {onnx_off_prob:.4f}")
        print(f"   ONNX language probs: {dict(zip(['en', 'hi', 'bn'], [f'{p:.4f}' for p in onnx_lang_probs]))}")

        # Check closeness
        diff_off = abs(off_prob - onnx_off_prob)
        diff_lang = np.max(np.abs(lang_probs - onnx_lang_probs))
        print(f"\n   Max diff (offensive): {diff_off:.6f}")
        print(f"   Max diff (language):  {diff_lang:.6f}")

        if diff_off < 0.001 and diff_lang < 0.001:
            print("   ONNX verification PASSED -- outputs match PyTorch!")
        else:
            print("   WARNING: Outputs differ slightly (may be acceptable)")

    except ImportError:
        print("   WARNING: onnxruntime not installed, skipping verification")

    # 7. Also save tokenizer to onnx dir for convenience
    tokenizer.save_pretrained(ONNX_SAVE_DIR)
    print(f"\n   Tokenizer saved to {ONNX_SAVE_DIR}")

    print("\n" + "=" * 60)
    print("  Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
