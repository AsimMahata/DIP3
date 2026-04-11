# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
FastAPI backend for Multilingual Abuse Detection.

Architecture: XLM-RoBERTa-Large backbone with two heads:
  - offensive_head → sigmoid → binary (toxic / clean)
  - language_head  → softmax → 3-class (english / hinglish / banglish)

Saved artefacts in ./model/:
  config.json + model-001.safetensors  → backbone weights
  heads.pt                             → offensive_head + language_head + dropout
  tokenizer.json + tokenizer_config.json
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
import re
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from lime.lime_text import LimeTextExplainer

try:
    import emoji
except ImportError:
    emoji = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

app = FastAPI(title="Multilingual Abuse Detection API")

# ── Paths ──────────────────────────────────────────────────
MODEL_PATH = "./model"
ONNX_MODEL_PATH = "./onnx_model/model.onnx"
MAX_LEN = 128
THRESHOLD = 0.5  # default; can be overridden

# ── Language mappings ──────────────────────────────────────
LANG_TO_IDX = {"english": 0, "hinglish": 1, "banglish": 2}
IDX_TO_LANG = {v: k for k, v in LANG_TO_IDX.items()}
NUM_LANGUAGES = 3

# ── Preprocessing (from training notebook) ─────────────────
ENG_ABBREVS = {
    "wtf": "what the fuck", "stfu": "shut the fuck up",
    "idk": "I don't know", "ngl": "not gonna lie",
    "lmao": "laughing my ass off", "omg": "oh my god",
    "u": "you", "ur": "your", "r": "are", "bc": "because",
    "tbh": "to be honest", "smh": "shaking my head",
    "af": "as fuck", "nvm": "never mind", "lol": "laugh out loud",
    "rofl": "rolling on the floor laughing", "wbu": "what about you",
    "brb": "be right back", "gtg": "got to go",
    "ttyl": "talk to you later", "fyi": "for your information",
    "imo": "in my opinion", "imho": "in my humble opinion",
    "btw": "by the way", "np": "no problem", "ty": "thank you",
    "wyd": "what you doing", "hmu": "hit me up",
    "iykyk": "if you know you know", "rn": "right now",
    "ikr": "I know right", "jk": "just kidding",
    "sfw": "safe for work", "nsfw": "not safe for work",
}
DESI_ABBREVS = {
    # "yaar": "friend", "bhai": "brother", "abbe": "hey",
    # "sala": "jerk", "saale": "jerk", "bakwas": "nonsense",
    # "chup": "silent", "aukaat": "worth", "ganda": "dirty",
    # "bura": "bad", "mast": "awesome", "kya": "what",
    # "toh": "then", "nahi": "no", "bc": "behanchod",
    # "bsdk": "bahenchod ka duffer", "mc": "madarchod",
    # "bhen": "sister", "behen": "sister",
    # "bhenchod": "sisterfucker", "chutiya": "idiot",
    # "gaandu": "asshole", "jatt": "cool", "jigri": "close friend",
    # "panga": "trouble", "dimaag": "brain", "pakao": "boring",
    # "tharki": "pervert", "chamcha": "sycophant",
    # "kamina": "scoundrel", "lauda": "penis", "chut": "vagina",
    # "lund": "penis",
    # "vai": "brother", "apu": "sister", "pagla": "crazy",
    # "baje": "bad", "dhor": "catch", "boro": "big",
    # "choto": "small", "khub": "very", "jibon": "life",
    # "bhalo": "good", "kharap": "bad", "ki": "what",
    # "keno": "why", "kothay": "where", "kobe": "when",
    # "kemon": "how", "tui": "you", "tumi": "you (formal)",
    # "apni": "you (very formal)", "amar": "my", "tomar": "your",
    # "tor": "your", "ei": "this", "sei": "that", "kintu": "but",
    # "tobe": "then", "ar": "and", "o": "also", "na": "no",
    # "haoa": "to be", "thako": "stay", "jao": "go",
    # "aslo": "came", "dibo": "will give", "nibo": "will take",
    # "koro": "do", "kris": "did", "korchis": "doing",
    # "korbos": "will do",
}


def preprocess_text(text: str) -> str:
    """Same preprocessing pipeline as the training notebook."""
    # Strip URLs / HTML tags
    text = re.sub(r"http\S+|www\S+", " ", str(text))
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Demojize
    if emoji is not None:
        text = emoji.demojize(text, delimiters=(" ", " "))
    # Collapse repeated chars
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    # Expand abbreviations (use english + desi combined)
    abbrevs = dict(ENG_ABBREVS)
    abbrevs.update(DESI_ABBREVS)
    tokens = text.lower().split()
    text = " ".join(abbrevs.get(t, t) for t in tokens)
    return text.strip()


# ── Model Definition ───────────────────────────────────────
class MultiTaskAbuseDetector(nn.Module):
    """Exact same architecture as the training notebook."""
    def __init__(self):
        super().__init__()
        # Use 'eager' attention so we can get attention weights out. 
        # SDPA (the default in PyTorch 2+) does not support output_attentions=True.
        self.backbone = AutoModel.from_pretrained(MODEL_PATH, attn_implementation="eager")
        hidden = self.backbone.config.hidden_size  # 1024
        self.dropout = nn.Dropout(0.1)
        self.offensive_head = nn.Linear(hidden, 1)
        self.language_head = nn.Linear(hidden, NUM_LANGUAGES)

    def forward(self, input_ids, attention_mask, output_attentions=False):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        cls = self.dropout(out.last_hidden_state[:, 0, :])  # CLS token
        return {
            "offensive_logits": self.offensive_head(cls),    # (B, 1)
            "language_logits": self.language_head(cls),       # (B, 3)
            "attentions": out.attentions if output_attentions else None,
        }


# ── Load ───────────────────────────────────────────────────
print("Initializing model...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Build the model and load the custom heads
model = MultiTaskAbuseDetector()
heads_path = os.path.join(MODEL_PATH, "heads.pt")
if os.path.exists(heads_path):
    heads = torch.load(heads_path, map_location="cpu", weights_only=True)
    model.offensive_head.load_state_dict(heads["offensive_head"])
    model.language_head.load_state_dict(heads["language_head"])
    model.dropout.load_state_dict(heads["dropout"])
    print("Custom heads loaded from heads.pt [OK]")
else:
    print("WARNING: heads.pt not found -- using random heads!")

model.to(DEVICE)
model.eval()
print(f"PyTorch model loaded on {DEVICE} [OK]")

# Try loading ONNX for faster inference
onnx_session = None
if ort is not None and os.path.exists(ONNX_MODEL_PATH):
    try:
        onnx_session = ort.InferenceSession(
            ONNX_MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )
        print("ONNX model loaded for inference [OK]")
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")

# LIME explainer (binary: Clean vs Offensive)
explainer = LimeTextExplainer(class_names=["Clean", "Offensive"])


# ── Inference helpers ──────────────────────────────────────
def predict_pytorch(text: str):
    """Run inference through the PyTorch model."""
    cleaned = preprocess_text(text)
    inputs = tokenizer(
        cleaned, return_tensors="pt",
        max_length=MAX_LEN, truncation=True, padding="max_length"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    off_prob = float(torch.sigmoid(out["offensive_logits"].squeeze(-1)).cpu())
    lang_probs = torch.softmax(out["language_logits"], dim=-1).cpu().numpy()[0]
    lang_idx = int(np.argmax(lang_probs))

    return {
        "offensive_prob": off_prob,
        "is_toxic": off_prob > THRESHOLD,
        "language": IDX_TO_LANG[lang_idx],
        "language_probs": {IDX_TO_LANG[i]: float(lang_probs[i]) for i in range(NUM_LANGUAGES)},
    }


def predict_onnx(text: str):
    """Run inference through the ONNX model."""
    cleaned = preprocess_text(text)
    inputs = tokenizer(
        cleaned, return_tensors="np",
        max_length=MAX_LEN, truncation=True, padding="max_length"
    )

    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    outputs = onnx_session.run(None, ort_inputs)
    offensive_logit = outputs[0][0, 0]  # (1, 1) → scalar
    language_logits = outputs[1][0]      # (1, 3) → (3,)

    # Sigmoid for offensive
    off_prob = float(1.0 / (1.0 + np.exp(-offensive_logit)))
    # Softmax for language
    lang_exp = np.exp(language_logits - np.max(language_logits))
    lang_probs = lang_exp / lang_exp.sum()
    lang_idx = int(np.argmax(lang_probs))

    return {
        "offensive_prob": off_prob,
        "is_toxic": off_prob > THRESHOLD,
        "language": IDX_TO_LANG[lang_idx],
        "language_probs": {IDX_TO_LANG[i]: float(lang_probs[i]) for i in range(NUM_LANGUAGES)},
    }


def predict(text: str):
    """Use ONNX if available, otherwise PyTorch."""
    if onnx_session is not None:
        return predict_onnx(text)
    return predict_pytorch(text)


def predict_probabilities_for_lime(texts):
    """
    LIME requires: list[str] → np.array of shape (n, 2)
    Columns: [P(clean), P(offensive)]
    """
    probs = []
    for text in texts:
        result = predict(text)
        off = result["offensive_prob"]
        probs.append([1.0 - off, off])
    return np.array(probs)


# ── Request / Response models ─────────────────────────────
class PredictRequest(BaseModel):
    text: str

class ExplainRequest(BaseModel):
    text: str
    target_class_idx: int = 1  # 0=clean, 1=offensive


# ── Endpoints ──────────────────────────────────────────────
@app.get("/")
def read_root():
    return {
        "message": "Multilingual Abuse Detection API is running",
        "model": "XLM-RoBERTa-Large (multi-task)",
        "onnx_loaded": onnx_session is not None,
    }


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    try:
        result = predict(req.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pipeline")
def pipeline_endpoint(req: PredictRequest):
    """
    Returns all intermediate processing steps for pipeline visualization:
    Raw Text → Preprocessed → Tokenized → Padded → Model → Output
    """
    try:
        raw_text = req.text

        # Step 1: Preprocessing
        preprocessed = preprocess_text(raw_text)

        # Step 2: Tokenization (without padding to see raw tokens)
        tokens_raw = tokenizer(preprocessed, add_special_tokens=True)
        raw_input_ids = tokens_raw["input_ids"]
        raw_tokens = tokenizer.convert_ids_to_tokens(raw_input_ids)

        # Step 3: Tokenization + Padding
        tokens_padded = tokenizer(
            preprocessed, return_tensors="pt",
            max_length=MAX_LEN, truncation=True, padding="max_length"
        )
        padded_ids = tokens_padded["input_ids"][0].tolist()
        attention_mask = tokens_padded["attention_mask"][0].tolist()
        num_real = sum(attention_mask)
        num_padded = MAX_LEN - num_real

        # Step 4: Model inference
        inputs = {k: v.to(DEVICE) for k, v in tokens_padded.items()}
        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_attentions=True,
            )

        off_logit = out["offensive_logits"].squeeze(-1).cpu().item()
        lang_logits = out["language_logits"].cpu().numpy()[0].tolist()
        off_prob = float(torch.sigmoid(torch.tensor(off_logit)))
        lang_probs_raw = torch.softmax(torch.tensor(lang_logits), dim=-1).numpy()

        # CLS attention (word-level)
        attn = out["attentions"][-1][0].mean(0)[0].cpu().numpy()
        attn_real = attn[:num_real]
        attn_real = attn_real / (attn_real.max() + 1e-9)

        # Merge subwords for word attention
        words, weights = [], []
        real_tokens = tokenizer.convert_ids_to_tokens(padded_ids[:num_real])
        for tok_s, w in zip(real_tokens, attn_real):
            clean_t = tok_s.replace("\u2581", "").replace("##", "")
            if clean_t in ("<s>", "</s>", "[CLS]", "[SEP]", "[PAD]", ""):
                continue
            if words and (tok_s.startswith("##") or not tok_s.startswith("\u2581")):
                words[-1] += clean_t
                weights[-1] = max(weights[-1], float(w))
            else:
                words.append(clean_t)
                weights.append(float(w))

        return {
            # Step 1: Raw
            "raw_text": raw_text,
            # Step 2: Preprocessed
            "preprocessed_text": preprocessed,
            # Step 3: Tokenization
            "raw_tokens": raw_tokens,
            "raw_token_ids": raw_input_ids,
            "num_raw_tokens": len(raw_input_ids),
            # Step 4: Padding
            "max_length": MAX_LEN,
            "num_real_tokens": num_real,
            "num_padded_tokens": num_padded,
            "attention_mask_sample": attention_mask[:20],  # first 20 for display
            # Step 5: Model outputs
            "offensive_logit": round(off_logit, 4),
            "offensive_prob": round(off_prob, 4),
            "is_toxic": off_prob > THRESHOLD,
            "language_logits": [round(l, 4) for l in lang_logits],
            "language_probs": {IDX_TO_LANG[i]: round(float(lang_probs_raw[i]), 4) for i in range(NUM_LANGUAGES)},
            "language": IDX_TO_LANG[int(np.argmax(lang_probs_raw))],
            # Word attention
            "attention_words": words[:25],
            "attention_weights": weights[:25],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain_lime")
def explain_lime(req: ExplainRequest):
    try:
        exp = explainer.explain_instance(
            req.text,
            predict_probabilities_for_lime,
            num_features=10,
            num_samples=300,
            labels=(req.target_class_idx,)
        )
        attributions = exp.as_list(label=req.target_class_idx)
        return {"attributions": attributions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain_attention")
def explain_attention(req: ExplainRequest):
    try:
        cleaned = preprocess_text(req.text)
        inputs = tokenizer(
            cleaned, return_tensors="pt",
            max_length=MAX_LEN, truncation=True, padding="max_length"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_attentions=True,
            )

        if out["attentions"] is None:
            raise HTTPException(
                status_code=500,
                detail="Attention weights not available"
            )

        # Last layer, average across heads
        last_layer_attention = out["attentions"][-1][0]    # (num_heads, seq_len, seq_len)
        mean_attention = last_layer_attention.mean(dim=0)  # (seq_len, seq_len)

        # Get real tokens (non-padding)
        real_n = int((inputs["input_ids"][0] != tokenizer.pad_token_id).sum())
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[:real_n]
        attn_matrix = mean_attention[:real_n, :real_n].cpu().numpy().tolist()

        return {
            "tokens": tokens,
            "attention_matrix": attn_matrix,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain_word_attention")
def explain_word_attention(req: ExplainRequest):
    """
    Returns CLS-row attention merged at the word level.
    This shows which words the model focused on most when classifying.
    Mirrors the training notebook's predict_with_attention() logic.
    """
    try:
        cleaned = preprocess_text(req.text)
        inputs = tokenizer(
            cleaned, return_tensors="pt",
            max_length=MAX_LEN, truncation=True, padding="max_length"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_attentions=True,
            )

        if out["attentions"] is None:
            raise HTTPException(
                status_code=500,
                detail="Attention weights not available"
            )

        # CLS-row attention from last layer, averaged across all heads
        attn = out["attentions"][-1][0].mean(0)[0].cpu().numpy()  # (seq_len,)

        # Get real (non-padding) tokens
        real_n = int((inputs["input_ids"][0] != tokenizer.pad_token_id).sum())
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[:real_n]
        attn = attn[:real_n]
        attn = attn / (attn.max() + 1e-9)  # normalize to [0, 1]

        # Merge subword tokens into whole words
        words, weights = [], []
        for tok_s, w in zip(tokens, attn):
            clean_t = tok_s.replace("\u2581", "").replace("##", "")
            # Skip special tokens
            if clean_t in ("<s>", "</s>", "[CLS]", "[SEP]", "[PAD]", ""):
                continue
            # Subword continuation: merge with previous word
            if words and (tok_s.startswith("##") or not tok_s.startswith("\u2581")):
                words[-1] += clean_t
                weights[-1] = max(weights[-1], float(w))
            else:
                words.append(clean_t)
                weights.append(float(w))

        # Also get the prediction result for context
        pred_result = predict(req.text)

        return {
            "words": words[:30],
            "weights": weights[:30],
            "offensive_prob": pred_result["offensive_prob"],
            "is_toxic": pred_result["is_toxic"],
            "language": pred_result["language"],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
