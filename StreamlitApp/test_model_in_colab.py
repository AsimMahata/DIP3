"""
Testing the isolated model on Google Colab:
1. Open a new notebook in Colab.
2. Install libraries in cell 1:
   !pip install transformers numpy onnxruntime huggingface_hub

3. Copy and paste all the code below into cell 2 and run it!
"""

import numpy as np
import time
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import onnxruntime as ort

MODEL_ID = "artyuishere/multi-lang-abuse-detection"
MAX_LEN = 128
THRESHOLD = 0.6

# The exact language mappings you use
LANGUAGES = ["english", "hinglish", "banglish"] 

print(f"1. Downloading tokenizer from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"2. Downloading ONNX data from {MODEL_ID} (This might take a minute)...")
# Download both parts so ONNX functions correctly!
hf_hub_download(repo_id=MODEL_ID, filename="model.onnx.data")
onnx_model_path = hf_hub_download(repo_id=MODEL_ID, filename="model.onnx")

print("3. Loading ONNX Inference Session...")
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
print("   -> ONNX Loaded Successfully!")

def test_prediction(text: str):
    print(f"\n--- Testing: '{text}' ---")
    start = time.time()
    
    inputs = tokenizer(
        text, return_tensors="np",
        max_length=MAX_LEN, truncation=True, padding="max_length"
    )
    
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    
    outputs = session.run(None, ort_inputs)
    
    offensive_logit = outputs[0][0, 0]
    language_logits = outputs[1][0]
    
    # Sigmoid for toxicity
    off_prob = float(1.0 / (1.0 + np.exp(-offensive_logit)))
    is_toxic = off_prob > THRESHOLD
    
    # Softmax for language
    lang_exp = np.exp(language_logits - np.max(language_logits))
    lang_probs = lang_exp / lang_exp.sum()
    lang_idx = int(np.argmax(lang_probs))
    language = LANGUAGES[lang_idx] if lang_idx < len(LANGUAGES) else f"lang_{lang_idx}"
    
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s")
    print(f"Toxicity: {off_prob*100:.1f}% -> {'TOXIC' if is_toxic else 'CLEAN'}")
    print(f"Language Detected: {language} (Confidence: {lang_probs[lang_idx]*100:.1f}%)")

print("\n4. Running Predictions...")
test_prediction("this is a clean and polite message.")
test_prediction("you are acting like a complete idiot right now")
test_prediction("tu pagal hai kya")
