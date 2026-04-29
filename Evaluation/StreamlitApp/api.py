
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import time
import os
import re
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from lime.lime_text import LimeTextExplainer
from huggingface_hub import hf_hub_download

try:
    import emoji
except ImportError:
    emoji = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

import uvicorn

app = FastAPI(title="Multilingual Abuse Detection API")

MODEL_ID = "artyuishere/multi-lang-abuse-detection"
MAX_LEN = 128
THRESHOLD = float(os.environ.get("THRESHOLD", 0.6))

def get_language_mappings(model_id):
    try:
        lang_file = hf_hub_download(repo_id=model_id, filename="languages.json")
        with open(lang_file, "r", encoding="utf-8") as f:
            langs = json.load(f)
        return {l.lower(): i for i, l in enumerate(langs)}
    except Exception as e:
        print(f"Warning: Could not load languages.json from hub: {e}")
            
    num_langs = 3 # default
    try:
        heads_path = hf_hub_download(repo_id=model_id, filename="heads.pt")
        heads = torch.load(heads_path, map_location="cpu", weights_only=True)
        if "weight" in heads["language_head"]:
            num_langs = heads["language_head"]["weight"].shape[0]
    except Exception as e:
        print(f"Warning: Could not read shape from heads.pt: {e}")
            
    langs = ["english", "hinglish", "banglish", "kannada", "malayalam", "tamil", "bengali", "hindi"]
    if num_langs > len(langs):
        for i in range(len(langs), num_langs):
            langs.append(f"language_{i}")
    return {l: i for i, l in enumerate(langs[:num_langs])}

LANG_TO_IDX = get_language_mappings(MODEL_ID)
IDX_TO_LANG = {v: k for k, v in LANG_TO_IDX.items()}
NUM_LANGUAGES = len(LANG_TO_IDX)


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
    "bc": "behanchod",
    "mc": "madarchod",
    "bhenchod": "sisterfucker",
    "mkc": "maa ki chut",
    "tmkc": "teri maa ki chut",
    "bck": "behenchod",
    "pk": "pagal",
    "gnd": "gandu",
    "chtiya": "chutiya",
    "bchd": "behenchod",
    "bsdk": "bhosdike",
    
    "bcoda": "bokachoda",
    "mgi": "magi",
    "bal": "baal",
    
    "otha": "ommala",
    "ommale": "ommaala",
    "punda": "pundai",
    "tp": "thevidiya paiyan",
    "mairu": "myre",
    "gotha": "gothaa",
    
    "myr": "myre",
    "thendi": "beggar",
    "thayoli": "motherfucker",
    "pooran": "asshole",
    
    "shata": "shata",
    "loosu": "loosu",
    
    "aiz": "aizavadya",
    "lavdya": "lavdya",
    "bhadkya": "bhadkhau",
    "raand": "raand",
}


def preprocess_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", " ", str(text))
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if emoji is not None:
        text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    abbrevs = dict(ENG_ABBREVS)
    abbrevs.update(DESI_ABBREVS)
    tokens = text.lower().split()
    text = " ".join(abbrevs.get(t, t) for t in tokens)
    return text.strip()


class MultiTaskAbuseDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(MODEL_ID, attn_implementation="eager")
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


print("Initializing model...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading tokenizer from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

onnx_session = None
if ort is not None:
    try:
        print(f"Downloading ONNX model and weights from {MODEL_ID}...")
        hf_hub_download(repo_id=MODEL_ID, filename="model.onnx.data")
        onnx_model_path = hf_hub_download(repo_id=MODEL_ID, filename="model.onnx")
        onnx_session = ort.InferenceSession(
            onnx_model_path,
            providers=["CPUExecutionProvider"]
        )
        print("ONNX model loaded for inference [OK]")
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")

_pytorch_model = None

def get_pytorch_model():
    global _pytorch_model
    if _pytorch_model is not None:
        return _pytorch_model
        
    print(f"Lazy loading PyTorch model from {MODEL_ID}...")
    model = MultiTaskAbuseDetector()
    try:
        heads_path = hf_hub_download(repo_id=MODEL_ID, filename="heads.pt")
        heads = torch.load(heads_path, map_location="cpu", weights_only=True)
        model.offensive_head.load_state_dict(heads["offensive_head"])
        model.language_head.load_state_dict(heads["language_head"])
        model.dropout.load_state_dict(heads["dropout"])
        print("Custom heads loaded from HF Hub [OK]")
    except Exception as e:
        print(f"WARNING: heads.pt not found on Hub -- using random heads! {e}")

    model.to(DEVICE)
    model.eval()
    _pytorch_model = model
    print(f"PyTorch model loaded on {DEVICE} [OK]")
    return _pytorch_model

explainer = LimeTextExplainer(class_names=["Clean", "Offensive"])


def predict_pytorch(text: str):
    cleaned = preprocess_text(text)
    inputs = tokenizer(
        cleaned, return_tensors="pt",
        max_length=MAX_LEN, truncation=True, padding="max_length"
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    model = get_pytorch_model()
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
    offensive_logit = outputs[0][0, 0]  
    language_logits = outputs[1][0]      

    off_prob = float(1.0 / (1.0 + np.exp(-offensive_logit)))
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
    if onnx_session is not None:
        return predict_onnx(text)
    return predict_pytorch(text)


def predict_probabilities_for_lime(texts):
    start_time = time.time()
    
    cleaned_texts = [preprocess_text(t) for t in texts]
    
    if onnx_session is not None:
        inputs = tokenizer(
            cleaned_texts, return_tensors="np",
            max_length=MAX_LEN, truncation=True, padding="max_length"
        )
        
        try:
            batch_size = 16
            all_logits = []
            
            for i in range(0, len(texts), batch_size):
                ort_inputs = {
                    "input_ids": inputs["input_ids"][i:i+batch_size].astype(np.int64),
                    "attention_mask": inputs["attention_mask"][i:i+batch_size].astype(np.int64),
                }
                outputs = onnx_session.run(None, ort_inputs)
                all_logits.append(outputs[0][:, 0])
                
            offensive_logits = np.concatenate(all_logits)  # shape: (N,)
            off_probs = 1.0 / (1.0 + np.exp(-offensive_logits))
            all_probs = np.column_stack((1.0 - off_probs, off_probs))
            
        except Exception as e:
            print(f"  [LIME] Full Batch ONNX failed ({e}), falling back to iteration...")
            all_probs = []
            for i in range(len(texts)):
                ort_inputs = {
                    "input_ids": inputs["input_ids"][i:i+1].astype(np.int64),
                    "attention_mask": inputs["attention_mask"][i:i+1].astype(np.int64),
                }
                outputs = onnx_session.run(None, ort_inputs)
                offensive_logit = outputs[0][0, 0]
                off_prob = float(1.0 / (1.0 + np.exp(-offensive_logit)))
                all_probs.append([1.0 - off_prob, off_prob])
            all_probs = np.array(all_probs)
            
        elapsed = time.time() - start_time
        print(f"  [LIME Summary] Evaluated {len(texts)} samples via ONNX in {elapsed:.2f}s")
        return all_probs
    else:        probs = []
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"  [LIME Batch] PyTorch Inference: {i}/{len(texts)} samples...")
            result = predict(text)
            off = result["offensive_prob"]
            probs.append([1.0 - off, off])
            
        elapsed = time.time() - start_time
        print(f"  [LIME Batch] Evaluated {len(texts)} samples via PyTorch in {elapsed:.2f}s")
        return np.array(probs)

class PredictRequest(BaseModel):
    text: str

class ExplainRequest(BaseModel):
    text: str
    target_class_idx: int = 1  # 0=clean, 1=offensive

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

    try:
        raw_text = req.text

        preprocessed = preprocess_text(raw_text)

        tokens_raw = tokenizer(preprocessed, add_special_tokens=True)
        raw_input_ids = tokens_raw["input_ids"]
        raw_tokens = tokenizer.convert_ids_to_tokens(raw_input_ids)

        tokens_padded = tokenizer(
            preprocessed, return_tensors="pt",
            max_length=MAX_LEN, truncation=True, padding="max_length"
        )
        padded_ids = tokens_padded["input_ids"][0].tolist()
        attention_mask = tokens_padded["attention_mask"][0].tolist()
        num_real = sum(attention_mask)
        num_padded = MAX_LEN - num_real

        inputs = {k: v.to(DEVICE) for k, v in tokens_padded.items()}
        model = get_pytorch_model()
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

        attn = out["attentions"][-1][0].mean(0)[0].cpu().numpy()
        attn_real = attn[:num_real]
        attn_real = attn_real / (attn_real.max() + 1e-9)

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
            "raw_text": raw_text,
            "preprocessed_text": preprocessed,
            "raw_tokens": raw_tokens,
            "raw_token_ids": raw_input_ids,
            "num_raw_tokens": len(raw_input_ids),
            "max_length": MAX_LEN,
            "num_real_tokens": num_real,
            "num_padded_tokens": num_padded,
            "attention_mask_sample": attention_mask[:20],  
            "offensive_logit": round(off_logit, 4),
            "offensive_prob": round(off_prob, 4),
            "is_toxic": off_prob > THRESHOLD,
            "language_logits": [round(l, 4) for l in lang_logits],
            "language_probs": {IDX_TO_LANG[i]: round(float(lang_probs_raw[i]), 4) for i in range(NUM_LANGUAGES)},
            "language": IDX_TO_LANG[int(np.argmax(lang_probs_raw))],
            "attention_words": words[:25],
            "attention_weights": weights[:25],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain_lime")
def explain_lime(req: ExplainRequest):
    try:
        print(f"\n{'='*50}")
        print(f"[LIME Explainer] Starting analysis...")
        print(f"[LIME Explainer] Target: '{req.text[:50]}...'")
        start_lime = time.time()
        
        exp = explainer.explain_instance(
            req.text,
            predict_probabilities_for_lime,
            num_features=10,
            num_samples=100,  
            labels=(req.target_class_idx,)
        )
        attributions = exp.as_list(label=req.target_class_idx)
        
        total_time = time.time() - start_lime
        print(f"[LIME Explainer] SUCCESS! All 100 samples evaluated.")
        print(f"[LIME Explainer] Total elapsed time: {total_time:.2f} seconds")
        print(f"{'='*50}\n")
        
        return {"attributions": attributions}
    except Exception as e:
        print(f"[LIME Explainer] ERROR: {e}")
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

        model = get_pytorch_model()
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

        last_layer_attention = out["attentions"][-1][0]    
        mean_attention = last_layer_attention.mean(dim=0)  

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

    try:
        cleaned = preprocess_text(req.text)
        inputs = tokenizer(
            cleaned, return_tensors="pt",
            max_length=MAX_LEN, truncation=True, padding="max_length"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        model = get_pytorch_model()
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

        attn = out["attentions"][-1][0].mean(0)[0].cpu().numpy()  # (seq_len,)

        real_n = int((inputs["input_ids"][0] != tokenizer.pad_token_id).sum())
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[:real_n]
        attn = attn[:real_n]
        attn = attn / (attn.max() + 1e-9)  # normalize to [0, 1]

        words, weights = [], []
        for tok_s, w in zip(tokens, attn):
            clean_t = tok_s.replace("\u2581", "").replace("##", "")
            if clean_t in ("<s>", "</s>", "[CLS]", "[SEP]", "[PAD]", ""):
                continue
            if words and (tok_s.startswith("##") or not tok_s.startswith("\u2581")):
                words[-1] += clean_t
                weights[-1] = max(weights[-1], float(w))
            else:
                words.append(clean_t)
                weights.append(float(w))

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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting uvicorn server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

