from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from optimum.onnxruntime import ORTModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

app = FastAPI(title="Slang Language Detection API")

# Configurations
MODEL_PATH = "./model"
ONNX_MODEL_PATH = "./onnx_model"

# LABEL_MAPPING can be easily updated. Keeping it versatile. 
# Defaults to Kaggle toxic categories if 6 are present, else fallback to LABEL_X
LABEL_MAPPING = {
    "LABEL_0": "Toxic",
    "LABEL_1": "Severe Toxic",
    "LABEL_2": "Obscene",
    "LABEL_3": "Threat",
    "LABEL_4": "Insult",
    "LABEL_5": "Identity Hate"
}

print("Initializing models...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
try:
    if os.path.exists(ONNX_MODEL_PATH):
        print("Loading ONNX Model for inference...")
        inference_model = ORTModelForSequenceClassification.from_pretrained(ONNX_MODEL_PATH)
        is_onnx = True
    else:
        print("ONNX model not found, falling back to PyTorch Model...")
        inference_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        is_onnx = False
except Exception as e:
    print(f"Error loading ONNX model, falling back to PyTorch: {e}")
    inference_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    is_onnx = False

# We use the pipeline for standard inference
try:
    from optimum.pipelines import pipeline as opt_pipeline
    if is_onnx:
        pipe = opt_pipeline("text-classification", model=inference_model, tokenizer=tokenizer, top_k=None)
    else:
        pipe = pipeline("text-classification", model=inference_model, tokenizer=tokenizer, top_k=None)
except Exception:
    pipe = pipeline("text-classification", model=inference_model, tokenizer=tokenizer, top_k=None)


# For Attention Visualization we load the original PyTorch model with output_attentions=True
print("Loading base PyTorch model for attention visualization...")
pt_attention_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, output_attentions=True)
pt_attention_model.eval()

explainer = LimeTextExplainer(class_names=list(LABEL_MAPPING.values())[:inference_model.config.num_labels])

class PredictRequest(BaseModel):
    text: str
    target_class_idx: int = 0

class ExplainRequest(BaseModel):
    text: str
    target_class_idx: int = 0

def predict_probabilities(texts):
    # LIME requires a function that takes a list of strings and returns probabilities as a numpy array.
    # We pass the entire list to pipe so it batches efficiently
    results = pipe(texts)
    
    probs_array = []
    for res in results:
        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
            res = res[0]
        # Sort by label name to ensure consistent index
        res_sorted = sorted(res, key=lambda x: int(x['label'].split('_')[-1]) if 'LABEL' in x['label'] else x['label'])
        probs = [x['score'] for x in res_sorted]
        probs_array.append(probs)
    return np.array(probs_array)

@app.get("/")
def read_root():
    return {"message": "Slang Language Detection API is running"}

@app.post("/predict")
def predict(req: PredictRequest):
    res = pipe(req.text)
    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
        res = res[0]
    
    # Map labels
    mapped_res = []
    for r in res:
        label_id = r['label']
        mapped_name = LABEL_MAPPING.get(label_id, label_id)
        mapped_res.append({"label": mapped_name, "id": label_id, "score": float(r['score'])})
    return {"predictions": mapped_res}

@app.post("/explain_lime")
def explain_lime(req: ExplainRequest):
    try:
        # Generate explanations for the requested target class index
        # 500 samples provides a good balance between execution speed and stability
        exp = explainer.explain_instance(
            req.text, 
            predict_probabilities, 
            num_features=10, 
            num_samples=500,
            labels=(req.target_class_idx,)
        )
        
        # Get the feature attributions
        attributions = exp.as_list(label=req.target_class_idx)
        return {"attributions": attributions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain_attention")
def explain_attention(req: ExplainRequest):
    try:
        inputs = tokenizer(req.text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = pt_attention_model(**inputs)
        
        attentions = outputs.attentions # Tuple of attention tensors, one for each layer
        # Output shape per layer: (batch_size, num_heads, sequence_length, sequence_length)
        
        # We will return the mean attention of the last layer across all heads to make visualization easier
        last_layer_attention = attentions[-1][0] # (num_heads, seq_len, seq_len)
        mean_attention = last_layer_attention.mean(dim=0) # (seq_len, seq_len)
        
        # Get tokens to match sequence length
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        # We can return the attention matrix and tokens for the frontend to render
        return {
            "tokens": tokens,
            # Returning the attention for each token to every other token is O(N^2)
            # which we can visualize as a heatmap
            "attention_matrix": mean_attention.numpy().tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
