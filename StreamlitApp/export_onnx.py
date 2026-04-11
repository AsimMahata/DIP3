import os
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_id = "./model"
save_dir = "./onnx_model"

print("Loading and exporting model to ONNX...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Export to ONNX
model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)

print("Saving ONNX model...")
tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
print("ONNX model saved successfully in", save_dir)
