"""
Instructions for testing your backend in Google Colab (Standalone Backend Test):

1. Open a new notebook at https://colab.research.google.com
2. Upload TWO files to the Colab files pane:
   - Your updated `api.py`
   - This `test_on_colab.py` file

3. In the first cell, install requirements:
   !pip install fastapi uvicorn transformers lime pydantic huggingface_hub onnxruntime httpx

4. In the second cell, run the test script:
   !python test_on_colab.py
"""

import time
from fastapi.testclient import TestClient

print("="*60)
print("1. IMPORTING API AND DOWNLOADING MODELS FROM HUGGING FACE...")
print("="*60)

# Importing api.py will automatically trigger the HF Hub downloads and model initialization!
import api
from api import app

print("\n" + "="*60)
print("2. MODELS LOADED. STARTING IN-MEMORY FASTAPI TEST SERVER...")
print("="*60)

client = TestClient(app)

def run_tests():
    texts_to_test = [
        "this is a very friendly and clean message",
        "you are a piece of trash", # Generic toxicity
        "tu pagal hai kya behenchod", # Hinglish toxicity 
    ]
    
    print("\n" + "="*60)
    print("3. HITTING /predict (Testing ONNX Base Inference)")
    print("="*60)
    for text in texts_to_test:
        print(f"\n[POST /predict] Payload: '{text}'")
        start = time.time()
        response = client.post("/predict", json={"text": text})
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            print(f"  -> Success! ({elapsed:.2f}s)")
            print(f"  -> Toxic: {result.get('is_toxic')}")
            print(f"  -> Language: {result.get('language')} ({max(result['language_probs'].values())*100:.1f}%)")
        else:
            print(f"  -> ERROR {response.status_code}: {response.text}")

    print("\n" + "="*60)
    print("4. HITTING /explain_attention (Testing Lazy PyTorch Fallback)")
    print("="*60)
    
    test_text = "I am so mad right now!"
    print(f"\n[POST /explain_attention] Payload: '{test_text}'")
    
    start = time.time()
    response = client.post("/explain_attention", json={"text": test_text, "target_class_idx": 1})
    elapsed = time.time() - start
    
    if response.status_code == 200:
        print(f"  -> Success! PyTorch model lazy-loaded correctly. ({elapsed:.2f}s)")
    else:
         print(f"  -> ERROR {response.status_code}: {response.text}")

    print("\n" + "="*60)
    print("TEST FINISHED. If there are no errors above, your backend is 100% READY for HF Spaces!")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_tests()
