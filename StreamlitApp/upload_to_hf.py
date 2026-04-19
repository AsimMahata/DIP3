import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.environ.get("HF_TOKEN")
if not TOKEN:
    print("Error: HF_TOKEN not found in .env file.")
    exit(1)

REPO_ID = "artyuishere/multi-lang-abuse-detection"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTORCH_MODEL_DIR = os.path.join(BASE_DIR, "model")
ONNX_MODEL_FILE = os.path.join(BASE_DIR, "onnx_model", "model.onnx")

def upload_models():
    api = HfApi(token=TOKEN)

    print(f"\nUploading PyTorch model files from '{PYTORCH_MODEL_DIR}'...")
    try:
        api.upload_folder(
            folder_path=PYTORCH_MODEL_DIR,
            repo_id=REPO_ID,
            repo_type="model",
            token=TOKEN
        )
        print("PyTorch model files uploaded successfully!")
    except Exception as e:
        print(f"Error uploading PyTorch model: {e}")

    print(f"\nUploading ONNX model structure '{ONNX_MODEL_FILE}'...")
    try:
        api.upload_file(
            path_or_fileobj=ONNX_MODEL_FILE,
            path_in_repo="model.onnx", 
            repo_id=REPO_ID,
            repo_type="model",
            token=TOKEN
        )
        print("ONNX model structure uploaded successfully!")
    except Exception as e:
        print(f"Error uploading ONNX model: {e}")

    print(f"\nUploading ONNX weights 'model.onnx.data' (this will take a while)...")
    try:
        onnx_data_path = os.path.join(BASE_DIR, "onnx_model", "model.onnx.data")
        api.upload_file(
            path_or_fileobj=onnx_data_path,
            path_in_repo="model.onnx.data", 
            repo_id=REPO_ID,
            repo_type="model",
            token=TOKEN
        )
        print("ONNX weights uploaded successfully!")
    except Exception as e:
        print(f"Error uploading ONNX model data: {e}")

    print(f"\nAll done! Your models are hosted at: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    upload_models()
