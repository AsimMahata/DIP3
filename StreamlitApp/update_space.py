import os
# Suppress strict SSL checks locally
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.environ.get("HF_TOKEN")

if not TOKEN:
    print("Error: HF_TOKEN not found in .env!")
    exit(1)

api = HfApi(token=TOKEN)
SPACE_ID = "artyuishere/multilingual-abuse-api"

print("\n1. Pushing optimized api.py to the cloud server...")
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    api.upload_file(
        path_or_fileobj=os.path.join(BASE_DIR, "api.py"), 
        path_in_repo="api.py", 
        repo_id=SPACE_ID, 
        repo_type="space"
    )
    print("\n   -> Files successfully updated!")
except Exception as e:
    print(f"Error uploading files: {e}")
    exit(1)

print("\n" + "="*60)
print(f"3. SUCCESS! Your backend is now restarting with optimizations!")
print(f"   Check your dashboard here: https://huggingface.co/spaces/{SPACE_ID}")
print("============================================================")
