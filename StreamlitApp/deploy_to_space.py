import os
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.environ.get("HF_TOKEN")

if not TOKEN:
    print("Error: HF_TOKEN not found in .env!")
    exit(1)

api = HfApi(token=TOKEN)

SPACE_ID = "artyuishere/multilingual-abuse-api"

print(f"\n1. Provisioning Cloud Server Space: {SPACE_ID}...")
try:
    create_repo(
        repo_id=SPACE_ID, 
        repo_type="space", 
        space_sdk="docker", 
        exist_ok=True, 
        token=TOKEN
    )
except Exception as e:
    print(f"Error creating space: {e}")
    exit(1)

print("2. Uploading backend files to your new server...")
try:
    # Use absolute paths like we did for the model upload to avoid folder issues
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    api.upload_file(
        path_or_fileobj=os.path.join(BASE_DIR, "api.py"), 
        path_in_repo="api.py", 
        repo_id=SPACE_ID, 
        repo_type="space"
    )
    api.upload_file(
        path_or_fileobj=os.path.join(BASE_DIR, "requirements.txt"), 
        path_in_repo="requirements.txt", 
        repo_id=SPACE_ID, 
        repo_type="space"
    )
    api.upload_file(
        path_or_fileobj=os.path.join(BASE_DIR, "Dockerfile"), 
        path_in_repo="Dockerfile", 
        repo_id=SPACE_ID, 
        repo_type="space"
    )
    print("\n   -> Files successfully pushed!")
except Exception as e:
    print(f"Error uploading files: {e}")
    exit(1)

print("\n" + "="*60)
print(f"3. SUCCESS! Your backend is now building live!")
print(f"   Check your dashboard here: https://huggingface.co/spaces/{SPACE_ID}")
print("============================================================")
