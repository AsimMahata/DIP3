# Multilingual Abuse Detection App

This application uses a custom multi-task XLM-RoBERTa-Large model to perform binary toxicity detection and language identification (English, Hinglish, Banglish). 

It features a **FastAPI backend** for efficient model inference (with ONNX support) and a **Streamlit frontend** with visual explainability tools (pipeline tracking, LIME token attribution, and Attention Heatmaps).

## Prerequisites
Before running, you must place the trained model files in the `model/` directory.

Ensure your `model/` folder contains exactly these files:
- `config.json`
- `heads.pt`
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`

## Setup & Running the App

Run these commands in your terminal inside the `StreamlitApp` directory.

### 1. Create a Virtual Environment
```powershell
python -m venv venv
```

### 2. Activate the Environment
```powershell
# On Windows:
.\venv\Scripts\activate

# On Mac/Linux:
# source venv/bin/activate
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Optimize the Model (ONNX Export)
This step converts the 2.2GB PyTorch weights into a highly optimized 3MB ONNX model for lightning-fast inference on the CPU.
```powershell
python export_onnx.py
```
*(You will see a "Export complete!" message once this finishes.)*

### 5. Start the Application
You can now start both the backend API and the frontend UI with a single command:
```powershell
.\start_app.bat
```

Once running:
- **Frontend UI:** Open your browser to `http://localhost:8501`
- **Backend API:** Running on `http://localhost:8000`

---
**Note:** To stop the app abruptly, press `Ctrl + C` in the two terminal windows that were spawned by the batch script.
