# Slang & Hate Speech Detection System

This project is a high-performance web application designed to categorize text into various slang and hate speech classifications. It features a FastAPI backend powered by ONNX Runtime for extremely low-latency inference, and a Streamlit frontend providing real-time probability charting alongside LIME token attributions and Transformer attention heatmaps!

## Architecture Highlights
* **Backend:** FastAPI (Python)
* **Inference:** Hugging Face `transformers` converted to **ONNX Runtime** for blistering speeds.
* **Explainability:** Features `lime_text` to show which tokens caused the prediction, and a base PyTorch loaded path to dynamically extract multi-head Self-Attention weights straight from the transformer architecture.
* **Frontend:** Interactive Streamlit Dashboard with Plotly visuals.

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-link>
cd <your-repo-name>
```

### 2. Prepare the Model Folders
Because the ML models are extremely large, they are excluded from this repository. 
You must place your raw Hugging Face PyTorch model into a folder named `model` in the root directory.

* Create a folder named `model` inside this root directory. 
* Place your model files inside it (`config.json`, `model.safetensors`, `tokenizer.json`, etc.)

### 3. Setup Virtual Environment
It is highly recommended to isolate the dependencies. By default, the startup script expects a `venv` environment.
```bash
python -m venv venv
```

Activate the environment:
* **Windows (PowerShell):** `.\venv\Scripts\Activate.ps1`
* **Windows (CMD):** `.\venv\Scripts\activate.bat`
* **Mac/Linux:** `source venv/bin/activate`

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Export to ONNX (Low Latency Runtime)
To achieve fast inference, convert the raw PyTorch model into an ONNX model. Simply run the included export script.
```bash
python export_onnx.py
```
*This will create a new folder called `onnx_model` containing the optimized model and extracted `vocab.txt`.*

---

## 🏃‍♂️ Running the Application

### The Easy Way (Windows)
Double click the `start_app.bat` script, or run it through the terminal:
```bash
.\start_app.bat
```
This script automatically activates the `venv`, launches the FastAPI backend in the background, waits a few seconds, and brings up the Streamlit frontend.

### Manual Launch
If you are on Mac/Linux or want to launch them independently:

**1. Start the API:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**2. Start the Frontend:**
Open a separate terminal window/tab:
```bash
streamlit run app.py
```

The UI will be accessible at `http://localhost:8501`.

---

## 🧠 Adjusting Categories
If you ever retrain your classifier with less categories (ex: Binary "Hate" vs "Not Hate" or map raw ID integers), simply modify the `LABEL_MAPPING` dictionary located at the top of `api.py`. The frontend will automatically adapt to the new mappings!
