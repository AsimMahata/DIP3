FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirement files and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API backend
COPY api.py .

# Hugging Face Spaces map the internet to port 7860 internally
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
