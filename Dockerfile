FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached on rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# HF Spaces requires port 7860
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
