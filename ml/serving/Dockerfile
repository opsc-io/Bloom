# Inference container for Mental Health Classification
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference server
COPY serve.py .

# Expose port
EXPOSE 8080

# Run server
CMD ["python", "serve.py"]
