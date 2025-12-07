# Custom training container for Vertex AI
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# Copy training script
COPY train_vertex.py .

# Set the entrypoint
ENTRYPOINT ["python", "train_vertex.py"]
