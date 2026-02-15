# Use a lightweight Python base image
FROM python:3.10-slim

# Install system libraries (You already did this, keep it!)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- NEW: SUPPRESS TENSORFLOW WARNINGS ---
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the server
CMD ["python", "server.py"]