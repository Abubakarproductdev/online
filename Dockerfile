# Use a lightweight Python base image
FROM python:3.10-slim

# --- THE FIX: INSTALL MISSING GRAPHICS LIBRARIES ---
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the server
CMD ["python", "server.py"]