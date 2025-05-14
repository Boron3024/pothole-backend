# Use PyTorch with CPU support (change to CUDA if needed)
FROM pytorch/pytorch:2.1.0-cpu

# Set working directory
WORKDIR /app

# Install system dependencies and Python packages in one layer to minimize image size
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Copy the rest of the app code
COPY . .

# Expose the port the app will run on
EXPOSE 8000

# Run FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
