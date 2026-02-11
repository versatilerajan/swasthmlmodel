FROM python:3.10-slim

WORKDIR /app

# Install minimal system dependencies — split for better caching & debugging
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgomp1 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install OCR & PDF dependencies (this was the failing part)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        libtesseract-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY train.py .
COPY app.py .

# Train models during build
RUN python train.py

# Verify models were created
RUN ls -la models/ && \
    test -f models/health_score_model.keras && \
    test -f models/risk_classification_model.keras && \
    echo "✓ All models verified successfully"

# Create necessary directories
RUN mkdir -p /tmp/uploads

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV TF_CPP_MIN_LOG_LEVEL=2

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Use gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "300", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
