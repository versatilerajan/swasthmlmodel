FROM python:3.10-slim

WORKDIR /app

# Install basic runtime dependencies for OpenCV / numpy / TF
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install PDF + OCR stack (tesseract + poppler)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        libtesseract-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (caching layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY train.py .
COPY app.py .

# Run training (happens at build time)
RUN python train.py

# Basic verification step — fails build if models are missing
RUN ls -la models/ && \
    test -f models/health_score_model.keras && \
    test -f models/risk_classification_model.keras && \
    echo "✓ All models verified successfully"

# Create temp upload folder
RUN mkdir -p /tmp/uploads

# Environment setup
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV TF_CPP_MIN_LOG_LEVEL=2

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "300", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
