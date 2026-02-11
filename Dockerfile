FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including Tesseract OCR
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy model (optional but recommended)
RUN python -m spacy download en_core_web_sm || echo "Spacy model download skipped"

# Copy application files
COPY train.py .
COPY app.py .

# Train models during build
RUN python train.py

# Verify models were created
RUN ls -la models/ && \
    test -f models/health_score_model.keras && \
    test -f models/risk_classification_model.keras && \
    echo "All models verified successfully"

# Create necessary directories
RUN mkdir -p /tmp/uploads

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Use gunicorn with appropriate settings for ML workload
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "300", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "app:app"]
