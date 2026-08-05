FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    PORT=10000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-liberation2 \
        libgomp1 \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-tha \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data /app/outputs /tmp/ai-cad-data \
    && chown -R appuser:appuser /app /tmp/ai-cad-data

USER appuser
EXPOSE 10000

CMD ["sh", "-c", "python app.py --server-name 0.0.0.0 --server-port ${PORT:-10000}"]
