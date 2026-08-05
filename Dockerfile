FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-liberation2 \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-tha \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

RUN useradd --create-home --uid 1000 appuser

COPY --chown=appuser:appuser . /app
RUN mkdir -p /app/data /app/outputs \
    && chown -R appuser:appuser /app/data /app/outputs

USER appuser

EXPOSE 10000

CMD ["sh", "-c", "python app.py --server-name 0.0.0.0 --server-port ${PORT:-10000}"]
