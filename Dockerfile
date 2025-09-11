FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# System deps (optional, helps some TF wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY export ./export
COPY server.py .

EXPOSE 8000
# 1 worker is fine for start; raise if you need more concurrency
CMD ["uvicorn","server:app","--host","0.0.0.0","--port","8000","--workers","1","--timeout-keep-alive","120"]
