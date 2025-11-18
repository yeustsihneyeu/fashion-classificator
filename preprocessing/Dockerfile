FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/ml/processing
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY preprocess.py .
CMD ["python", "preprocess.py"]
