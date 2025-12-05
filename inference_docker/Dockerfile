FROM public.ecr.aws/docker/library/python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/model

COPY inference_docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/fashion_model.py .
COPY inference_docker/app.py .

RUN printf '#!/usr/bin/env bash\ncd /opt/ml/model\nexec uvicorn app:app --host 0.0.0.0 --port 8080\n' > /usr/local/bin/serve && \
    chmod +x /usr/local/bin/serve
