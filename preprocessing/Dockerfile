FROM pytorch/pytorch:2.2.0-cpu
WORKDIR /opt/ml/processing
COPY preprocess.py .
RUN pip install torchvision boto3
ENTRYPOINT ["python", "preprocess.py"]
