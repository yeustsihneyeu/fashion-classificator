FROM pytorch/pytorch:2.2.0-cpu-py3.10
WORKDIR /opt/ml/processing
COPY preprocess.py .
RUN pip install torchvision boto3
ENTRYPOINT ["python", "preprocess.py"]
