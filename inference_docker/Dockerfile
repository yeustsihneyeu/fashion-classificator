FROM pytorch/pytorch:2.2.0-cpu
WORKDIR /opt/ml/model
COPY app.py .
COPY model.py .
RUN pip install fastapi uvicorn pillow torchvision
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
