# Guidance for AI coding agents

Quick, focused notes so an AI can be immediately productive in this repository.

- Project purpose: a small Fashion-MNIST classifier built with PyTorch, packaged into images for preprocessing and inference, and composed into a SageMaker pipeline (`src/sagemaker_pipeline.py`).

- Key components (read these files first):
  - `preprocessing/preprocess.py` — preprocessing job: downloads FashionMNIST, resizes to 28x28, normalizes and saves `train_dataset.pt` to the processing output dir (`SM_OUTPUT_DATA_DIR` / `/opt/ml/processing/output`).
  - `src/train.py` — training script: expects `train_dataset.pt` in `SM_CHANNEL_TRAINING` (default `/opt/ml/input/data/training`), trains `FashionCNN`, saves `model.pth` to `SM_MODEL_DIR` (`/opt/ml/model`).
  - `src/model.py` — model definition and `CLASSES` constant used by inference.
  - `inference_docker/app.py` — FastAPI inference service that loads `/opt/ml/model/model.pth` and exposes `/predict`.
  - `src/sagemaker_pipeline.py` — orchestrates preprocessing, training, model registration and serverless deployment. Uses ECR images constructed from env vars `ACCOUNT_ID` and `AWS_REGION`.

- Data & artifact conventions (important):
  - Preprocessed dataset filename: `train_dataset.pt` (PyTorch dataset object). Training loads this file directly with `torch.load()`.
  - Model artifact filename: `model.pth` (state_dict saved with `torch.save(model.state_dict(), ...)`). Inference uses `model.load_state_dict(torch.load(...))`.
  - Environment variables mirror SageMaker conventions: `SM_CHANNEL_INPUT`, `SM_OUTPUT_DATA_DIR`, `SM_CHANNEL_TRAINING`, `SM_MODEL_DIR` are used throughout. Honor these when running locally or inside containers.

- Local development tips / run patterns (concrete):
  - Run preprocessing locally (produces `train_dataset.pt`):
    - Easiest: cd into repository root and run `python preprocessing/preprocess.py`. The script uses sensible defaults (`/opt/ml/processing/input` and `/opt/ml/processing/output`) but you can set env vars to point to local folders.
  - Run training locally (module import nuance):
    - The `train.py` in `src/` imports `model` as a sibling module (`from model import FashionCNN`). Either run it from the `src` directory: `cd src && python train.py`, or run from repo root with `PYTHONPATH=src python src/train.py` so imports resolve.
  - Run inference locally: the FastAPI app expects `/opt/ml/model/model.pth` to exist; run `cd inference_docker && python app.py` after ensuring the model file is present (or adjust path). The app uses CPU and `map_location='cpu'`.

- Packaging & pipelines:
  - There are Dockerfiles for preprocessing (`preprocessing/Dockerfile`) and inference (`inference_docker/Dockerfile`). The SageMaker pipeline composes ECR image URIs from `ACCOUNT_ID` and `AWS_REGION` and re-uses the same image for preprocess and inference in the example.
  - The SageMaker PyTorch estimator (in `sagemaker_pipeline.py`) sets `source_dir='src'` and `entry_point='train.py'`: that makes intra-src imports (`from model import FashionCNN`) resolve inside the training container.

- Project-specific patterns an AI should follow when editing code:
  - Keep artifact names consistent: `train_dataset.pt` and `model.pth` are hard-coded in multiple places—when changing names, update all usages (`preprocess.py`, `src/train.py`, `inference_docker/app.py`).
  - Use SageMaker env-vars instead of hard-coded paths so the code runs both locally and in SageMaker.
  - Use CPU-friendly code in `inference_docker/app.py` (the app loads models to CPU); don't assume GPU there.

- Integration points and external dependencies to be careful with:
  - Torch/torchvision versions: requirements are split per-folder; when adding features check the matching versions in `preprocessing/requirements.txt`, `src/requirements.txt`, and `inference_docker/requirements.txt`.
  - ECR/SageMaker: `src/sagemaker_pipeline.py` expects `ACCOUNT_ID` and `AWS_REGION` env vars to build image URIs. Modifying pipeline steps requires understanding how the image is built and pushed (not included here).

- Quick examples to include in edits (copy into PR descriptions or commits):
  - If you change the dataset filename: update `preprocessing/preprocess.py` (torch.save path), `src/train.py` (torch.load path) and `src/sagemaker_pipeline.py` if you change S3 output names.
  - If you change model save/load behavior from `state_dict` to full model, update inference loading in `inference_docker/app.py` accordingly.

If anything in this guidance is incomplete or you want me to expand a section (examples for debugging, tests to add, or CI steps using `buildspec.yml`), tell me which area to elaborate and I'll update this file.
