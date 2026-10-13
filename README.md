# YOLO Dataset & Training Studio

Compact PySide6 desktop application for:
- creating a new dataset explicitly
- opening an existing raw annotation dataset
- opening an existing prepared YOLO dataset
- image annotation with YOLO bounding boxes
- class management
- YOLO dataset preparation to a user-selected output directory
- model selection across YOLO26, YOLO12, YOLO11, YOLOv10, YOLOv9, YOLOv8, YOLOv6 and YOLOv5
- CPU / CUDA GPU selection
- NVIDIA GPU detection
- optional CUDA-enabled PyTorch installation into the active Python environment
- YOLO training in a worker thread

## Safety behavior

The application never creates a `dataset/` folder at startup.

It never recursively deletes an output directory.

Preparation refuses to use the source dataset, the application root, or the Python `venv` as the output.

Only the explicitly selected YOLO output folder is created/modified by preparation.

## Dataset formats

Raw:
```
project/
  images/
  labels/          # or annotations/
  classes.txt      # optional when data.yaml provides names
```

Prepared YOLO:
```
project/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

If `classes.txt` is missing:
1. `data.yaml` names are used when available and `classes.txt` is created.
2. Otherwise class IDs are detected from label files and the UI asks for class names.
3. If no labels/classes exist, you can add classes manually before annotation.

## Install

Use the active virtual environment:

```
python -m pip install -r requirements.txt
python main.py
```

## CUDA note

The app can detect NVIDIA GPUs through `nvidia-smi` and check whether the current PyTorch build can use CUDA.

The optional CUDA installation installs a CUDA-enabled PyTorch wheel into the current Python environment. It does NOT install or update the NVIDIA display driver or the system CUDA Toolkit. A compatible NVIDIA driver is still required.

The included CUDA installer uses the official PyTorch CUDA 13.2 wheel index and installs torch 2.13.0 + torchvision 0.28.0 into the active environment. If your NVIDIA driver does not support that runtime, use the PyTorch selector to choose a compatible build instead.

## Model support note

The UI lists the requested Ultralytics YOLO families. Exact pretrained checkpoint availability depends on the installed Ultralytics release. YOLOv6, for example, may require a YAML architecture because current Ultralytics documentation does not publish YOLOv6 `.pt` checkpoints. A local `.pt`/`.yaml` can always be selected through Browse Model.

Official documentation:
https://docs.ultralytics.com/models
https://docs.ultralytics.com/quickstart
