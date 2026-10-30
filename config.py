from pathlib import Path
import os

APP_ROOT = Path(__file__).resolve().parent
APP_STATE_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".local" / "share"))) / "IntrinoTech" / "YOLODatasetStudio"

def app_state_file(name: str) -> Path:
    APP_STATE_DIR.mkdir(parents=True, exist_ok=True)
    return APP_STATE_DIR / name

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

YOLO_FAMILIES = [
    "YOLO26", "YOLO12", "YOLO11", "YOLOv10",
    "YOLOv9", "YOLOv8", "YOLOv6", "YOLOv5",
]

YOLO_VARIANTS = ["n", "s", "m", "l", "x"]

# Common pretrained detection checkpoint names. Ultralytics decides whether
# a particular name is available in the installed package.
MODEL_PRESETS = {
    "YOLO26": [f"yolo26{v}.pt" for v in YOLO_VARIANTS],
    "YOLO12": [f"yolo12{v}.pt" for v in YOLO_VARIANTS],
    "YOLO11": [f"yolo11{v}.pt" for v in YOLO_VARIANTS],
    "YOLOv10": [f"yolov10{v}.pt" for v in YOLO_VARIANTS],
    "YOLOv9": [f"yolov9{v}.pt" for v in YOLO_VARIANTS],
    "YOLOv8": [f"yolov8{v}.pt" for v in YOLO_VARIANTS],
    "YOLOv6": [f"yolov6{v}.pt" for v in YOLO_VARIANTS],
    "YOLOv5": [f"yolov5{v}u.pt" for v in YOLO_VARIANTS],
}

TASKS = ["detect", "segment", "classify", "pose", "obb"]

def normalize_class_name(name: str) -> str:
    return "_".join(name.strip().lower().split())
