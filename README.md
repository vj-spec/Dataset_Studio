# Dataset Studio

Dataset Studio is a local PySide6 desktop application for preparing object-detection datasets, annotating and validating images, training supported models, reviewing experiments, and running inference. It also includes dataset merging and a Compute Center for local scheduling and authenticated worker monitoring.

## Start here

- [Workflow guide](WORKFLOW.md) — recommended end-to-end tasks from dataset intake to training and inference.
- [Two-computer quick start](QUICK_START.md) — connect a controller and worker and submit a first remote run.
- [Phase 6 operating guide](PHASE_6_GUIDE.md) — detailed network setup, worker compatibility, transfers, recovery, and troubleshooting.
- [User manual](USER_MANUAL.md) — detailed UI instructions, dataset formats, recovery, training, Compute Center, and troubleshooting.
- [Project file guide](PROJECT_GUIDE.md) — what the main files and modules contain.
- [Development audit](DEVELOPMENT_AUDIT.md) — implementation scope, verification history, and limitations.

## Main capabilities

- Open or create raw annotation datasets; browse and edit bounding boxes and classes; validate data and undo supported operations.
- Prepare YOLO datasets or explicitly reconcile classes and merge compatible YOLO datasets into a new output folder.
- Train supported Ultralytics YOLO and RT-DETR models, review experiment history, compare metrics, evaluate runs, and register checkpoints.
- Run image, video, camera, and supported URL inference; export compatible models; annotate video tracks manually in the Video Annotator.
- Use Compute Center in Controller, Worker, or Hybrid mode. Enrolled workers can receive validated YOLO and RT-DETR jobs over the existing TLS connection, verify and cache transferred datasets, report progress, and return verified checkpoints and training files.

## Requirements and installation

Use a Python environment compatible with the installed PySide6, PyTorch, OpenCV, and Ultralytics versions. Current workspace checks use Windows, Python 3.11.9, PySide6 6.12.0, PyTorch 2.15.0 CPU-only, and Ultralytics 8.4.197. CUDA is not available in this environment.

From PowerShell in this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
```

If you already have a project virtual environment, activate and reuse it instead of creating another. `requirements.txt` lists the application’s base dependencies. ONNX Runtime and TensorRT are optional inference runtimes and are not included in the base requirements. GPU training requires a compatible NVIDIA driver and a CUDA-enabled PyTorch build; do not install a different PyTorch build without checking compatibility with the driver and Python version.

## Where files are saved

- Dataset annotations, classes, prepared datasets, and training outputs are saved under the selected dataset or output folders.
- Experiment and Compute Center SQLite state defaults to `%LOCALAPPDATA%\IntrinoTech\YOLODatasetStudio` on Windows. Set `DATASET_STUDIO_STATE_DIR` before launch to use a different app-state folder.
- Recovery snapshots for dataset edits are stored inside that dataset in `.dataset_studio_recovery`.
- Merge and fine-tuning outputs are written to new user-selected folders; source datasets and original model weights are not modified by those workflows.

## Safety and scope

The application does not create a dataset folder on startup or delete nonempty preparation outputs automatically. Dataset deletion and destructive annotation actions require confirmation and supported dataset edits have undo snapshots. Training and merge outputs should be saved to locations with sufficient free space.

Compute Center requires explicit user action to start a TLS controller listener or connect a worker. Built-in certificate setup creates/reuses the local CA and controller certificate; enrollment packages contain only the public CA and a one-time token. Worker enrollment also requires controller approval. Worker credentials and controller keys use Windows DPAPI. Remote jobs are limited to predefined YOLO and RT-DETR adapters and validated training parameters; workers do not accept shell commands or Python code. Dataset and artifact content is hash-verified. The listener binds to loopback by default: use Windows Firewall rules restricted to the Private network and trusted devices when enabling LAN access; never disable the firewall. This workspace has automated mocked-network verification but no physical two-computer acceptance result. See [Phase 6 operating guide](PHASE_6_GUIDE.md) and [two-computer quick start](QUICK_START.md) before connecting machines.

## Development status

Phases 1–5 are present. Phase 6 remote execution and Phase 6.1 security/reliability improvements are implemented and covered by automated tests for protocol validation, enrollment, scheduling, dataset/cache handling, mocked remote YOLO/RT-DETR execution, progress, and artifact receipt. Physical multi-computer/GPU testing, automatic process reconciliation after a worker restart, and automatic resume/retry after a long outage remain unverified or incomplete. See [development audit](DEVELOPMENT_AUDIT.md) for current evidence and limits.

## Tests

Run the suite from the project folder:

```powershell
python -m unittest discover -s tests -v
```

Some tests perform small real CPU training runs and can take a couple of minutes. To keep application state isolated during tests, set `DATASET_STUDIO_STATE_DIR` to a temporary or test-only directory before running them.
