# Project File Guide

This guide maps the main files in this repository to their responsibilities. The application is a single PySide6 program launched by `main.py`; the other Python files are modules loaded by that app or its internal training child process.

## Start-up and main application

| File | What it contains / does |
| --- | --- |
| `main.py` | Application entry point. Creates `QApplication`, opens `DrillAnnotationStudio`, and starts the Qt event loop. |
| `studio.py` | Main window, workspace navigation, dataset annotation/preparation/training UI, and integration with experiments, inference, and Compute Center. |
| `config.py` | Shared app paths, supported image extensions, model family/version presets, and class-name normalization. |
| `app_state.py` | Persists and restores the user's application/session state. |

## Dataset and annotation modules

| File | What it contains / does |
| --- | --- |
| `dataset_manager.py` | Dataset discovery, class definitions, validation, preparation, recoverable annotation/class operations, and undo snapshots. |
| `annotation_formats.py` | Readers/writers and conversion helpers for YOLO, COCO, Pascal VOC, LabelMe, and CVAT annotation data. |
| `dataset_browser.py` | Dataset thumbnail browser, composed class/dimension/format/count filters, annotation preview, and batch box actions. |
| `class_manager.py` | Class list, class-level operations, dataset statistics, validation report, and recovery controls. |
| `dataset_merge.py` | Merge planning/execution services: source discovery, explicit class mapping, duplicate handling, filename safety, split generation, staging, validation, and merge metadata. |
| `merge_studio.py` | Dataset Merge Studio UI for selecting sources/images, reconciling classes, previewing, validating, and confirming a merge. |
| `fine_tuning.py` | Fine-tuning dataset preparation and explicit mapping between dataset classes and pretrained model classes. |

## Training, compute, and experiment modules

| File | What it contains / does |
| --- | --- |
| `training_framework.py` | Training configuration, YOLO and RT-DETR adapters/workers, adapter selection, metadata lifecycle, and the parent-side isolated training bridge. |
| `yolo_trainer.py` | Existing Ultralytics YOLO training worker and model-training callbacks reused by the YOLO adapter. |
| `compute_training_process.py` | Internal child-process entry point. Loads a fixed local adapter/configuration, reports structured events, and watches a per-job control file for safe stop requests. It is not a separate application. |
| `compute_manager.py` | Hardware/resource profiles, local job scheduling/reservations, operating-mode manager, TLS worker enrollment/heartbeat services, and credential protection. |
| `compute_security.py` | Built-in controller CA/certificate lifecycle, SAN and key validation, DPAPI-protected controller keys, and public-CA one-time enrollment package validation. Called by Compute Center setup and enrollment flows. |
| `distributed_training.py` | Remote job validation and broker, worker suitability, SHA-256 dataset manifests, streaming archives, verified cache, resumable file receiver, remote adapter execution, and artifact receipt. Called by `compute_manager.py` and `compute_studio.py`; receives validated configs, dataset folders, and authenticated messages, and produces persistent job/experiment updates and verified files. |
| `compute_studio.py` | Compute Center UI: local hardware, workers, jobs, and compute settings. |
| `hardware.py` | CPU, memory, disk, GPU, CUDA, and training-runtime discovery. Optional tools may leave some telemetry unavailable. |
| `experiment_database.py` | SQLite schema and persistence for experiments, metrics, artifacts, model registry, compute profiles, jobs, workers, sequenced remote job progress, hashed enrollment-token state, job archival, revoked-worker removal, and security events. Schema version 5 migrates earlier databases in place. |
| `experiment_manager.py` | Experiment services for training history, evaluation, benchmarking, model versions, and artifact references. |
| `experiment_studio.py` | Experiments, comparison, evaluation/benchmark, and model registry UI panels. |

## Inference and video

| File | What it contains / does |
| --- | --- |
| `project_views.py` | Detection/inference UI, model loading, image/video/camera sources, class filters, and model exchange/export controls. |
| `video_annotator.py` | Separate manual video annotation window with tracks/keyframes, project save/load, annotated-video export, and YOLO dataset export. |
| `camera_sources.json` | Saved camera/source configuration used by the application. |

## Tests, documentation, and local assets

| Path | What it contains / does |
| --- | --- |
| `tests/` | `unittest` coverage for dataset operations, merge, experiments, training adapters, Compute Center, worker protocol, child-process training, and mocked distributed transfer/execution. |
| `requirements.txt` | Base Python package requirements. Optional ONNX/TensorRT runtimes are not included. |
| `README.md` | Project overview, installation, scope, and links to the detailed guides. |
| `WORKFLOW.md` | Short task-oriented workflow through dataset prep, training, inference, Compute Center, and video annotation. |
| `USER_MANUAL.md` | Detailed end-user instructions and troubleshooting. |
| `DEVELOPMENT_AUDIT.md` | Implementation decisions, phase checklists, test evidence, and known limitations. |
| `models/README.txt` | Notes about the local model files/directory. |
| `dataset/` | Example/local dataset content. Treat it as user data; do not assume it is disposable. |
| `*.pt`, `annotated_video.mp4` | Local model checkpoints and media assets; these are data, not application source code. |

## Remote training data flow

1. `ComputeCenterPanel` reads the current dataset and training values from the main `studio.py` window through the guided remote-training dialog.
2. `UnifiedComputeManager` passes the request to `RemoteJobBroker` in `distributed_training.py`. The broker validates a `TrainingConfig`, hashes dataset files, stores a remote job and central experiment record, and prepares a streaming ZIP archive.
3. An approved outbound worker polls on the existing TLS heartbeat channel. The broker checks reported architecture, device, RAM/disk, runtime versions, approval, and current job count before reserving and offering a job.
4. `RemoteWorkerAgent` requests bounded archive/model blocks. `DatasetCache` verifies each listed file and stages it under a content-derived ID. A worker-local YAML copy points at the staged directory; the original dataset and archived YAML remain unchanged.
5. The agent validates the architecture-specific config and starts `IsolatedTrainingWorker`, which uses `compute_training_process.py` and the existing YOLO or RT-DETR adapter. Progress and available metrics return as sequenced messages.
6. The worker sends allowlisted artifacts in verified blocks. The controller publishes them only after complete size and SHA-256 validation, then updates the central experiment record. The model registry is unchanged unless a user explicitly registers a model.

Remote jobs are not dispatched by `LocalJobScheduler`; that scheduler remains responsible for local training. Worker connections are outbound from the worker. The controller listener is TLS-protected, starts on loopback by default, and must be explicitly bound to a reachable interface. Long outages and worker restarts require user review; no automatic training retry is performed. Resource suitability is calculated from fresh heartbeat values, job requests, reservations and safety margins.

## Useful commands

Run the app from the project directory:

```powershell
python main.py
```

Run all automated tests:

```powershell
python -m unittest discover -s tests -v
```

For isolated test state, set `DATASET_STUDIO_STATE_DIR` to a test-only directory before invoking the test command. Avoid deleting files in dataset, model, media, or temporary-looking folders unless you have verified their origin and contents.
