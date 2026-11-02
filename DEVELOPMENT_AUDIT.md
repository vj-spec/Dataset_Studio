# Dataset Studio development audit and phase record

Audit date: 2026-11-01

## Environment

- OS: Windows 11 (10.0.26300), x64.
- System `python`: 3.11.9; project virtual environment: `venv`, Python 3.14.0.
- Installed project packages include PySide6 6.12.0, PyTorch 2.15.0+cpu, Ultralytics 8.4.197, OpenCV 5.0.0, PyYAML 6.0.3, NumPy 2.5.4 and psutil 7.2.2.
- PyTorch reports no CUDA runtime and no CUDA device. The application can still use CPU. NVIDIA utility discovery and GPU presence could not be confirmed from this shell.
- Available RAM: 15.71 GiB. Free space on the project drive: 11.04 GiB at audit time.
- `requirements.txt` covers the current application modules. No package changes are needed for Phase 1.

## Existing architecture and features

| Area | Status | Evidence / limits |
| --- | --- | --- |
| App entry and UI | Present | `main.py` starts a PySide6 `DrillAnnotationStudio` from `studio.py`; several workspace panels and dialogs are kept in a few large modules. |
| Dataset handling | Present | `dataset_manager.py` recognizes raw and split YOLO layouts, supported images, classes, and YOLO/COCO/VOC/LabelMe/CVAT annotations. |
| Image annotation | Present | `studio.py` draws, moves, resizes, assigns, and deletes rectangular boxes for the selected image. |
| Dataset browser | Partial | Prior to Phase 1 it was a `QFileSystemModel` tree with single-image preview, without class filters or annotation overlay. |
| Class management | Partial | Add and delete existed. Delete rewrote YOLO TXT labels directly without recovery and did not provide a transactional operation history. |
| Dataset preparation | Present | `DatasetManager.prepare_yolo` validates labels and writes to an explicitly selected separate output folder. |
| Training | Present | `yolo_trainer.py` uses Ultralytics from a Qt worker thread, supports stop/resume checkpoint flow. Other architecture adapters are missing. |
| Inference | Present | `project_views.py` includes model loading/inference UI and camera source management. |
| Hardware | Partial | `hardware.py` detects NVIDIA GPUs through `nvidia-smi` and PyTorch CUDA; the UI has a psutil-based performance dialog. CUDA is unavailable in this environment. |
| Cameras | Partial | Camera input sources and multi-model inference exist; there is no independent CCTV AI configuration per camera. |
| Experiments / distributed | Missing | No experiment database, model registry, remote worker controller or worker agent was found. |
| Dependencies / docs | Present | `requirements.txt`, `README.md`, and `USER_MANUAL.md`; no project-specific tests or migration framework were found in the initial file inventory. |

## Dataset structure observed

The project contains `dataset/images` and `dataset/labels` with train/val subfolders, `dataset/classes.txt`, and `dataset/data.yaml`; two local `.pt` checkpoints, an annotated video, and a virtual environment are also present. Dataset images and model binaries were not opened during the source audit. Existing user data was not changed by the audit.

## Existing workflows

- Dataset open detects directory structure and class names from `classes.txt`, `data.yaml`, or annotation metadata.
- Annotation reads/writes are dispatched through `annotation_formats.py`; YOLO preparation copies image/label pairs to a separately selected output.
- Training uses an Ultralytics model and the open dataset's `data.yaml`; it stores runs under the dataset.
- Inference supports image/video/camera sources and multiple model formats, subject to optional runtimes.
- Hardware reporting includes CPU/RAM/disk/network telemetry and NVIDIA GPU details when supported utilities are available.

## Gaps and risks found

- Dataset image browsing and class-based search were absent.
- Existing class deletion could leave a partially rewritten dataset if an I/O error occurred and had no undo.
- Validation was geared toward training readiness; statistics and per-class distribution were absent.
- Class names are positional IDs. Any rename/reorder must keep `classes.txt`, `data.yaml`, and format-specific metadata consistent.
- COCO/CVAT are shared annotation documents, so edits to one image must preserve records for all other images.
- The GUI currently performs most dataset scans synchronously; large datasets may make class statistics and full-dataset edits take time.
- Environment is CPU-only with 11.04 GiB free at audit time. CUDA training is unavailable until a compatible driver and CUDA PyTorch build are installed; no automatic install was performed.
- No extra dependencies are required for the Phase 1 work below.

## Proposed architecture and roadmap

Keep the current application structure and extract focused services incrementally. Preserve `DatasetManager` as the dataset-facing API, add transactional/recoverable operations there, and add dedicated UI modules where the current files are already large. Use SQLite for experiments and project metadata only in later phases. Keep remote workers outbound-connected with enrollment and revocable credentials when distributed work begins.

1. Phase 1: dataset/class browser, safe class and annotation operations, validation/statistics, recovery, documentation, focused verification.
2. Phase 2: dataset merge/mapping and separate fine-tuning preparation.
3. Phase 3: tested architecture adapters and unified training.
4. Phase 4: experiment tracking, comparison, evaluation and model registry.
5. Phase 5: local resource-aware parallel scheduling.
6. Phases 6–7: authenticated worker foundation, remote jobs, artifact transfer and recovery.
7. Phase 8: independent CCTV camera profiles and inference scheduling.
8. Phase 9: integration, backups, security/performance review and end-to-end verification.

## Phase 1 file plan and status

- Modify `dataset_manager.py`: summary statistics, class-scoped annotation edits, class ID remapping, and recoverable snapshots/undo.
- Modify `annotation_formats.py`: preserve explicit empty CVAT image records and edit existing LabelMe sidecars in place.
- Add `class_manager.py`: dedicated class list and safe operations UI.
- Replace `dataset_browser.py`: searchable thumbnail grid, filename/class/annotation filters, annotated preview and image metadata.
- Modify `studio.py`: open Class Manager and pass the dataset manager to the browser.
- Add `tests/test_dataset_operations.py`: focused temporary-dataset checks for selective edits, ID mapping, and undo.
- Update `USER_MANUAL.md` with Phase 1 operations and recovery behavior.
- No requirements changes.

Checklist:

- [x] Phase 1 completed and verified.

### Completed

- [x] Environment and source audit.
- [x] Searchable thumbnail browser, composable class combination/exclusion filters, dimension and format ranges, annotation count ranges, preview overlays and image metadata.
- [x] Multi-box preview, batch class reassignment and deletion with confirmation.
- [x] Explicit keep/delete/review choices for images that become unannotated.
- [x] Dataset-wide validation report with affected paths, suggestions and a confirmation-gated missing-label correction.
- [x] Class distribution and dataset summary counts.
- [x] Recoverable class add/rename/reorder/delete, class replacement, class-specific annotation removal, selected-image annotation removal and selected-image deletion.
- [x] Append-only snapshot history, sequential undo, restart recovery, and partial-failure recovery.
- [x] User manual update; no dependency changes.

## Phase 2 — Dataset Merge Studio

### Completed

- [x] Inspect and reuse `DatasetManager` as the dataset reader/validator; YOLO-only merge service, deterministic class remapping, staging, output validation and metadata are implemented.
- [x] Dedicated merge interface with multiple sources, source/image include controls, previews, distributions, explicit class mapping/exclusion, target class creation/rename/merge, duplicate policy, split controls and confirmation.
- [x] Background merge worker with progress, cancellation, staging cleanup on failure, and source immutability checks.
- [x] Focused merge tests for multiple datasets, differing IDs/names, explicit mapping, class exclusion/merge, remapping, duplicates, filename conflicts, split integrity, failure cleanup, and source preservation.
- [x] Phase 1 regression suite and merge dialog construction smoke check pass.
- [x] Fine-tuning preparation workflow inspects pretrained detection model classes in a background worker, requires explicit dataset class mapping/exclusion, retains/adds target classes, and writes a separately validated dataset without changing model weights or source data.
- [x] Pre-merge exact duplicate manual review displays image previews and supports keep-one or keep-all choices.
- [x] UI-level test covers source/image inclusion and explicit-mapping preview validation.
- [x] User manual updated for merge, fine-tuning preparation and current format scope.
- [x] Acceptance audit against the Phase 2 criteria; no Phase 3 work started.

### Pending

- Phase 3 is completed and regression-verified; see the Phase 3 record below.
- Phase 4 is in implementation and verification. Phase 5 has not started.

### Blocked

- None.

### Tested

- [x] Phase 2 merge service: multi-source YOLO loading, explicit mapping across differing IDs/names, class exclusion/merging, ID remapping, duplicate policies, deterministic split integrity, same-stem filename conflicts, source preservation, and failed-copy staging cleanup.
- [x] Merge and fine-tuning dialog construction smoke checks; fine-tuning data preparation uses the same covered merge/remapping/validation service.
- [x] Full regression suite: 25 tests passed (5 Phase 2, 20 Phase 1).
- [x] Python compilation for new and modified Python modules.
- [x] `git diff --check` (no whitespace errors; Git reports existing line-ending notices).

- [x] Python compile checks for changed modules.
- [x] Temporary-dataset operations, composable filter behavior, report checks, and undo/recovery tests.
- [x] UI startup and UI module import smoke checks.
- [x] YOLO dataset preparation regression.
- [x] TrainingWorker flow with a stub model.
- [x] One-epoch real CPU YOLO26 training regression on a temporary synthetic dataset.
- [x] Model loader flow with a stub model; bundled `yolo26n.pt` CPU inference smoke test.
- [x] COCO image deletion preserves other image records and undo restores the source JSON.
- [x] `git diff --check` reports no whitespace errors (only Git line-ending notices).

## Phase 5 — Unified Compute Management & Worker Infrastructure

### Architecture and decisions

- The existing PySide6 application and Phase 4 experiment SQLite database remain the application, UI, and persistent store. Compute tables are added with a backward-compatible schema migration; no second application or database manager was introduced.
- `UnifiedComputeManager` owns the operating mode, local scheduler, worker services, and shared resource profile. `ComputeCenterPanel` is embedded in the existing workspace. Controller, Worker, and Hybrid modes can be selected in settings; listeners and outbound connections require explicit user action.
- `LocalJobScheduler` owns local admission, priority ordering, duplicate active-run prevention, resource reservations, cancellation, and state persistence. Local YOLO/RT-DETR jobs submitted from the existing training workflow run through the existing adapters in an isolated child process. Child communication is a fixed JSON event/control protocol; it does not accept arbitrary commands.
- The controller accepts only enrollment, authentication, heartbeat, and disconnect messages over TLS. Enrollment tokens are one-use and stored as hashes. Per-worker credentials are salted PBKDF2 hashes in SQLite; the worker-side credential is protected using Windows DPAPI. Approval, disable, revoke, and heartbeat state are persisted.
- Hardware discovery uses existing system interfaces and optional `psutil`/`nvidia-smi` support. CPU-only execution remains available. Conservative defaults cap GPU use to one job per device when reliable VRAM data is unavailable.

### Completed checklist

- [x] Three operating modes are persisted and mode switching is blocked while local jobs are active.
- [x] CPU/RAM/disk/GPU/framework discovery is shown through a background hardware monitor; CPU-only and mocked-GPU cases are covered.
- [x] Resource profiles validate and persist. Local job reservations, CPU concurrency, conservative GPU admission, duplicate submissions, cancellation and release are scheduler responsibilities.
- [x] SQLite worker enrollment, authentication, approval, disable/revoke, heartbeat and disconnect handling are implemented.
- [x] A TLS controller listener and outbound TLS worker client are integrated into the existing app; no remote command or remote training operation is accepted.
- [x] Existing training submission is routed through the local scheduler and child-process adapter bridge; lifecycle state is reconciled on restart as failed when interrupted.
- [x] Compute Center overview/hardware, workers, local queue, and settings are present in the existing navigation.
- [x] User manual and this audit updated.

### Verification and limitations

- Root cause: parent shutdown closed the child's text stdin while a child control thread was blocked reading that same stream. On Windows, closing the stream can wait on the read lock and deadlock. Synchronous stdout reading also prevented timeout enforcement if output stalled.
- The bridge now uses a job-scoped control file for safe-stop signaling, a dedicated output-drain thread feeding a queue, a bounded process timeout, terminate/kill escalation, terminal-event and exit-code validation, and cleanup of temporary config/control files. Child events include UTC timestamps; parent progress/log/result signals retain the existing public contract.
- Full regression suite: 67 tests passed, 0 failed. This includes Phase 1–4 regressions, scheduler/Compute Center tests, existing CPU YOLO/RT-DETR tests, isolated child CPU YOLO/RT-DETR training, child failure propagation, timeout cleanup, cancellation cleanup, and sequential scheduler jobs.
- The isolated YOLO integration verifies that the parent receives progress and the actual checkpoint result path. RT-DETR child training returns the checkpoint path and epoch progress. Child failure reaches the parent with a nonzero exit code; timeout terminates the child; cancellation removes its control file and releases the process handle.
- No real TLS handshake or physical multi-computer connection was tested in this environment. Worker network protocol tests use a mocked TLS stream.
- Remote worker job execution, dataset/artifact transfer, remote scheduling/allocation, checkpoint return, and automatic network discovery are deferred to Phase 6. Worker mode does not yet execute remote training.
- DPAPI credential storage is Windows-only. GPU monitoring may lack utilization/VRAM when vendor tools are absent. CUDA is unavailable in the audited CPU-only environment.
- A local CPU job can be safely stopped through the current adapter stop protocol, but OS-level forced termination after an unresponsive training library may leave an interrupted job; restart reconciliation records it as failed.

### Phase 5 testing checklist

- [x] Compute manager and UI tests, including sequential scheduler execution.
- [x] Full regression suite (67 passed, 0 failed; Phase 1–4 and isolated CPU YOLO/RT-DETR included).
- [x] Parent-to-child isolated lifecycle, progress, result, error, timeout, cancellation, and cleanup checks.
- [x] Python compilation and `git diff --check` (no whitespace errors; line-ending notices only).

### Failed

- An initial test run used the host's system temporary directory, outside the writable workspace; fixtures were moved into the project workspace.
- Atomic replacement of recovery history failed on the Windows/OneDrive workspace. History now uses an append-only event log; restart and undo tests cover it.

### Phase 2 limitations

- YOLO TXT detection datasets are the only mergeable format; COCO/VOC/LabelMe/CVAT remain preview-only.
- Duplicate detection is exact SHA-256 only; perceptual near-duplicate detection is not implemented.
- Fine-tuning preparation inspects Ultralytics detection models and writes a separate dataset/config. It does not start training, and adaptation support depends on the model architecture.
- Grouping an entire source dataset can limit split distribution or make requested splits impossible when there are too few independent source groups; preflight reports this and refuses to publish.

## Phase 3 — Multi-Architecture Training Framework

### Completed

- [x] Existing YOLO `TrainingWorker` retained behind `TrainingAdapter`; RT-DETR has a separate Ultralytics worker and adapter.
- [x] Adapter-aware config validation, lifecycle and epoch metrics, checkpoint/resume hooks, background launch, hardware compatibility reporting, and persistent per-run JSON metadata.
- [x] CPU smoke training for YOLO26 and RT-DETR ran in the local Python 3.14 / PyTorch 2.15 CPU-only environment; checkpoints were written. CUDA was not tested.
- [x] Transformer and trusted local custom-PyTorch extension points are defined; they are not presented as implemented training architectures.
- [x] Training Studio supports both tested detection adapters and model-aware parameters.

## Phase 4 — Experiment Manager, Model Comparison & Registry

### Completed in implementation

- [x] Added SQLite metadata service in `experiment_database.py`, using the existing `%LOCALAPPDATA%\IntrinoTech\YOLODatasetStudio` app-state directory by default. `DATASET_STUDIO_STATE_DIR` can override it for isolated installs/tests.
- [x] Added `experiment_manager.py` for experiment metadata, legacy JSON import, metric extraction, YOLO/RT-DETR evaluation, and measured inference benchmark workers.
- [x] Added tables for schema version, experiments, epoch metrics, artifacts, registered model versions, evaluations, and benchmarks. Checkpoints and datasets stay on disk; records retain their paths.
- [x] Connected `TrainingManager` to optional SQLite persistence while retaining the Phase 3 JSON metadata format.
- [x] Added Experiments dashboard, run details/epoch chart, searchable/status-filtered history, notes, metadata-only deletion confirmation, output-folder opening, resume, evaluation and benchmarking.
- [x] Added configurable experiment/model comparison table and single-metric comparative chart, with missing metrics and compatibility caveats shown.
- [x] Added model registration/version uniqueness, preferred version, explicit copy action, evaluation and benchmark actions.
- [x] Fixed the existing application-startup regression: `studio.py` now imports the `QCheckBox` used by the Training tab.
- [x] Updated `USER_MANUAL.md` with database location/backup, metrics, model comparison and registry behavior.

### Testing checklist

- [x] Full regression suite: 51 tests passed across Phase 1–4, including dataset operations/merge, application startup, YOLO/RT-DETR adapter checks and CPU training smoke tests.
- [x] Phase 4 persistence, restart, metric normalization, failed status, safe metadata deletion/reimport tombstone, model version registry, mocked evaluation, batched benchmark, comparison UI, and dashboard/registry UI tests.
- [x] Python compilation: 23 Python files compiled.
- [x] `git diff --check`: no whitespace errors; Git emitted only existing line-ending notices.

### Known scope

- Evaluation and inference benchmarking are currently implemented for Ultralytics YOLO and RT-DETR detection checkpoints only. Other model families need an architecture-specific adapter.
- Benchmark results are stored with device, resolution, batch size, iteration counts, latency, and throughput; hardware is not standardized across machines, so results may not be directly comparable.
- The SQLite DB is a local index. Existing per-run JSON remains for compatibility. Missing artifact paths do not invalidate or delete history.
- Legacy JSON import is available when a dataset is opened and through `ExperimentManager.import_legacy_metadata`; it is not a full-project filesystem crawl on every launch. Deleted IDs are tombstoned to prevent those old JSON files from reappearing in the UI.
- Training-run artifacts and checkpoint files remain managed by their output folders; deleting experiment metadata never removes those files.
- Evaluation and benchmark execution paths are unit-tested with mocked Ultralytics model calls. No real checkpoint evaluation/benchmark was run as part of this phase; existing real one-epoch YOLO and RT-DETR CPU training smoke tests passed. No model downloads or dependency changes were made.

## Phase 6 — Distributed Training Engine (2026-11-02)

### Implemented in the current workspace

- [x] Extended the existing authenticated TLS heartbeat protocol with bounded, structured remote updates and controller polling. Only approved, enabled workers receive offers; no shell or Python command operation exists.
- [x] Added `distributed_training.py` with strict `TrainingConfig` field/architecture validation; YOLO and RT-DETR use the existing adapter IDs and isolated child process.
- [x] Added content-hashed dataset manifests, streaming ZIP creation, 512 KiB transfer blocks, worker cache integrity checks, safe path extraction, worker-local YAML generation, disk-space preflight, and explicit verified-cache cleanup.
- [x] Added controller-side job reservation, automatic/specific worker assignment, one active remote job per worker, worker suitability explanations, sequence-checked progress, and explicit remote cancellation requests.
- [x] Added worker-side remote execution, progress/available metrics reporting, transfer of selected existing `.pt`/`.yaml` model files, and verified return of `best.pt`, `last.pt`, `results.csv`, and `args.yaml` when present.
- [x] Added central experiment creation/update and artifact associations. Returned models are not added to Model Registry automatically.
- [x] Extended SQLite schema to version 3 with sequenced remote job updates; existing experiment/job/worker tables remain in place.
- [x] Added **Guided remote training…** in Compute Center, compatibility reasons, worker resource/version/architecture fields, job progress/checkpoint display, and explicit dataset-cache cleanup.
- [x] Added [PHASE_6_GUIDE.md](PHASE_6_GUIDE.md) and [QUICK_START.md](QUICK_START.md), and updated README, Project Guide, Workflow, and User Manual.

### Verification on this computer

- Environment: Windows workspace, Python 3.11.9, PySide6 6.12.0, PyTorch 2.15.0+cpu, Ultralytics 8.4.197; CUDA unavailable. Ultralytics settings and app state were redirected to writable project folders for tests because the global user settings path was denied by this environment.
- Phase 5 baseline before Phase 6 changes: full suite passed **67 tests** in 162.7 seconds, including real one-epoch CPU YOLO/RT-DETR runs and isolated process tests.
- Phase 6 tests cover manifest determinism, archive/file integrity, traversal rejection, cache reuse and cleanup, runtime YAML paths, resumable receiver, remote job authorization/assignment/progress/cancellation, TLS message exchange, mocked YOLO and RT-DETR remote adapter paths, model/dataset block transfer, and checkpoint receipt.
- Final full regression suite: **82 tests passed, 0 failed** in 300.4 seconds on this computer. This includes all 67 Phase 5 baseline tests, real CPU YOLO and RT-DETR isolated-child smoke tests, and 15 new Phase 6 protocol/transfer/worker tests. Python compilation passed for all changed runtime and test modules. `git diff --check` reported no whitespace errors; Git printed only its existing LF-to-CRLF working-copy notices.
- Remote YOLO and RT-DETR adapter execution was additionally exercised through a mocked worker/controller broker; no remote pretrained-model downloads were required. These automated and local CPU checks do not establish physical network behavior.

### Known limitations and remaining work

- Physical two-computer acceptance testing was not possible in this environment. TLS handshakes/network behavior are mocked. GPU remote training was not run.
- Remote progress updates and artifacts are persisted centrally, but a worker-process restart is not automatically reconciled with the controller. A controller restart marks active jobs interrupted/failed rather than retrying them. Long network outages can exceed the per-request timeout; no automatic job retry or worker reassignment occurs.
- If a remote worker restarts mid-job, its dataset cache and checkpoints remain on disk, but this release does not automatically resume that process or retrieve an unfinished checkpoint. The user must inspect the worker and decide whether to cancel/resume/re-submit.
- Compatibility checks require reported architecture, RAM/disk availability when present, requested device support, and PyTorch/Ultralytics version presence. They do not enforce a full exact Python/PyTorch/CUDA/Ultralytics compatibility matrix.
- Large dataset manifests are sent as one bounded controller offer. Offers exceeding the protocol size limit fail with a useful job error; paged manifests are not implemented. Dataset and artifact payload bytes themselves are transferred in bounded chunks without loading whole files into memory.
- Disconnected partial transfer data can remain in the worker cache root. **Clear worker dataset cache…** removes content-addressed dataset-version folders, including corrupted cache entries; it intentionally does not delete partial transfers, model cache files, training outputs, or checkpoints.
- No automatic model registration occurs. Benchmarking/evaluation of a returned checkpoint remains an explicit user action.
- A remote job that was active when the worker application terminated must be reviewed manually before resubmission to avoid duplicate work.
