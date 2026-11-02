# Dataset Studio Workflows

This is the short path through the application. For field-by-field instructions and format details, see [USER_MANUAL.md](USER_MANUAL.md).

## A. Prepare and annotate a dataset

1. Start the app with `python main.py` and choose **Open Dataset…** or **New Dataset…**.
2. Confirm the class order in **Annotate → Classes**. YOLO class IDs are positional: the first class is ID 0, the next is ID 1, and so on.
3. Add or review image boxes in **Annotate**. Use **Browse Files & Preview…** to filter by dimensions, format, annotation count, and class combinations. Use **Class Manager…** for class-wide operations and undo.
4. Run **Validate dataset**. Correct missing files, invalid IDs, boxes, and class-definition issues before preparing or training.
5. Choose the next route:
   - For one source dataset, use **Prepare** and select a separate output folder.
   - For multiple compatible YOLO sources, open **Dataset Merge Studio…**, include/exclude datasets or images, map every class explicitly, review duplicates, configure splits, inspect the preview, then confirm the merge.
6. Open the prepared output and verify its classes and splits. Keep original sources unchanged as a backup.

## B. Prepare a fine-tuning dataset

1. Open **Fine-Tuning Prep…** and select a pretrained detection model and a compatible merged dataset.
2. Review the model's existing classes and explicitly map each source class to a retained class, new target class, or exclusion.
3. Validate the mapping and select a new output folder. Fine-Tuning Prep writes a separate dataset/configuration; it does not change source data or weights.
4. If the number or ordering of classes changes, the model detection head may need adapting or reinitializing during training. Support depends on the chosen architecture.

## C. Train and review a model

1. Open the prepared dataset and go to **Dataset & Training → Train**.
2. Choose **YOLO** or **RT-DETR**, select an available model version or local weights/configuration, set the dataset, device, experiment name, output folder, and supported parameters.
3. Review the configuration and start **Train Model**. Training runs in the background; Compute Center schedules local jobs and reports progress/state.
4. Use **Training Results** or **Experiments** to inspect metrics, logs, and checkpoints. Use **Model Comparison** to compare selected runs and **Model Registry** to preserve named model versions.
5. Evaluation and benchmarking are supported for YOLO and RT-DETR detection checkpoints. Results depend on dataset, device, resolution, batch size, and runtime versions.

## D. Run inference or export

1. Open **Detect**, load a model, choose an image/video/camera or supported URL, and set confidence, image size, device, and class filters.
2. Run detection and save the annotated output if needed.
3. Use **Model Exchange** to export supported checkpoints to ONNX or TensorRT when the corresponding optional runtime is installed and compatible.

## E. Use Compute Center

1. Open the **Compute Center** workspace tab and select **Controller**, **Worker**, or **Hybrid** in Compute Settings.
2. Configure conservative CPU/GPU, memory, disk, and thread limits. Mode changes are blocked while local jobs are active.
3. For Controller or Hybrid setup, configure a TLS certificate/private key and start the listener explicitly. Create a one-time enrollment token for a trusted worker.
4. For Worker or Hybrid setup, set the controller address and trusted CA, provide the enrollment token on first connection, and connect outbound. Approve the new worker on the controller.
5. Confirm the worker is **Online**, approved, enabled, and reports its architecture/runtime capabilities. The **Guided remote training…** button in **Job Queue** uses the open training dataset and settings from the Train panel.
6. Choose automatic assignment or a specific compatible worker, review the staging explanation, and submit. The controller hashes and transfers the dataset; the worker verifies it, runs the existing isolated YOLO/RT-DETR adapter, and sends progress.
7. Watch **Progress** and **Latest checkpoint** in Job Queue or open **Experiments** to see the central experiment record. Checkpoints and available training files are copied back and verified. The worker keeps its local copy; dataset-cache cleanup is a separate, confirmed action.
8. If the worker disconnects, reconnect it and review the last status. The app does not automatically duplicate or retry a job after a long outage or worker restart. Request cancellation before deciding whether to submit again.

For a first-time two-computer walkthrough and network troubleshooting, follow [QUICK_START.md](QUICK_START.md) and [PHASE_6_GUIDE.md](PHASE_6_GUIDE.md). Physical multi-computer operation has not been verified in this workspace.

## F. Annotate a video

1. Open **Video Annotator** from the top bar and load a video.
2. Add classes, draw a box, create an object track, and add keyframes as the object moves.
3. Save the project JSON if you need to continue later. Export an annotated video or a YOLO dataset to a separate output folder.

## Safe working habits

- Keep an untouched copy of source datasets and pretrained weights.
- Use a separate, empty destination for preparation, merge, fine-tuning, and exports.
- Confirm class mappings by meaning; IDs and similar names are never sufficient grounds for merging classes.
- Review validation results before training. Use dataset undo for supported destructive edits.
- Back up the app-state database and output folders separately. Close Dataset Studio before copying SQLite files.
