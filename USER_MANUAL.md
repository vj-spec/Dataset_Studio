# YOLO Dataset & Training Studio — User Manual

This manual covers the desktop app from first launch through dataset preparation, model training, detection, model export, and video annotation.

## 1. What the app does

The studio has three main work areas:

- **Dataset & Training** — open or create a dataset, draw and manage image boxes, prepare a YOLO dataset, and train a model.
- **Training Results** — inspect completed runs, view metrics, save checkpoints, and send a trained `best.pt` model to detection.
- **Detect** — run a model on an image, video, camera, or supported URL; save annotated output.

The **Video Annotator** is a separate window for manually annotating objects over time and exporting an annotated video or YOLO image dataset.

## 2. Install and start

Open a terminal in the project folder and install the listed dependencies into the Python environment you plan to use:

```powershell
python -m pip install -r requirements.txt
python main.py
```

The app supports CPU use. For CUDA training or inference, you need a compatible NVIDIA driver and a CUDA-enabled PyTorch build. The training tab reports whether CUDA is available. **Check / Install CUDA PyTorch** can install the configured PyTorch CUDA wheels into the active Python environment; restart the app after that installation completes.

## 3. Understand the dataset types

### Raw annotation dataset

A raw dataset is a collection of images and labels. The app recognizes these object-detection formats:

| Format | Expected layout | Notes |
| --- | --- | --- |
| YOLO | `images/` with matching `labels/*.txt` files, or `annotations/*.txt` | One normalized box row per object. This is the format used for training after preparation. |
| COCO | `images/` and `annotations/instances_*.json` | Reads category names and bounding boxes from the COCO JSON. |
| Pascal VOC | `JPEGImages/` and `Annotations/*.xml` | Reads object names and `<bndbox>` coordinates. |
| LabelMe-style JSON | Matching per-image JSON files beside the images or under `annotations/` | Matches by embedded image path first, then by relative folder/name. Reads shapes and points as bounding rectangles. |
| CVAT | `images/` and `annotations.xml` | Imports image-mode bounding boxes and frame-based track boxes. Use **Open CVAT ZIP…** for the ZIP export. |

The standard YOLO layout is:

```text
my_dataset/
  images/
    photo_001.jpg
    photo_002.jpg
  labels/
    photo_001.txt
    photo_002.txt
  classes.txt
```

For YOLO labels, the label directory may instead be named `annotations/`. Each label filename must match its image filename, with a `.txt` extension. For example, `photo_001.jpg` uses `photo_001.txt`.

The annotation editor is for rectangular bounding boxes. When it reads LabelMe polygons or other shapes, it displays their enclosing rectangle. Saving those shapes from this editor writes rectangles and does not preserve the original polygon outline or segmentation mask.

The app can create the folder structure with **New Dataset…**. Then copy image files into its `images/` folder using File Explorer and click **Refresh** in the app. Supported image types include JPG/JPEG, PNG, BMP, WebP, and TIFF. Use **Browse Files & Preview…** to open a file-browser window for the current dataset and click an image to preview it.

For a CVAT ZIP, click **Open CVAT ZIP…**. The archive is extracted into the app’s local import cache and opened as a dataset; the original ZIP is not modified.

### Prepared YOLO dataset

A prepared dataset has train/validation folders and a `data.yaml` file:

```text
my_yolo_dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

Use **Open Dataset…** and select the dataset root folder. Existing prepared YOLO datasets can go directly to training if `data.yaml` and the expected image splits are present.

## 4. Standard workflow: annotate, prepare, train

### Step 1 — Open or create a dataset

- Click **Open Dataset…** and choose a dataset root folder, or choose **New Dataset…** and select a parent folder and dataset name.
- If the dataset has a `classes.txt` file, those class names are loaded. For a prepared YOLO dataset, the app can also read class names from `data.yaml`.
- If labels contain class IDs but no class names are available, the app asks for the name of each class ID. Keep the names in the same order as the IDs.
- If there are no classes yet, add them from the **Annotate** tab before labeling images.

### Step 2 — Add or review classes

In **Dataset & Training → Annotate → Classes**:

- **Add…** adds a class. Names are normalized to lowercase with underscores between words (for example, `Safety Guard` becomes `safety_guard`).
- **Delete** removes the selected class and adjusts later class IDs in the label files. This rewrites labels, so use it only when you intend to change the dataset’s class schema.

Finalize the class list before training. Class IDs are positional: class 0 is the first name in `classes.txt` or `data.yaml`, class 1 is the second, and so on.

### Step 3 — Annotate images

Select an image from the left-hand image list. Existing labels are displayed when present.

- Drag on the image to draw a bounding box around an object.
- Select a box, choose a class in the **Classes** selector, and click **Assign Class to Box**. You can also press **1–9** to assign one of the first nine classes.
- Drag a box to move it; drag a box handle to resize it.
- Use **Delete Box** or the `Delete` key to remove the selected box. **Clear All** removes every box on the current image after confirmation.
- Use **Prev** / **Next**, or the `A` / `D` keys, to move between images. The mouse wheel zooms; middle-button drag pans; **Fit** or `0` resets the view.
- Use **Save Annotation** or `Ctrl+S` to write the current labels. The image list shows whether an image has a label file.

If an image contains no target objects, create an empty matching `.txt` label file. Dataset preparation requires a label file for every image, including negative/background images.

When the first valid annotation is saved in a dataset with no recognized labels, the app asks whether to create annotation files and then asks which format to use. It uses that format for later saves in the same dataset. Choose YOLO for the simplest training flow; choose a format your other tools require when exchanging annotations.

### Step 4 — Prepare the YOLO dataset

1. In the **Prepare** tab, choose a new output folder that is empty or does not exist yet. Do not select the source dataset, the app folder, or its Python environment.
2. Set the **Validation split** (default 0.2, meaning 20% validation).
3. Click **Prepare YOLO Dataset**.

The app checks that every image has a readable label file, class IDs are valid, and normalized box coordinates are between 0 and 1. If it reports problems, fix the source annotations and prepare again. It will not delete a non-empty output folder; choose a different empty folder instead. On success, the app opens the prepared dataset.

### Step 5 — Configure and start training

In **Dataset & Training → Train**:

- Select a **Family** and **Preset**, or browse for a local `.pt` checkpoint or supported model YAML file. Preset availability depends on the installed Ultralytics version.
- Select a **Device**: Auto, CPU, or an available CUDA GPU.
- Set **Epochs** (how many training passes to run), **Image size** (training resolution), and **Batch**. Batch `-1` asks Ultralytics to choose a batch size automatically.
- Click **Train YOLO**. A progress window reports status and epoch metrics.

Training runs are saved under the prepared dataset’s `runs/train/` folder. The **Training Results** tab shows the run metrics and offers **Save best.pt…**, **Save last.pt…**, and **Use best.pt for detection**.

To continue an interrupted or paused run, reopen the prepared dataset and click **Resume from last.pt** when enabled. The studio saves `last.pt` at a safe epoch boundary when training is stopped through the app’s close/reset flow.

## 5. Detect with a model

Open the **Detect** tab. You can load one or more compatible models; use the checkboxes beside each loaded model to enable or disable it.

### Load a model

- Click **Add model files…** and choose one or more `.pt`, `.onnx`, or `.engine` files, or type semicolon-separated file paths and click **Load model and inspect**.
- **View checkpoint details…** shows metadata exposed by the model and loader.
- ONNX models require ONNX Runtime in the active environment. A CPU runtime can be installed with `python -m pip install onnxruntime`. For supported NVIDIA configurations, install the compatible `onnxruntime-gpu` package instead.
- TensorRT `.engine` files require a compatible NVIDIA GPU and TensorRT runtime. Engines can be tied to the image dimensions and TensorRT version used to build them. If an engine rejects the selected inference size, use the engine’s supported size or export a new engine at the needed size.

### Choose an input and options

- **Source type**: Image, Video, or Camera.
- For Image and Video, choose **File** and use **Browse…**, or choose **URL** and paste a direct image URL or supported video page URL. Video page URLs use `yt-dlp`; some sites or streams may not be downloadable.
- For Camera, choose a configured camera. Use **Manage…** to add a named USB camera index or network stream source.
- **Video output** is optional. Choose a path to save the annotated video; leave it blank to preview without writing an output file.
- **Confidence** controls the minimum detection confidence. Lower it to show more possible detections; raise it to reduce weaker detections.
- **Image size (px)** sets the square inference size. Presets include 320, 416, 512, 640, 768, 960, 1024, 1280, 1536, 1920, 2048, 2560, 3072, and 3840; a custom integer from 32 through 4096 can be typed. Larger sizes may help with small objects but use more memory and may run more slowly. Start at 640 if unsure.
- **Device** selects Auto, CPU, or a CUDA device where available.
- **AI inference FPS** caps inference rate for camera input.
- **Show bounding boxes** toggles overlays; **Opacity** adjusts their blend.
- Under **Per-model classes**, choose which model classes to run. Uncheck all classes for a model to run all of its classes.

### Run and save detection

1. Choose and load the model(s).
2. Choose the source type, source, image size, confidence, device, and optional output path.
3. Click **Run detection**. The right side shows the annotated result and inference timing.
4. For video, use **Pause**, the speed menu, and the seek bar while detection runs. Click **Stop** to end video or camera detection.
5. For an image, click **Save annotated image…** to save the displayed result. For video, set **Video output** before starting if you want an annotated file written.

The selected detection options and model paths are saved with the studio session and restored when the project is reopened, as long as those model files remain in place.

## 6. Export a model to ONNX or TensorRT

1. Click **Model Exchange** in the top bar.
2. Browse for a `.pt` checkpoint.
3. Choose **ONNX** or **TensorRT engine**, set the export **Image size**, then click **Export model**.
4. Wait for the export result and note the output path shown in the dialog.
5. Load the exported file from the **Detect** tab.

TensorRT export and use require a compatible NVIDIA/TensorRT setup. Export failures usually mean the required runtime or a compatible GPU installation is missing. Use the size selected during export as the starting point for inference with a static engine.

## 7. Video Annotator: track objects through a video

Click **Video Annotator** in the top bar to open the separate annotation window. This tool manually records object tracks across video frames; it does not automatically detect or track objects.

### Create a tracked annotation

1. Click **Open Video…** and select a video file.
2. Click **Add class…**, enter a class name, and select it in the **Class** list.
3. Navigate with **Play/Pause**, the timeline, frame-step buttons (`−10`, `−1`, `+1`, `+10`), or the `Space` and arrow keys.
4. Drag on a frame to draw a box around one object.
5. Click **Create new track from box**. Each separate object occurrence should have its own track ID.
6. Move to later frames as the object changes position. Draw its updated box and click **Add keyframe at this frame** (or **Update selected track at this frame**). Intermediate boxes are interpolated between keyframes.
7. Use **Set selected track start here** / **Set selected track end here** to limit the frames where a track appears. Change its class or delete a track with the matching controls.
8. Use **Save Project…** to save a `.json` annotation project. The project refers to its source video, so keep the video available; if it has moved, the app asks you to locate it when loading.

### Export video or a dataset

- **Export Annotated Video…** writes a rendered video with the tracked boxes and labels.
- **Export YOLO Dataset…** writes frames with tracks as YOLO box labels. Only frames containing annotated tracks are exported. Select an output folder intended for this export.
- The annotator also autosaves its current working session locally. Use **Save Project…** for a portable named project file.

## 8. Training Results, saved work, and reset

- **Training Results** lists runs for the currently open dataset. Select a run to inspect metrics and save its checkpoints.
- The **Performance** window shows CPU, memory, GPU, disk, and network information when the related system tools are available.
- Dataset annotations and prepared data are written to their dataset folders. Training results and weights are stored in `runs/` beside the prepared YOLO dataset.
- The studio restores the last dataset, selected tab, inference setup, and other session settings automatically.
- **Reset Everything…** clears saved app/session state and current unsaved workspace state after confirmation. It does not delete dataset files, saved annotations, training runs, or model checkpoints.

## 9. Keyboard shortcuts

| Shortcut | Action |
| --- | --- |
| `Ctrl+S` | Save current image annotation |
| `A` | Previous image |
| `D` | Next image |
| `Delete` | Delete selected image annotation box |
| `1`–`9` | Assign one of the first nine classes to the selected box |
| Mouse wheel | Zoom the image/video canvas |
| Middle-button drag | Pan the image annotation canvas |
| `0` | Fit/reset the image annotation view |
| `Space` | Toggle video playback in Video Annotator |
| Arrow keys | Step one frame in Video Annotator |

## 10. Common issues

- **Dataset cannot be opened:** choose the dataset root containing `images/`, or select the `images/` folder itself. Prepared datasets need `images/train`, `images/val` (or other supported split folders), or a valid `data.yaml` layout.
- **Preparation reports missing labels:** create/save a `.txt` for each image. Leave the file empty for images with no target objects.
- **Class ID out of range:** make sure `classes.txt` or `data.yaml` has one name for every ID used by the label files, in the correct order.
- **Training runs out of memory:** reduce training image size or batch size; try batch `-1`, or select CPU if necessary (CPU training may be much slower).
- **Detection runs out of memory or is slow:** lower the detection image size, reduce the number of enabled models, or use a compatible GPU.
- **ONNX model fails to load:** install a compatible ONNX Runtime in the active environment and restart the app.
- **TensorRT engine fails to load:** check that TensorRT is installed, the GPU is compatible with the engine, and the engine’s build/runtime versions and dimensions match the machine and chosen settings.
- **Camera does not open:** verify the USB index or network stream URL in **Manage…**, and close any other application currently using the camera.
- **Resume is disabled:** resume is available only when the open prepared dataset has `runs/train/weights/last.pt`.

## 11. Format recommendation

For this studio’s object-detection training workflow, use **YOLO labels** as the final prepared format because the trainer consumes the prepared YOLO dataset. Keep the original COCO, VOC, LabelMe, or CVAT export as an untouched source copy if it is your interchange/archive format. COCO JSON is a strong interchange choice when moving box annotations between different tools. Ultralytics also documents an NDJSON dataset format; that could be a useful future import option if you want one extensible metadata file or later need segmentation/keypoints support. See [Ultralytics detection dataset formats](https://docs.ultralytics.com/datasets/detect) and [Ultralytics NDJSON format](https://docs.ultralytics.com/datasets/detect/ndjson).

