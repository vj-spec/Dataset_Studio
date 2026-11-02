# Quick Start — Two Computers, One Remote Training Run

This short walkthrough connects a controller computer to a worker computer and submits one experiment. Both computers run the same Dataset Studio application.

## 1. Prepare both computers

1. Install/open Dataset Studio on both computers using the instructions in [README.md](README.md). Use the same project version and compatible Python/PyTorch/Ultralytics environments.
2. Connect both computers to a trusted network. The worker needs to reach the controller; the worker itself does not need an inbound port.
3. On the controller, have a TLS certificate and private key ready. If you do not already have a certificate trusted by the worker, ask your network administrator for a certificate/CA pair. The private key stays on the controller.
4. Make sure the controller firewall allows Dataset Studio’s configured port (default **8765**) from the trusted network. Do not expose it to the public internet.

## 2. Start the controller

1. Open Dataset Studio on Computer A and click **Compute Center**.
2. Set **Operating mode** to **Controller**, then click **Apply operating mode**.
3. In **Worker Manager**, browse to the TLS **Certificate** and **Private key** files.
4. Click **Start secure controller**.
5. Click **Create one-time enrollment token…**. Copy the token and send it privately to Computer B’s operator. The token expires and works once.

## 3. Connect the worker

1. Open Dataset Studio on Computer B and click **Compute Center**.
2. Set **Operating mode** to **Worker**, then click **Apply operating mode**.
3. Enter a **Worker name**, the controller computer’s network name or IP in **Controller host**, and port **8765** (or the port chosen on Computer A).
4. Browse to the controller certificate/CA file in **Controller trust certificate**. This tells the worker which controller to trust. Do not copy the controller private key.
5. Paste the one-time token into **Enrollment token** and click **Connect worker securely**.
6. On Computer A, select the worker row and click **Approve selected**. Wait for **Online**, **Approved**, and **Enabled**.

## 4. Open a prepared dataset and configure training

1. On Computer A, open a prepared YOLO dataset containing `data.yaml`, images, labels, and class names.
2. Open the main **Train** panel. Choose **YOLO** or **RT-DETR**, a model, and **CPU** for the simplest first run. Set a small number of **Epochs** (for example, 1) and a small image size suitable for a smoke run.
3. If you select a local model file on Computer A, Dataset Studio transfers that file to the worker. Otherwise, choose a model preset available in the worker’s Ultralytics installation.

## 5. Submit and watch the remote job

1. Go to **Compute Center → Job Queue** and click **Guided remote training…**.
2. Read the dataset/model summary. Choose **Automatic compatible worker** or select the connected worker by name.
3. Continue through the transfer explanation and review page, then click **Submit remote job**.
4. Watch **State**, **Progress**, and **Latest checkpoint** in **Job Queue**. Dataset Studio hashes dataset files, transfers them, verifies them on Computer B, and starts the existing isolated training adapter.
5. When the job completes, its experiment appears in the existing **Experiments** area. The controller receives and hash-verifies `best.pt`, `last.pt`, and available training files before listing them. The model is not registered automatically; use **Register from experiment…** if you want that.

## If something goes wrong

- If the worker is offline, check that Computer A’s controller is running, Computer B has the right host/port and certificate, and the controller firewall permits the private-network connection.
- If enrollment is rejected, create a new token on Computer A and enter it on Computer B, then approve the newly enrolled row.
- If the worker is unsuitable, check **Architectures**, **PyTorch / Ultralytics**, free RAM/disk, and GPU/VRAM in **Worker Manager**. Select CPU for a CPU-only worker.
- If dataset transfer fails, check free disk space on both computers and reconnect the worker. The original dataset is unchanged.
- If the network or worker application stops during training, inspect the worker’s local output before submitting again. Long interruptions and worker restarts are not automatically reconciled or retried in this release.

See [PHASE_6_GUIDE.md](PHASE_6_GUIDE.md) for detailed setup, transfer, recovery, and troubleshooting instructions. Automated remote execution tests use mocked networking; this project workspace has not performed physical two-computer acceptance testing.
