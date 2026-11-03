# Quick Start — Two Computers, One Remote Training Run

This short walkthrough connects a controller computer to a worker computer and submits one experiment. Both computers run the same Dataset Studio application.

## 1. Prepare both computers

1. Install/open Dataset Studio on both computers using the instructions in [README.md](README.md). Use the same project version and compatible Python/PyTorch/Ultralytics environments.
2. Connect both computers to a trusted network. The worker needs to reach the controller; the worker itself does not need an inbound port.
3. On the controller, Dataset Studio can create a local CA and controller certificate with detected host/IP names. No OpenSSL command is needed.
4. In Windows Firewall, allow the selected TCP port only on the **Private** network profile and, where possible, only from the worker computer. Do not disable the firewall or create a public-network rule.

## 2. Start the controller

1. Open Dataset Studio on Computer A and click **Compute Center**.
2. Set **Operating mode** to **Controller**, then click **Apply operating mode**.
3. In **Worker Manager**, set **Bind address** to the controller’s reachable LAN IP (the default `127.0.0.1` accepts only this computer).
4. Click **Set up or reuse controller certificates…**. Dataset Studio creates/reuses a CA, SAN certificate, and DPAPI-protected keys. **Renew controller certificate…** renews the leaf certificate; **Rotate CA…** also changes the trust authority and requires workers to enroll again.
5. Click **Start secure controller**.
6. Click **Create one-time enrollment token…**, then **Export worker package…**. Enter the controller LAN hostname/IP and save the package. Send it to Computer B only through a trusted private channel; it contains a one-use token and public CA certificate, never a private key or worker credential.

## 3. Connect the worker

1. Open Dataset Studio on Computer B and click **Compute Center**.
2. Set **Operating mode** to **Worker**, then click **Apply operating mode**.
3. Enter a **Worker name**, then click **Join existing Controller…** and import the package from Computer A. Check the imported hostname, port, and token expiry. The package installs only the public CA trust certificate.
4. Click **Connect worker securely**. The worker verifies both the certificate chain and controller hostname; do not bypass certificate errors.
5. On Computer A, select the worker row and click **Approve selected**. Wait for **Online**, **Approved**, and **Enabled**.

## 4. Open a prepared dataset and configure training

1. On Computer A, open a prepared YOLO dataset containing `data.yaml`, images, labels, and class names.
2. Open the main **Train** panel. Choose **YOLO** or **RT-DETR**, a model, and **CPU** for the simplest first run. Set a small number of **Epochs** (for example, 1) and a small image size suitable for a smoke run.
3. If you select a local model file on Computer A, Dataset Studio transfers that file to the worker. Otherwise, choose a model preset available in the worker’s Ultralytics installation.

## 5. Submit and watch the remote job

1. Go to **Compute Center → Job Queue** and click **Guided remote training…**.
2. Read the dataset/model summary. Choose **Automatic compatible worker** or select the connected worker by name.
3. Enter additional RAM/VRAM requirements if known; the configured safety margins are added to these amounts. Worker eligibility shows current RAM, reservations, requested memory, safety reserve, GPU memory, and required disk.
4. Continue through the transfer explanation and review page, then click **Submit remote job**.
5. Watch **State**, **Progress**, and **Latest checkpoint** in **Job Queue**. Dataset Studio hashes dataset files, transfers them, verifies them on Computer B, and starts the existing isolated training adapter.
6. When the job completes, its experiment appears in the existing **Experiments** area. The controller receives and hash-verifies `best.pt`, `last.pt`, and available training files before listing them. The model is not registered automatically; use **Register from experiment…** if you want that.

## If something goes wrong

- If the worker is offline, check that Computer A’s controller is running, Computer B has the right host/port and certificate, and the controller firewall permits the private-network connection.
- If enrollment is rejected, create a new token on Computer A and enter it on Computer B, then approve the newly enrolled row.
- If TLS validation fails, check that the worker connected using a hostname/IP listed in the controller certificate SANs and reimport the current CA package after a CA rotation.
- If the worker is unsuitable, check **Architectures**, **PyTorch / Ultralytics**, free RAM/disk, and GPU/VRAM in **Worker Manager**. Select CPU for a CPU-only worker.
- If dataset transfer fails, check free disk space on both computers and reconnect the worker. The original dataset is unchanged.
- If the network or worker application stops during training, inspect the worker’s local output before submitting again. Long interruptions and worker restarts are not automatically reconciled or retried in this release.

See [PHASE_6_GUIDE.md](PHASE_6_GUIDE.md) for detailed setup, transfer, recovery, and troubleshooting instructions. Automated remote execution tests use mocked networking; this project workspace has not performed physical two-computer acceptance testing.
