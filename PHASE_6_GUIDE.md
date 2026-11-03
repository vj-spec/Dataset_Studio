# Phase 6 — Remote Training Guide

This guide describes the current **Compute Center** controls for connecting a controller and one or more workers. It applies to the same Dataset Studio application in all operating modes. Remote training uses the existing YOLO and RT-DETR adapters and isolated training child process.

## Before connecting computers

1. Install and open the same Dataset Studio project version on both computers. Each computer needs the project’s existing Python, PySide6, PyTorch, and Ultralytics dependencies. Do not change or upgrade a worker’s PyTorch/CUDA installation just to match another computer without checking its driver and Python compatibility.
2. Put the controller and workers on a network where the worker can reach the controller. The worker starts the connection outbound; workers do not need public inbound ports.
3. The controller needs an X.509 TLS certificate and private key. Use a certificate trusted by the worker, or copy only the controller certificate/CA certificate to the worker as its trust file. **Never copy the private key or enrollment token to a worker.**
4. The controller listener defaults to port **8765**. Allow that port through the controller computer’s firewall only for the trusted network/devices that need access. The application does not open router ports or publish a Ray dashboard.
5. Allow enough free disk on both computers for a dataset archive and its extracted worker cache. A remote job may also transfer a selected local `.pt` or `.yaml` model file. Large files take longer to hash and transfer.

## Configure the controller

1. Open **Compute Center** in Dataset Studio.
2. Set **Operating mode** to **Controller** or **Hybrid**, then click **Apply operating mode**.
3. In **Worker Manager**, enter the controller **Bind address** and **Port**. Select a LAN address reachable by the worker; the default is `127.0.0.1` (local computer only).
4. Click **Set up or reuse controller certificates…**. The application creates/reuses a local CA and SAN certificate; no OpenSSL commands are needed. Private keys are protected with Windows DPAPI. **Renew controller certificate…** rotates the leaf certificate; **Rotate CA…** rotates trust and requires re-enrollment.
5. Click **Start secure controller**. If it fails, check certificate validity, port availability, and firewall rules.
6. Click **Create one-time enrollment token…**. The dialog supports copy/show, expiry countdown, consumed status, revocation, and generating another token. Click **Export worker package…** to export endpoint, public CA, one-time token and expiry. Share it only over a trusted private channel; it contains no private keys.

## Configure and approve a worker

1. On the second computer, open **Compute Center** and set **Operating mode** to **Worker** or **Hybrid**.
2. Enter a recognizable **Worker name**. Keep the generated **Persistent worker ID** unless you are deliberately enrolling a new identity.
3. Click **Join existing Controller…** and import the enrollment package. It installs only the public CA certificate and fills host, port, token, and expiry.
4. Click **Connect worker securely**. The worker connects outward using TLS 1.2 or newer, verifies the controller hostname and CA, and reports hardware/software information.
5. On the controller, refresh **Worker Manager**. Select the new row and click **Approve selected**. The row should say **Online**, **Approved**, and **Enabled**. Review the architecture, PyTorch/Ultralytics versions, free RAM/disk, and GPU/VRAM fields.
6. If the worker stays pending or offline, confirm the controller is running, the host/port resolve correctly, the certificate name matches the host, firewall access is allowed, and both clocks are reasonably accurate.

## Check compatibility

Use **Guided remote training…** in the **Job Queue** tab. It uses the currently open prepared dataset and the current selections in the main **Train** panel. The guide shows online approved workers and a reason when a worker is unsuitable.

The controller checks approval/enablement, heartbeat freshness, architecture, requested CPU/GPU device, RAM/VRAM/disk, runtime version presence, and active resource reservations. Compatibility details show reported free memory, reservations, requested memory, and safety margin. A worker can be connected and approved yet unsuitable for a particular job. Exact framework/CUDA version matching is not guaranteed; inspect reported versions when environments differ.

## Submit a remote experiment

1. In the main application, open a prepared YOLO dataset with `data.yaml` and choose **YOLO** or **RT-DETR** in the **Train** panel.
2. Set the desired model, device, epochs, image size, batch size, experiment name, and other supported parameters. A controller-side local checkpoint selected as a real `.pt`/`.yaml` file is copied to the worker and hash-verified. A model preset that is only a name must also be available through the worker’s Ultralytics installation or its normal model-download behavior.
3. Go to **Compute Center → Job Queue** and click **Guided remote training…**.
4. Read the dataset/model summary. Choose **Automatic compatible worker** or pin one suitable worker. CPU selection works with CPU-only workers; a CUDA selection requires reported usable CUDA.
5. Review the dataset staging explanation. Dataset Studio hashes files, transfers a streaming archive in bounded blocks, checks each extracted file, and stores the exact dataset version in the worker cache. The original dataset is not edited.
6. Review the training summary and click **Submit remote job**. The controller creates a central experiment entry and job record. A compatible worker starts the job when it polls.
7. In **Job Queue**, watch state, dataset transfer progress, epoch progress, and latest checkpoint. Select the experiment in **Experiments** for available metrics, worker/job metadata, and returned checkpoint paths.
8. After the worker reports completion, the controller verifies the checkpoint/log sizes and SHA-256 hashes before publishing them to the experiment record. The model is **not** added to Model Registry automatically; use **Register from experiment…** if you want it registered.

## Dataset cache and storage

The worker cache is content-addressed from a dataset manifest containing relative file paths, sizes, and SHA-256 hashes. An unchanged version is reused after verifying its listed files. Runtime YAML is written as a worker-local copy so paths point to the staged directory. The original dataset and archived YAML remain unchanged.

To clear cached dataset versions, first stop active remote work, then click **Clear worker dataset cache…** in **Worker Manager** and confirm. This removes content-addressed dataset cache folders, including entries that fail integrity checks. It does not remove original datasets, model-cache files, training outputs, or checkpoints. Interrupted partial-transfer files are not included in this cleanup action.

## Monitoring and cancellation

- The worker is limited to one active remote training job. Do not treat queued, preparing, or running as completed.
- **Cancel / stop selected job safely** cancels a queued job immediately or sends a stop request to its assigned worker. During training, the existing adapter stops at a safe checkpoint/epoch boundary when possible.
- Available epoch metrics are reported; metrics that the installed trainer does not expose are not invented.
- Worker network callbacks do not run training in the GUI thread. Transfer blocks and progress messages share the authenticated TLS channel.

## Disconnection and restart recovery

- If only the network drops while the worker application and training process remain running, training can continue locally. The worker reconnects using bounded exponential backoff with jitter. Each request has a bounded wait; a long outage can fail the remote job. The controller does not assign an already-reserved job to another worker.
- Check the worker’s local output folder and Compute Center before submitting another job. Do not submit a duplicate while the original process may still be training.
- If the worker application restarts, its cache and checkpoints remain on disk, but this version does not persist and reconcile an active worker process automatically. If the controller application restarts, an active job is recorded as interrupted/failed and is not automatically retried.
- If the last status is unclear, reconnect the worker, request cancellation from the controller when possible, confirm the worker process has stopped, and then decide whether to resume from a known `last.pt` or submit a new experiment. A worker-side unfinished checkpoint may need to be copied back manually if the transfer did not complete.
- No automatic retry loop starts duplicate training. Review a failure message and the worker’s local run before resubmitting.

## Troubleshooting

| What you see | What to check |
| --- | --- |
| Worker cannot connect | Controller listener state, host/IP, port 8765 or configured port, firewall scope, TLS certificate trust/name, and network route. |
| Enrollment rejected | Token spelling/expiry/one-use status; create a new token and enroll again. Approve the worker after it appears. |
| Worker is unsuitable | Confirm it is online, approved, enabled, reports YOLO/RT-DETR support, has the requested CPU/GPU mode, adequate free disk/RAM, and PyTorch/Ultralytics installed. |
| Dataset transfer fails | Check both disks, connection stability, dataset file readability, and worker cache integrity. The source dataset remains untouched. |
| Training says dataset invalid | Open the dataset locally, prepare a valid YOLO dataset with `data.yaml`, images, labels, and matching class names, then submit again. |
| Model cannot load | Use a model preset supported by the worker runtime or choose a real local `.pt`/`.yaml` file on the controller so it is transferred. A nonexistent controller path is rejected. |
| Checkpoint transfer fails | Preserve the worker run folder, reconnect, and inspect the job/experiment error. Do not register a checkpoint until it exists on the controller and can be opened. |
| Job appears active after worker restart | Request cancellation if the worker reconnects. The current release has no automatic worker-process reconciliation; verify the local process before retrying. |

## Verification status

Automated tests cover the mocked TLS protocol, remote YOLO and RT-DETR adapter paths, dataset/archive hashes and cache, progress, cancellation records, and artifact receipt. This development environment has one computer, so it has **not** completed the physical two-computer acceptance sequence. GPU remote execution is also untested here.

## Job history and security events

Job Queue provides **Archive selected**, **Delete selected**, **Clear terminal jobs**, and **Inspect job / logs**. Active jobs cannot be archived or deleted. A remote record remains protected until the worker confirms terminal state. Removing a job preserves its experiment record and files. Worker Manager displays audit events for enrollment, authentication, approval/revocation, assignment, cancellation, dataset transfer, and artifact retrieval. Enrollment tokens are single-use, expiring, revocable, and stored only as hashes in SQLite.
