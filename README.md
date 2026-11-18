# VolleyVision

VolleyVision brings Volley Labeler, court calibration, model training, data management, and match analysis together, with GPU and CPU-only deployment options.

## Requirements

- Docker Compose v2
- Linux/amd64 host
- An NVIDIA GPU and the NVIDIA Container Toolkit for the GPU image

## Build the Docker image locally

From the repository root, choose the GPU or CPU Dockerfile:

```bash
# CUDA-enabled image
docker build \
  --build-arg IMAGE_VERSION=dev \
  --tag volleyvision:latest \
  -f Dockerfile \
  .

# CPU-only image
docker build \
  --build-arg IMAGE_VERSION=dev-cpu \
  --tag volleyvision:cpu \
  -f Dockerfile.cpu \
  .
```

The final `.` is the build context and must be the repository root, where both Dockerfiles and `requirements-unified.txt` are located. The GPU build downloads CUDA-enabled PyTorch and other compiled dependencies. Expect it to use roughly **11–12 GB** of local disk space, plus temporary build space; the PyTorch install alone contributes about 6.8 GB. The CPU image avoids CUDA libraries but still includes PyTorch and compiled dependencies, so it remains several gigabytes.

Check local image sizes with:

```bash
docker image ls volleyvision
```

## Run from source

```bash
cp .env.example .env
# Edit the host paths in .env to match your data folders.

# GPU image (requires NVIDIA Container Toolkit)
docker compose up -d --build

# CPU-only image (Docker Compose v2.24 or newer)
docker compose -f compose.yaml -f compose.cpu.yaml up -d --build
```

Open `http://localhost:8095`. The `.env.example` binds to localhost and uses `./data` for local storage. Set `BIND_ADDRESS=0.0.0.0` in `.env` only if you intend to make the service reachable from other computers on a trusted network. Compose uses the same local paths and localhost binding when no `.env` is present.

The root `compose.yaml` and `compose.cpu.yaml` are the only Compose deployment files. The root `Dockerfile` and `Dockerfile.cpu` build the integrated application; there are no separate per-component Docker deployments.

Both images support training and match analysis. CPU mode uses PyTorch on the CPU and can be substantially slower, especially for video analysis and long training runs. The CPU Compose override removes the GPU reservation; no NVIDIA GPU or toolkit is needed.

## Persistent data

The Compose file mounts data from the host; source videos, labels, calibration files, training outputs, and analysis results are not part of the image. Configure these variables in `.env`:

- `VOLLEY_DATA_HOST_PATH`: default data root
- `LABELS_HOST_PATH`: native labels and YOLO exports
- `VIDEO_HOST_PATH`: source clips
- `CALIBRATIONS_HOST_PATH`: calibration data
- `OUTPUT_HOST_PATH`: training workspace and analysis output
- `EXISTING_VIDEO_HOST_PATH`: optional read-only video library
- `EXISTING_MODEL_HOST_PATH`: optional read-only model library

When using `.env.example`, these folders are under `./data`. The Compose defaults use those same relative paths if no `.env` is present.

Local `.env` files, datasets, videos, exports, model weights, Python environments, and caches are excluded from Git and Docker build contexts. Supply any required data or weights through the documented mounts instead. The `LABELER_*` environment variable names remain supported for existing deployments.

## Data manipulation and export

Use **Data manipulation & export** to search and manage videos in the writable library and filter, import, rename, download, or delete supported model weights. Imports are saved to the training models folder with the selected model target in their filename. Read-only external libraries can be downloaded from but not renamed or deleted.