FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 \
      python3.12-venv \
      python3-pip \
      ffmpeg \
      libglib2.0-0 \
      libgl1 \
      libstdc++6 \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv "$VIRTUAL_ENV"

ENV PYTHONPATH=/app

WORKDIR /app

ARG IMAGE_VERSION=dev
LABEL org.opencontainers.image.title="VolleyVision" \
      org.opencontainers.image.description="GPU-enabled volleyball labeling, training, calibration, and analysis" \
      org.opencontainers.image.version="${IMAGE_VERSION}"

RUN pip install --upgrade pip setuptools wheel \
    && pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126

COPY requirements-unified.txt .
RUN pip install -r requirements-unified.txt

COPY portal ./portal
COPY volley-labeler/app.py volley-labeler/export.py ./volley-labeler/
COPY volley-labeler/static ./volley-labeler/static
COPY volley-analyzer/app ./volley-analyzer/app
COPY volley-calibrator/app ./volley-calibrator/app
COPY volley-training-studio/app ./volley-training-studio/app

EXPOSE 8000
CMD ["uvicorn", "portal.main:app", "--host", "0.0.0.0", "--port", "8000"]