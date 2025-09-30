# CUDA + cuDNN + PyTorch runtime base to ensure GPU availability
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

# System deps for audio/video, V4L2, and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    v4l2loopback-dkms \
    v4l2loopback-utils \
    v4l-utils \
 && rm -rf /var/lib/apt/lists/*

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/.cache/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt \
 # Ensure torchaudio matches the CUDA 12.1 PyTorch in this base image
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 torchaudio==2.4.1+cu121

# App files
COPY . /app

# Default command does a quick GPU sanity check
CMD ["python", "verify_cuda.py"]
