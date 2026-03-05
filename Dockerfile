# =============================================================
# AURA — Dockerfile (NVIDIA CUDA + Python)
# =============================================================
# Uses CUDA 12.4 runtime for GPU-accelerated segmentation
# (Sa2VA-4B) and PyTorch inference.
# =============================================================

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- System packages + Python 3.11 via deadsnakes PPA ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        git libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies (cached layer) ---
# Install PyTorch with CUDA 12.4 support first
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cu124

# Then install remaining dependencies
# --ignore-installed blinker: the CUDA base image ships blinker 1.4 via
# distutils which pip cannot cleanly uninstall.
COPY requirements.txt .
RUN pip install --ignore-installed blinker -r requirements.txt

# --- Application code ---
COPY app.py image_analysis_module.py chatbot_module.py manual_manager.py ./
COPY .streamlit/ .streamlit/
COPY sample_manuals/ sample_manuals/

# Pre-built IPC-A-610F manual (indices + extracted figures)
COPY manuals/ manuals/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["python", "-m", "streamlit", "run", "app.py", \
            "--server.address=0.0.0.0", "--server.port=8501"]
