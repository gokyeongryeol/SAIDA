ARG PYTORCH="2.1.2"
ARG CUDA="12.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# --------------------
# System packages
# --------------------
RUN apt-get update && \
    apt-get install -y \
        ffmpeg \
        libsm6 \
        libxext6 \
        git \
        ninja-build \
        libglib2.0-0 \
        libxrender-dev \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# --------------------
# Python packages
# --------------------
RUN pip install "pip<23.1"

# MMEngine / MMCV
RUN pip install openmim && \
    mim install "mmengine>=0.8.0" && \
    pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

# --------------------
# Project
# --------------------
WORKDIR /workspace
COPY . /workspace

RUN pip install --no-cache-dir --no-build-isolation -e external/mmdetection

RUN pip install fairscale jupyter
RUN pip install numpy==1.26.4 opencv-python==4.11.0.86 requests
