FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:/root/.local/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV LIBRARY_PATH=${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}

# apt install by root user
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    python-is-python3 \
    python3.10-dev \
    python3-pip \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxrender1 \
    libxi6 \
    libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/*

# install base & torch
RUN pip install --upgrade pip setuptools ninja
RUN pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

# install gs3 requirements
COPY . /tmp
RUN echo "Installing GS^3 specific packages..." \
&& export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0+PTX" \
&& export TCNN_CUDA_ARCHITECTURES="90;86;80;75;70;61;60" \
&& cd /tmp \
&& pip install -r requirements.txt \
&& rm -rf /tmp/*
