# Start from a PyTorch image with CUDA
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND noninteractive
# Flash-attn work around @ https://github.com/Dao-AILab/flash-attention/issues/509
ENV FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE

# Set the working directory in the container
WORKDIR /app
RUN mkdir /dataset/
RUN mkdir /models/

COPY . .

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    wget \
    curl \
    dnsutils \
    nano \
    zip \
    unzip \
    git \
    s3cmd \
    ffmpeg \
    screen \
    fonts-freefont-ttf \
    inotify-tools \
    parallel \
    pciutils \
    ncdu \
    libbz2-dev \
    gettext \
    apt-transport-https \
    gnupg2 \
    time \
    openssl \
    redis-tools \
    ca-certificates \
    hdf5-tools\
    vim

RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install any dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -e .
RUN pip install -e ".[train]"
RUN pip install flash-attn --no-build-isolation

# Install lm-eval
RUN pip install git+https://github.com/ZhuohaoNi/lm_eval.git

# Install lmm-eval
RUN pip install git+https://github.com/ToviTu/lmms-eval.git@llava_plain
