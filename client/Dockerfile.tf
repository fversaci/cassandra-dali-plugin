FROM nvcr.io/nvidia/tensorflow:24.08-tf2-py3

RUN \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get update -y -q \
    && apt-get install -y \
       iproute2 vim fish nmon htop less \
    && rm -rf /var/lib/apt/lists/*

RUN pip install clize pillow tqdm
