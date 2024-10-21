FROM nvcr.io/nvidia/tensorflow:24.08-tf2-py3
# FROM tensorflow/tensorflow:2.17.0-gpu

ENV EXT_UID 1000
ENV EXT_GID 1000
RUN groupadd -g $EXT_GID user
RUN useradd -m --uid $EXT_UID --gid $EXT_GID user

RUN \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get update -y -q \
    && apt-get install -y \
       iproute2 vim fish nmon htop less \
    && rm -rf /var/lib/apt/lists/*

RUN pip install IPython tqdm clize
EXPOSE 5050
EXPOSE 5051

COPY ./code /home/user/code
RUN chown -R user:user /home/user/code
WORKDIR /home/user/code
USER user
