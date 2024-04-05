# Starting from NVIDIA PyTorch NGC Container
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
FROM nvcr.io/nvidia/pytorch:24.01-py3

# install some useful tools
RUN \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get update -y -q \
    && apt-get install -y \
    aptitude \
    automake \
    bash-completion \
    bison \
    build-essential \
    cmake \
    dnsutils \
    elinks \
    emacs-nox emacs-goodies-el \
    fish \
    flex \
    git \
    htop \
    iperf3 \
    iproute2 \
    iputils-ping \
    less \
    libtool \
    libopencv-dev \
    mc \
    nload \
    nmon \
    psutils \
    source-highlight \
    ssh \
    sudo \
    tmux \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

########################################################################
# Cassandra C++ and Python drivers
########################################################################

# install cassandra C++ driver
RUN \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get update -y -q \
    && apt-get install -y libuv1-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/* 

ARG CASS_DRIVER_VER=2.16.2
RUN \
    wget -nv "https://github.com/datastax/cpp-driver/archive/$CASS_DRIVER_VER.tar.gz" \
    && tar xfz $CASS_DRIVER_VER.tar.gz \
    && cd cpp-driver-$CASS_DRIVER_VER \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j \
    && make install


#install cassandra python driver + some python libraries
RUN \
    pip3 install --upgrade --no-cache matplotlib pandas clize \
      opencv-python cassandra-driver pybind11 tqdm tifffile pyyaml

########################################################################
# SPARK installation, to test examples
########################################################################
# download and install spark
ARG SPARK_V=3.5
RUN \
    export SPARK_VER=$(curl 'https://downloads.apache.org/spark/' | grep -o "$SPARK_V\.[[:digit:]]\+" | tail -n 1) \
    && cd /tmp && wget -nv "https://downloads.apache.org/spark/spark-$SPARK_VER/spark-$SPARK_VER-bin-hadoop3.tgz" \
    && cd / && tar xfz "/tmp/spark-$SPARK_VER-bin-hadoop3.tgz" \
    && ln -s "spark-$SPARK_VER-bin-hadoop3" spark

# Install jdk
RUN \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get update -y -q \
    && apt-get install -y openjdk-11-jdk

ENV PYSPARK_DRIVER_PYTHON=python3
ENV PYSPARK_PYTHON=python3
EXPOSE 8080
EXPOSE 7077
EXPOSE 4040

########################################################################
# Cassandra server installation, to test examples
########################################################################
ARG CASS_V=4.1
RUN \
    export CASS_VERS=$(curl 'https://downloads.apache.org/cassandra/' | grep -o "$CASS_V\.[[:digit:]]\+" | tail -n 1) \
    && cd /tmp && wget -nv "https://downloads.apache.org/cassandra/$CASS_VERS/apache-cassandra-$CASS_VERS-bin.tar.gz" \
    && cd / && tar xfz "/tmp/apache-cassandra-$CASS_VERS-bin.tar.gz" \
    && ln -s "apache-cassandra-$CASS_VERS" cassandra

EXPOSE 9042

########################################################################
# Download the Imagenette dataset
########################################################################
WORKDIR /tmp
RUN \
    wget -nv 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz' \
    && tar xfz 'imagenette2-320.tgz' \
    && rm 'imagenette2-320.tgz'

########################################################################
# Upgrade DALI, install plugin and run as user
########################################################################
# Fix for error given by "from nvidia.dali.plugin.pytorch import DALIGenericIterator"
# - https://forums.developer.nvidia.com/t/issues-building-docker-image-from-ngc-container-nvcr-io-nvidia-pytorch-22-py3/209034
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib:/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib"
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist \
    --upgrade nvidia-dali-cuda120==1.34
RUN \
    useradd -m -G sudo -s /usr/bin/fish -p '*' user \
    && sed -i 's/ALL$/NOPASSWD:ALL/' /etc/sudoers \
    && chown -R user.user /apache-cassandra-$CASS_V*

COPY . /home/user/cassandra-dali-plugin

# increase write timeout to 20 seconds, listen to all interfaces,
# enable SSL and increase max direct memory available
RUN \
    cp /home/user/cassandra-dali-plugin/varia/keystore /cassandra/conf/ \
    && python3 /home/user/cassandra-dali-plugin/varia/edit_cassandra_conf.py

RUN chown -R user.user '/home/user/cassandra-dali-plugin'
RUN chown -R user.user "/spark/"
# create data dir
RUN mkdir /data
RUN chown user.user '/data'
# install plugin
WORKDIR /home/user/cassandra-dali-plugin
RUN pip3 install .
USER user

