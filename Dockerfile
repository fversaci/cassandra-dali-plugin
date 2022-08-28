# Starting from NVIDIA PyTorch NGC Container
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
FROM nvcr.io/nvidia/pytorch:22.08-py3

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
    iperf \
    iproute2 \
    iputils-ping \
    ipython3 \
    less \
    libtool \
    libopencv-dev \
    mc \
    nload \
    nmon \
    psutils \
    python3-pip \
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

RUN \
    wget 'https://github.com/datastax/cpp-driver/archive/2.16.0.tar.gz' \
    && tar xfz 2.16.0.tar.gz \
    && cd cpp-driver-2.16.0 \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j \
    && make install


#install cassandra python driver + some python libraries
RUN \
    pip3 install --upgrade --no-cache matplotlib pandas clize \
      opencv-python cassandra-driver pybind11 tqdm tifffile

########################################################################
# SPARK installation, to test examples
########################################################################
# download and install spark
ARG SPARK_V=3.3
RUN \
    export SPARK_VER=$(curl 'https://downloads.apache.org/spark/' | grep -o "$SPARK_V\.." | tail -n 1) \
    && cd /tmp && wget "https://downloads.apache.org/spark/spark-$SPARK_VER/spark-$SPARK_VER-bin-hadoop3.tgz" \
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
ARG CASS_V=4.0
RUN \
    export CASS_VERS=$(curl 'https://downloads.apache.org/cassandra/' | grep -o "$CASS_V\.." | tail -n 1) \
    && cd /tmp && wget "https://downloads.apache.org/cassandra/$CASS_VERS/apache-cassandra-$CASS_VERS-bin.tar.gz" \
    && cd / && tar xfz "/tmp/apache-cassandra-$CASS_VERS-bin.tar.gz" \
    && ln -s "apache-cassandra-$CASS_VERS" cassandra

# increase write timeout to 20 seconds
RUN \
    sed -i 's/^\(write_request_timeout_in_ms:\)\(.*\)/\1 20000/' /cassandra/conf/cassandra.yaml

EXPOSE 9042

########################################################################
# Download the Imagenette dataset
########################################################################
WORKDIR /tmp
RUN \
    wget 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz' \
    && tar xfz 'imagenette2-320.tgz' \
    && rm 'imagenette2-320.tgz'

########################################################################
# Install plugin and run as user
########################################################################
RUN \
    useradd -m -G sudo -s /usr/bin/fish -p '*' user \
    && sed -i 's/ALL$/NOPASSWD:ALL/' /etc/sudoers \
    && chown -R user.user /apache-cassandra-$CASS_V*

COPY . /home/user/cassandra-dali-plugin
RUN chown -R user.user '/home/user/cassandra-dali-plugin'
WORKDIR /home/user/cassandra-dali-plugin
RUN pip3 install .
USER user

# Fix for error given by "from nvidia.dali.plugin.pytorch import DALIGenericIterator"
# - https://forums.developer.nvidia.com/t/issues-building-docker-image-from-ngc-container-nvcr-io-nvidia-pytorch-22-py3/209034
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"
