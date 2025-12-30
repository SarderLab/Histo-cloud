# This Dockerfile is used to generate the docker image dsarchive/histomicstk
# This docker image includes the HistomicsTK python package along with its
# dependencies.
#
# All plugins of HistomicsTK should derive from this docker image


# start from TensorFlow 2.x with GPU support
FROM tensorflow/tensorflow:2.15.0-gpu
LABEL com.nvidia.volumes.needed="nvidia_driver"

LABEL maintainer="Brendon Lutnick - Sarder Lab. <brendonl@buffalo.edu>"
LABEL description="HistomicsTK with TensorFlow 2.x and DeepLabV3+ support"

CMD echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! STARTING THE BUILD !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# RUN mkdir /usr/local/nvidia && ln -s /usr/local/cuda-10.0/compat /usr/local/nvidia/lib

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Remove bad repos
RUN rm \
    /etc/apt/sources.list.d/cuda.list

RUN apt-get update && \
    apt-get install --yes --no-install-recommends software-properties-common && \
    RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get --yes --no-install-recommends -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" dist-upgrade && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    #keyboard-configuration \
    git \
    wget \
    curl \
    ca-certificates \
    libcurl4-openssl-dev \
    libexpat1-dev \
    unzip \
    libhdf5-dev \
    software-properties-common \
    libssl-dev \
    # Standard build tools \
    build-essential \
    cmake \
    autoconf \
    automake \
    libtool \
    pkg-config \
    # useful later \
    libmemcached-dev && \
    #apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

CMD echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! CHECKPOINT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

RUN apt-get update ##[edited]
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

# RUN apt-get install software-properties-common -y
# RUN add-apt-repository ppa:graphics-drivers/ppa -y
# RUN apt-get update -y
# RUN apt-get upgrade -y
# RUN apt-get install nvidia-driver-455 -y

WORKDIR /
# Make Python3 the default and install pip. TF 2.15 comes with Python 3.11
RUN which python && \
    python --version

ENV build_path=$PWD/build

# HistomicsTK sepcific

# copy HistomicsTK files
ENV htk_path=$PWD/HistomicsTK
RUN mkdir -p $htk_path

RUN apt-get update && \
    apt-get install -y --no-install-recommends memcached && \
    #apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# RUN pip install torch
# RUN python -c 'import torch,sys;print(torch.cuda.is_available());sys.exit(not torch.cuda.is_available())'
COPY . $htk_path/
WORKDIR $htk_path

# Install HistomicsTK and its dependencies
# Upgrade setuptools and pip
RUN pip install --no-cache-dir --upgrade pip setuptools && \
    # Install large_image with memcached extras \
    pip install --no-cache-dir 'large-image[memcached]' && \
    # Install TF-Slim for TF2 compatibility \
    pip install --no-cache-dir 'tf-slim>=1.1.0' && \
    # Install pillow_lut \
    pip install --no-cache-dir 'pillow-lut' && \
    # Install HistomicsTK \
    pip install --no-cache-dir . --find-links https://girder.github.io/large_image_wheels && \
    # clean up \
    rm -rf /root/.cache/pip/*

# Show what was installed
RUN python --version && pip --version && pip freeze

# remove cuda compat
# RUN apt remove --purge cuda-compat-10-0 --yes

# pregenerate font cache
RUN python -c "from matplotlib import pylab"

# Note: TF2 deprecation warnings are handled via tf.compat.v1 API usage in code
# No need to suppress warnings in the TF installation

# define entrypoint through which all CLIs can be run
WORKDIR $htk_path/histomicstk/cli

# Test our entrypoint.  If we have incompatible versions of numpy and
# openslide, one of these will fail
RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python -m slicer_cli_web.cli_list_entrypoint SegmentWSI --help
RUN python -m slicer_cli_web.cli_list_entrypoint TrainNetwork --help
RUN python -m slicer_cli_web.cli_list_entrypoint ExtractFeaturesFromAnnotations --help
RUN python -m slicer_cli_web.cli_list_entrypoint IngestAperioXML --help

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]
