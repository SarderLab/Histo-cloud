# This Dockerfile is used to generate the docker image dsarchive/histomicstk
# This docker image includes the HistomicsTK python package along with its
# dependencies.
#
# All plugins of HistomicsTK should derive from this docker image


# start from nvidia/cuda 10.0
# FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
FROM tensorflow/tensorflow:1.15.4-gpu-py3
LABEL com.nvidia.volumes.needed="nvidia_driver"

LABEL maintainer="Brendon Lutnick - Sarder Lab. <brendonl@buffalo.edu>"

CMD echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! STARTING THE BUILD !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# RUN mkdir /usr/local/nvidia && ln -s /usr/local/cuda-10.0/compat /usr/local/nvidia/lib

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Remove bad repos
RUN rm \
    /etc/apt/sources.list.d/cuda.list

RUN apt-get update && \
    apt-get install --yes --no-install-recommends software-properties-common && \
    # As of 2018-04-16 this repo has the latest release of Python 2.7 (2.7.14) \
    # add-apt-repository ppa:jonathonf/python-2.7 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get --yes --no-install-recommends -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" dist-upgrade && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    #keyboard-configuration \
    git \
    wget \
    python-qt4 \
    python3-pyqt4 \
    curl \
    ca-certificates \
    libcurl4-openssl-dev \
    libexpat1-dev \
    unzip \
    libhdf5-dev \
    libpython-dev \
    libpython3-dev \
    python2.7-dev \
    python-tk \
    # We can't go higher than 3.7 and use tensorflow 1.x \
    python3.6-dev \
    python3.6-distutils \
    python3-tk \
    software-properties-common \
    libssl-dev \
    # Standard build tools \
    build-essential \
    cmake \
    autoconf \
    automake \
    libtool \
    pkg-config \
    # needed for supporting CUDA \
    # libcupti-dev \
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
# Make Python3 the default and install pip.  Whichever is done last determines
# the default python version for pip.
RUN rm /usr/bin/python && \
    ln /usr/bin/python3.6 /usr/bin/python && \
    rm /usr/local/bin/python && \
    ln /usr/bin/python3.6 /usr/local/bin/python && \
    ln /usr/bin/python3.6 /usr/local/bin/python3
RUN which  python && \
    python --version
# RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
RUN curl -O https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

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
#   Upgrade setuptools, as the version in Conda won't upgrade cleanly unless it
# is ignored.
RUN pip install --no-cache-dir --upgrade --ignore-installed pip setuptools && \
    # pip install --no-cache-dir 'tensorflow<2' && \
    # Install large_image memcached extras \
    pip install --no-cache-dir 'large-image[memcached]' && \
    # Install HistomicsTK \
    pip install --no-cache-dir . --find-links https://girder.github.io/large_image_wheels && \
    # Install tf-slim \
    pip install --no-cache-dir 'tf-slim>=1.1.0' && \
    # Install pillow_lut \
    pip install --no-cache-dir 'pillow-lut' && \
    # clean up \
    rm -rf /root/.cache/pip/*

# Show what was installed
RUN python --version && pip --version && pip freeze

# remove cuda compat
# RUN apt remove --purge cuda-compat-10-0 --yes

# pregenerate font cache
RUN python -c "from matplotlib import pylab"

# Suppress warnings
RUN sed -i 's/^_PRINT_DEPRECATION_WARNINGS = True/_PRINT_DEPRECATION_WARNINGS = False/g' /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/util/deprecation.py && \
    sed -i 's/rename = get_rename_v2(full_name)/rename = False/g' /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/util/module_wrapper.py

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
