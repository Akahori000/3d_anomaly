FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
ENV NV_CUDNN_VERSION=8.5.0.96

# update packages
RUN set -x && \
    apt update && \
    apt upgrade -y

# install command
RUN set -x && \
    apt install -y wget && \
    apt install -y sudo && \
    apt install -y zip

# install git
RUN apt-get install -y git

# Anaconda
RUN set -x && \
    wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh && \
    bash Anaconda3-2021.05-Linux-x86_64.sh -b && \
    rm Anaconda3-2021.05-Linux-x86_64.sh

# path setting
ENV PATH $PATH:/root/anaconda3/bin
