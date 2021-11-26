# Inspired by
# https://github.com/Paperspace/fastai-docker/blob/master/fastai-v4/Dockerfile.
FROM nvidia/cuda:9.2-base-ubuntu16.04

ENV LANG C.UTF-8
LABEL com.nvidia.volumes.needed="nvidia_driver"

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    ca-certificates \
    python-qt4 \
    libjpeg-dev \
    zip \
    unzip \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTHON_VERSION=3.9

RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install conda-build

ENV PATH=$PATH:/opt/conda/bin/

RUN conda install python>=${PYTHON_VERSION}

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

COPY requirements-pyg.txt /requirements-pyg.txt
RUN pip install -r requirements-pyg.txt

# Install SSH. Inspired by https://github.com/Paperspace/gradient-ssh-test.
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir /var/run/sshd && \
    sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd && \
    echo "export VISIBLE=now" >> /etc/profile

# Build WebSockets proxy.
RUN curl -L https://go.dev/dl/go1.17.3.linux-amd64.tar.gz -o go.tar.gz && \
    tar -C /usr/local -xzf go.tar.gz && \
    rm go.tar.gz
ENV GOPATH=/huproxy \
    PATH=${PATH}:/usr/local/go/bin
RUN go get github.com/google/huproxy

WORKDIR /notebooks

COPY run.sh /run.sh
CMD [ "/run.sh" ]
