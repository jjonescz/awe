# Inspired by
# https://github.com/Paperspace/fastai-docker/blob/master/fastai-v4/Dockerfile.
FROM nvidia/cuda:9.2-base-ubuntu16.04

ENV LANG C.UTF-8
LABEL com.nvidia.volumes.needed="nvidia_driver"

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget && \
    rm -rf /var/lib/apt/lists/*
RUN \
    # Add source for Google Chrome.
    wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list' && \
    # Install programs.
    apt-get update && apt-get install -y --no-install-recommends \
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
    libpng-dev \
    # Needed to work with processes using GPU.
    psmisc \
    # Needed for SSH below.
    openssh-server \
    # Needed to unzip SWDE.
    p7zip-full \
    # Needed for Puppeteer.
    google-chrome-stable && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Install Conda.
RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install conda-build

ENV PATH=/opt/conda/bin:$PATH

# Install Python.
ENV PYTHON_VERSION=3.9.7
RUN conda install python==${PYTHON_VERSION}

# Install Node.js.
ENV NODE_VERSION=17.1.0
RUN conda install -c conda-forge nodejs==${NODE_VERSION} && \
    corepack enable

# Install PyTorch Geometric.
COPY requirements-*.txt /
RUN pip install -r requirements-torch.txt && \
    pip install -r requirements-pyg.txt

# Configure SSH. Inspired by https://github.com/Paperspace/gradient-ssh-test.
RUN mkdir /var/run/sshd && \
    sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd && \
    echo "export VISIBLE=now" >> /etc/profile && \
    # Preserve environment variables in SSH sessions. Except Jupyter variables
    # (`DISABLE_JUPYTER` and `JUPYTER_TOKEN` are present in Gradient).
    echo "export \$(cat /proc/1/environ | tr '\\\0' '\\\n' | grep -v JUPYTER | xargs)" >> /etc/profile

# Build WebSockets proxy.
RUN curl -L https://go.dev/dl/go1.17.3.linux-amd64.tar.gz -o go.tar.gz && \
    tar -C /usr/local -xzf go.tar.gz && \
    rm go.tar.gz
ENV GOPATH=/huproxy \
    PATH=/usr/local/go/bin:$PATH
RUN go get github.com/google/huproxy

# Install Python packages.
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

# Persist HuggingFace cache.
ENV TRANSFORMERS_CACHE=/storage/.cache/huggingface/transformers

WORKDIR /storage/awe/src

COPY run.sh /run.sh
CMD [ "/run.sh" ]
