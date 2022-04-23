# Inspired by
# https://github.com/Paperspace/fastai-docker/blob/master/fastai-v4/Dockerfile.
# Docker image page: https://hub.docker.com/r/nvidia/cuda.
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
    sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list' && \
    # Install programs.
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    zip \
    unzip \
    libpng-dev \
    tmux \
    jq \
    # Needed to work with processes using GPU.
    psmisc \
    # Needed for SSH below.
    openssh-server \
    # Useful for CPython debugging.
    gdb \
    # Needed for Puppeteer.
    google-chrome-stable && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

# Install latest 7zip.
RUN wget https://downloads.sourceforge.net/project/sevenzip/7-Zip/21.07/7z2107-linux-x64.tar.xz \
    && tar xf 7z2107-linux-x64.tar.xz -C /usr/bin 7zz \
    && rm 7z2107-linux-x64.tar.xz

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
RUN conda install -y --freeze-installed python==${PYTHON_VERSION} \
    && conda clean -afy

# Install Node.js and GitHub CLI.
ENV NODE_VERSION=17.1.0
RUN conda install -y --freeze-installed -c conda-forge nodejs==${NODE_VERSION} gh==2.6.0 \
    && corepack enable \
    && conda clean -afy

# Install PyTorch.
COPY requirements-torch.txt /
RUN pip install --no-cache-dir -r /requirements-torch.txt

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
RUN go get github.com/google/huproxy@a131ee126ce30e57a8a47d259a0d02a835fa3189

# Install Google Drive CLI.
ENV GOPATH=/usr
RUN go get github.com/prasmussen/gdrive@2.1.1

# Install jless (JSON CLI).
RUN curl -fsSL https://github.com/PaulJuliusMartinez/jless/releases/download/v0.8.0/jless-v0.8.0-x86_64-unknown-linux-gnu.zip -o jless.zip \
    && unzip jless.zip -d /usr/bin \
    && rm jless.zip

# Install Python packages.
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

# Persist HuggingFace cache.
ENV HF_HOME=/storage/.cache/huggingface

WORKDIR /storage/awe/src

COPY run.sh /run.sh
CMD [ "/run.sh" ]
