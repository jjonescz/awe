# See https://github.com/microsoft/vscode-dev-containers/tree/main/containers/python-3.
FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.9

ENV \
    # Disable interactivity of `apt-get`.
    DEBIAN_FRONTEND="noninteractive"

# Install programs.
RUN apt-get update && apt-get install --yes --no-install-recommends \
    # Needed to unzip SWDE.
    p7zip-full \
    # Clean up.
    && rm -rf /var/lib/{apt,dpkg,cache,log}

# Install Node.js.
ARG NODE_VERSION="17.0.1"
RUN su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"
