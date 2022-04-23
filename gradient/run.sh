#!/bin/bash

# This script should be run after source code is mounted.
mkdir -p /storage/awe/src

# Persist VSCode directory and other config directories (used in `install.sh`).
mkdir -p /storage/awe/.vscode-server
ln -sT /storage/awe/.vscode-server ~/.vscode-server
ln -sT /storage/.tmux.conf ~/.tmux.conf

# Generate a random alphanumeric string of length 48 (like Jupyter notebook
# token, e.g., `c8de56fa4deed24899803e93c227592aef6538f93025fe01`). Inspired by
# https://gist.github.com/earthgecko/3089509.
if [ -z "$JUPYTER_TOKEN" ]; then
    JUPYTER_TOKEN=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 48 | head -n 1)
fi

# Print mocked Jupyter token, so that the port is exposed as if it were a
# notebook within Gradient.run. Inspired by
# https://github.com/Paperspace/gradient-coder.
echo "http://localhost:8888/lab?token=${JUPYTER_TOKEN}"
echo "http://127.0.0.1:8888/lab?token=${JUPYTER_TOKEN}"
echo "[I $(date +'%Y-%m-%d %T.123') ServerApp] http://localhost:8888/lab?token=${JUPYTER_TOKEN}"
echo "[I $(date +'%Y-%m-%d %T.123') ServerApp]  or http://127.0.0.1:8888/lab?token=${JUPYTER_TOKEN}"

# Print domain exposed by Gradient for localhost:8888.
echo ${PAPERSPACE_FQDN}

# Start SSH server.
echo "Password: ${JUPYTER_TOKEN}"
echo "root:${JUPYTER_TOKEN}" | chpasswd && /usr/sbin/sshd -eD &

# Listen through huproxy.
/huproxy/bin/huproxy -listen :8888
