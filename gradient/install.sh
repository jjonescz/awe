#!/bin/bash

# This script should be run once after VSCode is connected.

# Set remote-machine-specific settings.
cat << EOF > ~/.vscode-server/data/Machine/settings.json
{
    "jupyter.jupyterServerType": "remote"
}
EOF

# Setup `tmux`.
cat << EOF > ~/.tmux.conf
set -g mouse on
EOF

# Install extensions.
code --install-extension editorconfig.editorconfig
code --install-extension esbenp.prettier-vscode
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-toolsai.jupyter
code --install-extension streetsidesoftware.code-spell-checker
code --install-extension mutantdino.resourcemonitor
code --install-extension innerlee.nvidia-smi
