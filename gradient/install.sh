#!/bin/bash

# This script should be run once after VSCode is connected. Note that settings
# applied here are persisted by `run.sh`.

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
