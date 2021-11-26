#!/bin/bash

# Generate a random alphanumeric string of length 48 (like Jupyter notebook
# token, e.g., `c8de56fa4deed24899803e93c227592aef6538f93025fe01`). Inspired by
# https://gist.github.com/earthgecko/3089509.
if [ -z "$JUPYTER_TOKEN" ]; then
    JUPYTER_TOKEN=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 48 | head -n 1)
fi

# Print mocked Jupyter token, so that we can run this container as if it were a
# notebook within Gradient.run. Inspired by
# https://github.com/Paperspace/gradient-coder.
echo "http://localhost:8888/?token=${JUPYTER_TOKEN}"
echo "http://localhost:8888/\?token\=${JUPYTER_TOKEN}"

echo "root:${JUPYTER_TOKEN}" | chpasswd && /usr/sbin/sshd -eD &
/huproxy/bin/huproxy -listen :8888
