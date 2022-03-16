# See https://www.gitpod.io/docs/config-docker.
FROM gitpod/workspace-full

# Install Google Chrome.
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add - \
    && sudo sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list' \
    && sudo apt-get update && sudo apt-get install -y --no-install-recommends google-chrome-stable \
    && sudo rm -rf /var/lib/{apt,dpkg,cache,log}

# Install Google Drive CLI and jless (JSON CLI).
RUN brew install gdrive jless
