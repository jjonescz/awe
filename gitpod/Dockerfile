# Image for Gitpod development. See `docs/dev/env.md`.
# See also https://www.gitpod.io/docs/config-docker.

# Base image defined in `gradient/Dockerfile`. Exact tag is specified so Gitpod
# rebuilds the workspace when the base image changes (and the tag is manually
# updated).
FROM janjones/awe-gradient:1650739890

ARG USER="gitpod"
ENV HOME=/home/${USER}

# Install `sudo`.
RUN apt-get update && apt-get install -y --no-install-recommends sudo \
    && rm -rf /var/lib/apt/lists/*

# Create user. Based on https://github.com/gitpod-io/workspace-images/blob/481f7600b725e0ab507fbf8377641a562a475625/base/Dockerfile.
# '-l': see https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
RUN if [ "${USER}" != "root" ] ; then \
    useradd -l -u 33333 -G sudo -md /home/${USER} -s /bin/bash -p ${USER} ${USER} \
    # passwordless sudo for users in the 'sudo' group
    && sed -i.bkp -e 's/%sudo\s\+ALL=(ALL\(:ALL\)\?)\s\+ALL/%sudo ALL=NOPASSWD:ALL/g' /etc/sudoers ; fi
WORKDIR $HOME
USER ${USER}
# use sudo so that user does not get sudo usage info on (the first) login
RUN sudo echo "Running 'sudo' for ${USER}: success" && \
    # create .bashrc.d folder and source it in the bashrc
    mkdir -p $HOME/.bashrc.d && \
    (echo; echo "for i in \$(ls -A \$HOME/.bashrc.d/); do source \$HOME/.bashrc.d/\$i; done"; echo) >> $HOME/.bashrc

# Keep the container running if used on its own.
CMD while sleep 1000; do :; done
