# See `/gradient/Dockerfile`.
FROM janjones/awe-gradient:latest

# Install programs.
RUN apt-get update && apt-get install --yes --no-install-recommends \
    # Needed to create non-root user with sudo priviledges below.
    sudo \
    # Clean up.
    && rm -rf /var/lib/{apt,dpkg,cache,log}

# Create non-root user.
ARG USER="developer"
ARG UID=1000
ARG GID=1000
RUN if [ "$USER" != "root" ] ; then \
    groupadd --gid $GID $USER \
    && useradd -s /bin/bash --uid $UID --gid $GID -m $USER \
    && echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER ; fi

# Add Gitpod user to sudoers.
ARG USER="gitpod"
RUN echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER

USER $USER

# Keep the container running if used on its own.
CMD while sleep 1000; do :; done
