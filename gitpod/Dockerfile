# See `/gradient/Dockerfile`.
FROM janjones/awe-gradient:latest

# Install programs.
RUN apt-get update && apt-get install --yes --no-install-recommends \
    # Needed to create non-root user with sudo priviledges below.
    sudo \
    # Clean up.
    && rm -rf /var/lib/{apt,dpkg,cache,log}

# Add Gitpod user to sudoers.
ARG USER="gitpod"
RUN echo $USER ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER
