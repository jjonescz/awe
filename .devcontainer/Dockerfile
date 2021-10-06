FROM continuumio/miniconda3

ENV \
    # Disable interactivity of `apt-get`.
    DEBIAN_FRONTEND="noninteractive"

# Install programs.
RUN apt-get update && apt-get install --yes --no-install-recommends \
    # Needed to create non-root user with sudo priviledges below.
    sudo \
    # Needed to sign Git commits using GPG.
    gnupg2 \
    # Needed to unzip SWDE.
    p7zip-full \
    # Clean up.
    && rm -rf /var/lib/{apt,dpkg,cache,log}

# Install Python.
RUN conda install -y python=3.6
