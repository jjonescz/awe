FROM continuumio/miniconda3

ENV \
    # Disable interactivity of `apt-get`.
    DEBIAN_FRONTEND="noninteractive"

# Install programs.
RUN apt-get update && apt-get install --yes --no-install-recommends \
    # Needed to sign Git commits using GPG.
    gnupg2 \
    # Needed to unzip SWDE.
    p7zip-full \
    # Clean up.
    && rm -rf /var/lib/{apt,dpkg,cache,log}

# Install Python and PyTorch Geometric.
RUN conda install -y python=3.6 pyg=2.0.1 -c pyg -c conda-forge
