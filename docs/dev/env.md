# Development environment

As IDE, we use [Visual Studio Code](https://code.visualstudio.com/).
For developing both Python and TypeScript code,
use Docker image `janjones/awe-gradient`.

## Docker image

The development Docker image can be built either

- locally,

  ```bash
  cd gradient
  docker build -t janjones/awe-gradient .
  ```

- by executing GitHub Action
  [Gradient Docker Image](../../.github/workflows/gradient-docker-image.yml)
  which also pushes the image to [Docker Hub](https://hub.docker.com/)
  as `janjones/awe-gradient`.

Remote environments described below usually assume
existence of `janjones/awe-gradient` on [Docker Hub](https://hub.docker.com/).

## Locally

To develop locally, either reproduce your environment
according to `gradient/Dockerfile`[^1],
or develop inside a Docker image as described below.

[^1]: Note that all Python package versions are frozen in
[`awe/requirements.txt`](../../awe/requirements.txt).

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).

2. Download the Docker image
   (or build it locally as described [above](#docker-image)).

   ```bash
   docker pull janjones/awe-gradient
   ```

3. Start the Docker image.

   ```bash
   # Go up to repository root.
   cd ..
   # (Optional) Remove persistent volume to start fresh.
   docker volume rm awe
   # Run the container with persistent storage in a named volume.
   docker run --rm -it -v awe:/storage -p 22:22 janjones/awe-gradient
   ```

4. Now SSH can be used as below. Password is displayed in Docker output.

   ```bash
   ssh root@localhost
   ```

5. Continue with [VS Code SSH](#vs-code-ssh) steps.

## CPU on Gitpod

If CPU is enough (e.g., to process data or experiment with a simple model),
[Gitpod](https://www.gitpod.io/) is configured and can be used.

## GPU on Gradient.run

When GPU is needed, [Gradient.run](https://gradient.run/) cloud can be used.
Since it does not support direct SSH connections, a little trick is needed[^2].

[^2]: This trick is based on directing SSH over WebSockets
which might stop working if Gradient.run infrastructure is changed
in a non-trivial way.

1. Create and start Gradient notebook.
   In advanced options
   - ensure Git repository is not set,
   - select the development Docker image (`janjones/awe-gradient:latest`)
     as a [custom container](https://docs.paperspace.com/gradient/explore-train-deploy/notebooks/create-a-notebook/notebook-containers),
     and
   - set the command to `/run.sh`.

2. Install [huproxy](https://github.com/google/huproxy) locally
   and configure SSH as shown below.
   Replace `<JUPYTER_LAB_ID>` by ID from URL that opens in Gradient
   when clicking on "Open in Jupyter Lab" button.
   As SSH password, use token from that URL.
   Alternatively, look into terminal output in Gradient web environment
   after the notebook starts, both the token and public URL should appear there.

   ```ssh_config
   Host 127.0.0.1
     User root
     ProxyCommand <FULL_PATH_TO_huproxyclient.exe> wss://<JUPYTER_LAB_ID>.paperspacegradient.com/proxy/%h/%p
   ```

3. Continue with [VS Code SSH](#vs-code-ssh) steps.

## VS Code SSH

1. Connect VS Code via
   [Remote SSH](https://code.visualstudio.com/docs/remote/ssh)
   and open directory `/storage/awe/src`.

2. If this is the first time,
   clone the repository, configure Git and initialize VSCode:

   ```sh
   git clone https://github.com/jjonescz/awe .
   ./sh/configure.sh
   ./gradient/install.sh
   ```
