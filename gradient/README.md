# Gradient setup

This folder contains files used to setup development environment on
[Gradient](https://gradient.run).

1. Build the Docker by manually executing GitHub Action workflow
   [`gradient-docker-image`](https://github.com/jjonescz/awe/actions/workflows/gradient-docker-image.yml).

2. (Optional) Test that the container works locally.

   ```sh
   # Go up to repository root.
   cd ..
   # (Optional) Remove the volume to start fresh.
   docker volume rm awe
   # Run the container with persistent storage in named volume.
   docker run --rm -it -p 8888:8888 -v awe:/storage janjones/awe-gradient
   ```

3. Create and start Gradient notebook. Skip if testing locally.

   - Ensure Git repository is not set.
   - Select the pushed Docker image (`janjones/awe-gradient`) as [custom
     container](https://docs.paperspace.com/gradient/explore-train-deploy/notebooks/create-a-notebook/notebook-containers).
     Note that you may need to specify exact tag to overwrite previously cached
     version. Set the command to `/run.sh`.

4. Install [huproxy](https://github.com/google/huproxy) locally and configure
   SSH as shown below. Replace `<JUPYTER_LAB_ID>` by ID from URL that opens in
   Gradient when clicking on "Open in Jupyter Lab" button. For SSH password, use
   token shown in terminal after the container is started.

   ```ssh_config
   Host 127.0.0.1
     User root
     ProxyCommand <FULL_PATH_TO_huproxyclient.exe> wss://<JUPYTER_LAB_ID>.paperspacegradient.com/proxy/%h/%p

   Host localhost
     User root
     ProxyCommand <FULL_PATH_TO_huproxyclient.exe> ws://localhost:8888/proxy/%h/%p
   ```

   Note that the first is for Gradient.run and the second is for local testing.

5. Connect via [VS Code](https://code.visualstudio.com/) with [Remote
   SSH](https://code.visualstudio.com/docs/remote/ssh) and open directory
   `/storage/awe/src`.

6. If this is the first time, clone the repository, configure Git and initialize
   VSCode:

   ```sh
   git clone https://github.com/jjonescz/awe .
   ./sh/configure.sh
   ./gradient/install.sh
   ```
