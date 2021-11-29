# Gradient setup

This folder contains files used to setup development environment on
[Gradient](https://gradient.run).

1. Build the Docker image.

   ```sh
   cd gradient # go to this directory
   docker build -t janjones/awe-gradient .
   ```

2. Push the Docker image.

   ```sh
   docker push janjones/awe-gradient
   ```

3. (Optional) Test that the container works locally.

   ```sh
   # Go up to repository root.
   cd ..
   # Clone repository into named volume (skip if already done).
   docker volume rm awe
   docker run --rm -it -v awe:/storage janjones/awe-gradient git clone https://github.com/jjonescz/awe .
   # Run the container.
   docker run --rm -it -p 8888:8888 -v awe:/storage janjones/awe-gradient
   ```

4. Create and start Gradient notebook. Skip if testing locally.

   - Provide Git repository as workspace (`https://github.com/jjonescz/awe`).
   - Select the pushed Docker image (`janjones/awe-gradient`) as [custom
     container](https://docs.paperspace.com/gradient/explore-train-deploy/notebooks/create-a-notebook/notebook-containers).
     Set the command to `/run.sh`.

5. Install [huproxy](https://github.com/google/huproxy) locally and configure
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

6. Connect via [VS Code](https://code.visualstudio.com/) with [Remote
   SSH](https://code.visualstudio.com/docs/remote/ssh) and open directory
   `/storage/awe/src`.

7. If this is the first time, initialize VSCode:

   ```sh
   ./gradient/install.sh
   ```
