# Gradient setup

This folder contains files used to setup development environment on
[Gradient](https://gradient.run).

1. Build the Docker image by manually executing GitHub Action workflow
   [`gradient-docker-image`](https://github.com/jjonescz/awe/actions/workflows/gradient-docker-image.yml).

   Alternatively, build the image locally by executing:

   ```sh
   cd gradient
   docker build -t janjones/awe-gradient .
   ```

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
   token from that URL. Alternatively, look into terminal output in Gradient web
   environment after the notebook starts, both the token and public URL should
   appear there.

   ```ssh_config
   Host 127.0.0.1
     User root
     ProxyCommand <FULL_PATH_TO_huproxyclient.exe> wss://<JUPYTER_LAB_ID>.paperspacegradient.com/proxy/%h/%p

   Host localhost
     User root
     ProxyCommand <FULL_PATH_TO_huproxyclient.exe> ws://localhost:8888/proxy/%h/%p
   ```

   Note that the first is for Gradient.run and the second is for local testing.

   Alternatively, [Tailscale](https://tailscale.com/) is also installed in the
   container and can be used. It can be more resilient than huproxy, although it
   requires login and the connection is routed through some peer nodes. Simply
   install and start it on your local machine, then you can connect to the
   remote machine directly (the first time, you will need to authenticate the
   remote machine through a link displayed in Gradient machine output). You can
   either use its IP or (more persistent) host name if MagicDNS is enabled in
   Tailscale configuration.

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

7. Jupyter notebooks live only as long as VSCode is not disconnected. To use
   persistent Jupyter server which lives as long as the Gradient machine,
   instruct VSCode to connect to Jupyter remote server `http://localhost:8890/`
   (it's started automatically in script `run.sh`).

## Build image with data

1. Open [a dummy repository](https://github.com/4GeeksAcademy/html-hello) in
   Gitpod, so that `.gitpod.yml` of this repository is not used.

2. Remove all contents and clone this repository instead:

   ```bash
   rm -rf .* *
   git clone https://github.com/jjonescz/awe
   ```

3. Download prepared data (e.g., from GDrive).

4. Build Docker image:

   ```bash
   docker build -t janjones/awe-data -f gradient/Dockerfile.data .
   docker push janjones/awe-data
   ```

## Training Workflow

To run training as a CI job inside
[Gradient Workflows](https://docs.paperspace.com/gradient/workflows/):

1. Login to [Gradient CLI](https://docs.paperspace.com/gradient/cli/):

   ```bash
   gradient apiKey <api_key>
   ```

2. Create datasets `swde` and `apify` and note their IDs. Also create dataset
   `awe-model` to store the trained model.

3. Upload prepared data (`<vertical>.zip` files to `swde` and `<website>.zip`
   files to `apify`):

   ```bash
   gradient datasets versions create --id <dataset_id>
   for f in *.zip
   do gradient datasets files put --id <dataset_id>:<version_id> --source-path $f
   done
   gradient datasets version commit --id <dataset_id>:<version_id>
   ```

4. Create a workflow (and copy the resulting ID):

   ```bash
   gradient workflows create --name crossval --projectId <project_id>
   ```

5. Specify training parameters in `data/params.json`. Run this command to create
   new file or validate existing:

   ```bash
   python -m awe.training.params
   ```

6. Upload the params to Gradient.

   ```bash
   gradient secrets set project --id <project_id> --name MODEL_PARAMS --value "$(cat data/params.json)"
   ```

7. Run the workflow:

   ```bash
   gradient workflows run --id <workflow_id> --path ./gradient/crossval.yml
   ```
