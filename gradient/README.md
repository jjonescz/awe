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
   cd .. # go up to repository root
   docker run --rm -it -p 8888:8888 -v$(pwd):/notebooks janjones/awe-gradient
   ```

4. Create Gradient notebook.

   - Provide Git repository as workspace (`https://github.com/jjonescz/awe`).
   - Select the pushed Docker image (`janjones/awe-gradient`) as [custom
     container](https://docs.paperspace.com/gradient/explore-train-deploy/notebooks/create-a-notebook/notebook-containers).
     Set the command to `/run.sh`.

5. Start the notebook and then optionally switch to Jupyter Lab for better
   terminal support.
