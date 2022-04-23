# Training

Training is performed in the same Docker image as [development](dev/env.md).
You might also want to [setup data](data.md).

## Parameters

Parameters are set in file `data/params.json` which is not tracked by Git.
To generate one with default values or validate an existing one, run:

```bash
python -m awe.training.params
```

They are documented in the script `awe/training/params.py`.

## Developing

For experiments, use notebook `awe/training/training.ipynb`
or script `awe/training/train.py`.

```bash
python -m awe.training.train
```

## Cross-validation

Full cross-validation experiment can be performed via a Python script
(if you have enough RAM to fit all training and testing data),

```bash
python -m awe.training.crossval
```

or a bash script
(which runs each fold as a separate process
to ensure only necessary data are loaded to memory).

```bash
./sh/crossval.sh
```

### Params

It is recommended to set `val_subset` to `1` (since testing will be performed)
and disable all checkpoint creation (`save_*` parameters).

### SSH

To perform an experiment over SSH without losing it on hangup,
`tmux` can be used to start the experiment,
then detached,
and its output obtained continuously using:

```bash
tmux capture-pane -S - && tmux save-buffer $(pwd)/out.txt
```

### Mean

After cross-validation, compute mean results of all runs:

```sh
python -m awe.training.crossval_mean <first_version_num>
```

## GitHub Action

Training can be run on CPU
as GitHub Action [Training](../.github/workflows/training.yml).
To pass `data/params.json` as input, minify them using:

```bash
jq -c . < data/params.json
```

## Gradient Workflow

Training can be run on GPU as
[a Gradient Workflow](https://docs.paperspace.com/gradient/workflows/).

1. Prepare data (including GloVe embeddings by running `python -m awe.prepare`)
   and build a Docker image containing them:

   ```bash
   docker build -t janjones/awe-data -f gradient/Dockerfile.data .
   docker push janjones/awe-data
   export TIMESTAMP=$(date +%s)
   docker tag janjones/awe-data janjones/awe-data:$TIMESTAMP
   docker push janjones/awe-data:$TIMESTAMP
   ```

2. Login to [Gradient CLI](https://docs.paperspace.com/gradient/cli/):

   ```bash
   gradient apiKey <api_key>
   ```

3. Create a workflow (and copy the resulting ID):

   ```bash
   gradient workflows create --name crossval --projectId <project_id>
   ```

4. Run the workflow[^1]:

   ```bash
   jq --null-input --arg params "$(cat data/params.json)" '{ "params": { "value": $params } }' > data/input.json
   gradient workflows run --id <workflow_id> --path ./gradient/crossval.yml --inputPath data/input.json
   ```

[^1]: The workflow spec [`crossval.yml`](../gradient/crossval.yml)
can be freely edited without pushing to Git,
it is read by the Gradient CLI only locally.
