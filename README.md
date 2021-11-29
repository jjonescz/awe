# AI-based web extractor

This repository contains source code for AI-based structured web data extractor.

## Directory structure

- [`awe`](awe): Python source code (data loading and machine learning).
- [`js`](js): Node.js source code (headless browser for visual attribute
  extraction).

## Usage

We use [VS Code](https://code.visualstudio.com/) with [Dev
Containers](https://code.visualstudio.com/docs/remote/containers). For training
on GPU in cloud, we have [instructions](gradient/README.md) to run on
[Gradient](https://gradient.run) via [Remote
SSH](https://code.visualstudio.com/docs/remote/ssh).

### Running

See available commands via `doit list`. Their source code is in `dodo.py`.

1. Prepare dataset by running `doit`.
2. See `notebook.ipynb`.

## Development

**Adding new Python package:** Add the package to `requirements.txt` and/or
`gradient/requirements.txt` and run `doit install` (if using Dev Containers) or
rebuild the Docker image (if using Remote SSH).
