# AI-based web extractor

This repository contains source code for AI-based structured web data extractor.

- Author: [Jan Joneš](https://github.com/jjonescz)
- Thesis: [assignment](https://is.cuni.cz/studium/dipl_st/index.php?id=&tid=&do=main&doo=detail&did=241832)

## Directory structure

- [`awe`](awe): Python source code (data loading and machine learning).
- [`js`](js): Node.js source code (headless browser for visual attribute
  extraction).

## Usage

We use [VS Code IDE](https://code.visualstudio.com/).
- For training on GPU in cloud, we have [instructions](gradient/README.md) to
run on [Gradient](https://gradient.run) via [Remote
SSH](https://code.visualstudio.com/docs/remote/ssh) (the same container can be
also setup locally for testing).
- When CPU is enough, [Gitpod](https://www.gitpod.io/) can be used.

### Running

See available commands via `doit list`. Their source code is in `dodo.py`.

1. Prepare dataset by running `doit`.
2. See `*.ipynb` Jupyter notebooks.

### Development

**Adding new Python package:** Add the package to `gradient/requirements.txt`,
try if it works by running `pip install -r gradient/requirements.txt`, and
rebuild the Docker image.

**Debugging training code:** Set `num_workers=0`. GPUs can be enabled.

**Kill processes taking up GPU:** Run `fuser -k /dev/nvidia0`.
