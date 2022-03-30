# AI-based web extractor

This repository contains source code for AI-based structured web data extractor.

- 👨‍💻 Author: [Jan Joneš](https://github.com/jjonescz)
- 📜 Thesis: [assignment](https://is.cuni.cz/studium/dipl_st/index.php?id=&tid=&do=main&doo=detail&did=241832)
- 🚀 Demo: [live](https://bit.ly/awedemo), [DockerHub](https://hub.docker.com/repository/docker/janjones/awe-demo)

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

- For data manipulations, [Gitpod](https://www.gitpod.io/) can be used.

### Running

See available commands via `doit list`. Their source code is in `dodo.py`.

1. Prepare dataset by running `doit`.
2. See `*.ipynb` Jupyter notebooks.

### Development

**Add new Python package:** Add the package to `gradient/requirements.txt`, try
if it works by running `pip install -r gradient/requirements.txt`, and rebuild
the Docker image.

**Debug training code:** Set `num_workers=0`. GPUs can be enabled.

**Kill processes taking up GPU:** Run `fuser -k /dev/nvidia0`.

**Inspect HTML in the dataset:** Run `pnpx -y http-server` and navigate to the
page through web browser.

**Debug CPython code:** Run `gdb -ex=r --args python <path_to_script>.py`. Then
issue GDB command `backtrace`.

**Get HTML from a scraping log**: For example, if it's on line 11, run
`sed '11!d' data/scraping-logs/2022-03-21T13-26-38.056Z.txt | jq -r '.html' > data.html`.

### Release

#### Upload pre-trained model

```bash
tar czf logs.tar.gz logs/1-version-name/
gh auth login
gh release upload v0.1 logs.tar.gz
```
