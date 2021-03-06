# AI-based web extractor

This repository contains source code of AI-based structured web data extractor.

- π¨βπ» Author: [Jan JoneΕ‘](https://github.com/jjonescz)
- π Thesis: [PDF](https://github.com/jjonescz/awe/releases/download/v1.0/jan-jones-master-thesis.pdf), [assignment](https://is.cuni.cz/studium/dipl_st/index.php?id=&tid=&do=main&doo=detail&did=241832)
- π Demo: [live](https://bit.ly/awedemo),
  [Docker Hub](https://hub.docker.com/repository/docker/janjones/awe-demo)
- ποΈ Data: [SWDE with visuals](https://github.com/jjonescz/swde-visual)

## Directory structure

- π [`awe/`](awe): Python module (data manipulation and machine learning).
  See [`awe/README.md`](awe/README.md).
- π [`js/`](js): Node.js app (visual attribute extractor and inference demo).
  See [`js/README.md`](js/README.md).
- π [`docs/`](docs)
  - π [`dev/`](docs/dev)
    - π [`env.md`](docs/dev/env.md): development environment setup.
    - π [`tips.md`](docs/dev/tips.md): development guidelines and bash snippets.
  - π [`data.md`](docs/data.md): dataset preparation.
  - π [`extractor.md`](docs/extractor.md): running the visual extractor.
  - π [`train.md`](docs/train.md): training instructions.
  - π [`release.md`](docs/release.md): release instructions.
  - π [`demo/`](docs/demo)
    - π [`run.md`](docs/demo/run.md): developing and running the demo.
    - π [`deploy.md`](docs/demo/deploy.md): online demo deployment.

## Quickstart

### Running the pre-trained demo locally

```bash
docker pull janjones/awe-demo
docker run --rm -it -p 3000:3000 janjones/awe-demo
```

Open a web browser and navigate to <http://localhost:3000/>.

For more details, see [`docs/demo/run.md`](docs/demo/run.md).

### Training on the SWDE dataset

```bash
docker pull janjones/awe-gradient
docker run --rm -it -v awe:/storage -p 3000:3000 janjones/awe-gradient bash
```

Then, run inside the Docker container:

```bash
git clone https://github.com/jjonescz/awe .
git clone https://github.com/jjonescz/swde-visual data/swde
python -m awe.training.params
python -m awe.training.train
# Model is trained, now you can run the demo.
cd js
pnpm install
DEBUG=1 pnpm run server
```

For more details, see
1. [`docs/dev/env.md`](docs/dev/env.md),
2. [`docs/data.md`](docs/data.md),
3. [`docs/train.md`](docs/train.md), and
4. [`docs/demo/run.md`](docs/demo/run.md).
