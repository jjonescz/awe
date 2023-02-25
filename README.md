# AI-based web extractor

This repository contains source code of AI-based structured web data extractor.

- 👨‍💻 Author: [Jan Joneš](https://github.com/jjonescz)
- 📜 Thesis: [PDF](https://github.com/jjonescz/awe/releases/download/v1.0/jan-jones-master-thesis.pdf), [assignment](https://is.cuni.cz/studium/dipl_st/index.php?id=&tid=&do=main&doo=detail&did=241832), [submission](http://hdl.handle.net/20.500.11956/174143)
- 🚀 Demo: [live](https://bit.ly/awedemo),
  [Docker Hub](https://hub.docker.com/repository/docker/janjones/awe-demo)
- 🗃️ Data: [SWDE with visuals](https://github.com/jjonescz/swde-visual)

## Directory structure

- 📂 [`awe/`](awe): Python module (data manipulation and machine learning).
  See [`awe/README.md`](awe/README.md).
- 📂 [`js/`](js): Node.js app (visual attribute extractor and inference demo).
  See [`js/README.md`](js/README.md).
- 📂 [`docs/`](docs)
  - 📂 [`dev/`](docs/dev)
    - 📄 [`env.md`](docs/dev/env.md): development environment setup.
    - 📄 [`tips.md`](docs/dev/tips.md): development guidelines and bash snippets.
  - 📄 [`data.md`](docs/data.md): dataset preparation.
  - 📄 [`extractor.md`](docs/extractor.md): running the visual extractor.
  - 📄 [`train.md`](docs/train.md): training instructions.
  - 📄 [`release.md`](docs/release.md): release instructions.
  - 📂 [`demo/`](docs/demo)
    - 📄 [`run.md`](docs/demo/run.md): developing and running the demo.
    - 📄 [`deploy.md`](docs/demo/deploy.md): online demo deployment.

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
