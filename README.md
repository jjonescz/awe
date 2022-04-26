# AI-based web extractor

This repository contains source code of AI-based structured web data extractor.

- 👨‍💻 Author: [Jan Joneš](https://github.com/jjonescz)
- 📜 Thesis: [assignment](https://is.cuni.cz/studium/dipl_st/index.php?id=&tid=&do=main&doo=detail&did=241832)
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
