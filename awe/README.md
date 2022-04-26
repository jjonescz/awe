# AWE Python module

This folder contains Python module
with data manipulation and machine learning code.
Further documentation is available in code.

## Overview

- 📂 `data/`: data manipulation.
  - 📂 `set/`: HTML dataset loading and pre-processing.
    - 📄 `pages.py`: common abstractions.
    - 📄 `swde.py`, `apify.py`: implementations of two datasets.
  - 📂 `graph/`: DOM processing.
  - 📂 `visual/`: visual attribute processing.
    - 📄 `attribute.py`: list of available visual attributes.
  - 📄 `sampling.py`: data loading for training.
  - 📄 `validation.py`: data validation.
- 📂 `features/`: feature extraction.
  - 📄 `feature.py`: common abstraction.
  - 📄 `text.py`, `dom.py`, `visual.py`: three feature groups.
- 📂 `model/`: PyTorch model definitions.
  - 📄 `classifier.py`: the main model.
  - 📄 `word_lstm.py`: RNN sub-model.
  - 📄 `eval.py`, `metrics.py`: model evaluation.
  - 📄 `decoding.py`: prediction processing.
- 📂 `training/`: code for training and evaluation.
  - 📄 `trainer.py`: tha main entrypoint containing training loop.
  - 📄 `crossval.py`: running cross-validation experiments.
  - 📄 `params.py`: hyper-parameters.
