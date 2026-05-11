# Augur: Visually Explainable Subtype Classification from Whole Slide Images

This repository contains the code for the paper "Augur: Visually Explainable Subtype Classification from Whole Slide Images". The code is implemented in Python and uses PyTorch Lightning for training and evaluation. The directory is organized as follows:

- `configs/`: Contains the configuration files for dataloader, model training and evaluation.
- `containers/`: Contains the code to create apptainer images.
- `data/`: Contains the configuration files for data download and preprocessing.
- `envs/`: Contains the environment files for creating apptainer images.
- `scripts/`: Contains the scripts for data downloading, model training and evaluation, and visualization.
- `slurm/`: Contains the SLURM scripts for training and evaluation on a cluster.
- `tests/`: Contains the unit tests for the code.
- `augur/`: Contains the code for the Augur model and its components.
  - `datasets/`: Contains the code for the datasets used in the paper.
  - `models/`: Contains the code for the Augur model and its components, split into `tile_level/` (tile encoders, decoders, classifiers) and `slide_level/` (MIL aggregation, attention).
  - `utils/`: Contains the utility functions for data loading, model training and evaluation, and visualization.
