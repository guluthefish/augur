# Vex-DR: $\mathbb V$isually $\mathbb{EX}$plainable $\mathbb D$river Mutation $\mathbb R$ecognition using Whole Slide Images

This repository contains the code for the paper "Vex-DR: Visually Explainable Driver Mutation Recognition using Whole Slide Images". The code is implemented in Python and uses PyTorch Lightning for training and evaluation. The repository is organized as follows:

- `configs/`: Contains the configuration files for dataloader, model training and evaluation.
- `containers/`: Contains the code to create apptainer images.
- `data/`: Contains the configuration files for data download and preprocessing.
- `envs/: Contains the environment files for creating apptainer images.
- `notebooks/`: Contains the Jupyter notebooks for usage illustration.
- `scripts/`: Contains the scripts for data downloading, model training and evaluation, and visualization.
- `slurm`: Contains the SLURM scripts for training and evaluation on a cluster.
- `tests/`: Contains the unit tests for the code.
- `Vex-DR/`: Contains the code for the Vex-DR model and its components.
  - `datasets/`: Contains the code for the datasets used in the paper.
  - `models/`: Contains the code for the Vex-DR model and its components
  - `utils/`: Contains the utility functions for data loading, model training and evaluation, and visualization.
