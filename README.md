# ADLR project repository for Team 17

## Overview
This project aims to adapt the Network described in [Mackowiak, 2020](literature/Mackowiak_paper.pdf) for a tactile classification task, then benchmark it against the classifier architecture described in [Tulbure et al., 2018](literature/Baueml_paper.pdf). We will be using Google Compute Engine and Python for all implementations.

This repository is structured to facilitate navigation within the project.

## Directory Structure

```
.
|-- data/                   # data used for the models (tactmat, imagenet)
|
|-- literature/             # .pdf files of supporting literature
|
|-- models/
|   |-- TactNetII_model/    # NN adapted from paper "Superhuman Performance and Tactile Material Classification"
|   |-- TBD/                # --
|
|-- notebooks/
|
|-- scripts/
|   |-- train/              # Script for training models
|   |-- uncertainty/        # Script for Monte Carlo Dropout (Uncertainty Metrics)
|   |-- visualization/      # Script for visualization of all metrics
|
|-- output/                 # Storage of all outputs
|
|-- ibinn_model/            # Existing Information Bottleneck Neural Network repo
|
|-- requirements.txt        # List of dependencies for the project
|-- requirements_cpu.txt        # List of dependencies for the project when used in CPU
|-- README.md               # Project description and setup instructions
```
## Setup Instructions

### 1. Cloning the Repository
Clone this repository to your local machine using:
```
git clone <repo-url>
```

### 2. Install Dependencies
Create a virtual environment and install the dependencies from `requirements.txt` (use `requirements_cpu.txt` if working on CPU):

```bash
pip install -r requirements.txt
```

## External Repositories

This project makes use of an adapted version of the code from the [ibinn_imagenet repository](https://github.com/RayDeeA/ibinn_imagenet), developed by RayDeeA. Proper credit is given according to the original licensing terms.

## External Datasets

This project uses the dataset from [DLR-TactMat](https://dlr-alr.github.io/dlr-tactmat/). Proper credit is given according to the original licensing terms.

