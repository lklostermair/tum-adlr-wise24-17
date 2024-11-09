# ADLR project repository for Team 17

## Overview
This project aims to adapt the Network described in [Mackowiak 2020][] for a tactile classification task, then benchmark it against the classifier architecture described in [Tulbure 2018][]. We will be using Google Compute Engine and Python for all implementations.

This repository is structured to facilitate navigation within the project.

## Directory Structure

```
.
|-- data/
|
|-- models/
|   |-- existing_model/    # Existing neural network to be adapted
|   |-- benchmark_model/   # Benchmark classifier architecture
|
|-- notebooks/
|
|-- scripts/
|
|-- results/
|
|-- visualization/
|
|-- requirements.txt       # List of dependencies for the project
|-- README.md              # Project description and setup instructions
```
## Setup Instructions

### 1. Cloning the Repository
Clone this repository to your local machine using:
```
git clone <repo-url>
```

### 2. Install Dependencies
Create a virtual environment and install the dependencies from `requirements.txt`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Running Notebooks on Google Compute Engine
1. Connect to Google Compute Engine instance.
2. Upload the repository or clone it directly.
3. Install Jupyter Notebook using the following command:
   ```bash
   pip install jupyter
   ```
4. Launch Jupyter Notebook to work on `.ipynb` files:
   ```bash
   jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
   ```
   Access the notebook via external IP address of your Google Compute Engine instance.

### 6. Evaluating the Models
After training, model checkpoints and metrics will be saved in the `results/` directory. You can evaluate the performance and visualize results using the stored metrics.

