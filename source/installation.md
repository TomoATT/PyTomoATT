# Installation

## Conda environment

### Create a new Conda environment

We recommend installing PyTomoATT in a conda environment to ensure compatibility with the required dependencies.

```{note}
If you don't already have Conda installed, you can download and install it from the official Conda website: [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
```

Create a new Conda environment for Pytomoatt with the following command:
```
conda create -n pytomoatt python=3.11
```

This creates a new environment named "pytomoatt" with Python version 3.10 installed.


### Activate the Conda environment

Activate the newly created environment with the following command:

```
conda activate pytomoatt
```

## Installing stable version via PyPI

```
pip install pytomoatt
```

## Installing development version from source

- Clone the repository

Users can install PyTomoATT from the source code available on GitHub. First, clone the PyTomoATT repository:
```
git clone --branch=devel https://github.com/MIGG-NTU/PyTomoATT.git
```

- Install using `pip`

```
pip install .
```
