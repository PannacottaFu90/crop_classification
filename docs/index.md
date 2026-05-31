# Crop Classification Project Documentation

## Overview

This repository contains a Python pipeline for crop classification using Sentinel-2 imagery and Azure storage. The project is structured to be portfolio-ready with reusable modules, a command-line interface, and documentation.

## What is included

- `crop_classification/` — core package modules
- `main.py` — command-line interface for pipeline steps
- `scripts/download_s2.py` — example STAC search/download script
- `requirements.txt` — dependency list
- `pyproject.toml` — package metadata and build configuration
- `README.md` — project overview and usage guide
- `tests/` — unit tests for core functions
- `.github/workflows/python-package.yml` — CI workflow for syntax checking

## Why `pyproject.toml`?

`pyproject.toml` is the modern Python packaging configuration file:

- it defines how the package is built
- it lists the package dependencies
- it enables `pip install .` for local installation
- it makes the project ready for publishing to PyPI in the future

## Getting started

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy the configuration template:

   ```bash
   cp crop_classification/config_template.py config.py
   ```

4. Edit `config.py` and add your Azure Storage and Copernicus credentials.

## How to try the code

The repository is mainly a pipeline template. To try it locally you need:

- a `config.py` file with valid Azure / Copernicus credentials
- Sentinel-2 assets or a matching Azure container
- a `train_mask.tif` file for training labels

Example commands:

```bash
python main.py search --start-date 2023-02-01 --end-date 2023-10-31 --max-cloud 20
python main.py composite --month 2023-06
python main.py train --mask-path train_mask.tif --output-model checkpoints/rf_model_final.joblib
python main.py predict --model checkpoints/rf_model_final.joblib --stack-path data/stack.tif --output-path results/classification_map.tif
python main.py visualize --input-path results/classification_map.tif --lut-path crop_dictionary.csv --output-path results/classification_map.png
```

## What you are showing on GitHub

This repository shows:

- a clean Python project package
- data-processing and ML workflow modules
- a simple CLI for running pipeline steps
- a documentation page and GitHub Actions CI config
- a portfolio-ready structure with `README.md`, `LICENSE`, and `pyproject.toml`

## What is immediately testable?

Without data, the best immediate proof is:

- install dependencies
- import the package modules
- run `python main.py --help`
- run the unit tests (after installing `pytest`)

That means anyone can see the project is well-organized and can run the CLI scaffolding.
