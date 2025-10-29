# Multi-Task Protein Disorder Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A cross-platform PyTorch package for multi-task prediction of protein disorder: joint classification of intrinsic disorder (DisProt_Label) and prediction of per-residue flexibility (rmsf).

## Features

- 🚀 Cross-Platform Compatible (CUDA, MPS, CPU)
- 🧬 Multi-Task Learning (simultaneous disorder classification and flexibility regression)
- 🏗️ Simple Feedforward Neural Network architecture (easy to modify)
- 📊 Trains on your CSV, predicts on new data
- ⚡ Easily extensible to new features or additional disorder/flexibility tasks

## Quick Start

**1. Install:**
poetry install
poetry run pip install torch pandas scikit-learn joblib


**2. Train on your CSV:**

Your CSV must contain: rmsf, bfactors, plddt, gscore, pLDDT_flex, residue_letter, DisProt_Label

poetry run python mtl_protein_disorder/train.py your_data.csv


**3. Predict on new data:**

Your CSV must contain: rmsf, bfactors, plddt, gscore, pLDDT_flex, residue_letter
poetry run python mtl_protein_disorder/inference.py test_inference_data.csv


Produces a CSV with predicted disorder probability and predicted flexibility for each residue.

## License

MIT License


