#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="mtl-protein-disorder",
    version="0.1.0",
    description="Multi-Task Learning Network for Protein Disorder Prediction",
    author="Daryna Sova",
    author_email="daryna.sova@tum.de",
    url="https://github.com/DarynaSova/mtl-protein-disorder",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.3.3",
        "joblib>=1.4.0"
    ],
    entry_points={
        "console_scripts": [
            "mtl-train=mtl_protein_disorder.train:main",
            "mtl-infer=mtl_protein_disorder.inference:main",
            "mtl-generate-data=mtl_protein_disorder.generate_data:main",
        ],
    },
)
