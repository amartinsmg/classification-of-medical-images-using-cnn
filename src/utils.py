#!/usr/bin/env python
# coding: utf-8

"""
TRANSFERLEARNING FOR RADIOLOGICAL IMAGE CLASSIFICATION

utils.py: Utility functions for model evaluation and threshold optimization.

Expected directory structure:

base/
└── results/
      └── experiment_name
            ├── run01/
            │    └── predictions.csv
            ├── run02/
            │    └── predictions.csv
            └── run03/
                 └── predictions.csv


"""

import pathlib

import numpy as np
import pandas as pd
import sklearn as sk

# ================================================
# OPTIMAL THRESHOLD CALCULATION USING YOUDEN'S J STATISTIC
# ================================================


def optimal_threshold(experiment_path: str | pathlib.Path):
    exp_path = pathlib.Path(experiment_path)

    run_dirs = sorted([d for d in exp_path.iterdir() if d.is_dir()])

    if not run_dirs:
        raise Exception(f"No runs found in {exp_path}")

    dfs = [pd.read_csv(run_dir / "predictions.csv") for run_dir in run_dirs]
    combined = pd.concat(dfs)

    fpr, tpr, thresholds = sk.metrics.roc_curve(
        combined["y_true"], combined["y_scores"]
    )

    youden_j = np.argmax(tpr - fpr)
    threshold = float(thresholds[youden_j])

    return threshold
