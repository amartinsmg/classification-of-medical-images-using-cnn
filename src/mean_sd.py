#!/usr/bin/env python
# coding: utf-8

"""
TRANSFERLEARNING FOR RADIOLOGICAL IMAGE CLASSIFICATION

Exemple task: Classify chest X-ray images as normal or pneumonia.

Expected directory structure:

base/
└── results/
      └── experiment_name
            ├── run01/
            │    ├── config.json
            │    ├── history.json
            │    └── metrics.json
            ├── run02/
            │    ├── history.json
            │    └── metrics.json
            └── run03/
                 ├── history.json
                 └── metrics.json

"""

import json
import glob

def calculate_mean_and_std(
    base_dir: str,
    experiment_name: str = "",
    total_runs: int = 3,
):
    summary = [
        "decision_threshold",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "specificity",
        "auc-roc",
    ]

    

    return None
