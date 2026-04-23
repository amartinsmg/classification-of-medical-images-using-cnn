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
            │    ├── metrics.json
            │    └── config.json
            ├── run02/
            │    └── metrics.json
            └── run03/
                 └── metrics.json

"""


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
