#!/usr/bin/env python
# coding: utf-8

"""
TRANSFERLEARNING FOR RADIOLOGICAL IMAGE CLASSIFICATION

Expected directory structure:

base/
└── results/
      └── experiment_name
            ├── run01/
            │    ├── config.json
            │    ├── history.json
            │    └── metrics.json
            ├── run02/
            │    ├── config.json
            │    ├── history.json
            │    └── metrics.json
            └── run03/
                 ├── config.json
                 ├── history.json
                 └── metrics.json

"""

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import scipy

# ================================================
# LOAD AND AGGREGATE EXPERIMENT RESULTS
# ================================================

def load_runs(experiment_path) -> dict:
    exp_path = pathlib.Path(experiment_path)

    runs_dirs = sorted([d for d in exp_path.iterdir() if d.is_dir()])

    if not runs_dirs:
        raise Exception(f"No runs found in {exp_path}")

    metrics_rows = []
    history_dfs = []
    tprs = []
    aucs = []
    cm_totals = {"TN": 0, "FP": 0, "FN": 0, "TP": 0}
    common_fpr = np.linspace(0, 1, 200)

    for run_dir in runs_dirs:
        # -- metrics.json --
        metrics_path = run_dir / "metrics.json"

        if not metrics_path.exists():
            raise Exception(f"Metrics file not found: {metrics_path}")

        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)

        summary = {
            k: v
            for k, v in metrics_data["summary"].items()
            if k != "decision-threshold"
        }
        metrics_rows.append(summary)

        cm = metrics_data["details"]["confusion-matrix"]

        for k in cm_totals.keys():
            cm_totals[k] += cm[k]

        roc = metrics_data["details"]["roc-curve"]
        fpr = np.array(roc["fpr"])
        tpr = np.array(roc["tpr"])
        interp_fn = scipy.interpolate.interp1d(
            fpr, tpr, kind="linear", bounds_error=False, fill_value=(0.0, 1.0)
        )
        tprs.append(interp_fn(common_fpr))
        aucs.append(float(np.trapezoid(tpr, fpr)))

        # -- history.csv --

        history_path = run_dir / "history.csv"

        if not history_path.exists():
            warnings.warn(f"History file not found: {history_path}")
        else:
            history_dfs.append(pd.read_csv(history_path))

        # CREATE METRICS DATAFRAME

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.index = [d.name for d in runs_dirs]
        metrics_df.index.name = "run"

        # AGGREGATE TRAINING HISTORY

        history_df = _aggregate_history(history_dfs)

        # CALCULATE ROC AND AUC STATISTICS

        tprs_arr = np.array(tprs)
        roc_df = pd.DataFrame(
            {
                "fpr": common_fpr,
                "tpr-mean": tprs_arr.mean(axis=0),
                "tpr-std": tprs_arr.std(axis=0),
            }
        )
        auc = {
            "mean": float(np.mean(aucs)),
            "std": float(np.std(aucs)),
        }

        # CREATE CONFUSION MATRIX DATAFRAME

        cm_df = pd.DataFrame([cm_totals])

    return {
        "name": exp_path.name,
        "metrics": metrics_df,
        "history": history_df,
        "roc": roc_df,
        "auc": auc,
        "confusion-matrix": cm_df,
    }


def load_experiments(base_result_dir, experiment_names: list[str]):
    base = pathlib.Path(base_result_dir)

    return [load_runs(base / exp_name) for exp_name in experiment_names]


def _aggregate_history(history_dfs: list[pd.DataFrame]):
    cols = history_dfs[0].columns.to_list()
    stacked = np.stack([df[cols].values for df in history_dfs], axis=0)

    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)

    result = {}
    for i, col in enumerate(cols):
        result[f"{col}-mean"] = mean[:, i]
        result[f"{col}-std"] = std[:, i]

    df = pd.DataFrame(result)
    df.index = range(1, len(df) + 1)
    df.index.name = "epoch"
    return df
