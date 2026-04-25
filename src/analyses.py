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

import numpy as np
import pandas as pd
import scipy


def load_runs(experiment_path) -> list[dict]:
    exp_path = pathlib.Path(experiment_path)

    runs_dirs = [d for d in exp_path.iterdir() if d.is_dir()]

    if not runs_dirs:
        raise Exception(f"No runs found in {exp_path}")

    runs = []
    for run_dir in runs_dirs:
        run = {}
        for fname in ["config.json", "metrics.json", "history.csv"]:
            fpath = run_dir / fname
            if not fpath.exists():
                print(f"File not found: {fpath}")
            elif fname.endswith(".json"):
                with open(fpath) as f:
                    run[fname.replace(".json", "")] = json.load(f)
            elif fname.endswith(".csv"):
                with open(fpath) as f:
                    run[fname.replace(".csv", "")] = pd.DataFrame(pd.read_csv(f))
            else:
                raise Exception(f"Unknown file type: {fname}")

        runs.append(run)

    return runs


def load_experiments(base_result_dir, experiment_names: list[str]):
    base = pathlib.Path(base_result_dir)
    experiments = []

    for exp_name in experiment_names:
        experiments.append({"name": exp_name, "runs": load_runs(base / exp_name)})

    return experiments


def aggregate_metrics(experiment: list[dict]):
    runs = experiment["runs"]
    keys = runs[0]["metrics"]["summary"].keys()
    metrics = {}

    for key in keys:
        values = [run["metrics"]["summary"][key] for run in runs]
        metrics[key] = (
            {"mean": np.mean(values), "std": np.std(values)}
            if not key == "decision-threshold"
            else values[0]
        )

    return metrics


def aggregate_history(experiment: list[dict]):
    runs = experiment["runs"]
    keys = runs[0]["history"].keys()
    result = {}

    for key in keys:
        matrix = np.array([run["history"][key] for run in runs])
        result[key] = {
            "mean": matrix.mean(axis=0).tolist(),
            "std": matrix.std(axis=0).tolist(),
        }

    return result


def aggregate_cm(experiment: list[dict]):
    runs = experiment["runs"]
    keys = ("TN", "FP", "FN", "TP")
    totals = {k: 0 for k in keys}

    for run in runs:
        cm = run["metrics"]["details"]["confusion-matrix"]
        for k in keys:
            totals[k] += cm[k]

    return totals


def aggregate_roc(experiment: list[dict], n_points: int = 200):
    runs = experiment["runs"]
    common_fpr = np.linspace(0, 1, n_points)
    tprs = []
    aucs = []

    for run in runs:
        roc = run["metrics"]["details"]["roc-curve"]
        fpr = np.array(roc["fpr"])
        tpr = np.array(roc["tpr"])

        inter_fn = scipy.interpolate.interp1d(
            fpr, tpr, kind="linear", bounds_error=False, fill_value=(0.0, 1.0)
        )
        tprs.append(inter_fn(common_fpr))

        aucs.append(float(np.trapz(tpr, fpr)))

    tprs = np.array(tprs)

    return {
        "fpr": common_fpr.tolist(),
        "tpr_mean": tprs.mean(axis=0).tolist(),
        "tpr_std": tprs.std(axis=0).tolist(),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
    }
