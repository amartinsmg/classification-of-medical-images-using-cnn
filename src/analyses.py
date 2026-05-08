#!/usr/bin/env python
# coding: utf-8

"""
TRANSFERLEARNING FOR RADIOLOGICAL IMAGE CLASSIFICATION

analyses.py: Analysis and visualization tools for summarizing results from multiple runs and experiments.

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

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy

# ================================================
# LOAD AND AGGREGATE EXPERIMENTS RESULTS
# ================================================

# LOAD ALL RUNS FOR A SINGLE EXPERIMENT


def load_runs(experiment_path) -> dict:
    exp_path = pathlib.Path(experiment_path)

    run_dirs = sorted([d for d in exp_path.iterdir() if d.is_dir()])

    if not run_dirs:
        raise Exception(f"No runs found in {exp_path}")

    metrics_rows = []
    history_dfs = []
    tprs = []
    aucs = []
    cm_totals = {"TN": 0, "FP": 0, "FN": 0, "TP": 0}
    common_fpr = np.linspace(0, 1, 200)

    for run_dir in run_dirs:
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
    metrics_df.index = [d.name for d in run_dirs]
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


# LOAD MULTIPLE EXPERIMENTS


def load_experiments(base_result_dir, experiment_names: list[str]):
    base = pathlib.Path(base_result_dir)

    return [load_runs(base / exp_name) for exp_name in experiment_names]


# AGGREGATE TRAINING HISTORY FROM MULTIPLE RUNS


def _aggregate_history(history_dfs: list[pd.DataFrame]):
    cols = history_dfs[0].columns.to_list()
    stacked = np.stack([df[cols].values for df in history_dfs], axis=0)

    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)

    result = {}
    for i, col in enumerate(cols):
        result[f"{col}_mean"] = mean[:, i]
        result[f"{col}_std"] = std[:, i]

    df = pd.DataFrame(result)
    df.index = range(1, len(df) + 1)
    df.index.name = "epoch"
    return df


# ================================================
# GENERATE SUMMARY TABLES
# ================================================


def metrics_table(experiments: list[dict]) -> pd.DataFrame:
    cols = experiments[0]["metrics"].columns.tolist()
    rows = []
    for exp in experiments:
        stacked = exp["metrics"][cols].values
        row = {"experiment": exp["name"]}
        for i, col in enumerate(cols):
            row[f"{col}-mean"] = stacked[:, i].mean()
            row[f"{col}-std"] = stacked[:, i].std()
        rows.append(row)

    return pd.DataFrame(rows).set_index("experiment")


# ================================================
# PLOTTING FUNCTIONS
# ================================================

# COLORS FOR PLOTTING

_PALETE = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# PLOT TRAINING AND VALIDATION METRICS WITH SHADING FOR STD ACROSS RUNS


def plot_training_history(
    experiments: list[dict],
    metrics: list[str] = ("accuracy", "loss", "AUC"),
    val: bool = True,
    figsize_per_col: tuple = (5, 3.5),
) -> plt.Figure:

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_col[0] * n, figsize_per_col[1]))
    if n == 1:
        axes = [axes]

    for col, metric in enumerate(metrics):
        ax = axes[col]
        for i, exp in enumerate(experiments):
            color = _PALETE[i % len(_PALETE)]
            history = exp["history"]
            label = exp["name"]
            epochs = history.index

            mean_train = history[f"{metric}_mean"].values
            std_train = history[f"{metric}_std"].values

            ax.plot(epochs, mean_train, color=color, label=f"{label} - train")
            ax.fill_between(
                epochs,
                mean_train - std_train,
                mean_train + std_train,
                alpha=0.15,
                color=color,
            )

            val_mean_col = f"val_{metric}_mean"
            val_std_col = f"val_{metric}_std"
            if val and val_mean_col in history.columns:
                mean_val = history[val_mean_col].values
                std_val = history[val_std_col].values
                ax.plot(
                    epochs,
                    mean_val,
                    color=color,
                    linestyle="--",
                    label=f"{label} - val",
                )
                ax.fill_between(
                    epochs,
                    mean_val - std_val,
                    mean_val + std_val,
                    alpha=0.10,
                    color=color,
                )

        ax.set_title(metric.upper())
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# PLOT ROC CURVES WITH SHADING FOR STD ACROSS RUNS


def plot_roc_curves(
    experiments: list[dict],
    figsize: tuple = (5, 5),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    for i, exp in enumerate(experiments):
        color = _PALETE[i % len(_PALETE)]
        roc = exp["roc"]
        auc = exp["auc"]
        label = exp["name"]

        ax.plot(
            roc["fpr"],
            roc["tpr-mean"],
            color=color,
            label=f"{label} (AUC = {auc["mean"]:.3f} ± {auc["std"]:.3f})",
        )
        ax.fill_between(
            roc["fpr"],
            roc["tpr-mean"] - roc["tpr-std"],
            roc["tpr-mean"] + roc["tpr-std"],
            alpha=0.15,
            color=color,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth="0.8", label="Random")
    ax.set_title("ROC Curve (mean ± std across runs)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# PLOT CONFUSION MATRICES WITH SHADING FOR STD ACROSS RUNS


def plot_confusion_matrices(
    experiments: list[dict],
    class_names: list[str] = ("Negative", "Positive"),
    cmap: str = "Blues",
    figsize_per_col: tuple = (3.5, 3),
) -> plt.Figure:

    n = len(experiments)
    fig, axes = (
        plt.subplots(1, n, figsize=(figsize_per_col[0] * n, figsize_per_col[1]))
        if n < 4
        else plt.subplots(n, 1, figsize=(figsize_per_col[0], figsize_per_col[1] * n))
    )
    if n == 1:
        axes = [axes]

    for col, exp in enumerate(experiments):
        ax = axes[col]
        cm_row = exp["confusion-matrix"].iloc[0]
        cm = np.array([[cm_row["TN"], cm_row["FP"]], [cm_row["FN"], cm_row["TP"]]])

        ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_title(exp["name"], fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        thresh = cm.max() / 2
        for r in range(2):
            for c in range(2):
                ax.text(
                    c,
                    r,
                    str(cm[r, c]),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if cm[r, c] > thresh else "black",
                )

    fig.suptitle("Confusion Matrices (sum of runs)", y=1.02)
    fig.tight_layout()
    return fig


# ===========================================================================
# FULL COMPARISON FUNCTION TO LOAD, ANALYZE AND PLOT MULTIPLE EXPERIMENTS
# ===========================================================================


def full_comparison(
    base_result_dir,
    experiment_names: list[str],
    history_metrics: list[str] = ("accuracy", "loss", "AUC"),
    class_names: list[str] = ("Negative", "Positive"),
    show_plot: bool = False,
    save_dir: pathlib.Path | str | None = None,
    separe_archs: bool = False,
    archs: list[str] = ["resnet", "densenet", "efficientnet"],
):
    experiments = load_experiments(base_result_dir, experiment_names)

    metrics = metrics_table(experiments)

    fig_hist = plot_training_history(experiments, metrics=history_metrics)
    fig_roc = plot_roc_curves(experiments)
    fig_cm = plot_confusion_matrices(experiments, class_names=class_names)

    # SAVE RESULTS

    if save_dir is not None:
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        fig_hist.savefig(save_dir / "history.png", bbox_inches="tight", dpi=300)
        fig_roc.savefig(save_dir / "roc.png", bbox_inches="tight", dpi=300)
        fig_cm.savefig(save_dir / "cm.png", bbox_inches="tight", dpi=300)
        metrics.to_csv(save_dir / "metrics.csv")

        if separe_archs:
            save_separeted_archs(
                experiments,
                save_dir,
                archs=archs,
                history_metrics=history_metrics,
                class_names=class_names,
            )

        print(f"\nResults saved in {save_dir}")

    # DISPLAY RESULTS

    if show_plot:
        try:
            from IPython.display import display

            display(metrics)
        except ImportError:
            print(metrics.to_string())

        plt.show()
    else:
        plt.close("all")

    return experiments, metrics, fig_hist, fig_roc, fig_cm


# SAVE SEPARATED PLOTS FOR EACH ARCHITECTURE (ASSUMING EXPERIMENT NAMES START WITH ARCHITECTURE NAME)


def save_separeted_archs(
    experiments: list[dict],
    save_dir: pathlib.Path,
    archs: list[str] = ["resnet", "densenet", "efficientnet"],
    history_metrics: list[str] = ("accuracy", "loss", "AUC"),
    class_names: list[str] = ("Negative", "Positive"),
) -> None:
    for arch in archs:
        archs_exp = [exp for exp in experiments if exp["name"].startswith(arch)]
        save_dir_arch = save_dir / arch
        save_dir_arch.mkdir(parents=True, exist_ok=True)

        fig_hist = plot_training_history(archs_exp, metrics=history_metrics)
        fig_hist.savefig(save_dir_arch / "history.png", bbox_inches="tight", dpi=300)
        fig_roc = plot_roc_curves(archs_exp)
        fig_roc.savefig(save_dir_arch / "roc.png", bbox_inches="tight", dpi=300)
        fig_cm = plot_confusion_matrices(archs_exp, class_names=class_names)
        fig_cm.savefig(save_dir_arch / "cm.png", bbox_inches="tight", dpi=300)
        plt.close("all")
