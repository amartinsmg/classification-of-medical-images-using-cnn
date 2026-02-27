#!/usr/bin/env python
# coding: utf-8

"""
TRANSFERLEARNING FOR RADIOLOGICAL IMAGE CLASSIFICATION

Exemple task: Classify chest X-ray images as normal or pneumonia.

Expected directory structure:

base/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── models/
└── results/

"""

import os
import argparse
import tensorflow as tf
import keras
import typing
import numpy as np
import sklearn
import json
import tabulate

# ======================================
# PATH CONFIGURATION
# ======================================


def configure_paths(base_dir: str, experiment_name: str = "", run_id: int = 1):
    test_dir = os.path.join(base_dir, "data", "test")
    model_path = os.path.join(base_dir, "models", "model.keras")
    metrics_path = ""
    if len(experiment_name) == 0:
        metrics_path = os.path.join(base_dir, "results", "metrics.json")
    else:
        metrics_path = os.path.join(
            base_dir, "results", experiment_name, f"run{run_id:d2}", "metrics.json"
        )

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    return test_dir, model_path, metrics_path


# ======================================
# DATASET LOADING
# ======================================


def load_test_data(
    test_dir: str,
    normalization: str = "recaling",
    base_model: str = "resnet",
    image_size: typing.Tuple[int, int] = (224, 224),
    batch_size: int = 32,
):

    # DATASET LOADING

    test_data = keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )

    # NORMALIZATION

    normalization_layer = None

    if normalization == "rescaling":
        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    elif normalization == "preprocess_input":
        if base_model == "resnet":
            normalization_layer = keras.applications.resnet.preprocess_input
        elif base_model == "densenet":
            normalization_layer = keras.applications.densenet.preprocess_input
        elif base_model == "efficientnet":
            normalization_layer = keras.applications.efficientnet.preprocess_input

    test_data = test_data.map(lambda x, y: (normalization_layer(x), y))

    # PERFORMANCE OPTIMIZATION

    test_data = test_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return test_data


# ======================================
# METRICS CALCULATION
# ======================================


def calculate_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5):

    # THRESHOLDING TO GET PREDICTED LABELS

    y_pred = (y_scores > threshold).astype(int)

    # METRICS CALCULATION

    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)

    TN, FP, FN, TP = confusion_matrix.ravel()

    summary = {
        "decision_threshold": threshold,
        "accuracy": float(sklearn.metrics.accuracy_score(y_true, y_pred)),
        "precision": float(sklearn.metrics.precision_score(y_true, y_pred)),
        "recall": float(sklearn.metrics.recall_score(y_true, y_pred)),
        "f1_score": float(sklearn.metrics.f1_score(y_true, y_pred)),
        "specificity": float(TN / (TN + FP) if (TN + FP) > 0 else 0),
        "auc-roc": float(sklearn.metrics.roc_auc_score(y_true, y_scores)),
    }

    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_scores)

    details = {
        "confusion_matrix": {
            "TN": int(TN),
            "FP": int(FP),
            "FN": int(FN),
            "TP": int(TP),
        },
        "roc-curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }

    return summary, details


# ======================================
# METRICS REPORTING
# ======================================


def save_and_report(
    metrics_path: str, summary: dict, details: dict, experiment_name: str = ""
):

    # SAVE METRICS TO JSON

    with open(metrics_path, "w") as f:
        json.dump({"summary": summary, "details": details}, f, indent=4)

    # PRINT SUMMARY TABLE

    summary_headers = [
        "Decision Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "Specificity",
        "AUC ROC",
    ]

    print(f"\n Experiment Summary ({experiment_name})")
    print(
        tabulate.tabulate([summary.values()], headers=summary_headers, tablefmt="grid")
    )


# ======================================
# MODEL TESTING AND EVALUATION
# ======================================


def test_pipeline(
    base_dir: str,
    experiment_name: str = "",
    run_id: int = 1,
    normalization: str = "rescaling",
    base_model: str = "resnet",
    image_size: typing.Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    threshold: float = 0.5,
):

    # PATH CONFIGURATION AND LOADING OF TEST DATA AND MODEL

    test_dir, model_path, metrics_path = configure_paths(
        base_dir, experiment_name=experiment_name, run_id=run_id
    )

    test_data = load_test_data(
        test_dir,
        normalization=normalization,
        base_model=base_model,
        image_size=image_size,
        batch_size=batch_size,
    )

    model = keras.models.load_model(model_path)

    # EXTRACTION OF TRUE LABELS AND PREDICTING SCORES

    y_true = np.concatenate([y for x, y in test_data], axis=0).flatten()

    y_scores = model.predict(test_data).flatten()

    # CALCULATE AND METRICS

    summary, details = calculate_metrics(y_true, y_scores, threshold=threshold)

    save_and_report(metrics_path, summary, details, experiment_name=experiment_name)

    return y_true, y_scores


# ======================================
# MAIN FUNCTION
# ======================================


def main(args):
    test_pipeline(base_dir=args.base_dir)


if __name__ == "__main__":

    # ARGUMENT PARSING

    parser = argparse.ArgumentParser(
        description="Test a CNN model for radiological image classification."
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        help="Base project directory. Default is the parent directory of the script.",
    )

    args = parser.parse_args()

    main(args)
