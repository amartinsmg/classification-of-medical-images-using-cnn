#!/usr/bin/env python
# coding: utf-8

"""
TRANSFERLEARNING FOR RADIOLOGICAL IMAGE CLASSIFICATION

test.py: Testing and evaluation pipeline for CNN models trained on radiological images.

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

import argparse
import json
import pathlib
import tabulate
import typing

import keras
import numpy as np
import pandas as pd
import sklearn as sk
import tensorflow as tf

# ======================================
# PATH CONFIGURATION
# ======================================


def _configure_paths(
    base_dir: str | pathlib.Path,
    experiment_name: str = "",
    run_id: int = 1,
    versioning_models: bool = False,
):
    base_path = pathlib.Path(base_dir).resolve()

    if len(experiment_name) == 0:
        MODELS_DIR = base_path / "models"
        RESULT_DIR = base_path / "results"
    else:
        if versioning_models:
            MODELS_DIR = base_path / "models" / experiment_name / f"run{run_id:02d}"
        else:
            MODELS_DIR = base_path / "models" / f"run{run_id:02d}"
        RESULT_DIR = base_path / "results" / experiment_name / f"run{run_id:02d}"

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "model.keras"

    test_dir = base_path / "data" / "test"

    metrics_path = RESULT_DIR / "metrics.json"

    predictions_path = RESULT_DIR / "predictions.csv"

    return test_dir, model_path, metrics_path, predictions_path


# ======================================
# DATASET LOADING
# ======================================


def _load_test_data(
    test_dir: str | pathlib.Path,
    normalization: str = "rescaling",
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
    elif normalization == "preprocess-input":
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


def _calculate_metrics(
    y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5
):

    # THRESHOLDING TO GET PREDICTED LABELS

    y_pred = (y_scores > threshold).astype(int)

    # METRICS CALCULATION

    confusion_matrix = sk.metrics.confusion_matrix(y_true, y_pred)

    TN, FP, FN, TP = confusion_matrix.ravel()

    summary = {
        "decision-threshold": threshold,
        "accuracy": float(sk.metrics.accuracy_score(y_true, y_pred)),
        "precision": float(sk.metrics.precision_score(y_true, y_pred)),
        "recall": float(sk.metrics.recall_score(y_true, y_pred)),
        "f1-score": float(sk.metrics.f1_score(y_true, y_pred)),
        "specificity": float(TN / (TN + FP) if (TN + FP) > 0 else 0),
        "auc-roc": float(sk.metrics.roc_auc_score(y_true, y_scores)),
    }

    fpr, tpr, _ = sk.metrics.roc_curve(y_true, y_scores)

    details = {
        "confusion-matrix": {
            "TN": int(TN),
            "FP": int(FP),
            "FN": int(FN),
            "TP": int(TP),
        },
        "roc-curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }

    return summary, details


# ======================================
# METRICS SAVING AND REPORTING
# ======================================


def _save_metrics_and_report(
    metrics_path: str | pathlib.Path,
    summary: dict,
    details: dict,
    experiment_name: str = "",
    run_id: int = 1,
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

    print(f"\n Experiment Summary ({experiment_name}: run {run_id})")
    print(
        tabulate.tabulate([summary.values()], headers=summary_headers, tablefmt="grid")
    )


# ======================================
# PREDICTIONS SAVING
# ======================================


def save_predictions(
    predictions_path: str | pathlib.Path,
    y_true: np.ndarray,
    y_scores: np.ndarray,
):
    df = pd.DataFrame({"y_true": y_true, "y_scores": y_scores})
    df.to_csv(predictions_path, index=False)


# ======================================
# MODEL TESTING AND EVALUATION
# ======================================


def test_pipeline(
    base_dir: str | pathlib.Path,
    experiment_name: str = "",
    run_id: int = 1,
    normalization: str = "rescaling",
    base_model: str = "resnet",
    image_size: typing.Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    threshold: float = 0.5,
    versioning_models: bool = False,
    save_predictions: bool = False,
):

    # PATH CONFIGURATION AND LOADING OF TEST DATA AND MODEL

    test_dir, model_path, metrics_path, predictions_path = _configure_paths(
        base_dir,
        experiment_name=experiment_name,
        run_id=run_id,
        versioning_models=versioning_models,
    )

    test_data = _load_test_data(
        test_dir,
        normalization=normalization,
        base_model=base_model,
        image_size=image_size,
        batch_size=batch_size,
    )

    model = keras.models.load_model(model_path)

    # EXTRACTION OF TRUE LABELS AND PREDICTING SCORES

    y_true = np.concatenate([y for _, y in test_data], axis=0).flatten()

    y_scores = model.predict(test_data).flatten()

    # SAVE PREDICTIONS TO CSV OR CALCULATE METRICS AND SAVE REPORT

    if save_predictions:
        save_predictions(predictions_path, y_true, y_scores)

        print("Test completed. Predictions saved.")

        return None
    else:
        summary, details = _calculate_metrics(y_true, y_scores, threshold=threshold)

        _save_metrics_and_report(
            metrics_path,
            summary,
            details,
            experiment_name=experiment_name,
            run_id=run_id,
        )

        print("Test completed. Metrics saved.")

        return summary


# ======================================
# MAIN EXECUTION
# ======================================

if __name__ == "__main__":

    # ARGUMENT PARSING

    parser = argparse.ArgumentParser(
        description="Test a CNN model for radiological image classification."
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default=pathlib.Path(__file__).resolve().parents[1],
        help="Base project directory. Default is the parent directory of the script.",
    )

    args = parser.parse_args()

    # EXECUTE THE TEST PIPELINE

    test_pipeline(base_dir=args.base_dir)
