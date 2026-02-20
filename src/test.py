#!/usr/bin/env python
# coding: utf-8

"""
TRANSFERLEARNING WITH RESNET50 FOR RADIOLOGICAL IMAGE CLASSIFICATION

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
import sklearn
import json

# ======================================
# PATH CONFIGURATION
# ======================================


def configure_paths(base_path):
    test_dir = os.path.join(base_path, "data", "test")
    model_path = os.path.join(base_path, "models", "xray_images.keras")
    result_path = os.path.join(base_path, "results", "xray_test_results.json")

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    return test_dir, model_path, result_path


# ======================================
# DATASET LOADING
# ======================================


def load_test_data(test_dir, image_size=(224, 224), batch_size=32):
    test_data = keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )

    # NORMALIZATION AND PERFORMANCE OPTIMIZATION

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

    test_data = test_data.map(lambda x, y: (normalization_layer(x), y))

    test_data = test_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return test_data


# ======================================
# MODEL TESTING AND EVALUATION
# ======================================


def test_pipeline(base_dir, image_size=(224, 224), batch_size=32):
    test_dir, model_path, result_path = configure_paths(base_dir)

    test_data = load_test_data(test_dir, image_size=image_size, batch_size=batch_size)

    model = keras.models.load_model(model_path)

    results = model.evaluate(test_data, return_dict=True)
    y_true = []
    y_pred = []

    for images, labels in test_data:
        predictions = model.predict(images)
        predictions = (predictions > 0.5).astype(int)

        y_true.extend(labels.numpy())
        y_pred.extend(predictions.flatten())

    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)

    TN, FP, FN, TP = confusion_matrix.ravel().tolist()

    results["confusion_matrix"] = {
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
        "TP": int(TP),
    }

    results["f1_score"] = sklearn.metrics.f1_score(y_true, y_pred)

    results["specificity"] = TN / (TN + FP)

    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    print(json.dumps(results, indent=4))

    return y_true, y_pred, results


# ======================================
# MAIN FUNCTION
# ======================================


def main(args):
    test_pipeline(
        base_dir=args.base_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )


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
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Image size for testing. Default is 224 X 224.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for testing. Default is 32.",
    )

    args = parser.parse_args()

    args.image_size = tuple(args.image_size)

    main(args)
