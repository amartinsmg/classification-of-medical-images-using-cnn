#!/usr/bin/env python
# coding: utf-8

"""
TRANSFERLEARNING WITH RESNET50 FOR RADIOLOGICAL IMAGE CLASSIFICATION

Exemple task: Classify chest X-ray images as normal or pneumonia.

Expected directory structure:

base/
├── data/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── val/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── models/
└── results/

"""

import os
import argparse
import tensorflow as tf
import keras
import json
import typing


# ======================================
# PATH CONFIGURATION
# ======================================


def configure_paths(base_path):
    train_dir = os.path.join(base_path, "data", "train")
    val_dir = os.path.join(base_path, "data", "val")
    model_path = os.path.join(base_path, "models", "xray_images.keras")
    model_weights_path = os.path.join(base_path, "models", "xray_images.weights.h5")
    result_path = os.path.join(base_path, "results", "xray_images.json")
    config_path = os.path.join(base_path, "results", "config.json")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    return train_dir, val_dir, model_path, model_weights_path, result_path, config_path


# ======================================
# DATASET LOADING
# ======================================


def load_datasets(train_dir, val_dir, image_size=(224, 224), batch_size=32, seed=42):

    train_data = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=True,
        seed=seed,
    )

    val_data = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )

    # NORMALIZATION, DATA AUGMENTATION AND PERFORMANCE OPTIMIZATION

    normalization_layer = keras.layers.Rescaling(1.0 / 255)

    train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
    val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

    data_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomZoom(0.1),
        ]
    )

    train_data = train_data.map(lambda x, y: (data_augmentation(x, training=True), y))

    train_data = train_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_data = val_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_data, val_data


# ======================================
# MODEL DEFINITION AND COMPILATION
# ======================================


def build_model():

    base_model = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.models.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.Precision(name="precision"),
        ],
    )

    return model


# ======================================
# MODEL TRAINING
# ======================================


def train_model(model, train_data, val_data, epochs=10):

    history = model.fit(train_data, validation_data=val_data, epochs=epochs)

    return model, history


# ======================================
# RESULTS SAVING
# ======================================


def save_results(
    model,
    history,
    config_dict,
    model_path,
    model_weights_path,
    result_path,
    config_path,
):
    model.save(model_path)

    model.save_weights(model_weights_path)

    history_dict = history.history

    with open(result_path, "w") as f:
        json.dump(history_dict, f, indent=4)

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)


# ======================================
# TRAINING PIPELINE
# ======================================


def train_pipeline(
    base_dir: str,
    image_size: typing.Tuple[int, int],
    batch_size: int,
    epochs: int,
    seed: int,
):
    train_dir, val_dir, model_path, model_weights_path, result_path, config_path = (
        configure_paths(base_dir)
    )

    train_data, val_data = load_datasets(
        train_dir,
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )

    model = build_model()

    model, history = train_model(model, train_data, val_data, epochs=epochs)

    config_dict = {
        "base_model": "ResNet50",
        "weights": "imagenet",
        "optimizer": "adam",
        "preprocessing": "rescaling, data augmentation",
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
    }

    save_results(
        model,
        history,
        config_dict,
        model_path,
        model_weights_path,
        result_path,
        config_path,
    )

    print("Training completed. Model and results saved.")


# ======================================
# MAIN FUNCTION
# ======================================


def main(args):
    train_pipeline(base_dir=args.base_dir)


if __name__ == "__main__":

    # ARGUMENT PARSING

    parser = argparse.ArgumentParser(
        description="Train a CNN model for radiological image classification."
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        help="Base project directory. Default is the parent directory of the script directory.",
    )

    args = parser.parse_args()

    main(args)


# ======================================
# ACADEMIC DISCLAIMER
# ======================================

# This model performs statistical image classification and should be considered a decision support tool.
# It is not intended for real-world clinical use or medical diagnosis, and should be used for academic and research purposes only.
# Always consult with medical professionals and follow ethical guidelines when working with medical data.
