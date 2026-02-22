#!/usr/bin/env python
# coding: utf-8

"""
TRANSFERLEARNING FOR RADIOLOGICAL IMAGE CLASSIFICATION

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


def configure_paths(base_dir: str, experiment_name: str = "", run_id: int = 1):
    if len(experiment_name) == 0:
        RESULT_DIR = RESULT_DIR = os.path.join(base_dir, "results")
    else:
        RESULT_DIR = os.path.join(
            base_dir, "results", experiment_name, f"run{run_id:d2}"
        )

    train_dir = os.path.join(base_dir, "data", "train")
    val_dir = os.path.join(base_dir, "data", "val")
    model_path = os.path.join(base_dir, "models", "model.keras")
    model_weights_path = os.path.join(base_dir, "models", "model.weights.h5")
    history_path = os.path.join(RESULT_DIR, "history.json")
    config_path = os.path.join(RESULT_DIR, "config.json")

    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    return train_dir, val_dir, model_path, model_weights_path, history_path, config_path


# ======================================
# DATASET LOADING
# ======================================


def load_datasets(
    train_dir: str,
    val_dir: str,
    normalization: str = "rescaling",
    base_model: str = "resnet",
    data_aug: bool = True,
    image_size: typing.Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    seed: int = 42,
):

    keras.utils.set_random_seed(seed)

    train_data = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=True,
    )

    val_data = keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        shuffle=False,
    )

    # NORMALIZATION

    if normalization == "rescaling":
        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    elif normalization == "preproccess_input":
        if base_model == "resnet":
            normalization_layer = keras.applications.resnet.preprocess_input
        elif base_model == "densenet":
            normalization_layer = keras.applications.densenet.preprocess_input
        elif base_model == "efficientnet":
            normalization_layer = keras.applications.efficientnet.preprocess_input

    train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
    val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

    # DATA AUGMENTATION

    if data_aug:
        data_augmentation = keras.Sequential(
            [
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.05),
                keras.layers.RandomZoom(0.1),
            ]
        )
        train_data = train_data.map(lambda x, y: (data_augmentation(x, training=True), y))

    # PERFORMANCE OPTMIZATION

    train_data = train_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_data = val_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_data, val_data


# ======================================
# MODEL DEFINITION AND COMPILATION
# ======================================


def build_model(
    base_model_arch: str = "resnet",
    input_shape: typing.Tuple[int, int, int] = (224, 224, 3),
):
    base_model = None
    if base_model_arch == "resnet":
        base_model = keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    elif base_model_arch == "densenet":
        base_model = keras.applications.DenseNet121(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
    elif base_model_arch == "efficientnet":
        base_model = keras.applications.EfficientNetB0(
            weights="imagenet", include_top=False, input_shape=input_shape
        )

    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
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
            keras.metrics.AUC(name="AUC"),
        ],
    )

    return model


# ======================================
# MODEL TRAINING
# ======================================


def train_model(model: keras.models.Model, train_data, val_data, epochs: int = 10):

    history = model.fit(train_data, validation_data=val_data, epochs=epochs)

    return model, history


# ======================================
# RESULTS SAVING
# ======================================


def save_results(
    model: keras.models.Model,
    history: keras.callbacks.History,
    config_dict: dict,
    model_path: str,
    model_weights_path: str,
    history_path: str,
    config_path: str,
):
    model.save(model_path)

    model.save_weights(model_weights_path)

    history_dict = history.history

    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=4)

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)


# ======================================
# TRAINING PIPELINE
# ======================================


def train_pipeline(
    base_dir: str,
    expereriment_name: str = "",
    run_id: int = 1,
    base_model: str = "resnet",
    normalization: str = "rescaling",
    data_augmentation: bool = True,
    image_size: typing.Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    epochs: int = 10,
    seed: int = 42,
):
    config_dict = {
        "base_model": "",
        "weights": "imagenet",
        "optimizer": "adam",
        "preprocessing": [normalization],
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
    }

    if base_model == "resnet":
        config_dict["base_model"] = "ResNet50"
    elif base_model == "densenet":
        config_dict["base_model"] = "DenseNet121"
    elif base_model == "efficientnet":
        config_dict["base_model"] = "EfficientNetB0"

    if data_augmentation:
        config_dict["preprocessing"].append("data augmentation")

    train_dir, val_dir, model_path, model_weights_path, history_path, config_path = (
        configure_paths(base_dir, experiment_name=expereriment_name, run_id=run_id)
    )

    train_data, val_data = load_datasets(
        train_dir,
        val_dir,
        normalization=normalization,
        base_model=base_model,
        data_aug=data_augmentation,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )

    input_shape = image_size + (3,)

    model = build_model(base_model_arch=base_model, input_shape=input_shape)

    model, history = train_model(model, train_data, val_data, epochs=epochs)

    save_results(
        model,
        history,
        config_dict,
        model_path,
        model_weights_path,
        history_path,
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
