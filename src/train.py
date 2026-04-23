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

import pathlib
import argparse
import tensorflow as tf
import keras
import json
import typing
import numpy as np
import sklearn as sk


# ======================================
# PATH CONFIGURATION
# ======================================


def configure_paths(base_dir: pathlib.Path, experiment_name: str = "", run_id: int = 1):
    if len(experiment_name) == 0:
        RESULT_DIR = base_dir / "results"
    else:
        RESULT_DIR = base_dir / "results" / experiment_name / f"run{run_id:02d}"

    train_dir = base_dir / "data" / "train"
    val_dir = base_dir / "data" / "val"
    model_path = base_dir / "models" / "model.keras"
    model_weights_path = base_dir / "models" / "model.weights.h5"
    history_path = RESULT_DIR / "history.json"
    config_path = RESULT_DIR / "config.json"

    pathlib.Path.mkdir

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

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
    # LOAD DATASETS

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

    # NORMALIZATION

    if normalization == "rescaling":
        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    elif normalization == "preprocess_input":
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
        train_data = train_data.map(
            lambda x, y: (data_augmentation(x, training=True), y)
        )

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
    learning_rate: float = 0.001,
):

    # BASE MODEL LOADING

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

    # MODEL BUILDING

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.models.Model(inputs, outputs)

    # OPTIMIZER CONFIGURATION

    optimezer = keras.optimizers.Adam(learning_rate=learning_rate)

    # MODEL COMPILATION

    model.compile(
        optimizer=optimezer,
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


def train_model(
    model: keras.models.Model,
    train_data,
    val_data,
    epochs: int = 10,
    class_weights: bool = False,
):

    history = None

    if not class_weights:
        history = model.fit(train_data, validation_data=val_data, epochs=epochs)
    else:
        y_train = np.concatenate([y for _, y in train_data], axis=0)

        classes = np.unique(y_train)
        weights = sk.utils.class_weight.compute_class_weight(
            class_weights="balanced",
            classes=classes,
            y=y_train,
        )
        class_weights_dict = dict(zip(classes, weights))

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            class_weight=class_weights_dict,
        )

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
    class_weights: bool = False,
    learning_rate: float = 0.001,
):
    # SET SEEDS AND FORCE DETERMINISTIC GPU OPERATIONS

    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    # CONFIGURATION DICTIONARY FOR RESULTS SAVING

    config_dict = {
        "base_model": "",
        "weights": "imagenet",
        "optimizer": "adam",
        "preprocessing": [normalization],
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "class_weights": class_weights,
        "optimizer": {
            "name": "adam",
            "learning_rate": learning_rate,
        },
    }

    if base_model == "resnet":
        config_dict["base_model"] = "ResNet50"
    elif base_model == "densenet":
        config_dict["base_model"] = "DenseNet121"
    elif base_model == "efficientnet":
        config_dict["base_model"] = "EfficientNetB0"

    if data_augmentation:
        config_dict["preprocessing"].append("data augmentation")

    # CONFIGURE PATHS AND LOAD DATASETS

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

    # BUILD AND COMPILE MODEL

    model = build_model(base_model_arch=base_model, input_shape=input_shape)

    model, history = train_model(
        model, train_data, val_data, epochs=epochs, class_weights=class_weights
    )

    # SAVE MODEL AND RESULTS

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
    train_pipeline(base_dir=pathlib.Path(args.base_dir).resolve())


if __name__ == "__main__":

    # ARGUMENT PARSING

    parser = argparse.ArgumentParser(
        description="Train a CNN model for radiological image classification."
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parents[1]),
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
