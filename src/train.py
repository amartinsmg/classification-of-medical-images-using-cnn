#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import tensorflow as tf
import keras
import json


parser = argparse.ArgumentParser(
    description="Train a CNN model for medical image classification."
)

parser.add_argument(
    "--base-dir",
    type=str,
    default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    help="Base project directory. Default is the parent directory of the script.",
)

args = parser.parse_args()

BASE_PATH = args.base_dir
TRAIN_DIR = os.path.join(BASE_PATH, "data", "train")
VAL_DIR = os.path.join(BASE_PATH, "data", "val")
MODEL_PATH = os.path.join(BASE_PATH, "models", "xray_images.keras")
MODEL_WEIGHTS_PATH = os.path.join(BASE_PATH, "models", "xray_images.weights.h5")
RESULT_PATH = os.path.join(BASE_PATH, "results", "xray_images.json")


os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_WEIGHTS_PATH), exist_ok=True)
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42


train_data = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED,
)

val_data = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)


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


AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)


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


model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)


model.save(MODEL_PATH)

model.save_weights(MODEL_WEIGHTS_PATH)

history_dict = history.history

with open(RESULT_PATH, "w") as f:
    json.dump(history_dict, f)
