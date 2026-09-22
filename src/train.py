"""Train the CNN digit classifier on MNIST.

Run from the project root:
    python -m src.train
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

from . import config

tf.random.set_seed(config.SEED)
np.random.seed(config.SEED)


def load_mnist(path):
    with np.load(path) as data:
        x_train, y_train = data["x_train"], data["y_train"]
        x_test, y_test = data["x_test"], data["y_test"]
    return (x_train, y_train), (x_test, y_test)


def build_model():
    return keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),

        keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="last_conv"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(10, activation="softmax"),
    ])


def main():
    print("Loading MNIST...")
    (x_train, y_train), (x_test, y_test) = load_mnist(config.MNIST_PATH)
    print(f"Train shape: {x_train.shape}")
    print(f"Test shape : {x_test.shape}")

    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Stronger augmentation than clean MNIST to better match real phone photos
    datagen = ImageDataGenerator(
        rotation_range=12,
        width_shift_range=0.12,
        height_shift_range=0.12,
        zoom_range=0.12,
        shear_range=8,
        fill_mode="nearest",
    )
    datagen.fit(x_train)

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            config.BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\nStarting training...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=50,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    model.save(config.FINAL_MODEL_PATH)
    print(f"Model saved to: {config.FINAL_MODEL_PATH}")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{config.MODEL_DIR}/training_history.png")
    plt.show()

    print("\nTraining finished successfully!")


if __name__ == "__main__":
    main()