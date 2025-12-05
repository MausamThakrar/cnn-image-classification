from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 1e-3


def load_mnist_data():
    """
    Load the MNIST handwritten digits dataset from Keras.
    Returns train and test sets with images normalised to [0, 1].
    """
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # MNIST images are 28x28 grayscale -> add channel dimension
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)  # (num_samples, 28, 28, 1)
    x_test = np.expand_dims(x_test, -1)

    return (x_train, y_train), (x_test, y_test)


def build_cnn_model(input_shape=(28, 28, 1), num_classes: int = 10) -> keras.Model:
    """
    Build a simple CNN for digit classification.
    """
    model = keras.Sequential(
        [
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def plot_training_history(history: keras.callbacks.History) -> None:
    """
    Plot training and validation accuracy and loss over epochs.
    """
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)

    # Accuracy
    plt.figure()
    plt.plot(epochs, acc, label="Training accuracy")
    plt.plot(epochs, val_acc, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Loss
    plt.figure()
    plt.plot(epochs, loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    config = TrainingConfig(batch_size=128, epochs=5, learning_rate=1e-3)

    print("TensorFlow version:", tf.__version__)

    # 1. Load data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    num_classes = 10

    # 2. Build model
    model = build_cnn_model(input_shape=x_train.shape[1:], num_classes=num_classes)
    model.summary()

    # 3. Compile model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        metrics=["accuracy"],
    )

    # 4. Train model
    history = model.fit(
        x_train,
        y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_split=0.1,
        verbose=2,
    )

    # 5. Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # 6. Plot training curves
    plot_training_history(history)

    # 7. Save model
    model.save("mnist_cnn_model.h5")
    print("Saved trained model to 'mnist_cnn_model.h5'")


if __name__ == "__main__":
    main()
