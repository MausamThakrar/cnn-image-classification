# CNN Image Classification on MNIST

This project implements a **Convolutional Neural Network (CNN)** using
TensorFlow/Keras to classify handwritten digits from the **MNIST** dataset.

## Main Features

- Loads MNIST digit images (28x28 grayscale)
- Builds a simple CNN with:
  - Two convolution + max-pooling blocks
  - Dense layer with ReLU activation
  - Dropout for regularisation
  - Softmax output layer
- Trains the model using Adam optimiser
- Evaluates accuracy on a held-out test set
- Plots training & validation accuracy and loss
- Saves the trained model to disk

## Project Structure

```text
cnn-image-classification/
│── src/
│   └── train_cnn.py
│── requirements.txt
└── README.md
