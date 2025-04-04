This repository presents a complete implementation of a supervised learning pipeline for handwritten digit recognition based on the MNIST dataset. 
The classification task is addressed using a Multi-Layer Perceptron (MLP) neural network model, built and trained with the PyTorch deep learning framework.

Overview:
The MNIST dataset is a classical benchmark in machine learning, consisting of 70,000 grayscale images of handwritten digits (0–9), each of size 28×28 pixels. This project demonstrates:
-Preprocessing and loading of the MNIST dataset.
-Construction of a fully-connected neural network.
-Model training, validation, and testing using stochastic optimization.
-Visualization of training dynamics and classification performance, including a confusion matrix

Technical Highlights
*Dataset split:
-48,000 training samples
-12,000 validation samples
-10,000 test samples

*Evaluation metrics:
-Validation and test accuracy
-Loss across epochs
-Confusion matrix to visualize classification errors

*Hardware support:
Automatic detection and usage of CUDA (GPU) if available.

Application mnist.py is partially based on the project: https://github.com/jarek-pawlowski/machine-learning-applications/blob/main/mnist_in_3_flavours.ipynb


Requirements:
Python 3.8+
PyTorch
torchvision
matplotlib
seaborn
scikit-learn
NumPy
