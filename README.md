This repository presents a complete implementation of a supervised learning pipeline for handwritten digit recognition based on the MNIST dataset.  
The classification task is addressed using a Multi-Layer Perceptron (MLP) neural network model, built and trained with the PyTorch deep learning framework.

---

### Overview

The MNIST dataset is a classical benchmark in machine learning, consisting of **70,000 grayscale images** of handwritten digits (0–9), each of size **28×28 pixels**.  
This project demonstrates:

- Preprocessing and loading of the MNIST dataset  
- Construction of a fully-connected neural network  
- Model training, validation, and testing using stochastic optimization  
- Visualization of training dynamics and classification performance, including a confusion matrix  

---

### Technical Highlights

**Dataset split:**

- 48,000 training samples  
- 12,000 validation samples  
- 10,000 test samples  

**Evaluation metrics:**

- Validation and test accuracy  
- Loss across epochs  
- Confusion matrix to visualize classification errors  

**Hardware support:**

- Automatic detection and usage of **CUDA (GPU)** if available  

> **Note**: Application `mnist.py` is partially based on the project:  
> https://github.com/jarek-pawlowski/machine-learning-applications/blob/main/mnist_in_3_flavours.ipynb

---

### Requirements

- Python 3.8+  
- PyTorch  
- torchvision  
- matplotlib  
- seaborn  
- scikit-learn  
- NumPy  

---

### Example Output

**From terminal:**

![image](https://github.com/user-attachments/assets/fd676b23-9dee-47ff-b763-680a5ef8b757)

**Confusion matrix:**

![image](https://github.com/user-attachments/assets/eefd7e42-39bc-4d0a-92e9-15eef8216364)

**Training and validation loss:**

![image](https://github.com/user-attachments/assets/51f1efd1-cfd5-438e-a2eb-8641e620dc8a)

