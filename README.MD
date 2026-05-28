# Numpy-only-Neural-Net

A fully connected Neural Network implemented from scratch using only NumPy.

This repository contains no TensorFlow, no PyTorch, and no high-level machine learning frameworks. The purpose of this project is to deeply understand the internal mechanics of deep learning by manually implementing the underlying mathematics and structural components.

The network is currently evaluated on the MNIST handwritten digit dataset, achieving approximately 87%+ accuracy, and is structured to scale to additional datasets and classification tasks.

---

## Architecture

The baseline network uses a multi-layer perceptron configuration:

784 -> 32 -> 16 -> 10

- **Input Layer:** 784 neurons (28x28 pixel images, flattened)
- **Hidden Layer 1:** 32 neurons
- **Hidden Layer 2:** 16 neurons
- **Output Layer:** 10 neurons (representing categorical classes, such as digits 0-9)

---

## Features

- Pure NumPy execution without high-level library abstractions.
- Manual implementation of forward propagation and backpropagation.
- ReLU activation functions for hidden layers and Softmax for the output layer.
- Categorical target processing via One-Hot encoding.
- Parameter optimization using Gradient Descent and He weight initialization.
- Live performance metrics showing accuracy tracking across training epochs.
- Modular design intended for expansion to other datasets and deeper architectures.

---

## Tech Stack

- Python
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Project Goal

This project serves as a first-principles engineering exercise. Instead of using high-level APIs that obscure the underlying linear algebra and calculus, this implementation focuses entirely on uncovering:

- The exact flow of gradients across multi-layered networks.
- The explicit matrix calculations that dictate backpropagation.
- The interaction of matrix dimensions and tensor shapes during weight updates.
- The practical impact of activation functions and proper weight initialization.

---

## Mathematical Implementation

### Forward Propagation

For each layer $l$, the linear combination and subsequent activation are calculated as:

$$Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + B^{[l]}$$

$$A^{[l]} = g(Z^{[l]})$$

Where the activation function $g(z)$ is defined as:

- **ReLU** (Hidden Layers): $g(z) = \max(0, z)$
- **Softmax** (Output Layer): $g(z)_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$

### Backpropagation

Gradients are calculated manually using the chain rule to derive partial derivatives for the loss function with respect to weights ($dW$) and biases ($dB$). Parameters are iteratively updated via Gradient Descent:

$$W = W - \alpha \cdot dW$$

$$B = B - \alpha \cdot dB$$

Where $\alpha$ denotes the learning rate.

---

## Datasets and Scalability

While the primary validation environment uses the MNIST Handwritten Digits dataset, the underlying engine is built to be dataset-agnostic. The pipeline easily scales to handle other tabular and image classification problems (such as Fashion-MNIST and binary/multiclass variations) by updating the input dimensions and categorical configurations.

### Current Baseline Dataset (MNIST)

- 60,000 training samples
- 10,000 testing samples
- 28x28 grayscale images processed as 784-dimensional vectors

---

## Folder Structure

```text
Numpy-only-Neural-Net/
│
├── MNIST_Handwritten_Digits/
│   ├── MNIST-Neural-Net.ipynb
│   ├── train.csv
│   └── test.csv
├── .gitignore
├── LICENSE
└── README.md
```

---

## Future Improvements and Expansion

The roadmap for this repository focuses on turning this foundational implementation into a broader, generalized scratch-built deep learning library:

- **Optimization:** Implement Mini-Batch Gradient Descent and the Adam Optimizer.
- **Loss Functions:** Integrate native Cross-Entropy Loss optimization.
- **Regularization:** Add Dropout layers to mitigate overfitting.
- **Architecture Scaling:** Support dynamic layer adding to effortlessly run new datasets.
- **Utility Features:** Build model serialization mechanisms (saving/loading trained weights).
- **Refactoring:** Transition the workspace from Jupyter Notebooks into standard, modular Python packages (`.py` files).

---

## Learning Outcomes

Developing this project provided concrete experience with:

- Resolving tensor and matrix shape mismatches systematically.
- Translating raw mathematical formulas into highly vectorized NumPy code.
- Understanding why initial weight variance matters to prevent vanishing or exploding gradients.
- Managing input normalization boundaries to stabilize network training.

---

## Run Locally

Clone the repository:

```bash
git clone [https://github.com/Pushkal-Gupta/Numpy-only-Neural-Net.git](https://github.com/Pushkal-Gupta/Numpy-only-Neural-Net.git)
cd Numpy-only-Neural-Net
```

Install the required packages:

```bash
pip install numpy matplotlib pandas notebook
```

Launch the environment:

```bash
jupyter notebook
```

---

## Sample Output

Example inference verification:

```text
Prediction: 7
Label: 7
```

Training cycle logging:

```text
Iteration: 0     Accuracy: 12%
Iteration: 100   Accuracy: 71%
Iteration: 300   Accuracy: 84%
Iteration: 500   Accuracy: 87%
```

---

## Why This Project Matters

Frameworks like TensorFlow and PyTorch are industry standards for production optimization, but they hide the fundamental mechanics of deep learning behind opaque API abstractions. This repository builds core engineering intuition, proving that complex architectures are simply composed of elegant, foundational mathematical blocks.

---

## Author

**Pushkal Gupta**

Kaggle:
[MNIST Digits Notebook](https://www.kaggle.com/code/pushkalg/mnist-from-scratch-only-numpy)

GitHub: [Pushkal-Gupta](https://github.com/Pushkal-Gupta)

---

## License

This project is open-source and available under the MIT License.
