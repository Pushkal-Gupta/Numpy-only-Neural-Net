# MNIST Handwritten Digits — A Neural Network Derived by Hand

This is **step 1** of the [Numpy-only-Neural-Net](../README.MD) series, and it's the one written as a teaching document. The goal here was not accuracy — it was to understand, line by line, what a framework like PyTorch is actually doing when you call `loss.backward()`. Everything is pure NumPy: the forward pass, the backprop, the gradients, all of it written out by hand.

If you read only one notebook in this repo closely, read this one. The later projects assume you already understand the math worked out here.

> **Before this:** it really helps to have gone through [micrograd](https://github.com/Pushkal-Gupta/micrograd) first. micrograd does autodiff on a single scalar — one value, one gradient, one chain-rule step at a time. This notebook opens with a scalar-to-matrix mapping that takes that exact idea and lifts it into vectorized NumPy. If the scalar version makes sense to you, the matrix version is the same thing with more bookkeeping.

---

## The task

Classic MNIST: 28×28 grayscale images of handwritten digits, flattened to 784-dimensional vectors, sorted into 10 classes (0–9).

```text
784 -> 32 -> 16 -> 10
```

- **784 inputs** — one per pixel.
- **Two hidden layers** (32 and 16 units), ReLU.
- **10 outputs**, Softmax.

---

## What's actually in the notebook

This notebook is heavy on markdown on purpose. Alongside the code you'll find:

- **Long-form explanations** of why parameters are initialized the way they are — why the mean should be zero so the network isn't biased before training starts, and why the variance has to be controlled so signals don't vanish or explode as they move through layers.
- **A full chain-rule walkthrough** across all three layers, deriving backprop rather than just stating it.
- **A scalar-to-matrix mapping** that bridges single-neuron calculus to the vectorized NumPy version — this is the bit that connects micrograd's world to this one.
- **A symbol glossary** for every letter used (Z, A, W, b, dZ, dW, dA, g), so the equations and the code line up.
- **Inline comments on every line** of the forward and backward passes, explaining each matrix shape and why each operation produces the dimensions it does.

Read it the way you'd read a textbook chapter. The code is secondary to the explanations.

---

## The math, in short

**Forward pass**, for each layer:

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}, \qquad A^{[l]} = g^{[l]}(Z^{[l]})$$

with ReLU on the hidden layers and a numerically stable Softmax on the output.

**Loss** is categorical cross-entropy averaged over the batch.

**Backprop** starts from the clean result that falls out of the Softmax–cross-entropy pairing:

$$dZ^{[L]} = A^{[L]} - Y$$

and then walks backwards through each layer:

$$dZ^{[l]} = \left( W^{[l+1]^\top} dZ^{[l+1]} \right) \odot g'^{[l]}(Z^{[l]})$$

One of the things this notebook makes concrete is that `dZ = A - Y` isn't a magic shortcut — it's what you get when you actually do the calculus through the Softmax.

**Weights** are initialized uniformly on $[-0.5, 0.5]$ (`np.random.rand(...) - 0.5`). The notebook discusses He initialization as the theoretically better choice for ReLU — that improvement gets implemented later in the series, in [Letter Recognition](../Letter_Recognition).

---

## Training setup and result

| | |
|---|---|
| Architecture | 784 → 32 → 16 → 10 |
| Learning rate | 0.1 |
| Epochs | 500 |
| Batch | Full batch (vanilla gradient descent) |
| Train / dev split | 80 / 20 (33,600 train, 8,400 dev) |

| | Accuracy |
|---|---|
| Training set | 88.06% |
| Development set | 87.68% |

The dev accuracy sitting just under the train accuracy means the network generalized cleanly — no real overfitting. And ~88% with a hand-written two-hidden-layer net trained by full-batch gradient descent is exactly the point: it's not state of the art, it's *understood*.

---

## Data convention

The matrix is transposed right after loading so that **each column of `X` is one sample**. That's so the code lines up with the math notation ($Z = W \cdot A$ rather than $Z = A \cdot W$) and the equations and the NumPy stay visually aligned. This convention carries through every notebook in the series.

`train.csv` / `test.csv` are the Kaggle MNIST CSVs (42,000 labelled samples; first column is the label, the rest are pixel intensities in `[0, 255]`, normalized to `[0, 1]` before training).

---

## Running it

```bash
pip install numpy pandas matplotlib notebook
jupyter notebook Digits-MNIST-Neural-Net.ipynb
```

CSVs are tracked with Git LFS — run `git lfs pull` if they didn't come down with the clone.

There's also a Kaggle version: [MNIST from scratch, only NumPy](https://www.kaggle.com/code/pushkalg/mnist-from-scratch-only-numpy).

---

## Where this sits in the series

1. **MNIST Handwritten Digits** — the math, derived by hand ← you are here
2. [Fashion MNIST](../MNIST_Fashion) — deeper, with mini-batches
3. [Letter Recognition](../Letter_Recognition) — the dynamic, list-driven engine + He init
4. [ECG Heartbeat Recognition](../ECG%20Heartbeat%20Recognition) — transfer learning on that engine
