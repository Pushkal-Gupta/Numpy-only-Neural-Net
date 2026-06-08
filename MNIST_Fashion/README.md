# Fashion MNIST — Same Engine, Deeper, with Mini-Batches

This is **step 2** of the [Numpy-only-Neural-Net](../README.MD) series. If [step 1](../MNIST_Handwritten_Digits) was the textbook chapter, this one is the lab. The opening cell sets the tone — *"Code is the ultimate source of truth."* The math was all derived in the digits notebook; by this point you're expected to understand it, so the commentary is intentionally sparse and the focus shifts to seeing how the implementation scales.

Still pure NumPy. Still no ML library.

---

## The task

Fashion MNIST is a drop-in replacement for digits with the same shape — 28×28 grayscale images, flattened to 784 vectors — but the classes are clothing items instead of numbers, and it's a genuinely harder problem.

```text
784 -> 64 -> 32 -> 16 -> 10
```

Ten classes:

```text
0  T-shirt/Top      5  Sandal
1  Trouser          6  Shirt
2  Pullover         7  Sneaker
3  Dress            8  Bag
4  Coat             9  Ankle Boot
```

---

## What's new compared to the digits notebook

- **A deeper network.** The architecture grows from `784 → 32 → 16 → 10` to `784 → 64 → 32 → 16 → 10` — a third hidden layer and wider ones. This is deliberate: telling a Pullover from a Coat from a Shirt needs more representational capacity than separating a 3 from an 8.
- **Mini-batch gradient descent.** Instead of computing the gradient over the entire training set at once, the same forward and backward passes are wrapped in a loop over small batches (size 100). This is faster and the gradients are steadier.
- **A label decoder** that turns the 10 class indices into their clothing names.
- **A prediction visualizer** that draws a sample image with the predicted and actual class names as the plot title.

The forward pass, backprop, and gradient math are otherwise the same engine from the digits notebook — ReLU hidden layers, numerically stable Softmax output, cross-entropy loss, and the clean `dZ = A - Y` at the output layer.

---

## Why mini-batching matters

This is the main lesson of the notebook. Full-batch gradient descent (what the digits notebook used) takes one big, accurate step per epoch. Mini-batching takes many smaller, noisier steps per epoch instead. In practice that noise is a feature, not a bug — it speeds up training and helps the optimizer make steady progress. Seeing the difference firsthand, on the same engine, is the point.

---

## Training setup and result

| | |
|---|---|
| Architecture | 784 → 64 → 32 → 16 → 10 |
| Initialization | Uniform $[-0.5, 0.5]$ |
| Learning rate | 0.1 |
| Epochs | 200 |
| Batch size | 100 |
| Train / dev split | 80 / 20 (48,000 train, 12,000 dev) |

| | Accuracy |
|---|---|
| Training set | 93.06% |
| Development set | 85.49% |

The wider train–dev gap here (compared to digits) is expected. Fashion MNIST is a substantially harder visual task, and a deeper network with no regularization starts to overfit before the 200 epochs are up. That gap is itself a useful thing to see — it's the motivation for the regularization techniques on the repo's roadmap.

---

## Data

`fashion-mnist_train.csv` / `fashion-mnist_test.csv` from the Kaggle Fashion-MNIST distribution (60,000 labelled samples), exactly the same CSV layout as the digits dataset: first column is the label, the remaining 784 are pixel intensities in `[0, 255]`, normalized to `[0, 1]` before training. As everywhere in the series, the matrix is transposed so each column of `X` is one sample.

---

## Running it

```bash
pip install numpy pandas matplotlib notebook
jupyter notebook Fashion-MNIST-Neural-Net.ipynb
```

CSVs are tracked with Git LFS — run `git lfs pull` if they didn't come down with the clone.

---

## Where this sits in the series

1. [MNIST Handwritten Digits](../MNIST_Handwritten_Digits) — the math, derived by hand
2. **Fashion MNIST** — deeper, with mini-batches ← you are here
3. [Letter Recognition](../Letter_Recognition) — the dynamic, list-driven engine + He init
4. [ECG Heartbeat Recognition](../ECG%20Heartbeat%20Recognition) — transfer learning on that engine
