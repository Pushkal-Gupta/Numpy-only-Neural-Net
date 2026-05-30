# Numpy-only-Neural-Net

A fully connected feedforward neural network implemented from scratch using only NumPy — no TensorFlow, no PyTorch, no Keras, no Scikit-learn for the model itself.

This repository exists to make the internals of deep learning legible. Every matrix, every gradient, every shape change is written out by hand. The point is not benchmark accuracy — it is to understand, line by line, what a framework like PyTorch is actually doing when you call `loss.backward()`.

The project is structured as a progression. The first notebook (MNIST Handwritten Digits) is written as a teaching document — heavy markdown explanations, derivations of the math, and inline reasoning behind every matrix shape. The second notebook (Fashion MNIST) is the same engine extended one layer deeper and trained with mini-batch gradient descent, written under the philosophy that *code is the ultimate source of truth*.

---

## Recommended Reading Order

This repository is meant to be read in sequence. Each notebook builds on the intuition established by the previous one.

### 1. Start here → `MNIST_Handwritten_Digits/Digits-MNIST-Neural-Net.ipynb`

This is the **pedagogical notebook**. It contains:

- Long-form markdown explanations of why parameters are initialized the way they are
- A full walk-through of the chain rule applied across three layers
- A scalar-to-matrix mapping section that bridges single-neuron calculus to vectorized NumPy
- A glossary of every symbol used (Z, A, W, b, dZ, dW, dA, g)
- Inline comments on every line of the forward and backward passes explaining the matrix dimensions and why each operation produces the shape it does

Read this notebook the way you would read a textbook chapter. The code is secondary to the explanations.

### 2. Then move to → `MNIST_Fashion/Fashion-MNIST-Neural-Net.ipynb`

This notebook is **intentionally sparse on commentary**. The opening cell sets the tone: *"Code is the ultimate source of truth."* By this point, the reader is expected to understand the underlying math and only needs to see how the implementation scales.

What is new in this notebook:

- A fourth hidden layer (architecture grows from `784 → 32 → 16 → 10` to `784 → 64 → 32 → 16 → 10`)
- A mini-batch gradient descent loop wrapped around the same forward and backward passes
- A label decoder for the ten Fashion-MNIST classes
- A prediction visualizer that renders predicted and actual class names as the plot title

---

## Architecture

### MNIST Handwritten Digits

```text
784 -> 32 -> 16 -> 10
```

- Input layer: 784 neurons (28×28 grayscale image flattened)
- Hidden layer 1: 32 neurons, ReLU
- Hidden layer 2: 16 neurons, ReLU
- Output layer: 10 neurons, Softmax

### Fashion MNIST

```text
784 -> 64 -> 32 -> 16 -> 10
```

- Input layer: 784 neurons (28×28 grayscale image flattened)
- Hidden layer 1: 64 neurons, ReLU
- Hidden layer 2: 32 neurons, ReLU
- Hidden layer 3: 16 neurons, ReLU
- Output layer: 10 neurons, Softmax

The deeper architecture for Fashion MNIST is intentional. Fashion images are visually richer than digits — distinguishing a Pullover from a Coat or a Shirt requires more representational capacity than separating a 3 from an 8.

---

## Features

- Pure NumPy implementation of forward and backward propagation across arbitrary fully connected depths
- ReLU activations on hidden layers and a numerically stable Softmax on the output
- Cross-entropy loss derived analytically through the Softmax layer (the simplified `dZ = A - Y` form)
- One-hot label encoding implemented manually
- Mini-batch gradient descent in the Fashion MNIST notebook
- Train / dev splits with explicit transposition so each column of `X` represents one sample, matching the math notation used throughout the explanations
- Pixel normalization to the `[0, 1]` range
- Per-epoch accuracy logging
- Inline visualization of individual predictions using Matplotlib

---

## Tech Stack

- Python
- NumPy
- Pandas (only used to read the CSV files)
- Matplotlib (only used to visualize sample predictions)
- Jupyter Notebook

No machine learning library is imported in either notebook. Pandas exists purely as a CSV reader and Matplotlib purely as an image viewer.

---

## Mathematical Implementation

### Forward Propagation

For each layer $l$, the pre-activation and activation are:

$$Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]}$$

$$A^{[l]} = g^{[l]}(Z^{[l]})$$

Where $A^{[0]} = X$ and:

- **ReLU** (hidden layers): $g(z) = \max(0, z)$
- **Softmax** (output layer, numerically stabilized):

$$g(z)_i = \frac{e^{z_i - \max(z)}}{\sum_{j} e^{z_j - \max(z)}}$$

The `- max(z)` subtraction prevents overflow when any logit is large, without changing the result.

### Loss

Categorical cross-entropy, averaged over $m$ examples:

$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} Y^{(i)}_k \log A^{[L](i)}_k$$

### Backpropagation

The output-layer gradient collapses to a clean form because cross-entropy and Softmax compose nicely:

$$dZ^{[L]} = A^{[L]} - Y_{\text{one-hot}}$$

For every preceding layer:

$$dZ^{[l]} = \left( W^{[l+1]^\top} \cdot dZ^{[l+1]} \right) \odot g'^{[l]}(Z^{[l]})$$

Where $\odot$ denotes the element-wise (Hadamard) product, and for ReLU:

$$g'(z) = \mathbf{1}[z > 0]$$

The gradients with respect to parameters are:

$$dW^{[l]} = \frac{1}{m} \cdot dZ^{[l]} \cdot A^{[l-1]^\top}$$

$$db^{[l]} = \frac{1}{m} \sum_{i=1}^{m} dZ^{[l](i)}$$

### Parameter Update

Vanilla gradient descent:

$$W^{[l]} := W^{[l]} - \alpha \cdot dW^{[l]}$$

$$b^{[l]} := b^{[l]} - \alpha \cdot db^{[l]}$$

### Weight Initialization

Both notebooks initialize parameters by sampling from `np.random.rand(...) - 0.5`, which gives a uniform distribution on $[-0.5, 0.5]$:

$$W^{[l]}, b^{[l]} \sim \mathcal{U}(-0.5, 0.5)$$

The digits notebook contains an extended markdown discussion of why the mean should be zero (to avoid biasing the network in one direction before training begins) and why the variance must be controlled (to avoid vanishing or exploding activations as signals propagate through layers). He initialization is discussed as the theoretically correct choice for ReLU networks but is left as a future improvement.

---

## Training Configuration

| | MNIST Digits | Fashion MNIST |
|---|---|---|
| Architecture | 784 → 32 → 16 → 10 | 784 → 64 → 32 → 16 → 10 |
| Learning rate ($\alpha$) | 0.1 | 0.1 |
| Epochs | 500 | 200 |
| Batch size | Full batch | 100 |
| Train / dev split | 80 / 20 | 80 / 20 |
| Train samples | 33,600 | 48,000 |
| Dev samples | 8,400 | 12,000 |
| Optimizer | Vanilla gradient descent | Mini-batch gradient descent |

---

## Datasets

### MNIST Handwritten Digits

- Source: `train.csv` from the Kaggle MNIST distribution (42,000 labelled samples)
- 28×28 grayscale, flattened to 784-dimensional column vectors
- First column of the CSV is the label, remaining 784 columns are pixel intensities in `[0, 255]`

### Fashion MNIST

- Source: `fashion-mnist_train.csv` from the Kaggle Fashion-MNIST distribution (60,000 labelled samples)
- Same 28×28 grayscale format and same CSV layout as the digits dataset
- Ten class labels:

```text
0  T-shirt/Top      5  Sandal
1  Trouser          6  Shirt
2  Pullover         7  Sneaker
3  Dress            8  Bag
4  Coat             9  Ankle Boot
```

Both notebooks transpose the matrix after loading so that each column of `X` is one sample. This matches the convention used in the math (the matrices line up with $Z = W \cdot A$ rather than $Z = A \cdot W$), which keeps the code and the equations visually aligned.

---

## Folder Structure

```text
Numpy-only-Neural-Net/
│
├── MNIST_Handwritten_Digits/
│   ├── Digits-MNIST-Neural-Net.ipynb
│   ├── train.csv
│   └── test.csv
│
├── MNIST_Fashion/
│   ├── Fashion-MNIST-Neural-Net.ipynb
│   ├── fashion-mnist_train.csv
│   └── fashion-mnist_test.csv
│
├── .gitattributes
├── .gitignore
├── LICENSE
└── README.MD
```

The dataset CSV files are tracked with Git LFS.

---

## Results

### MNIST Handwritten Digits

| | Accuracy |
|---|---|
| Training set (after 500 epochs) | 88.06% |
| Development set | 87.68% |

The dev accuracy sits just under the training accuracy, indicating that the network has generalized cleanly without overfitting.

### Fashion MNIST

| | Accuracy |
|---|---|
| Training set (after 200 epochs) | 93.06% |
| Development set | 85.49% |

The wider train–dev gap is expected — Fashion MNIST is a substantially harder visual classification task, and the deeper network with no regularization begins to overfit before the schedule completes.

---

## Sample Output

Per-epoch logging during training:

```text
Epoch :  0
0.12455...
Epoch :  100
0.71190...
Epoch :  300
0.84720...
Epoch :  480
0.88062...
```

Inference on a single image:

```text
Prediction:  7
Label:       7
```

Fashion MNIST inference uses the class-name decoder:

```text
Predicted: Ankle Boot
Actual:    Ankle Boot
```

---

## Run Locally

Clone the repository:

```bash
git clone https://github.com/Pushkal-Gupta/Numpy-only-Neural-Net.git
cd Numpy-only-Neural-Net
```

Install the dependencies:

```bash
pip install numpy pandas matplotlib notebook
```

Launch Jupyter:

```bash
jupyter notebook
```

Then open `MNIST_Handwritten_Digits/Digits-MNIST-Neural-Net.ipynb` first, work through it, and then move on to `MNIST_Fashion/Fashion-MNIST-Neural-Net.ipynb`.

The dataset CSVs are tracked with Git LFS. If `git clone` does not pull them automatically, install Git LFS and run `git lfs pull`.

---

## Future Improvements

The roadmap for turning this implementation into a more general scratch-built deep learning library:

- **Initialization:** Replace uniform `[-0.5, 0.5]` initialization with He initialization for ReLU layers
- **Optimization:** Implement momentum, RMSProp, and Adam
- **Regularization:** Add dropout and L2 weight decay
- **Architecture:** Support dynamic layer specification so a new architecture is one list change away
- **Persistence:** Add weight serialization for saving and loading trained networks
- **Refactoring:** Move from notebooks to a clean Python package with reusable layer, optimizer, and loss classes
- **Diagnostics:** Add explicit loss curves and confusion matrix visualizations
- **Datasets:** Extend beyond image classification to a tabular and a binary classification example

---

## Learning Outcomes

Building this from scratch made the following concrete in a way that using a framework never does:

- Resolving tensor shape mismatches by reasoning about what each axis represents, not by trial and error
- Translating the chain rule directly into vectorized NumPy operations
- Seeing how `dZ = A - Y` at the output layer falls out of the Softmax–cross-entropy composition rather than being a magic shortcut
- Feeling why initial weight variance matters — too small and the network learns nothing, too large and the gradients explode within a few epochs
- Understanding why pixel values are normalized before training, and what happens when they are not
- Seeing the difference mini-batching makes to both training speed and gradient stability

---

## Why This Project Matters

Frameworks like TensorFlow and PyTorch are industry standards for production training, but they hide the fundamental mechanics of deep learning behind opaque API abstractions. Calling `loss.backward()` does not teach you what backpropagation is — it just runs it for you.

This repository takes the opposite approach. Every matrix is written out, every gradient is derived, every shape is justified. The goal is **clarity, not speed or scale**. Once the mechanics are understood at this level, every framework on top of NumPy becomes a convenience rather than a mystery.

---

## Author

**Pushkal Gupta**

Kaggle:
[MNIST Digits Notebook](https://www.kaggle.com/code/pushkalg/mnist-from-scratch-only-numpy)

GitHub: [Pushkal-Gupta](https://github.com/Pushkal-Gupta)

---

## License

This project is open-source and available under the MIT License.
