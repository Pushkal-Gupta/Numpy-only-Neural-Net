# Numpy-only-Neural-Net

A fully connected feedforward neural network implemented from scratch using only NumPy — no TensorFlow, no PyTorch, no Keras, no Scikit-learn for the model itself.

This repository exists to make the internals of deep learning legible. Every matrix, every gradient, every shape change is written out by hand. The point is not benchmark accuracy — it is to understand, line by line, what a framework like PyTorch is actually doing when you call `loss.backward()`.

The project is structured as a progression. The first notebook (MNIST Handwritten Digits) is written as a teaching document — heavy markdown explanations, derivations of the math, and inline reasoning behind every matrix shape. The second notebook (Fashion MNIST) is the same engine extended one layer deeper and trained with mini-batch gradient descent, written under the philosophy that *code is the ultimate source of truth*. The third notebook (Letter Recognition) turns the engine into a dynamic MLP. The forward, backward, and update passes now loop over a list of layers instead of hard-coding each one, so the depth no longer lives inside those passes. The one exception is `init_params`, which still writes the layers out by hand on purpose, because it reads more clearly that way. This notebook also switches to He initialization and reshuffles the data before every epoch.

The fourth notebook (ECG Heartbeat Recognition) puts that dynamic engine to work on transfer learning. A deep seven-layer network is trained to classify arrhythmias from ECG signals, and then its learned feature layers are frozen and reused as a feature extractor: a tiny new head is trained on top of them to detect myocardial infarction on a completely different ECG dataset. This follows the setup from the paper the dataset comes with (*ECG Heartbeat Classification: A Deep Transferable Representation*, Kachuee et al., 2018), and shows the whole thing was implemented with nothing but the NumPy primitives built up across the first three notebooks.

---

## Recommended Reading Order

This repository is meant to be read in sequence. Each notebook builds on the intuition established by the previous one.

### 0. Prerequisite → [micrograd](https://github.com/Pushkal-Gupta/micrograd)

It helps to go through [micrograd](https://github.com/Pushkal-Gupta/micrograd) first. This project builds on the intuition it gives you: micrograd does autodiff on a single scalar, one value and one gradient at a time, so you can see the chain rule run without any matrix machinery in the way.

The first notebook here starts with a scalar-to-matrix mapping that takes that same idea into NumPy. If the scalar version already makes sense to you, the matrix version is just the same thing with more bookkeeping. It's not required, since everything is still worked out from scratch below, but it makes the rest go down a lot easier.

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

### 3. Finally → `Letter_Recognition/Letter_Recognition.ipynb`

This notebook generalizes the engine. In the first two notebooks each layer was a named variable (`W1`, `W2`, `W3`, …) and the forward and backward passes were written out by hand. Here the parameters live in Python lists (`wArr`, `bArr`) and every pass loops over them, so to change the network you change the initializer and leave the math alone.

It also uses a different kind of dataset. Instead of raw pixels, the UCI Letter Recognition dataset describes each glyph with 16 hand-engineered features, and the job is to classify it as one of the 26 letters `A`–`Z`.

What's new here:

- **Dynamic architecture** — `forward_pass`, `backward_pass`, and `update_params` loop over the list of weights and biases, so the depth `16 → 128 → 64 → 32 → 26` is just data the passes read, not something written into them
- **He initialization** — weights come from $\mathcal{N}(0, \sqrt{2/n_{\text{in}}})$ and biases start at zero, which is the right choice for ReLU layers and the improvement the digits notebook only talked about
- **Proper shuffling in SGD** — the training set is re-permuted with `np.random.permutation` at the start of each epoch before it's cut into mini-batches, so no batch repeats from one epoch to the next
- Input is 16 features rather than flattened pixels
- A one-hot encoder that maps each letter through `ord(c) - ord('A')`

### 4. Then → `ECG Heartbeat Recognition/ECG_Heartbeat_Detection.ipynb`

This is the first notebook that does something other than train one network end to end. It reuses the dynamic engine from Letter Recognition unchanged and applies it to **transfer learning** on ECG signals, following the recipe in the dataset's source paper.

It runs in two stages:

1. **Source task** — a deep `187 → 512 → 256 → 128 → 64 → 32 → 16 → 5` network is trained on the MIT-BIH arrhythmia dataset to sort each heartbeat into one of five rhythm classes.
2. **Transfer** — the early layers of that trained network (everything up to the 32-feature layer) are frozen and treated as a fixed feature extractor. A small `32 → 20 → 10 → 2` head is then trained on those features to detect myocardial infarction on the PTB dataset, which the base network never saw.

What's new here:

- **Transfer learning by feature extraction** — the frozen layers are run once as a forward pass to turn each PTB heartbeat into a 32-dimensional feature vector, and only the new head is trained on top
- A neat trick for reusing the existing `forward_pass`: it always softmaxes the last layer, so to read off the ReLU'd 32-feature layer instead, one extra layer is included in the pass and its softmax output is simply thrown away (`aArr[-2]`)
- A **confusion matrix** built by hand to see *which* classes get confused, not just the overall accuracy
- Two binary/multi-class ECG datasets with a fixed 187-sample signal length, merged and shuffled for the binary task

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

### Letter Recognition

```text
16 -> 128 -> 64 -> 32 -> 26
```

- Input layer: 16 neurons (the 16 statistical features describing each glyph)
- Hidden layer 1: 128 neurons, ReLU
- Hidden layer 2: 64 neurons, ReLU
- Hidden layer 3: 32 neurons, ReLU
- Output layer: 26 neurons, Softmax (one per letter `A`–`Z`)

This architecture is not hard-coded the way the first two are. It is produced by `init_params`, and the forward and backward passes loop over the resulting parameter lists, so swapping in a different depth requires editing only the initializer.

### ECG Heartbeat Recognition

**Source network (MIT-BIH, 5-class arrhythmia):**

```text
187 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 5
```

- Input layer: 187 neurons (the 187 signal samples of one segmented heartbeat)
- Six hidden layers, ReLU, narrowing from 512 down to 16
- Output layer: 5 neurons, Softmax (`N`, `S`, `V`, `F`, `Q`)

**Transfer head (PTB, binary MI detection):**

```text
[ 187 -> 512 -> 256 -> 128 -> 64 -> 32 ]  ->  20 -> 10 -> 2
        (frozen, reused as feature extractor)      (trained from scratch)
```

- The bracketed layers are taken from the trained source network and frozen — their weights are never updated again
- Each PTB heartbeat is pushed through them once to produce a 32-dimensional feature vector
- A fresh `32 → 20 → 10 → 2` head is trained on those vectors to output normal vs. abnormal

The depth here is the same dynamic-list construction as Letter Recognition — the source network just happens to be seven weight layers deep instead of four. The transfer head is a second, independent parameter list, and the final assembled classifier is literally `wArr[0:5] + ptbdb_wArr` concatenated together.

---

## Features

- Pure NumPy implementation of forward and backward propagation across arbitrary fully connected depths
- **Dynamic, list-driven architecture** (Letter Recognition notebook) — layers live in lists and every pass loops over them, so depth is configuration rather than hand-unrolled code
- ReLU activations on hidden layers and a numerically stable Softmax on the output
- Cross-entropy loss derived analytically through the Softmax layer (the simplified `dZ = A - Y` form)
- Both uniform `[-0.5, 0.5]` initialization (digits, fashion) and **He initialization** for ReLU layers (Letter Recognition)
- One-hot label encoding implemented manually, including a character-based encoder for the `A`–`Z` letter labels
- Mini-batch gradient descent in the Fashion MNIST and Letter Recognition notebooks, with **per-epoch reshuffling** in the latter
- Train / dev splits with explicit transposition so each column of `X` represents one sample, matching the math notation used throughout the explanations
- Pixel normalization to the `[0, 1]` range (image notebooks)
- Per-epoch accuracy logging
- Inline visualization of individual predictions using Matplotlib
- **Transfer learning via feature extraction** (ECG notebook) — frozen early layers reused as a feature extractor, with only a small new head trained on top of the extracted features
- **A hand-written confusion matrix** (ECG notebook) for per-class error analysis on the 5-class arrhythmia task and the binary MI task
- Handling of both multi-class (5-way) and binary (2-way) tasks, including merging and shuffling two separate CSV files into one labelled set

---

## Tech Stack

- Python
- NumPy
- Pandas (only used to read the CSV files)
- Matplotlib (only used to visualize sample predictions)
- Jupyter Notebook

No machine learning library is imported in any of the notebooks. Pandas exists purely as a CSV reader and Matplotlib purely as an image viewer.

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

The two image notebooks initialize parameters by sampling from `np.random.rand(...) - 0.5`, which gives a uniform distribution on $[-0.5, 0.5]$:

$$W^{[l]}, b^{[l]} \sim \mathcal{U}(-0.5, 0.5)$$

The digits notebook contains an extended markdown discussion of why the mean should be zero (to avoid biasing the network in one direction before training begins) and why the variance must be controlled (to avoid vanishing or exploding activations as signals propagate through layers). He initialization is discussed there as the theoretically correct choice for ReLU networks.

The Letter Recognition notebook actually does this. Weights come from a zero-mean Gaussian scaled by $\sqrt{2/n_{\text{in}}}$ and biases start at zero:

$$W^{[l]} \sim \mathcal{N}\!\left(0,\ \tfrac{2}{n^{[l-1]}}\right), \qquad b^{[l]} = \mathbf{0}$$

This is He initialization. It keeps the activation variance roughly steady as the signal moves through a deep ReLU stack, which is what you need to train the four-weight-layer `16 → 128 → 64 → 32 → 26` network without it blowing up or stalling.

The ECG notebook leans on this even harder. Its source network is seven weight layers deep (`187 → 512 → 256 → 128 → 64 → 32 → 16 → 5`), and the same He initialization is what lets a stack that deep train at all with plain SGD and no normalization layers.

---

## Training Configuration

| | MNIST Digits | Fashion MNIST | Letter Recognition | ECG (MIT-BIH base) |
|---|---|---|---|---|
| Architecture | 784 → 32 → 16 → 10 | 784 → 64 → 32 → 16 → 10 | 16 → 128 → 64 → 32 → 26 | 187 → 512 → 256 → 128 → 64 → 32 → 16 → 5 |
| Initialization | Uniform $[-0.5, 0.5]$ | Uniform $[-0.5, 0.5]$ | He ($\sqrt{2/n_{\text{in}}}$) | He ($\sqrt{2/n_{\text{in}}}$) |
| Learning rate ($\alpha$) | 0.1 | 0.1 | 0.02 | 0.01 |
| Epochs | 500 | 200 | 200 | 50 |
| Batch size | Full batch | 100 | 100 | 1,000 |
| Train / dev split | 80 / 20 | 80 / 20 | 80 / 20 | Predefined train/test |
| Train samples | 33,600 | 48,000 | 16,000 | 87,553 |
| Dev samples | 8,400 | 12,000 | 4,000 | 21,891 |
| Optimizer | Vanilla gradient descent | Mini-batch gradient descent | Mini-batch SGD with per-epoch shuffling | Mini-batch SGD with per-epoch shuffling |

The ECG **transfer head** is trained separately: a `32 → 20 → 10 → 2` network on top of the frozen feature extractor, He initialized, learning rate `0.001`, batch size `500`, 50 epochs, on the PTB dataset (80 / 20 split of 14,550 merged normal + abnormal heartbeats).

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

### Letter Recognition

- Source: `letter-recognition.csv`, the UCI Letter Recognition dataset (20,000 labelled samples)
- Each sample is **not an image** but 16 integer-valued statistical features (box dimensions, pixel counts, mean positions, variances, edge counts, etc.) extracted from a distorted glyph
- The first column is the letter label (`A`–`Z`); the remaining 16 columns are the features
- 26 class labels, one per capital letter, encoded with `ord(c) - ord('A')`

### ECG Heartbeat Recognition

From the Kaggle [ECG Heartbeat Categorization](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) dataset, which bundles two preprocessed ECG collections. Every signal is cropped, downsampled to 125 Hz, and zero-padded to a fixed length, so each row is a sequence of signal samples followed by a single class label in the last column.

- **MIT-BIH Arrhythmia** (`mitbih_train.csv`, `mitbih_test.csv`) — ~109,000 heartbeats with 187 signal samples each, labelled into 5 rhythm classes: `N` (normal, 0), `S` (supraventricular, 1), `V` (ventricular, 2), `F` (fusion, 3), `Q` (unknown, 4). Ships with a predefined train/test split.
- **PTB Diagnostic ECG** (`ptbdb_normal.csv`, `ptbdb_abnormal.csv`) — ~14,550 heartbeats in the same 187-sample format, split across two files by class: normal vs. abnormal (myocardial infarction). The notebook concatenates the two files, shuffles, and makes its own 80 / 20 split.

All notebooks transpose the matrix after loading so that each column of `X` is one sample. This matches the convention used in the math (the matrices line up with $Z = W \cdot A$ rather than $Z = A \cdot W$), which keeps the code and the equations visually aligned.

---

## Folder Structure

```text
Numpy-only-Neural-Net/
│
├── MNIST_Handwritten_Digits/
│   ├── README.md
│   ├── Digits-MNIST-Neural-Net.ipynb
│   ├── train.csv
│   └── test.csv
│
├── MNIST_Fashion/
│   ├── README.md
│   ├── Fashion-MNIST-Neural-Net.ipynb
│   ├── fashion-mnist_train.csv
│   └── fashion-mnist_test.csv
│
├── Letter_Recognition/
│   ├── README.md
│   ├── Letter_Recognition.ipynb
│   └── letter-recognition.csv
│
├── ECG Heartbeat Recognition/
│   ├── README.md
│   ├── ECG_Heartbeat_Detection.ipynb
│   ├── mitbih_train.csv
│   ├── mitbih_test.csv
│   ├── ptbdb_normal.csv
│   └── ptbdb_abnormal.csv
│
├── .gitattributes
├── .gitignore
├── LICENSE
└── README.MD
```

Each project folder also has its own `README.md` that goes into more depth on the individual project than this top-level overview. The dataset CSV files are tracked with Git LFS.

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

### Letter Recognition

| | Accuracy |
|---|---|
| Training set (after 200 epochs) | 99% |
| Test set | 93.6% |

This is the best result in the repo. With He initialization, the deeper `16 → 128 → 64 → 32 → 26` network, and shuffling every epoch, it fits the 16-feature data almost perfectly and still holds up in the low 90s on letters it hasn't seen.

### ECG Heartbeat Recognition

**Source task — MIT-BIH arrhythmia (5 classes):**

| | Accuracy |
|---|---|
| Training set (after 50 epochs) | 96.93% |
| Test set | 96.55% |

The train and test numbers sit right on top of each other, so the deep base network generalizes cleanly with no real overfitting.

**Transfer task — PTB myocardial infarction (binary):**

| | Accuracy |
|---|---|
| Training set (head only, 50 epochs) | ~80% |
| Test set | 75.56% |

The interesting part is *how* this number was reached. The base network's first five layers were frozen and reused as a feature extractor, and only a small `32 → 20 → 10 → 2` head — under a thousand parameters — was trained on top. Without retraining a single weight of the original network, just the features it had already learned for arrhythmias carry enough signal to detect myocardial infarction at ~76%. That gap from the base task's 96% is the cost of never fine-tuning the body, and it's exactly the trade-off transfer learning is about.

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

The Letter Recognition notebook logs mini-batch accuracy every ten epochs as it climbs toward 99%:

```text
Epoch :  0
0.30...
Epoch :  100
0.97...
Epoch :  190
0.99...
```

The ECG notebook prints a confusion matrix so you can see which classes get mixed up, not just the headline accuracy. For the 5-class arrhythmia task it is strongly diagonal, with the off-diagonal mass concentrated in the rarer rhythm classes:

```text
[[18118    65    ...]
 [   ...   ...   ...]]   # rows = actual class, columns = predicted class
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

Then open `MNIST_Handwritten_Digits/Digits-MNIST-Neural-Net.ipynb` first, work through it, move on to `MNIST_Fashion/Fashion-MNIST-Neural-Net.ipynb`, then `Letter_Recognition/Letter_Recognition.ipynb`, and finish with `ECG Heartbeat Recognition/ECG_Heartbeat_Detection.ipynb`.

The dataset CSVs are tracked with Git LFS. If `git clone` does not pull them automatically, install Git LFS and run `git lfs pull`.

---

## Future Improvements

The roadmap for turning this implementation into a more general scratch-built deep learning library:

- **Initialization:** Replace uniform `[-0.5, 0.5]` initialization with He initialization for ReLU layers — done in the Letter Recognition notebook
- **Architecture:** Support dynamic layer specification so a new architecture is one list change away — done in the Letter Recognition notebook; the passes now loop over parameter lists
- **Transfer learning:** Freeze a trained network's early layers and train a new head on a different task — done in the ECG notebook
- **Diagnostics:** Add a confusion matrix for per-class error analysis — done in the ECG notebook; loss-curve plots are still to come
- **Datasets:** Extend beyond image classification to tabular and binary tasks — done across the Letter Recognition (tabular, 26-class) and ECG (signal, binary + multi-class) notebooks
- **Optimization:** Implement momentum, RMSProp, and Adam
- **Regularization:** Add dropout and L2 weight decay
- **Persistence:** Add weight serialization for saving and loading trained networks
- **Refactoring:** Move from notebooks to a clean Python package with reusable layer, optimizer, and loss classes
- **Fine-tuning:** Unfreeze the transferred ECG layers and fine-tune the whole network end to end, to close the gap from the frozen-feature 75% toward the base task's 96%

---

## Learning Outcomes

Building this from scratch made the following concrete in a way that using a framework never does:

- Resolving tensor shape mismatches by reasoning about what each axis represents, not by trial and error
- Translating the chain rule directly into vectorized NumPy operations
- Seeing how `dZ = A - Y` at the output layer falls out of the Softmax–cross-entropy composition rather than being a magic shortcut
- Feeling why initial weight variance matters — too small and the network learns nothing, too large and the gradients explode within a few epochs
- Understanding why pixel values are normalized before training, and what happens when they are not
- Seeing the difference mini-batching makes to both training speed and gradient stability
- Turning hand-written layers into a loop, and getting the forward/backward indexing right once it's written in terms of the layer count instead of fixed names
- Seeing for myself how much He initialization and per-epoch shuffling actually help on a deeper network
- Understanding transfer learning from the inside — that "freezing layers" is just running a forward pass once and training a new head on the activations, with no magic involved
- Reading a confusion matrix to find *where* a classifier fails, instead of trusting a single accuracy number
- Feeling the trade-off transfer learning makes: reusing frozen features is cheap and fast, but leaves accuracy on the table compared to training end to end

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
