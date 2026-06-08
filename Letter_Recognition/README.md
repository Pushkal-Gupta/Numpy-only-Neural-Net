# Letter Recognition — The Engine Becomes Dynamic

This is **step 3** of the [Numpy-only-Neural-Net](../README.MD) series, and it's where the engine stops being hard-coded. In the first two notebooks every layer was a named variable (`W1`, `W2`, `W3`, …) and the forward and backward passes were written out by hand, one layer at a time. Here the parameters live in plain Python lists and every pass is a loop over those lists — so the network's depth is no longer baked into the code. Change the initializer, and the math takes care of itself.

It also picks up two improvements the earlier notebooks had only talked about: **He initialization** and **proper per-epoch shuffling**. And it's the first project that isn't an image dataset.

Still pure NumPy. Still no ML library.

---

## The task

The UCI Letter Recognition dataset. Each sample is **not an image** — it's 16 hand-engineered statistical features (box dimensions, pixel counts, mean positions, variances, edge counts, and so on) extracted from a distorted glyph. The job is to classify it as one of the 26 capital letters `A`–`Z`.

```text
16 -> 128 -> 64 -> 32 -> 26
```

- **16 inputs** — the 16 features.
- **Three hidden layers** (128, 64, 32), ReLU.
- **26 outputs**, Softmax, one per letter.

---

## What's new here

### Dynamic architecture

This is the headline change. The three core functions — `forward_pass`, `backward_pass`, and `update_params` — all loop over a list of weight matrices (`wArr`) and bias vectors (`bArr`) instead of referring to `W1`, `W2`, and friends by name:

```python
def forward_pass(X, wArr, bArr):
    n = len(wArr)              # depth is just the length of the list
    activationVal = X
    zArr, aArr = [], []
    for i in range(n):
        zArr.append(wArr[i].dot(activationVal) + bArr[i])
        activationVal = softmax(zArr[-1]) if i == n - 1 else ReLU(zArr[-1])
        aArr.append(activationVal)
    return zArr, aArr
```

So the depth `16 → 128 → 64 → 32 → 26` is **data the passes read**, not something written into them. To build a different network you change one function — `init_params` — and nothing else.

The one place layers are still written out by hand is `init_params` itself, and that's on purpose: spelling the shapes out there keeps the architecture readable in a single glance. Everything downstream is generic. (This same engine, unchanged, is what [the ECG notebook](../ECG%20Heartbeat%20Recognition) later reuses for a seven-layer network and transfer learning.)

### He initialization

The earlier notebooks initialized weights uniformly on $[-0.5, 0.5]$ and discussed He init as the better choice for ReLU networks. This notebook actually does it — weights drawn from a zero-mean Gaussian scaled by $\sqrt{2/n_{\text{in}}}$, biases starting at zero:

$$W^{[l]} \sim \mathcal{N}\!\left(0,\ \tfrac{2}{n^{[l-1]}}\right), \qquad b^{[l]} = \mathbf{0}$$

This keeps activation variance roughly steady as the signal moves through the ReLU stack, which is what lets the deeper network train stably.

### Proper shuffling in SGD

The mini-batch loop now re-permutes the entire training set at the start of **every epoch** before slicing it into batches:

```python
shuffled_indices = np.random.permutation(Y.size)
X_shuffled = X[:, shuffled_indices]
Y_shuffled = Y[shuffled_indices]
```

So the network never sees the same batch composition twice. Without this, mini-batch SGD keeps re-using identical batches every epoch, which biases the gradient estimates.

### A character-based one-hot encoder

Since the labels are letters, not integers, the one-hot encoder maps each one through `ord(c) - ord('A')` to get a row index in `[0, 25]`.

---

## Training setup and result

| | |
|---|---|
| Architecture | 16 → 128 → 64 → 32 → 26 |
| Initialization | He ($\sqrt{2/n_{\text{in}}}$) |
| Learning rate | 0.02 |
| Epochs | 200 |
| Batch size | 100 |
| Train / test split | 80 / 20 (16,000 train, 4,000 test) |

| | Accuracy |
|---|---|
| Training set | 99% |
| Test set | 93.6% |

This is the strongest result in the repo. The combination of He init, the deeper network, and shuffling every epoch lets it fit the 16-feature data almost perfectly while still holding up in the low 90s on letters it hasn't seen.

---

## Data

`letter-recognition.csv` — the UCI Letter Recognition dataset, 20,000 labelled samples. The first column is the letter label (`A`–`Z`); the remaining 16 columns are the integer features. As everywhere in the series, the matrix is transposed after loading so each column of `X` is one sample.

---

## Running it

```bash
pip install numpy pandas matplotlib notebook
jupyter notebook Letter_Recognition.ipynb
```

The CSV is tracked with Git LFS — run `git lfs pull` if it didn't come down with the clone.

---

## Where this sits in the series

1. [MNIST Handwritten Digits](../MNIST_Handwritten_Digits) — the math, derived by hand
2. [Fashion MNIST](../MNIST_Fashion) — deeper, with mini-batches
3. **Letter Recognition** — the dynamic, list-driven engine + He init ← you are here
4. [ECG Heartbeat Recognition](../ECG%20Heartbeat%20Recognition) — transfer learning on this engine
