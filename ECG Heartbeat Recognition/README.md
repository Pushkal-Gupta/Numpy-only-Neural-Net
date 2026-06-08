# ECG Heartbeat Recognition — Transfer Learning from Scratch

This is the fourth project in the [Numpy-only-Neural-Net](../README.MD) series, and the first one that isn't just "train a network and read off the accuracy." Here the network is trained once on one ECG task, and then its learned layers are reused — frozen — as a feature extractor for a second, different ECG task. The whole thing, transfer learning included, is written with the same hand-rolled NumPy engine built up over the first three projects. No TensorFlow, no PyTorch, no autograd.

The setup follows the paper the dataset ships with: **Kachuee, Fazeli, and Sarrafzadeh, "ECG Heartbeat Classification: A Deep Transferable Representation" (2018)** — [arXiv:1805.00794](https://arxiv.org/pdf/1805.00794).

---

## The idea

ECG signals are how you read the electrical activity of a heartbeat. Two well-known datasets get used here:

- **MIT-BIH Arrhythmia** — heartbeats labelled into 5 rhythm classes (normal and four kinds of abnormal).
- **PTB Diagnostic ECG** — heartbeats labelled normal vs. abnormal, where abnormal means myocardial infarction (a heart attack).

The bet behind transfer learning is that a network that has learned to *read heartbeats well enough to classify arrhythmias* has, in its hidden layers, already learned a general-purpose representation of what a heartbeat looks like. If that's true, you shouldn't need to train a fresh network from scratch for the MI task — you should be able to reuse those representations and only train a small classifier on top.

So the plan is:

1. Train a deep network on MIT-BIH to classify the 5 arrhythmia classes.
2. Chop off its last couple of layers and freeze everything that's left.
3. Run the PTB heartbeats through the frozen part once to get a feature vector for each.
4. Train a tiny new network on those features to do the binary MI classification.

---

## Stage 1 — the source network (MIT-BIH, 5 classes)

A deep, narrowing MLP:

```text
187 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 5
```

- **187 inputs** — each heartbeat is a segmented, downsampled signal of 187 samples.
- **5 outputs** — `N` (normal, 0), `S` (supraventricular, 1), `V` (ventricular, 2), `F` (fusion, 3), `Q` (unknown, 4), via Softmax.
- ReLU on every hidden layer, He initialization throughout. At seven weight layers deep this is the deepest network in the whole repo, and He init is what makes a stack this deep trainable with plain SGD and no normalization tricks.

Trained with mini-batch SGD (batch size 1000, learning rate 0.01, 50 epochs, data reshuffled every epoch).

**Result:** **96.93%** on the MIT-BIH train set, **96.55%** on its test set. The two numbers sitting on top of each other means it generalized cleanly with essentially no overfitting.

A hand-written confusion matrix confirms the errors are where you'd expect — concentrated in the rarer rhythm classes, not spread randomly.

---

## Stage 2 — transfer to PTB (binary MI detection)

This is the interesting half.

### Freezing layers is just a forward pass

There's no special "freeze" machinery. To freeze the early layers, you simply **don't pass them to the optimizer** — you run them forward once to turn each input into a feature vector, and then forget about them. The frozen layers here are the first five, the ones that produce the 32-dimensional layer:

```text
[ 187 -> 512 -> 256 -> 128 -> 64 -> 32 ]   ->   features
        frozen, taken from the trained source net
```

### The Softmax trick

There's one wrinkle worth calling out. The reusable `forward_pass` always applies Softmax to the *last* layer it's given. But the features wanted here are the **ReLU** activations of the 32-unit layer, not a Softmax'd output.

The fix is small and a little cheeky: pass in *six* layers instead of five, let `forward_pass` Softmax the sixth, and then just **throw the sixth away** and read off the second-to-last activation (`aArr[-2]`). That gives the clean ReLU'd 32-feature vector without touching the forward-pass code at all.

### The new head

A small classifier is trained on those 32-dimensional features:

```text
32 -> 20 -> 10 -> 2
```

He initialized, learning rate 0.001, batch size 500, 50 epochs. Under a thousand parameters in total. None of the source network's weights are touched — only this head learns.

The final assembled model is literally the frozen body concatenated with the new head: `wArr[0:5] + ptbdb_wArr`.

**Result:** about **80%** on the head's training data and **75.56%** on the held-out PTB test set.

---

## What this shows

Without retraining a single weight of the original network, the features it learned for *arrhythmia* classification carry enough signal to detect *myocardial infarction* at ~76% — on a dataset the network was never trained on, using a head with under a thousand parameters.

That ~20-point gap from the source task's 96% is the honest cost of never fine-tuning the body. It's the central trade-off of transfer learning: reusing frozen features is cheap, fast, and needs very little new data, but training end to end would do better. Closing that gap by unfreezing the body and fine-tuning the whole thing is the obvious next step.

---

## Files

| File | What it is |
|---|---|
| `ECG_Heartbeat_Detection.ipynb` | The full notebook: source training, transfer, and evaluation |
| `mitbih_train.csv` / `mitbih_test.csv` | MIT-BIH arrhythmia, 5 classes, predefined split (~109k heartbeats) |
| `ptbdb_normal.csv` / `ptbdb_abnormal.csv` | PTB diagnostic ECG, split by class into two files (~14.5k heartbeats) |

Every row is 187 signal samples followed by a single class label in the last column. Signals are cropped, downsampled to 125 Hz, and zero-padded to a fixed length. Dataset: [Kaggle — ECG Heartbeat Categorization](https://www.kaggle.com/datasets/shayanfazeli/heartbeat).

---

## Running it

```bash
pip install numpy pandas matplotlib notebook
jupyter notebook ECG_Heartbeat_Detection.ipynb
```

Run the cells top to bottom — Stage 1 trains and evaluates the source network, then Stage 2 builds the transfer head on top of it. The CSVs are tracked with Git LFS; if they didn't come down with `git clone`, run `git lfs pull`.

---

## Where this sits in the series

This is **step 4** of [Numpy-only-Neural-Net](../README.MD):

1. [MNIST Handwritten Digits](../MNIST_Handwritten_Digits) — the math, derived by hand
2. [Fashion MNIST](../MNIST_Fashion) — deeper, with mini-batches
3. [Letter Recognition](../Letter_Recognition) — the dynamic, list-driven engine + He init
4. **ECG Heartbeat Recognition** — transfer learning on that engine ← you are here

It reuses the dynamic engine from step 3 unchanged. The only genuinely new idea is the transfer-learning workflow itself — and the point of building it from scratch is to see that "transfer learning" is nothing more exotic than a forward pass through frozen layers plus a small network trained on the result.
