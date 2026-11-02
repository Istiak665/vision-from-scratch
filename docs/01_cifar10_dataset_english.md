# Lesson 01: Understanding the CIFAR-10 Dataset

> **Project:** Vision From Scratch
> **Notebook:** `notebooks/00_data_exploration.ipynb`
> **Prerequisite:** Basic Python, CIFAR-10 downloaded with `scripts/download_cifar10.py`
> **Next lesson:** Loading all batches and writing `scratchvision/data/cifar10.py`

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain the structure, classes, and train/test distribution of CIFAR-10.
- Build an RGB image from the 3,072 values in one raw row.
- Use the notation for vectors, matrices, and image tensors.
- Explain the difference between the `(N, C, H, W)` and `(N, H, W, C)` shapes.
- Calculate the dataset's file size and memory requirements by hand.
- Run code in JupyterLab to verify the dataset.

## 1. Why Understand the Dataset Before the Model?

A machine learning model learns from data. If the input shape, label meaning, or pixel range is wrong, the code may still run, but the results can be wrong. Before writing a model, we will answer four questions:

1. What does one sample look like, and how is it stored?
2. What do the label numbers mean?
3. What are the shape, type, and value range of the data?
4. Are the classes balanced?

## 2. CIFAR-10 Overview

CIFAR-10 is a labeled dataset of small color images. It is widely used for learning and testing image classification.

| Property | Value |
|---|---|
| Total images | 60,000 |
| Training images | 50,000 |
| Test images | 10,000 |
| Image size | 32 × 32 pixels |
| Color channels | 3 (Red, Green, Blue) |
| Number of classes | 10 |
| Images per class | 6,000 |
| Training images per class | 5,000 |
| Test images per class | 1,000 |

### Classes

| Label | Class | Label | Class |
|---:|---|---:|---|
| 0 | airplane | 5 | dog |
| 1 | automobile | 6 | frog |
| 2 | bird | 7 | horse |
| 3 | cat | 8 | ship |
| 4 | deer | 9 | truck |

The `automobile` and `truck` classes do not overlap. Cars such as sedans and SUVs belong to `automobile`, while big trucks belong to `truck`.

## 3. Downloaded Files

After extraction, the `datasets/cifar-10-batches-py/` folder contains these files:

| File | Contents |
|---|---|
| `data_batch_1` … `data_batch_5` | 10,000 training images each |
| `test_batch` | 10,000 test images |
| `batches.meta` | Metadata, including class names |
| `readme.html` | Link to the dataset website |

Each batch file is a Python **pickle** file. Loading it returns a dictionary:

| Key | Type | Meaning |
|---|---|---|
| `b"data"` | `numpy.ndarray` | Shape `(10000, 3072)`, dtype `uint8` |
| `b"labels"` | `list` | 10,000 class labels |
| `b"filenames"` | `list` | Original image filenames |
| `b"batch_label"` | `bytes` | Name of the batch |

> **Note:** The `b` before a key means it is a **bytes** string. `batch["data"]` will not work, so write `batch[b"data"]`.
>
> **Security note:** Loading a pickle file can execute code. Only load pickle files from sources you trust.

## 4. Mathematical Notation

| Symbol | Meaning | CIFAR-10 value |
|---|---|---|
| $N$ | Number of images | 50,000 (training) |
| $H$ | Image height | 32 |
| $W$ | Image width | 32 |
| $C$ | Number of channels | 3 |
| $D$ | Values per flattened image | 3,072 |
| $K$ | Number of classes | 10 |
| $\mathbf{I}_i$ | The $i$-th image tensor | Shape $(32, 32, 3)$ |
| $\mathbf{x}_i$ | The $i$-th flattened image | Length 3,072 |
| $y_i$ | Label of the $i$-th image | $0, 1, \dots, 9$ |
| $\mathbf{X}$ | Data matrix | Shape $(N, D)$ |
| $\mathbf{y}$ | Label vector | Shape $(N,)$ |

The dataset can be written as:

$$
\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}
$$

One image and its label are defined as:

$$
\mathbf{I}_i \in \{0, 1, \dots, 255\}^{H \times W \times C}
$$

$$
\mathbf{x}_i \in \{0, 1, \dots, 255\}^{D}
$$

$$
y_i \in \{0, 1, \dots, K-1\}
$$

The length of a flattened image is:

$$
D = H \times W \times C = 32 \times 32 \times 3 = 3072
$$

When $N$ images are stored together, each row is one image:

$$
\mathbf{X} =
\begin{bmatrix}
\mathbf{x}_1^\top \\
\mathbf{x}_2^\top \\
\vdots \\
\mathbf{x}_N^\top
\end{bmatrix}
\in \{0, 1, \dots, 255\}^{N \times D}
$$

## 5. Structure of One Raw Row

Each channel has $32 \times 32 = 1024$ values, and the channels are stored one after another in the row:

```text
Index:   0 ............ 1023 | 1024 ......... 2047 | 2048 ......... 3071
Value:   Red channel (1024)  | Green channel (1024) | Blue channel (1024)
```

Inside each channel, pixels are stored in **row-major order**. The 32 pixels of the top row come first, followed by the next row.

### Finding the Index of a Pixel

Let:

- $k$ = channel index (Red = 0, Green = 1, Blue = 2)
- $r$ = pixel row, where $0 \le r \le 31$
- $c$ = pixel column, where $0 \le c \le 31$

The index in the raw row is:

$$
j = 1024k + 32r + c
$$

**Example 1:** The first pixel of the Green channel, where $k=1$, $r=0$, and $c=0$.

$$
j = 1024(1) + 32(0) + 0 = 1024
$$

**Example 2:** The last pixel of the Blue channel, where $k=2$, $r=31$, and $c=31$.

$$
j = 1024(2) + 32(31) + 31 = 2048 + 992 + 31 = 3071
$$

## 6. Hand Calculations

### File Size

Each `uint8` value takes 1 byte. The image data in one batch takes:

$$
10000 \times 3072 \times 1 = 30{,}720{,}000 \text{ bytes}
$$

In the KB units used by Windows:

$$
\frac{30{,}720{,}000}{1024} = 30{,}000 \text{ KB}
$$

Windows shows each batch file as about 30,309 KB. The extra space holds the labels, filenames, and pickle structure.

### Memory Requirement

The memory needed for the pixel data of the 50,000 training images is:

$$
\text{Memory} = N \times D \times (\text{bytes per value})
$$

| Data type | Bytes per value | Memory |
|---|---:|---:|
| `uint8` | 1 | 153,600,000 bytes (153.6 MB) |
| `float32` | 4 | 614,400,000 bytes (614.4 MB) |
| `float64` | 8 | 1,228,800,000 bytes (1.2288 GB) |

Here, 1 MB = $10^6$ bytes and 1 GB = $10^9$ bytes.

### Why `uint8` Needs Care

`uint8` can only store values from 0 to 255, so it cannot store negative values. Subtraction can therefore cause **overflow wraparound**:

```python
import numpy as np

a = np.array([3], dtype=np.uint8)
b = np.array([5], dtype=np.uint8)

print(a - b)  # [254], not [-2]
```

For this reason, data must be converted to a floating-point type before computing kNN distances or subtracting the mean.

## 7. NCHW and NHWC Shapes

| Format | Shape | Meaning |
|---|---|---|
| Flat | `(N, D)` | Each row is one flattened image |
| NCHW | `(N, C, H, W)` | The channel axis comes first |
| NHWC | `(N, H, W, C)` | The channel axis comes last |

The raw data is arranged by channel, so we first reshape it to NCHW:

```text
(10000, 3072) -> reshape -> (10000, 3, 32, 32)
```

Matplotlib expects the channel axis last, so we change the axis order:

```text
Old axes:  0  1  2  3
Shape:    (N, C, H, W)

transpose(0, 2, 3, 1)

New shape: (N, H, W, C)
```

`transpose` changes only the axis order. It does not change any pixel values.

## 8. Run Instructions (JupyterLab)

We run the notebooks in JupyterLab because it combines Markdown text cells and code cells.

### Step 1: Start JupyterLab

Run this in PowerShell:

```powershell
cd E:\9_Self_ML_DL_Foundations\vision-from-scratch
conda activate dl_projects
jupyter lab
```

JupyterLab opens in your browser. Keep the terminal window open while you work.

### Step 2: Create the Notebook

1. Open the `notebooks/` folder in the left sidebar.
2. Select **File > New > Notebook**.
3. Choose **Python 3 (ipykernel)** as the kernel.
4. Name the notebook `00_data_exploration.ipynb`.

### Step 3: Verify the Environment

Run this in the first code cell:

```python
import sys

print(sys.executable)
```

The output path must contain `dl_projects`.

### Step 4: Use the Cells

- Select **Markdown** or **Code** from the cell type dropdown in the toolbar.
- Write the title of each step below in a Markdown cell.
- Write the code for each step in its own code cell, then press `Shift + Enter`.

## 9. Hands-On Notebook

### Step 1: Load One Batch

```python
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

# JupyterLab runs this notebook from the notebooks folder
project_root = Path.cwd().parent
data_folder = project_root / "datasets" / "cifar-10-batches-py"
batch_path = data_folder / "data_batch_1"

print("File exists:", batch_path.exists())

# "rb" means read binary
with open(batch_path, "rb") as file:
    batch = pickle.load(file, encoding="bytes")

print(type(batch))
print(batch.keys())
```

**Expected output:**

```text
File exists: True
<class 'dict'>
dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
```

### Step 2: Inspect the Images and Labels

```python
raw_images = batch[b"data"]
labels = np.array(batch[b"labels"])

print("Image shape:", raw_images.shape)
print("Image dtype:", raw_images.dtype)
print("Minimum pixel:", raw_images.min())
print("Maximum pixel:", raw_images.max())
print("Label shape:", labels.shape)
print("First labels:", labels[:10])
```

**Expected result:** Image shape `(10000, 3072)`, dtype `uint8`, label shape `(10000,)`, and pixel values between 0 and 255.

### Step 3: Read the Class Names

```python
meta_path = data_folder / "batches.meta"

with open(meta_path, "rb") as file:
    meta = pickle.load(file, encoding="bytes")

class_names = []

for name in meta[b"label_names"]:
    class_names.append(name.decode("utf-8"))

print(class_names)
```

### Step 4: Build One Image by Hand

```python
first_row = raw_images[0]

# Split the 3072 values into 3 color channels
red = first_row[0:1024]
green = first_row[1024:2048]
blue = first_row[2048:3072]

# Convert each channel into a 32 x 32 image
red_image = red.reshape(32, 32)
green_image = green.reshape(32, 32)
blue_image = blue.reshape(32, 32)

# Stack channels into one color image
image = np.stack([red_image, green_image, blue_image], axis=2)

print("Image shape:", image.shape)
print("Class:", class_names[labels[0]])

plt.imshow(image, interpolation="nearest")
plt.title(class_names[labels[0]])
plt.axis("off")
plt.show()
```

### Step 5: Reshape All Images at Once

```python
number_of_images = raw_images.shape[0]

# Shape: (N, C, H, W)
images_nchw = raw_images.reshape(number_of_images, 3, 32, 32)

# Shape: (N, H, W, C)
images = images_nchw.transpose(0, 2, 3, 1)

print("NCHW shape:", images_nchw.shape)
print("NHWC shape:", images.shape)
print("Matches manual image:", np.array_equal(images[0], image))
```

**Expected output:**

```text
NCHW shape: (10000, 3, 32, 32)
NHWC shape: (10000, 32, 32, 3)
Matches manual image: True
```

### Step 6: Check the Class Distribution

```python
class_counts = np.bincount(labels, minlength=10)

for class_index in range(10):
    print(class_names[class_index], class_counts[class_index])

plt.bar(class_names, class_counts)
plt.xticks(rotation=45)
plt.ylabel("Number of images")
plt.title("Class distribution in data_batch_1")
plt.show()
```

**Expected result:** Each class has about 1,000 images. The counts may not be exactly equal within one training batch, but the five batches together contain exactly 5,000 images per class.

### Step 7: View One Image from Each Class

```python
figure, axes = plt.subplots(2, 5, figsize=(10, 4))

for class_index in range(10):
    # Find every image of this class
    image_indices = np.where(labels == class_index)[0]

    # Use the first image from this class
    first_index = image_indices[0]

    # Find the position in the 2 x 5 grid
    row = class_index // 5
    column = class_index % 5

    axes[row, column].imshow(images[first_index], interpolation="nearest")
    axes[row, column].set_title(class_names[class_index])
    axes[row, column].axis("off")

plt.tight_layout()
plt.show()
```

## 10. Common Errors

| Problem | Cause | Fix |
|---|---|---|
| `File exists: False` | The notebook is running from another folder | Check the location with `print(Path.cwd())` and keep the notebook in `notebooks/` |
| `KeyError: 'data'` | The key is a bytes string | Write `batch[b"data"]` |
| `ModuleNotFoundError` | Wrong kernel | Activate `dl_projects`, then start JupyterLab |
| Strange image colors or shapes | Wrong reshape order | Reshape to `(N, 3, 32, 32)` first, then apply `transpose(0, 2, 3, 1)` |
| The image looks blurry | Display interpolation or small image size | Use `interpolation="nearest"`, and remember that a 32 × 32 image is small |

## 11. Presentation Summary

> CIFAR-10 contains 60,000 labeled 32 × 32 RGB images from 10 classes, split into 50,000 training images and 10,000 test images. Each raw image is stored as a row of 3,072 `uint8` values: 1,024 red, 1,024 green, and 1,024 blue. We reshape each row to recover its spatial structure and inspect the data type, value range, and class distribution before training a model.

## 12. Quiz

Write your own answers first, then open the answers.

1. Why does one CIFAR-10 image contain 3,072 values?
2. Index 1024 of a raw row is the first value of which channel?
3. What is the shape of the data matrix after flattening 500 CIFAR-10 images?
4. Which `transpose` axes convert `(N, C, H, W)` to `(N, H, W, C)`, and why?
5. How many bytes are needed for the pixel data of 50,000 training images stored as `float64`?
6. What is the raw index of the Red channel pixel at $(r=2, c=5)$?
7. Why must `uint8` data be converted before computing distances?

<details>
<summary><strong>Show answers</strong></summary>

**1.** Because $32 \times 32 \times 3 = 3072$.

**2.** The Green channel, because the Red channel occupies indices 0 to 1023.

**3.** `(500, 3072)`.

**4.** `transpose(0, 2, 3, 1)`. Axis 0 ($N$), axis 2 ($H$), and axis 3 ($W$) move to the front, and axis 1 ($C$) moves to the end.

**5.** $50000 \times 3072 \times 8 = 1{,}228{,}800{,}000$ bytes, or 1.2288 GB.

**6.** $j = 1024(0) + 32(2) + 5 = 69$.

**7.** `uint8` cannot store negative values, so subtraction can wrap around and produce wrong distances.

</details>

## 13. Practice

1. Load `data_batch_2` and find its class distribution.
2. Load `test_batch` and show that each class has exactly 1,000 images.
3. Plot the Red, Green, and Blue channels of one image as separate grayscale images.

## References

1. A. Krizhevsky, [*Learning Multiple Layers of Features from Tiny Images*](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Technical Report, University of Toronto, 2009.
2. [The CIFAR-10 and CIFAR-100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html), University of Toronto.
3. [CS231n: Image Classification notes](https://cs231n.github.io/classification/), Stanford University.
