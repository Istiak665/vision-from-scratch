# Lesson 01: Understanding the CIFAR-10 Dataset

> **Project:** Vision From Scratch
> **Notebook:** `notebooks/00_data_exploration.ipynb`
> **Prerequisite:** Basic Python, CIFAR-10 downloaded with `scripts/download_cifar10.py`
> **Next lesson:** Loading all batches and writing `scratchvision/data/cifar10.py`

## Learning Objectives

এই lesson শেষে তুমি পারবে:

- CIFAR-10-এর structure, classes আর train/test distribution ব্যাখ্যা করতে।
- একটা raw row-এর 3,072টা value থেকে RGB image তৈরি করতে।
- Vector, matrix আর image tensor-এর notation ব্যবহার করতে।
- `(N, C, H, W)` আর `(N, H, W, C)` shape-এর পার্থক্য বুঝতে।
- Dataset-এর file size আর memory requirement হাতে হিসাব করতে।
- JupyterLab-এ code চালিয়ে dataset verify করতে।

## 1. Model-এর আগে Dataset কেন বুঝতে হবে?

Machine learning model data থেকেই শেখে। Input shape, label meaning বা pixel range ভুল হলে code চললেও result ভুল হতে পারে। তাই model লেখার আগে আমরা চারটা প্রশ্নের উত্তর জানব:

1. একটা sample দেখতে কেমন, আর সেটা কীভাবে store করা আছে?
2. Label-এর number গুলোর মানে কী?
3. Data-এর shape, type আর value range কত?
4. Classes কি balanced?

## 2. CIFAR-10 Overview

CIFAR-10 হলো ছোট color image-এর একটা labeled dataset, যা image classification শেখা ও test করার জন্য বহুল ব্যবহৃত।

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

`automobile` আর `truck` আলাদা, একে অপরের সাথে overlap করে না। Sedan, SUV-এর মতো গাড়ি `automobile`-এ আছে, আর বড় truck আছে `truck`-এ।

## 3. Downloaded Files

Extract করার পর `datasets/cifar-10-batches-py/` folder-এ এই files পাওয়া যায়:

| File | Contents |
|---|---|
| `data_batch_1` … `data_batch_5` | প্রতিটায় 10,000 training images |
| `test_batch` | 10,000 test images |
| `batches.meta` | Class names সহ metadata |
| `readme.html` | Dataset website-এর link |

প্রতিটা batch file একটা Python **pickle** file, যা load করলে একটা dictionary পাওয়া যায়:

| Key | Type | Meaning |
|---|---|---|
| `b"data"` | `numpy.ndarray` | Shape `(10000, 3072)`, dtype `uint8` |
| `b"labels"` | `list` | 10,000টা class label |
| `b"filenames"` | `list` | Original image filenames |
| `b"batch_label"` | `bytes` | Batch-এর নাম |

> **Note:** Key-এর সামনের `b` মানে **bytes** string। তাই `batch["data"]` কাজ করবে না, লিখতে হবে `batch[b"data"]`।
>
> **Security note:** Pickle file load করার সময় code execute হতে পারে। তাই শুধু বিশ্বস্ত source থেকে আসা pickle file load করবে।

## 4. Mathematical Notation

| Symbol | Meaning | CIFAR-10 value |
|---|---|---|
| $N$ | Number of images | 50,000 (training) |
| $H$ | Image height | 32 |
| $W$ | Image width | 32 |
| $C$ | Number of channels | 3 |
| $D$ | Values per flattened image | 3,072 |
| $K$ | Number of classes | 10 |
| $\mathbf{I}_i$ | $i$-তম image tensor | Shape $(32, 32, 3)$ |
| $\mathbf{x}_i$ | $i$-তম flattened image | Length 3,072 |
| $y_i$ | $i$-তম image-এর label | $0, 1, \dots, 9$ |
| $\mathbf{X}$ | Data matrix | Shape $(N, D)$ |
| $\mathbf{y}$ | Label vector | Shape $(N,)$ |

Dataset-কে এভাবে লেখা যায়:

$$
\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}
$$

একটা image আর তার label:

$$
\mathbf{I}_i \in \{0, 1, \dots, 255\}^{H \times W \times C}
$$

$$
\mathbf{x}_i \in \{0, 1, \dots, 255\}^{D}
$$

$$
y_i \in \{0, 1, \dots, K-1\}
$$

Flattened image-এর length:

$$
D = H \times W \times C = 32 \times 32 \times 3 = 3072
$$

$N$টা image একসাথে রাখলে প্রতিটা row একটা image হয়:

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

## 5. একটা Raw Row-এর Structure

প্রতিটা channel-এ $32 \times 32 = 1024$টা value থাকে, আর row-তে channel গুলো ধারাবাহিকভাবে রাখা হয়:

```text
Index:   0 ............ 1023 | 1024 ......... 2047 | 2048 ......... 3071
Value:   Red channel (1024)  | Green channel (1024) | Blue channel (1024)
```

প্রতিটা channel-এর ভেতরে pixel গুলো **row-major order**-এ থাকে। মানে প্রথমে উপরের row-এর 32টা pixel, তারপর পরের row।

### Pixel থেকে Index বের করার সূত্র

ধরি:

- $k$ = channel index (Red = 0, Green = 1, Blue = 2)
- $r$ = pixel row, যেখানে $0 \le r \le 31$
- $c$ = pixel column, যেখানে $0 \le c \le 31$

তাহলে raw row-এর index:

$$
j = 1024k + 32r + c
$$

**Example 1:** Green channel-এর প্রথম pixel, যেখানে $k=1$, $r=0$, $c=0$।

$$
j = 1024(1) + 32(0) + 0 = 1024
$$

**Example 2:** Blue channel-এর শেষ pixel, যেখানে $k=2$, $r=31$, $c=31$।

$$
j = 1024(2) + 32(31) + 31 = 2048 + 992 + 31 = 3071
$$

## 6. Hand Calculations

### File Size

প্রতিটা `uint8` value 1 byte নেয়। এক batch-এর image data:

$$
10000 \times 3072 \times 1 = 30{,}720{,}000 \text{ bytes}
$$

Windows-এর KB হিসাবে:

$$
\frac{30{,}720{,}000}{1024} = 30{,}000 \text{ KB}
$$

Windows-এ batch file প্রায় 30,309 KB দেখায়। বাড়তি অংশটা labels, filenames আর pickle structure-এর জন্য।

### Memory Requirement

50,000 training images-এর pixel data-এর জন্য memory:

$$
\text{Memory} = N \times D \times (\text{bytes per value})
$$

| Data type | Bytes per value | Memory |
|---|---:|---:|
| `uint8` | 1 | 153,600,000 bytes (153.6 MB) |
| `float32` | 4 | 614,400,000 bytes (614.4 MB) |
| `float64` | 8 | 1,228,800,000 bytes (1.2288 GB) |

এখানে 1 MB = $10^6$ bytes আর 1 GB = $10^9$ bytes ধরা হয়েছে।

### কেন `uint8` সাবধানে ব্যবহার করতে হয়

`uint8` শুধু 0 থেকে 255 রাখতে পারে, negative value রাখতে পারে না। তাই বিয়োগ করলে **overflow wraparound** হতে পারে:

```python
import numpy as np

a = np.array([3], dtype=np.uint8)
b = np.array([5], dtype=np.uint8)

print(a - b)  # [254], not [-2]
```

kNN distance বা mean subtraction-এর আগে data তাই floating-point type-এ convert করতে হবে।

## 7. NCHW এবং NHWC Shape

| Format | Shape | Meaning |
|---|---|---|
| Flat | `(N, D)` | প্রতিটা row একটা flattened image |
| NCHW | `(N, C, H, W)` | Channel axis আগে থাকে |
| NHWC | `(N, H, W, C)` | Channel axis শেষে থাকে |

Raw data channel অনুযায়ী সাজানো, তাই প্রথমে NCHW-তে reshape করা হয়:

```text
(10000, 3072) -> reshape -> (10000, 3, 32, 32)
```

Matplotlib channel শেষে চায়, তাই axis order বদলাতে হয়:

```text
Old axes:  0  1  2  3
Shape:    (N, C, H, W)

transpose(0, 2, 3, 1)

New shape: (N, H, W, C)
```

`transpose` শুধু axis order বদলায়, pixel value বদলায় না।

## 8. Run Instructions (JupyterLab)

আমরা notebooks JupyterLab-এ চালাব, কারণ এতে Markdown text cell আর code cell একসাথে রাখা যায়।

### Step 1: JupyterLab চালু করো

PowerShell-এ চালাও:

```powershell
cd E:\9_Self_ML_DL_Foundations\vision-from-scratch
conda activate dl_projects
jupyter lab
```

Browser-এ JupyterLab খুলবে। কাজ চলাকালীন Terminal window বন্ধ করবে না।

### Step 2: Notebook তৈরি করো

1. Left sidebar থেকে `notebooks/` folder খোলো।
2. **File > New > Notebook** select করো।
3. Kernel হিসেবে **Python 3 (ipykernel)** নাও।
4. Notebook-এর নাম দাও `00_data_exploration.ipynb`।

### Step 3: Environment verify করো

প্রথম code cell-এ চালাও:

```python
import sys

print(sys.executable)
```

Output-এর path-এ `dl_projects` থাকতে হবে।

### Step 4: Cell ব্যবহার করো

- Toolbar-এর dropdown থেকে cell type **Markdown** বা **Code** select করো।
- প্রতিটা নিচের step-এর শিরোনাম Markdown cell-এ লেখো।
- Step-এর code আলাদা code cell-এ লিখে `Shift + Enter` চাপো।

## 9. Hands-On Notebook

### Step 1: একটা batch load করো

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

### Step 2: Images আর labels দেখো

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

**Expected result:** Image shape `(10000, 3072)`, dtype `uint8`, label shape `(10000,)`, আর pixel range 0 থেকে 255-এর মধ্যে।

### Step 3: Class names পড়ো

```python
meta_path = data_folder / "batches.meta"

with open(meta_path, "rb") as file:
    meta = pickle.load(file, encoding="bytes")

class_names = []

for name in meta[b"label_names"]:
    class_names.append(name.decode("utf-8"))

print(class_names)
```

### Step 4: হাতে হাতে একটা image তৈরি করো

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

### Step 5: সব images একসাথে reshape করো

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

### Step 6: Class distribution দেখো

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

**Expected result:** প্রতিটা class প্রায় 1,000 হবে। একটা training batch-এ count ঠিক সমান না-ও হতে পারে, তবে পাঁচটা batch মিলিয়ে প্রতিটা class-এ ঠিক 5,000 images থাকে।

### Step 7: প্রতিটা class-এর একটা image দেখো

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
| `File exists: False` | Notebook অন্য folder থেকে চলছে | `print(Path.cwd())` দিয়ে location check করো, notebook `notebooks/`-এ রাখো |
| `KeyError: 'data'` | Key bytes type | `batch[b"data"]` লেখো |
| `ModuleNotFoundError` | ভুল kernel | `dl_projects` activate করে JupyterLab চালু করো |
| Image-এর color বা shape অদ্ভুত | Reshape order ভুল | আগে `(N, 3, 32, 32)`, তারপর `transpose(0, 2, 3, 1)` |
| Image ঘোলা দেখায় | Display interpolation বা ছোট size | `interpolation="nearest"` ব্যবহার করো, 32 × 32 image ছোটই |

## 11. Presentation Summary

> CIFAR-10 contains 60,000 labeled 32 × 32 RGB images from 10 classes, split into 50,000 training images and 10,000 test images. Each raw image is stored as a row of 3,072 `uint8` values: 1,024 red, 1,024 green, and 1,024 blue. We reshape each row to recover its spatial structure and inspect the data type, value range, and class distribution before training a model.

## 12. Quiz

আগে নিজে উত্তর লেখো, তারপর answer খোলো।

1. একটা CIFAR-10 image-এ কেন 3,072টা value থাকে?
2. Raw row-এর index 1024 কোন channel-এর প্রথম value?
3. 500টা CIFAR-10 image flatten করলে data matrix-এর shape কী হবে?
4. `(N, C, H, W)` থেকে `(N, H, W, C)`-তে যেতে `transpose`-এর axes কী হবে, আর কেন?
5. 50,000 training images `float64`-এ রাখলে pixel data-এর জন্য কত bytes লাগবে?
6. Red channel-এর pixel $(r=2, c=5)$-এর raw index কত?
7. `uint8` array থেকে distance হিসাব করার আগে data convert করতে হয় কেন?

<details>
<summary><strong>Answers দেখো</strong></summary>

**1.** কারণ $32 \times 32 \times 3 = 3072$।

**2.** Green channel-এর, কারণ Red channel index 0 থেকে 1023 পর্যন্ত থাকে।

**3.** `(500, 3072)`।

**4.** `transpose(0, 2, 3, 1)`। Axis 0 ($N$), axis 2 ($H$) আর axis 3 ($W$) সামনে আসে, আর axis 1 ($C$) শেষে যায়।

**5.** $50000 \times 3072 \times 8 = 1{,}228{,}800{,}000$ bytes, অর্থাৎ 1.2288 GB।

**6.** $j = 1024(0) + 32(2) + 5 = 69$।

**7.** `uint8` negative value রাখতে পারে না, তাই বিয়োগে wraparound হয়ে ভুল distance আসতে পারে।

</details>

## 13. Practice

1. `data_batch_2` load করে তার class distribution বের করো।
2. `test_batch` load করে দেখাও যে প্রতিটা class-এ ঠিক 1,000 images আছে।
3. একটা image-এর Red, Green আর Blue channel আলাদা grayscale image হিসেবে plot করো।

## References

1. A. Krizhevsky, [*Learning Multiple Layers of Features from Tiny Images*](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Technical Report, University of Toronto, 2009.
2. [The CIFAR-10 and CIFAR-100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html), University of Toronto.
3. [CS231n: Image Classification notes](https://cs231n.github.io/classification/), Stanford University.
