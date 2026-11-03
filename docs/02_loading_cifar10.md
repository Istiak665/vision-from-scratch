# Lesson 02: Loading the Full CIFAR-10 Dataset

> **Project:** Vision From Scratch
> **Code:** `scratchvision/data/cifar10.py`
> **Tests:** `tests/test_shapes.py`
> **Notebook:** `notebooks/00_data_exploration.ipynb`
> **Prerequisite:** [Lesson 01: Understanding the CIFAR-10 Dataset](01_cifar10_dataset.md)
> **Next lesson:** Train/validation split and preprocessing

**Language note:** Each section is written in English first, followed by Bengali. Code, equations, and tables are shared by both languages.

**ভাষা সম্পর্কে:** প্রতিটা section-এ আগে English, তারপর Bengali লেখা আছে। Code, equation আর table দুই ভাষার জন্য একই।

## Learning Objectives

By the end of this lesson, you will be able to:

- Turn exploration code into reusable functions.
- Combine five training batches into one array.
- Explain how `np.concatenate` changes array shapes.
- Find the source batch of any training image by hand.
- Verify a data loader with automated tests.

এই lesson শেষে তুমি পারবে:

- Notebook-এর exploration code থেকে reusable function তৈরি করতে।
- পাঁচটা training batch জুড়ে একটা array বানাতে।
- `np.concatenate` কীভাবে array-এর shape বদলায়, তা ব্যাখ্যা করতে।
- যেকোনো training image কোন batch থেকে এসেছে, তা হাতে হিসাব করতে।
- Automated test দিয়ে data loader verify করতে।

## 1. Why Do We Need a Loader?

In Lesson 01, we explored one batch inside a notebook. Every future model needs all 50,000 training images, so repeating that code in every notebook would be slow and error-prone. A **loader** keeps this logic in one tested place, and every notebook can import it.

Lesson 01-এ আমরা notebook-এর ভেতরে একটা batch দেখেছিলাম। কিন্তু পরের প্রতিটা model-এর দরকার 50,000টা training image। প্রতিটা notebook-এ একই code লিখলে সময় নষ্ট হবে এবং ভুলের সম্ভাবনা বাড়বে। **Loader** এই logic-কে একটা tested জায়গায় রাখে, আর প্রতিটা notebook সেখান থেকে import করতে পারে।

The loader has one job:

Loader-এর কাজ একটাই:

```text
data_batch_1 ... data_batch_5  ->  X_train (50000, 32, 32, 3), y_train (50000,)
test_batch                     ->  X_test  (10000, 32, 32, 3), y_test  (10000,)
```

## 2. Notation

| Symbol | Meaning | Value |
|---|---|---|
| $N_{\text{train}}$ | Number of training images | 50,000 |
| $N_{\text{test}}$ | Number of test images | 10,000 |
| $M$ | Images per batch | 10,000 |
| $B$ | Number of training batches | 5 |
| $i$ | Index in `X_train` | $0, 1, \dots, 49999$ |
| $b$ | Source batch number | $1, 2, \dots, 5$ |
| $p$ | Position inside the batch | $0, 1, \dots, 9999$ |

The size of the training set is:

Training set-এর size হলো:

$$
N_{\text{train}} = B \times M = 5 \times 10000 = 50000
$$

## 3. How Concatenation Works

`np.concatenate` joins arrays along one chosen axis. The sizes on all other axes must match. The chosen axis is added together, and the others stay the same.

`np.concatenate` একটা নির্দিষ্ট axis বরাবর array জোড়া লাগায়। বাকি সব axis-এর size একই হতে হয়। নির্বাচিত axis-এর size যোগ হয়, আর বাকিগুলো বদলায় না।

### Toy Example

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])  # Shape: (2, 2)
b = np.array([[5, 6]])          # Shape: (1, 2)

joined = np.concatenate([a, b], axis=0)

print(joined)
print(joined.shape)  # (3, 2)
```

Axis 0 is the row axis, so $2 + 1 = 3$ rows. The column count stays 2.

Axis 0 হলো row axis, তাই $2 + 1 = 3$টা row হয়। Column সংখ্যা 2-ই থাকে।

### Applying It to CIFAR-10

Each batch has shape `(10000, 32, 32, 3)`. Joining along axis 0 gives:

প্রতিটা batch-এর shape `(10000, 32, 32, 3)`। Axis 0 বরাবর জুড়লে পাই:

$$
(10000 + 10000 + 10000 + 10000 + 10000,\ 32,\ 32,\ 3) = (50000,\ 32,\ 32,\ 3)
$$

> **Warning:** With `axis=1`, NumPy gives no error. The shape becomes `(10000, 160, 32, 3)`, which silently stacks five images into one tall image. A running program is not proof of correct data, so we check shapes with tests.
>
> **সতর্কতা:** `axis=1` দিলে NumPy কোনো error দেয় না। Shape হয় `(10000, 160, 32, 3)`, অর্থাৎ পাঁচটা image নিঃশব্দে একটা লম্বা image হয়ে যায়। Code চলা মানেই data ঠিক থাকা নয়, তাই আমরা test দিয়ে shape check করি।

## 4. Finding the Source Batch by Hand

Because batches are joined in order, each index in `X_train` maps to exactly one batch and position:

Batch গুলো ক্রমানুসারে জোড়া হয়, তাই `X_train`-এর প্রতিটা index একটা নির্দিষ্ট batch ও position-এ পড়ে:

$$
b = \left\lfloor \frac{i}{M} \right\rfloor + 1
$$

$$
p = i \bmod M
$$

Here, $\lfloor \cdot \rfloor$ rounds down to a whole number, and $\bmod$ gives the remainder.

এখানে $\lfloor \cdot \rfloor$ মানে নিচের পূর্ণসংখ্যায় নামানো, আর $\bmod$ মানে ভাগশেষ।

**Example:** For $i = 23456$:

**উদাহরণ:** $i = 23456$ হলে:

$$
b = \left\lfloor \frac{23456}{10000} \right\rfloor + 1 = 2 + 1 = 3
$$

$$
p = 23456 \bmod 10000 = 3456
$$

So `X_train[23456]` is image 3456 of `data_batch_3`.

তাই `X_train[23456]` হলো `data_batch_3`-এর 3456 নম্বর image।

## 5. Design Decisions

| Decision | Reason | কারণ |
|---|---|---|
| Keep pixels as `uint8` | Uses 153.6 MB instead of 1.2288 GB | Memory প্রায় 8 গুণ কম লাগে |
| Convert to float later | Each model can choose the type it needs | প্রতিটা model নিজের দরকারমতো type নেবে |
| Return NHWC | Ready for Matplotlib and matches Lesson 01 | Matplotlib-এ সরাসরি দেখানো যায় |
| Use `pathlib` | Works across operating systems | সব OS-এ path একইভাবে কাজ করে |
| Keep the test set separate | The final evaluation must use unseen data | Final evaluation-এ অদেখা data লাগে |

We only separate the data here. The validation split comes in the next lesson.

এখানে আমরা শুধু data আলাদা রাখছি। Validation split পরের lesson-এ করব।

## 6. Writing `cifar10.py`

Open `scratchvision/data/cifar10.py` and write this code:

`scratchvision/data/cifar10.py` খুলে এই code লেখো:

```python
from pathlib import Path
import pickle

import numpy as np

# cifar10.py -> data -> scratchvision -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_FOLDER = PROJECT_ROOT / "datasets" / "cifar-10-batches-py"


def load_batch(batch_path):
    """Load one CIFAR-10 batch file.

    Returns:
        images: uint8 array of shape (N, 32, 32, 3)
        labels: int64 array of shape (N,)
    """
    # "rb" means read binary
    with open(batch_path, "rb") as file:
        batch = pickle.load(file, encoding="bytes")

    raw_images = batch[b"data"]
    labels = np.array(batch[b"labels"], dtype=np.int64)

    number_of_images = raw_images.shape[0]

    # Shape: (N, 3072) -> (N, 3, 32, 32)
    images = raw_images.reshape(number_of_images, 3, 32, 32)

    # Shape: (N, 3, 32, 32) -> (N, 32, 32, 3)
    images = images.transpose(0, 2, 3, 1)

    return images, labels


def load_class_names(data_folder=DEFAULT_DATA_FOLDER):
    """Return the 10 class names as a list of strings."""
    meta_path = Path(data_folder) / "batches.meta"

    with open(meta_path, "rb") as file:
        meta = pickle.load(file, encoding="bytes")

    class_names = []

    for name in meta[b"label_names"]:
        class_names.append(name.decode("utf-8"))

    return class_names


def load_cifar10(data_folder=DEFAULT_DATA_FOLDER):
    """Load the full CIFAR-10 dataset.

    Returns:
        X_train: uint8 array of shape (50000, 32, 32, 3)
        y_train: int64 array of shape (50000,)
        X_test: uint8 array of shape (10000, 32, 32, 3)
        y_test: int64 array of shape (10000,)
    """
    data_folder = Path(data_folder)

    image_batches = []
    label_batches = []

    # Load data_batch_1 to data_batch_5
    for batch_number in range(1, 6):
        batch_path = data_folder / f"data_batch_{batch_number}"
        images, labels = load_batch(batch_path)
        image_batches.append(images)
        label_batches.append(labels)

    # Join the 5 batches along the first axis
    X_train = np.concatenate(image_batches, axis=0)
    y_train = np.concatenate(label_batches, axis=0)

    # Load the test set separately
    X_test, y_test = load_batch(data_folder / "test_batch")

    return X_train, y_train, X_test, y_test
```

### Code Explanation

| Code | Explanation | ব্যাখ্যা |
|---|---|---|
| `.parent.parent.parent` | Moves up from the file to the project root | File থেকে তিন ধাপ উপরে project root-এ যায় |
| `DEFAULT_DATA_FOLDER` | Stores the default dataset location | Dataset-এর default location রাখে |
| `data_folder=DEFAULT_DATA_FOLDER` | Lets callers use another folder if needed | দরকার হলে অন্য folder দেওয়া যায় |
| `dtype=np.int64` | Stores labels as whole numbers | Label পূর্ণসংখ্যা হিসেবে রাখে |
| `reshape(...)` | Rebuilds the channels and pixels | Raw row থেকে channel ও pixel সাজায় |
| `transpose(0, 2, 3, 1)` | Moves channels to the last axis | Channel axis শেষে নেয় |
| `f"data_batch_{batch_number}"` | Inserts the number into the filename | Filename-এ batch number বসায় |
| `range(1, 6)` | Gives 1, 2, 3, 4, 5 | 1 থেকে 5 দেয়, 6 বাদ |
| `return X_train, ...` | Returns all four arrays | চারটা array একসাথে ফেরত দেয় |

## 7. Automated Tests

Open `tests/test_shapes.py` and write this code:

`tests/test_shapes.py` খুলে এই code লেখো:

```python
import numpy as np

from scratchvision.data.cifar10 import load_cifar10
from scratchvision.data.cifar10 import load_class_names

X_train, y_train, X_test, y_test = load_cifar10()


def test_training_shapes():
    assert X_train.shape == (50000, 32, 32, 3)
    assert y_train.shape == (50000,)


def test_test_shapes():
    assert X_test.shape == (10000, 32, 32, 3)
    assert y_test.shape == (10000,)


def test_pixel_range():
    assert X_train.dtype == np.uint8
    assert X_train.min() >= 0
    assert X_train.max() <= 255


def test_class_balance():
    train_counts = np.bincount(y_train, minlength=10)
    test_counts = np.bincount(y_test, minlength=10)

    assert np.all(train_counts == 5000)
    assert np.all(test_counts == 1000)


def test_class_names():
    class_names = load_class_names()

    assert len(class_names) == 10
    assert class_names[0] == "airplane"
    assert class_names[9] == "truck"
```

Each `assert` checks one fact. If the fact is false, the test fails and tells us where.

প্রতিটা `assert` একটা করে তথ্য check করে। তথ্যটা ভুল হলে test fail করে এবং কোথায় সমস্যা, তা দেখায়।

These tests need the local dataset, so run the download script first on a new computer.

এই test-গুলোর জন্য local dataset দরকার, তাই নতুন computer-এ আগে download script চালাতে হবে।

## 8. Run Instructions

### Step 1: Run the Tests

Open a new PowerShell tab with the **+** button, then run:

**+** button দিয়ে নতুন PowerShell tab খুলে এগুলো চালাও:

```powershell
cd E:\9_Self_ML_DL_Foundations\vision-from-scratch
conda activate dl_projects
python -m pytest tests -v
```

**Expected result:** The last line shows `5 passed`.

**Expected result:** শেষ line-এ `5 passed` দেখাবে।

### Step 2: Open the Notebook

Start JupyterLab if it is not running, then open `notebooks/00_data_exploration.ipynb`. Select **Kernel > Restart Kernel** so that the notebook sees your new code.

JupyterLab চালু না থাকলে চালু করো, তারপর `notebooks/00_data_exploration.ipynb` খোলো। **Kernel > Restart Kernel** select করো, যাতে notebook নতুন code দেখতে পায়।

> **Rule:** Restart the kernel whenever you change a `.py` file in `scratchvision/`.
>
> **নিয়ম:** `scratchvision/`-এর কোনো `.py` file বদলালে kernel restart করবে।

### Step 3: Add Part 2 to the Notebook

Add a Markdown cell titled `Part 2: Loading the Full Dataset`, then run each step below in its own code cell.

`Part 2: Loading the Full Dataset` শিরোনামে একটা Markdown cell যোগ করো, তারপর নিচের প্রতিটা step আলাদা code cell-এ চালাও।

## 9. Hands-On Notebook

### Step 1: Load the Dataset

```python
import matplotlib.pyplot as plt
import numpy as np

from scratchvision.data.cifar10 import DEFAULT_DATA_FOLDER
from scratchvision.data.cifar10 import load_batch
from scratchvision.data.cifar10 import load_cifar10
from scratchvision.data.cifar10 import load_class_names

X_train, y_train, X_test, y_test = load_cifar10()
class_names = load_class_names()

print("X_train:", X_train.shape, X_train.dtype)
print("y_train:", y_train.shape, y_train.dtype)
print("X_test:", X_test.shape, X_test.dtype)
print("y_test:", y_test.shape, y_test.dtype)
```

**Expected output:**

```text
X_train: (50000, 32, 32, 3) uint8
y_train: (50000,) int64
X_test: (10000, 32, 32, 3) uint8
y_test: (10000,) int64
```

### Step 2: Check Memory Usage

```python
train_bytes = X_train.nbytes
train_megabytes = train_bytes / 1_000_000

print("Training pixel memory:", train_bytes, "bytes")
print("Training pixel memory:", train_megabytes, "MB")
```

**Expected output:**

```text
Training pixel memory: 153600000 bytes
Training pixel memory: 153.6 MB
```

This matches the hand calculation $50000 \times 32 \times 32 \times 3 = 153{,}600{,}000$ bytes.

এটা হাতের হিসাব $50000 \times 32 \times 32 \times 3 = 153{,}600{,}000$ bytes-এর সাথে মিলে যায়।

### Step 3: Check the Class Balance

```python
train_counts = np.bincount(y_train, minlength=10)
test_counts = np.bincount(y_test, minlength=10)

for class_index in range(10):
    print(class_names[class_index], train_counts[class_index], test_counts[class_index])
```

**Expected result:** Each class shows 5000 training images and 1000 test images.

**Expected result:** প্রতিটা class-এ 5000 training image আর 1000 test image দেখাবে।

### Step 4: Verify the Index Formula

```python
index = 23456

# Calculate the source batch and position
batch_number = index // 10000 + 1
position = index % 10000

print("Batch number:", batch_number)
print("Position:", position)

# Load the source batch directly
batch_path = DEFAULT_DATA_FOLDER / f"data_batch_{batch_number}"
batch_images, batch_labels = load_batch(batch_path)

# Compare both copies
same_image = np.array_equal(X_train[index], batch_images[position])
same_label = y_train[index] == batch_labels[position]

print("Same image:", same_image)
print("Same label:", same_label)
```

**Expected output:**

```text
Batch number: 3
Position: 3456
Same image: True
Same label: True
```

### Step 5: Visualize Samples from Each Class

```python
samples_per_class = 5
rng = np.random.default_rng(42)

figure, axes = plt.subplots(10, samples_per_class, figsize=(6, 12))

for class_index in range(10):
    # Find every training image of this class
    class_indices = np.where(y_train == class_index)[0]

    # Pick 5 different images
    chosen_indices = rng.choice(class_indices, size=samples_per_class, replace=False)

    for column in range(samples_per_class):
        image_index = chosen_indices[column]
        axes[class_index, column].imshow(X_train[image_index], interpolation="nearest")
        axes[class_index, column].axis("off")

    axes[class_index, 0].set_title(class_names[class_index], fontsize=8, loc="left")

plt.tight_layout()
plt.show()
```

**Expected result:** A grid with 10 rows and 5 columns. Each row shows images of one class.

**Expected result:** 10টা row আর 5টা column-এর একটা grid। প্রতিটা row-তে এক class-এর image থাকবে।

Using `default_rng(42)` fixes the random seed, so the same images appear each time.

`default_rng(42)` random seed ঠিক করে দেয়, তাই প্রতিবার একই image আসে।

## 10. Common Errors

| Problem | Cause | Fix | সমাধান |
|---|---|---|---|
| `ModuleNotFoundError: scratchvision` | Wrong environment or kernel | Use `dl_projects` | `dl_projects` environment ব্যবহার করো |
| `FileNotFoundError` | Dataset is missing | Run `python scripts\download_cifar10.py` | Download script চালাও |
| `ImportError: cannot import name` | Old code in the kernel, or the file was not saved | Save the file and restart the kernel | File save করে kernel restart করো |
| Shape `(10000, 160, 32, 3)` | `axis=1` was used | Use `axis=0` | `axis=0` ব্যবহার করো |
| Strange image colors | Wrong reshape or transpose order | Compare with Section 6 | Section 6-এর সাথে মিলিয়ে দেখো |
| `0 items` or fewer than 5 tests | Test file was not saved | Save `tests/test_shapes.py` | Test file save করো |

## 11. Presentation Summary

> We wrote a reusable CIFAR-10 loader that reads five training batches and one test batch. Each batch is reshaped from `(N, 3072)` to `(N, 32, 32, 3)`, and the training batches are joined along axis 0 to form `X_train` with shape `(50000, 32, 32, 3)`. We keep pixels as `uint8` to save memory and verify the result with automated tests for shapes, value range, and class balance.

> আমরা একটা reusable CIFAR-10 loader লিখেছি, যা পাঁচটা training batch আর একটা test batch পড়ে। প্রতিটা batch `(N, 3072)` থেকে `(N, 32, 32, 3)` shape-এ reshape হয়, আর training batch গুলো axis 0 বরাবর জুড়ে `(50000, 32, 32, 3)` shape-এর `X_train` তৈরি হয়। Memory বাঁচাতে pixel গুলো `uint8` রাখা হয়েছে, আর shape, value range ও class balance automated test দিয়ে verify করা হয়েছে।

## 12. Quiz

Write your own answers first, then open the answers.

আগে নিজে উত্তর লেখো, তারপর answer খোলো।

1. Why do we use `axis=0` to join the training batches?

   Training batch জোড়া লাগাতে `axis=0` কেন ব্যবহার করি?

2. What shape do five arrays of shape `(10000, 32, 32, 3)` give with `axis=1`? Why is this dangerous?

   `(10000, 32, 32, 3)` shape-এর পাঁচটা array `axis=1`-এ জুড়লে shape কী হবে? এটা বিপজ্জনক কেন?

3. In which batch and position is `X_train[34567]`?

   `X_train[34567]` কোন batch-এর কোন position-এ আছে?

4. Why do we store pixels as `uint8` in the loader?

   Loader-এ pixel গুলো `uint8` হিসেবে কেন রাখি?

5. Why do we keep the test set separate from training data?

   Test set-কে training data থেকে আলাদা কেন রাখি?

6. Why are there three `.parent` calls in `PROJECT_ROOT`?

   `PROJECT_ROOT`-এ তিনবার `.parent` কেন আছে?

7. Why must the notebook kernel be restarted after editing `cifar10.py`?

   `cifar10.py` বদলানোর পর notebook kernel restart করতে হয় কেন?

<details>
<summary><strong>Show answers / উত্তর দেখো</strong></summary>

**1.** Axis 0 counts images, so joining along it adds more images while keeping each image's shape.

Axis 0 image গণনা করে। তাই এই axis-এ জুড়লে image-এর সংখ্যা বাড়ে, কিন্তু প্রতিটা image-এর shape একই থাকে।

**2.** `(10000, 160, 32, 3)`. NumPy gives no error, but the data is wrong.

`(10000, 160, 32, 3)`। NumPy কোনো error দেয় না, কিন্তু data ভুল হয়ে যায়।

**3.** $b = \lfloor 34567 / 10000 \rfloor + 1 = 4$ and $p = 34567 \bmod 10000 = 4567$, so it is image 4567 of `data_batch_4`.

$b = \lfloor 34567 / 10000 \rfloor + 1 = 4$ এবং $p = 34567 \bmod 10000 = 4567$, তাই এটা `data_batch_4`-এর 4567 নম্বর image।

**4.** It stores the original values exactly and uses one-eighth of the memory of `float64`.

এতে original value হুবহু থাকে, আর memory লাগে `float64`-এর এক-অষ্টমাংশ।

**5.** The test set estimates performance on unseen data. Using it for training or tuning would make that estimate too optimistic.

Test set অদেখা data-এ performance মাপে। Training বা tuning-এ ব্যবহার করলে result বাস্তবের চেয়ে ভালো দেখাবে।

**6.** The steps go from `cifar10.py` to `data/`, then `scratchvision/`, then the project root.

ধাপগুলো হলো `cifar10.py` থেকে `data/`, তারপর `scratchvision/`, তারপর project root।

**7.** Python keeps an imported module in memory, so the running kernel continues using the old version.

Python import করা module memory-তে রেখে দেয়, তাই চলমান kernel পুরনো version-ই ব্যবহার করতে থাকে।

</details>

## 13. Practice

1. Find the batch and position of `X_train[49999]`, then verify your answer in code.
2. Compute the mean pixel value of `X_train` and `X_test`.
3. Change `axis=0` to `axis=1` in a notebook cell and print the shape, without changing `cifar10.py`.

অনুশীলন:

1. `X_train[49999]`-এর batch ও position বের করো, তারপর code দিয়ে verify করো।
2. `X_train` আর `X_test`-এর mean pixel value বের করো।
3. `cifar10.py` না বদলে, notebook cell-এ `axis=0`-এর বদলে `axis=1` দিয়ে shape print করো।

## References

1. A. Krizhevsky, [*Learning Multiple Layers of Features from Tiny Images*](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Technical Report, University of Toronto, 2009.
2. [The CIFAR-10 and CIFAR-100 datasets](https://www.cs.toronto.edu/~kriz/cifar.html), University of Toronto.
3. [NumPy documentation: `numpy.concatenate`](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html).
4. [pytest documentation](https://docs.pytest.org/).
