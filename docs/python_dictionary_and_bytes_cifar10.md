# Python Dictionary and Bytes: Understanding the CIFAR-10 Batch

**Project:** [Vision From Scratch](https://github.com/Istiak665/vision-from-scratch)  
**Notebook:** `notebooks/00_data_exploration.ipynb`  
**Topics:** Python Dictionary (`dict`) and Bytes (`bytes`)  
**Level:** Beginner

---

# Part 1 — English

## 1. What is a Python Dictionary?

A **dictionary** is a Python data structure that stores information as **key–value pairs**.

- A **key** identifies a piece of information.
- A **value** is the information stored under that key.

Imagine storing information about a student:

```python
student = {
    "name": "Rahim",
    "age": 25,
    "department": "ECE"
}
```

| Key | Value |
|---|---|
| `"name"` | `"Rahim"` |
| `"age"` | `25` |
| `"department"` | `"ECE"` |

To retrieve a specific value, use its key:

```python
print(student["name"])
```

**Output:**

```text
Rahim
```

To check the object's type:

```python
print(type(student))
```

**Output:**

```text
<class 'dict'>
```

This tells us that `student` is a Python dictionary.

## 2. What Does the CIFAR-10 Batch Output Mean?

In `notebooks/00_data_exploration.ipynb`, we load a CIFAR-10 batch:

```python
# "rb" means read binary.
with open(batch_path, "rb") as file:
    batch = pickle.load(file, encoding="bytes")

print(type(batch))
print(batch.keys())
```

**Output:**

```text
<class 'dict'>
dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
```

Here's how to read this output:

- `type(batch)` returns `<class 'dict'>`, so `batch` is a dictionary.
- `batch.keys()` shows the keys available in that dictionary.
- `dict_keys(...)` is Python's representation of a *view* of the dictionary's keys; it is not a list of the image pixels.

| CIFAR-10 key | What its value contains |
|---|---|
| `b'batch_label'` | A description of the batch |
| `b'labels'` | Class labels for the images (integers from 0 to 9) |
| `b'data'` | Image pixel values |
| `b'filenames'` | Original image filenames |

For example, we can access the labels using:

```python
labels = batch[b"labels"]
print(type(labels))
```

Each CIFAR-10 training batch contains **10,000 images**. We will inspect the exact image array shape in a later lesson.

## 3. Why Is There a `b` Before Each Key?

Python distinguishes between a **string** (`str`) and a **bytes object** (`bytes`).

```python
name = "Rahim"    # String (str)
name_bytes = b"Rahim"  # Bytes (bytes)

print(type(name))
print(type(name_bytes))
```

**Output:**

```text
<class 'str'>
<class 'bytes'>
```

A bytes object represents a sequence of byte values, with each byte ranging from **0 to 255**. The `b` prefix tells Python that a literal is bytes rather than a normal text string.

In our CIFAR-10 experiment, the file was saved in an older Python pickle format. When we load it with:

```python
batch = pickle.load(file, encoding="bytes")
```

its original string keys are loaded as **bytes keys**, such as `b'data'`.

This means the following keys are **not interchangeable**:

```python
batch[b"data"]  # Correct: bytes key exists
batch["data"]   # KeyError: string key does not exist
```

**Important distinction:** The `b` belongs to the **dictionary key**, not to the image array itself. We will inspect the array's data type separately.

## 4. Try It in Our Notebook

Run this in the next cell of `notebooks/00_data_exploration.ipynb`:

```python
print(type(batch[b"data"]))
print(type(batch[b"labels"]))
print(len(batch[b"labels"]))
```

- `type(...)` tells us the Python type of an object.
- `len(...)` tells us how many elements are in a collection.

**Expected result:** The first expression shows a NumPy array type, the second shows a list type, and the third shows `10000`.

## 5. Quick Quiz

1. What is the difference between a **key** and a **value** in a dictionary?
2. Are `"data"` and `b"data"` the same Python data type?
3. What information does `batch.keys()` provide?

---

# Part 2 — বাংলা

## ১. Python Dictionary কী?

Python-এ **Dictionary** হলো এমন একটি data structure যেখানে তথ্য **key–value pair** আকারে সংরক্ষণ করা হয়।

- **Key** হলো তথ্যের নাম বা পরিচয়।
- **Value** হলো সেই Key-এর অধীনে রাখা তথ্য।

ধরো, একজন student-এর তথ্য রাখতে চাই:

```python
student = {
    "name": "Rahim",
    "age": 25,
    "department": "ECE"
}
```

| Key | Value |
|---|---|
| `"name"` | `"Rahim"` |
| `"age"` | `25` |
| `"department"` | `"ECE"` |

একটি নির্দিষ্ট Value বের করতে তার Key ব্যবহার করি:

```python
print(student["name"])
```

**Output:**

```text
Rahim
```

Variable-এর type দেখতে:

```python
print(type(student))
```

**Output:**

```text
<class 'dict'>
```

অর্থাৎ, `student` একটি Python Dictionary।

## ২. আমাদের CIFAR-10 Batch Output-এর অর্থ কী?

`notebooks/00_data_exploration.ipynb`-এ আমরা একটি CIFAR-10 batch load করেছি:

```python
# "rb" মানে read binary।
with open(batch_path, "rb") as file:
    batch = pickle.load(file, encoding="bytes")

print(type(batch))
print(batch.keys())
```

**Output:**

```text
<class 'dict'>
dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
```

এখানে:

- `type(batch)` থেকে বোঝা যাচ্ছে, `batch` একটি Dictionary।
- `batch.keys()` Dictionary-তে থাকা সব Key দেখাচ্ছে।
- `dict_keys(...)` হলো Dictionary-এর Keys দেখানোর Python representation বা *view*। এটি image pixel-এর list নয়।

| CIFAR-10 Key | কী তথ্য থাকে |
|---|---|
| `b'batch_label'` | Batch-এর description |
| `b'labels'` | প্রতিটি image-এর class label (0–9) |
| `b'data'` | Image pixel data |
| `b'filenames'` | Original image filenames |

উদাহরণস্বরূপ, labels বের করতে পারি:

```python
labels = batch[b"labels"]
print(type(labels))
```

প্রতিটি CIFAR-10 training batch-এ **10,000টি image** আছে। পরবর্তী lesson-এ image array-এর exact shape দেখব।

## ৩. প্রতিটি Key-এর আগে `b` কেন?

Python-এ সাধারণ **String (`str`)** এবং **Bytes (`bytes`)** দুটি আলাদা data type।

```python
name = "Rahim"          # String (str)
name_bytes = b"Rahim"   # Bytes (bytes)

print(type(name))
print(type(name_bytes))
```

**Output:**

```text
<class 'str'>
<class 'bytes'>
```

একটি Bytes object-এর প্রতিটি byte-এর value **0 থেকে 255** পর্যন্ত হতে পারে। `b` prefix বোঝায় যে এটি সাধারণ text String নয়, বরং Bytes literal।

আমাদের CIFAR-10 file একটি পুরোনো Python pickle format-এ save করা। আমরা যখন লিখি:

```python
batch = pickle.load(file, encoding="bytes")
```

তখন original string Keys-গুলো **Bytes key** হিসেবে load হয়। যেমন: `b'data'`।

তাই নিচের দুটি এক নয়:

```python
batch[b"data"]  # Correct: Bytes key আছে
batch["data"]   # KeyError: এই String key নেই
```

**মনে রাখবে:** এখানে `b` হলো Dictionary-এর **Key-এর অংশ**। এর অর্থ এই নয় যে image array নিজেই একটি Bytes object। Image array-এর data type আমরা আলাদাভাবে পরীক্ষা করব।

## ৪. Notebook-এ নিজে Experiment করো

`notebooks/00_data_exploration.ipynb`-এর পরবর্তী cell-এ লিখো:

```python
print(type(batch[b"data"]))
print(type(batch[b"labels"]))
print(len(batch[b"labels"]))
```

- `type(...)` কোনো object-এর Python type দেখায়।
- `len(...)` কোনো collection-এ কতগুলো element আছে তা দেখায়।

**Expected result:** প্রথমটি NumPy array type, দ্বিতীয়টি list type এবং তৃতীয়টি `10000` দেখাবে।

## ৫. ছোট Quiz

1. Dictionary-তে **Key** এবং **Value**-এর পার্থক্য কী?
2. `"data"` এবং `b"data"` কি একই Python data type?
3. `batch.keys()` আমাদের কী information দেয়?

---

**Next topic:** Understanding the structure and shape of CIFAR-10 image data (`batch[b"data"]`).
