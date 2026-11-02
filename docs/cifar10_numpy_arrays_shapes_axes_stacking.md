# From CIFAR-10 Raw Data to an RGB Image: NumPy Foundations

**Project:** Vision From Scratch v1.0 · `scratchvision`  
**Related notebook:** `notebooks/00_data_exploration.ipynb`  
**Suggested location:** `docs/cifar10_numpy_arrays_shapes_axes_stacking.md`  
**Level:** Beginner · **Dependencies:** Python, NumPy, Matplotlib

---

# Part I — English

## 0. What are we trying to do?

The CIFAR-10 Python batch stores each 32 × 32 RGB image as **3,072 numbers in one row**. We will extract one row, divide it into red, green, and blue channels, turn each channel into a 32 × 32 grid, combine them into a color image, read its class name, and display it.

```python
first_row = raw_images[0]

red   = first_row[0:1024]
green = first_row[1024:2048]
blue  = first_row[2048:3072]

red_image   = red.reshape(32, 32)
green_image = green.reshape(32, 32)
blue_image  = blue.reshape(32, 32)

image = np.stack([red_image, green_image, blue_image], axis=2)

print("Image shape:", image.shape)
print("Class:", class_names[labels[0]])

plt.imshow(image, interpolation="nearest")
plt.title(class_names[labels[0]])
plt.axis("off")
plt.show()
```

Expected image shape: `(32, 32, 3)`. The displayed class depends on the actual first label. No specific class is assumed here.

## 1. Array indexing and slicing

**Definition:** An index selects one element; a slice selects a range. Python uses **zero-based indexing**.

```python
import numpy as np

a = np.array([10, 20, 30, 40, 50, 60])
print(a[0])       # 10
print(a[1:4])     # [20 30 40]
print(a[:3])      # [10 20 30]
print(a[3:])      # [40 50 60]
print(a[::2])     # [10 30 50]
```

The notation `a[start:stop:step]` includes `start` but excludes `stop`. An omitted `start`, `stop`, or `step` uses the corresponding default.

| Expression | Input shape | Output shape | Meaning |
|---|---|---|---|
| `a[0]` | `(6,)` | scalar `()` | One value |
| `a[1:4]` | `(6,)` | `(3,)` | Values at indices 1, 2, 3 |
| `first_row[0:1024]` | `(3072,)` | `(1024,)` | Red channel |
| `first_row[1024:2048]` | `(3072,)` | `(1024,)` | Green channel |
| `first_row[2048:3072]` | `(3072,)` | `(1024,)` | Blue channel |

Why 1,024? Each 32 × 32 color channel has `32 * 32 = 1024` values, and `3 * 1024 = 3072`. **CIFAR-10's stored channel order is red block, green block, blue block**, not alternating RGBRGB values in the flattened row.

## 2. Dimensions, shape, and axes

**Definition:** An array's **number of dimensions** (`ndim`) tells us how many indices are needed to locate one scalar element. Its **shape** is a tuple giving the length of each dimension. An **axis** is a numbered dimension, starting at 0.

```python
v = np.array([10, 20, 30])
M = np.array([[1, 2, 3], [4, 5, 6]])
T = np.zeros((2, 3, 4))
B = np.zeros((5, 32, 32, 3))

print(v.shape, v.ndim)  # (3,) 1
print(M.shape, M.ndim)  # (2, 3) 2
print(T.shape, T.ndim)  # (2, 3, 4) 3
print(B.shape, B.ndim)  # (5, 32, 32, 3) 4
```

| Array | Shape | Valid existing axes | Typical interpretation |
|---|---|---|---|
| 1D | `(3,)` | 0 | three values |
| 2D | `(2, 3)` | 0, 1 | two rows, three columns |
| 3D | `(2, 3, 4)` | 0, 1, 2 | two grids of 3 × 4 |
| 3D RGB image | `(32, 32, 3)` | 0, 1, 2 | height, width, channels |
| 4D image batch | `(5, 32, 32, 3)` | 0, 1, 2, 3 | images, height, width, channels |

A 3D NumPy array means **three indexing dimensions**, not necessarily 3D physical space.

A 2D matrix `M` has `M[0, 1] == 2`: first row (axis 0), second column (axis 1). For a color image, `image[10, 20, 2]` means row 10, column 20, blue channel.

For a hypothetical 5D array with shape `(2, 4, 3, 32, 32)`, axes 0, 1, 2, 3, 4 correspond to the five positions in that tuple. **Axis numbers do not have universal meanings**; their interpretation depends on the chosen data layout.

## 3. Reshape: 1D to 2D

**Definition:** `reshape()` gives the same elements a new shape, without changing their sequential order under the default C-order convention. The total number of elements must stay equal.

```python
a = np.array([1, 2, 3, 4, 5, 6])    # shape (6,)
b = a.reshape(2, 3)                 # shape (2, 3)
print(b)
# [[1 2 3]
#  [4 5 6]]
```

Hand calculation: `6 = 2 × 3`. A request such as `a.reshape(4, 2)` fails because `4 × 2 != 6`.

For CIFAR-10: `red` has shape `(1024,)`, while `red.reshape(32, 32)` has shape `(32, 32)`. The same applies to green and blue. **Reshaping only changes organization; it does not invent pixels or merge channels.**

## 4. Understanding `axis` deeply

**Core rule:** Existing axes are numbered from 0 to `ndim - 1`. But the meaning of an `axis` parameter depends on the NumPy function.

### 4.1 Axes when indexing a matrix

```python
A = np.array([[1, 2],
              [3, 4]])
```

- Axis 0: row positions (two rows).
- Axis 1: column positions (two columns).
- `A[1, 0] == 3`: second row, first column.

### 4.2 Axes in `np.stack`: where to INSERT a NEW axis

**Definition:** `np.stack([A, B], axis=k)` combines arrays of identical shape by inserting one new dimension at position `k`.

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.stack([A, B], axis=0))
# [[[1 2]
#   [3 4]]
#
#  [[5 6]
#   [7 8]]]

print(np.stack([A, B], axis=1))
# [[[1 2]
#   [5 6]]
#
#  [[3 4]
#   [7 8]]]

print(np.stack([A, B], axis=2))
# [[[1 5]
#   [2 6]]
#
#  [[3 7]
#   [4 8]]]
```

All three outputs have shape `(2, 2, 2)` **only because this tiny example uses all sizes 2**. The *element arrangements* are different:

- `axis=0`: `out[0]` is all of `A`; `out[1]` is all of `B`.
- `axis=1`: `out[:, 0, :]` is `A`; `out[:, 1, :]` is `B`.
- `axis=2`: `out[:, :, 0]` is `A`; `out[:, :, 1]` is `B`.

To see shape differences, stack **three** arrays of shape `(2, 4)`:

| Stack choice | New output shape | Axis interpretation |
|---|---|---|
| `axis=0` | `(3, 2, 4)` | channel/group, row, column |
| `axis=1` | `(2, 3, 4)` | row, channel/group, column |
| `axis=2` | `(2, 4, 3)` | row, column, channel/group |

**General formula:** If the input shape has `n` dimensions, `np.stack()` produces `n + 1` dimensions. The allowed nonnegative insertion positions are `axis=0` through `axis=n`, inclusive. For example, stacking 3D inputs allows `axis=0, 1, 2, 3`; stacking 4D inputs allows `axis=0, 1, 2, 3, 4`.

### 4.3 Why CIFAR-10 uses `axis=2`

```python
image = np.stack([red_image, green_image, blue_image], axis=2)
# inputs: 3 × (32, 32)
# output: (32, 32, 3)
```

We want `image[row, column]` to be a triple `[R, G, B]`. Adding the color-channel dimension at the end (axis 2) gives the standard `(height, width, channels)` arrangement understood by `plt.imshow`.

```python
print(image[0, 0])     # [R_of_first_pixel, G_of_first_pixel, B_of_first_pixel]
print(image[:, :, 0].shape)  # (32, 32), red channel
```

Using `axis=0` instead would produce shape `(3, 32, 32)` (channel-first); this is useful for some ML conventions but **not** the expected RGB layout for ordinary `plt.imshow(image)`.

### 4.4 Axes in `np.sum` or `np.mean`: REDUCE an existing axis

This differs from `np.stack`.

```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])

print(M.sum(axis=0))  # [5 7 9] (sum down rows; preserve columns)
print(M.sum(axis=1))  # [6 15]  (sum across columns; preserve rows)
```

`axis=0` in `sum` does **not** insert a dimension. It says which existing dimension to sum over. Axis meanings are **operation-specific**.

## 5. Nested indexing: `class_names[labels[0]]`

**Definition:** An expression inside square brackets can itself be an indexing operation. Evaluate the inner operation first.

```python
labels = np.array([3, 1, 0])
class_names = ["cat", "dog", "bird", "fish"]

first_label = labels[0]           # 3
first_name = class_names[3]      # "fish"
print(class_names[labels[0]])    # fish
```

In the real CIFAR-10 notebook, `labels[0]` is the integer category ID of the first image, and `class_names[...]` translates that ID into a readable name. **The class ID is not the image itself.**

## 6. Matplotlib: `imshow`, `interpolation`, `title`, `axis`, `show`

```python
import matplotlib.pyplot as plt

plt.imshow(image, interpolation="nearest")
plt.title(class_names[labels[0]])
plt.axis("off")
plt.show()
```

- `plt.imshow(image)`: displays the pixel array as an image. For an RGB image, expected input shape is `(H, W, 3)`.
- `interpolation="nearest"`: when the display scales a small image, use nearest-neighbor sampling, preserving a blocky pixel appearance instead of smoothing between neighboring values. It does **not** change the original NumPy array.
- `plt.title(...)`: adds the readable class name above the image.
- `plt.axis("off")`: hides the plotted coordinate axes, ticks, and labels. **This Matplotlib `axis` setting is unrelated to the NumPy `axis=2` parameter.**
- `plt.show()`: renders the figure.

CIFAR-10 pixel arrays normally contain integer values 0–255, which `imshow` can display as RGB values when the shape is `(32, 32, 3)`.

## 7. A mini end-to-end experiment

This synthetic 2 × 2 image demonstrates the same transformation before applying it to 32 × 32 images:

```python
import numpy as np

flat = np.array([
    255, 0,   0,   255,   # 4 red values
    0,   255, 0,   255,   # 4 green values
    0,   0,   255, 255    # 4 blue values
], dtype=np.uint8)

r = flat[0:4].reshape(2, 2)
g = flat[4:8].reshape(2, 2)
b = flat[8:12].reshape(2, 2)
rgb = np.stack([r, g, b], axis=2)

print(rgb.shape)   # (2, 2, 3)
print(rgb[0, 0])   # [255   0   0] => red
print(rgb[0, 1])   # [  0 255   0] => green
print(rgb[1, 0])   # [  0   0 255] => blue
print(rgb[1, 1])   # [255 255 255] => white
```

**Expected use later:** understanding shape, slicing, and axes is essential for feature extraction, vectorized kNN, neural-network inputs, and avoiding mismatched tensor shapes.

## 8. Quick quiz

1. Why does `a[1:4]` return three elements, not four?
2. What are the shape and `ndim` of a 32 × 32 RGB image?
3. Three arrays each have shape `(2, 4)`. What are the output shapes for `np.stack(..., axis=0)`, `axis=1`, and `axis=2`?
4. What is the difference between `np.stack(..., axis=0)` and `np.sum(..., axis=0)`?
5. If `labels[0] == 3`, how does `class_names[labels[0]]` find a name?

---

# Part II — বাংলা (Bengali)

## ০. আমরা কী করছি, কেন করছি?

CIFAR-10 dataset-এ একটি 32 × 32 RGB image প্রথমে **3072টি সংখ্যা-সহ একটি 1D row** হিসেবে থাকে। আমরা সেই row থেকে Red, Green, Blue আলাদা করব, প্রতিটিকে 32 × 32 matrix বানাব, আবার একসঙ্গে জুড়ে একটি RGB image তৈরি করব। তারপর class name বের করে image দেখব।

```python
first_row = raw_images[0]
red   = first_row[0:1024]
green = first_row[1024:2048]
blue  = first_row[2048:3072]

red_image = red.reshape(32, 32)
green_image = green.reshape(32, 32)
blue_image = blue.reshape(32, 32)

image = np.stack([red_image, green_image, blue_image], axis=2)
```

Expected shape: `(32, 32, 3)`। Actual class name নির্ভর করবে `labels[0]`-এর মানের ওপর।

## ১. Array indexing এবং slicing

**Definition:** Index দিয়ে একটি element নেওয়া হয়। Slicing দিয়ে একাধিক element-এর একটি অংশ নেওয়া হয়। Python-এ index শুরু **0** থেকে।

```python
import numpy as np
a = np.array([10, 20, 30, 40, 50, 60])
print(a[0])     # 10
print(a[1:4])   # [20 30 40]
print(a[:3])    # [10 20 30]
print(a[3:])    # [40 50 60]
print(a[::2])   # [10 30 50]
```

`a[start:stop:step]`-এ `start`-এর element নেওয়া হয়, কিন্তু `stop`-এর element নেওয়া হয় **না**। তাই `a[1:4]` মানে index 1, 2, 3।

CIFAR-10 code-এ:

- `first_row[0:1024]` → Red-এর 1024টি value, output shape `(1024,)`
- `first_row[1024:2048]` → Green-এর 1024টি value, output shape `(1024,)`
- `first_row[2048:3072]` → Blue-এর 1024টি value, output shape `(1024,)`

কারণ একটি color channel-এ `32 × 32 = 1024` values। তিন channel-এ `1024 × 3 = 3072`। এখানে প্রথমে সব Red, তারপর Green, তারপর Blue values রাখা আছে।

## ২. 1D, 2D, 3D, ND, shape এবং axis

**Dimension (`ndim`)** মানে একটি scalar element খুঁজতে কয়টি index লাগে। **Shape** বলে প্রতিটি dimension-এর size। **Axis** হলো dimension-এর নম্বর, যা 0 থেকে শুরু হয়।

```python
v = np.array([10, 20, 30])
M = np.array([[1, 2, 3], [4, 5, 6]])
T = np.zeros((2, 3, 4))
B = np.zeros((5, 32, 32, 3))

print(v.shape, v.ndim)  # (3,) 1
print(M.shape, M.ndim)  # (2, 3) 2
print(T.shape, T.ndim)  # (2, 3, 4) 3
print(B.shape, B.ndim)  # (5, 32, 32, 3) 4
```

| Array | Shape | Axis | সহজ অর্থ |
|---|---|---|---|
| 1D | `(3,)` | 0 | 3টি value |
| 2D | `(2, 3)` | 0, 1 | 2 rows, 3 columns |
| 3D | `(2, 3, 4)` | 0, 1, 2 | 2টি 3 × 4 matrix |
| 3D RGB image | `(32, 32, 3)` | 0, 1, 2 | height, width, channels |
| 4D image batch | `(5, 32, 32, 3)` | 0, 1, 2, 3 | image count, height, width, channels |

উদাহরণ: `M[0, 1]` মানে first row এবং second column। RGB image-এর `image[10, 20, 2]` মানে row 10, column 20-এর Blue value।

**মনে রাখবে:** NumPy 3D array মানেই বাস্তব জগতের 3D geometry নয়। এর মানে 3টি indexing dimension। 5D shape `(2, 4, 3, 32, 32)` হলে axis 0, 1, 2, 3, 4 থাকবে। কোন axis-এর অর্থ কী হবে তা data layout নির্ধারণ করে।

## ৩. `reshape()`

**Definition:** `reshape()` element-এর মোট সংখ্যা ও default order অপরিবর্তিত রেখে arrangement বদলায়।

```python
a = np.array([1, 2, 3, 4, 5, 6])
b = a.reshape(2, 3)
print(b)
# [[1 2 3]
#  [4 5 6]]
```

প্রথমে shape `(6,)`, পরে `(2, 3)`। কারণ `6 = 2 × 3`। কিন্তু 6টি value-কে `(4, 2)` বানানো যাবে না, কারণ সেখানে 8টি value লাগবে।

তোমার `red.reshape(32, 32)` 1024টি value-কে 32 rows × 32 columns-এ সাজায়। Green ও Blue-এর ক্ষেত্রেও একই কাজ হয়।

## ৪. Axis — সবচেয়ে গুরুত্বপূর্ণ concept

**মূল ধারণা:** Axis = dimension number। Existing axes হলো `0` থেকে `ndim - 1`। কিন্তু একটি function-এর `axis` parameter **কী কাজ করবে** তা function অনুযায়ী আলাদা।

### ৪.১ 2D array-এর axis

```python
A = np.array([[1, 2],
              [3, 4]])
```

- `axis=0`: row positions-এর dimension
- `axis=1`: column positions-এর dimension
- `A[1, 0]` = 3, কারণ এটি second row-এর first value।

### ৪.২ `np.stack()`-এ axis মানে কোথায় NEW dimension বসবে

**Definition:** `np.stack()` একই shape-এর কয়েকটি array একসঙ্গে নিয়ে একটি **নতুন dimension** যোগ করে।

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.stack([A, B], axis=0))
# [[[1 2]
#   [3 4]]
#  [[5 6]
#   [7 8]]]

print(np.stack([A, B], axis=1))
# [[[1 2]
#   [5 6]]
#  [[3 4]
#   [7 8]]]

print(np.stack([A, B], axis=2))
# [[[1 5]
#   [2 6]]
#  [[3 7]
#   [4 8]]]
```

এই বিশেষ example-এ তিনটির shape-ই `(2, 2, 2)`। **Shape একই হলেও arrangement এক নয়।**

- `axis=0`: প্রথম position-এ A/B-এর নতুন dimension। `out[0] == A`।
- `axis=1`: row ও column dimension-এর মাঝখানে নতুন dimension। `out[:, 0, :] == A`।
- `axis=2`: সবার শেষে নতুন dimension। `out[:, :, 0] == A`।

Shape-এর পার্থক্য দেখতে **তিনটি `(2, 4)` array** stack করার কথা ভাবো:

| New axis | Output shape | Dimension order |
|---|---|---|
| 0 | `(3, 2, 4)` | group, row, column |
| 1 | `(2, 3, 4)` | row, group, column |
| 2 | `(2, 4, 3)` | row, column, group |

**General rule:** Input যদি `n` dimensional হয়, `np.stack()` output হবে `n+1` dimensional। তাই 2D input-এর জন্য `axis=0,1,2`; 3D input-এর জন্য `axis=0,1,2,3`; 4D input-এর জন্য `axis=0,1,2,3,4`। এখানে শেষ position পর্যন্ত allowed, কারণ নতুন dimension ঢোকানো হচ্ছে।

### ৪.৩ CIFAR-10-এ `axis=2` কেন?

```python
image = np.stack([red_image, green_image, blue_image], axis=2)
# (32, 32), (32, 32), (32, 32) -> (32, 32, 3)
```

আমরা চাই প্রতিটি pixel-এর তিনটি color value পাশাপাশি থাকুক:

```python
image[0, 0]       # [R, G, B] values of the first pixel
image[:, :, 0]    # full Red channel, shape (32, 32)
```

সেজন্য নতুন **channel** axis-টি শেষে বসাই: `(height, width, channels)`। `axis=0` দিলে `(3, 32, 32)` হবে, যা channel-first layout। সাধারণ `plt.imshow(image)`-এ RGB দেখানোর জন্য `(32, 32, 3)` দরকার।

### ৪.৪ `np.sum(axis=...)` কিন্তু আলাদা

`np.stack()` নতুন axis **যোগ করে**। `np.sum()` নির্দিষ্ট existing axis বরাবর যোগ করে সেটি **কমিয়ে দেয়**।

```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])

print(M.sum(axis=0))  # [5 7 9] — rows বরাবর sum
print(M.sum(axis=1))  # [6 15]  — columns বরাবর sum
```

এখানে axis 0 বা 1 মানে নতুন dimension বসানোর জায়গা নয়। **একই parameter name, ভিন্ন operation**।

## ৫. `class_names[labels[0]]` — Nested indexing

**Definition:** একটি indexing-এর result দিয়ে আরেকটি list/array index করা যায়। ভেতরের অংশ আগে evaluate হয়।

```python
labels = np.array([3, 1, 0])
class_names = ["cat", "dog", "bird", "fish"]

print(labels[0])              # 3
print(class_names[3])          # fish
print(class_names[labels[0]])  # fish
```

প্রথমে `labels[0]` থেকে class ID 3 পাওয়া গেল। তারপর `class_names[3]` থেকে `fish` পাওয়া গেল। CIFAR-10 notebook-এ একইভাবে প্রথম image-এর numerical label থেকে readable class name পাওয়া যায়।

## ৬. `plt.imshow(image, interpolation="nearest")`

```python
import matplotlib.pyplot as plt
plt.imshow(image, interpolation="nearest")
plt.title(class_names[labels[0]])
plt.axis("off")
plt.show()
```

- `imshow(image)`: NumPy pixel values দিয়ে image দেখায়। RGB image-এর shape `(H, W, 3)` হওয়া উচিত।
- `interpolation="nearest"`: 32 × 32 image বড় করে দেখানোর সময় closest pixel-এর color ব্যবহার করে, ফলে pixel boundaries স্পষ্ট থাকে; smoothing হয় না। **মূল array বদলায় না।**
- `title(...)`: image-এর ওপরে class name লেখে।
- `axis("off")`: plot-এর x/y coordinate axes ও tick labels লুকায়। এটি `np.stack(axis=2)`-এর axis নয়।
- `show()`: figure render করে।

CIFAR-10-এর RGB pixel values সাধারণত 0 থেকে 255 পর্যন্ত integer হয়।

## ৭. হাতে-কলমে 2 × 2 RGB example

32 × 32-এর আগে ছোট example-এ একই idea পরীক্ষা করি:

```python
import numpy as np

flat = np.array([
    255, 0,   0,   255,   # Red channel
    0,   255, 0,   255,   # Green channel
    0,   0,   255, 255    # Blue channel
], dtype=np.uint8)

red = flat[0:4].reshape(2, 2)
green = flat[4:8].reshape(2, 2)
blue = flat[8:12].reshape(2, 2)
image = np.stack([red, green, blue], axis=2)

print(image.shape)  # (2, 2, 3)
print(image[0, 0])  # [255   0   0] = red
print(image[0, 1])  # [  0 255   0] = green
print(image[1, 0])  # [  0   0 255] = blue
print(image[1, 1])  # [255 255 255] = white
```

**Future connection:** Image classification, HOG/color features, kNN vectorization, neural-network tensor shapes, সব জায়গায় slicing, reshape ও axis বুঝতে হবে।

## ৮. Short quiz

1. `a[1:4]` কেন 4টি নয়, 3টি value দেয়?
2. একটি RGB image-এর shape `(32, 32, 3)` হলে `ndim` কত?
3. তিনটি `(2, 4)` array-কে `axis=0`, `1`, `2` দিয়ে stack করলে আলাদাভাবে output shape কী হবে?
4. `np.stack(axis=0)` এবং `np.sum(axis=0)`-এর difference কী?
5. `class_names[labels[0]]`-এ কোন অংশটি আগে evaluate হয়?

---

**Learning note:** These are independent explanations and examples for a self-learning NumPy reimplementation inspired by CS231n. Keep course-specific submissions compliant with your course's academic integrity rules.
