## **Foxutils**

Utils for PyTorch-based deep-learning study. Most of the codes are from the author's daily work.

------

### **List of utils:**

(To be updated...)

#### **trainerX:**

Out-of-the-box neural network training code. Supports training log recording, automatic weight saving, storage of training parameters with YAML format and other functions. Training of different neural networks can be achieved only by simply inheriting and overloading some functions.

#### **plotter:**

- animation: Save pyTorch tensor to gif images.
- field: Encapsulated function and classes for fields plot.
- line: Encapsulated function and classes for lines plot.

#### **network:**

Basic network architectures like attention, normalization and UNet.

#### **helper:**

Some other useful functions.

------

### Installation

```bash
python3 setup.py sdist bdist_wheel
cd dist
pip install foxutils-*.whl
```

