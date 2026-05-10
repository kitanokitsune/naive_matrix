# Naive Matrix Arithmetic Library

A lightweight, intuitive Python library for matrix arithmetic, supporting exact rational arithmetic, complex numbers, and various matrix decompositions.

## 🌟 Features

- **Comprehensive Arithmetic**: Supports addition (`+`), subtraction (`-`), scalar multiplication (`*`), matrix multiplication (`@`), matrix division (`/`), and integer powers (`**`).
- **High Precision**: Seamless integration with Python's `fractions.Fraction` and `decimal.Decimal` for exact calculations, avoiding floating-point errors.
- **Advanced Operations**:
  - Determinant (`det()`)
  - Rank (`rank()`)
  - Inverse (`inv()`)
  - Moore-Penrose Pseudo-inverse (`pinv()`)
  - Transpose (`T()` or `transpose()`)
  - Adjoint (Conjugate Transpose) (`adjoint()`)
- **Type Conversion**: Easily switch between exact rational representation (`Frac()`) and floating-point representation (`Float()`).
- **Pretty Printing**: Built-in support for formatted matrix output for better readability.

## 📋 Requirements

- Python 3.x

This library relies on several modules to perform Gauss-Jordan elimination and complex arithmetic. Ensure the following files are available in the same directory:

- `ratcomplex.py` (for rational complex numbers)
- `gaussjordan.py` (for core matrix algorithms)

## 🚀 Quick Start

```python
from naive_matrix import NaiveMatrix, NaiveIdentityMatrix

# 1. Create a matrix
A = NaiveMatrix([[1, 2], [3, 4]])
B = NaiveMatrix([[5, 6], [7, 8]])
A.pprint("Matirx A=")
B.pprint("Matirx B=")

# 2. Basic Arithmetic
print("Scalar Multiplication:\n", -1 * A)
print("Scalar Division:\n", A / 2)
print("Addition:\n", A + B)
print("Matrix Multiplication:\n", A @ B)
print("Matrix Division:\n", A / B)  # equivlent to: A @ B.inv()

# 3. Exact Rational Arithmetic
# Convert to Fraction to avoid floating-point errors
C = NaiveMatrix([[0.1, 0.2], [0.3, 0.4]]).Frac(max_denominator=10)
print("Exact Matrix:")
C.pprint()

# 4. Advanced Operations
print("Determinant:", C.det())
print("Inverse:\n", C.inv())  # equivalent to: 1/A or A**-1
print("Transpose:\n", C.T())

# 5. Special Matrices
I = NaiveIdentityMatrix(3)
print("Identity Matrix:\n", I)
```

## 🛠 API Reference

### `NaiveMatrix(mat=None)`
The core class. `mat` can be a list of lists or another `NaiveMatrix` instance.

| Method | Description |
|:---|:---|
| `.inv()` | Returns the inverse matrix. (equivalent to `1/A`, `A**-1`) |
| `.pinv()` | Returns the Moore-Penrose pseudo-inverse. |
| `.det()` | Returns the determinant. |
| `.rank()` | Returns the rank of the matrix. |
| `.T()` / `.transpose()` | Returns the transposed matrix. |
| `.adjoint()` | Returns the conjugate transpose. |
| `.Frac(max_denominator=None)` | Converts elements to `Fraction`. |
| `.Float()` | Converts elements to `float`. |
| `.pprint(prefix='')` | Pretty-prints the matrix to the console. |
| `.spprint(prefix='')` | Returns the output of `.pprint()` as a string. |


### Utility Functions
- `NaiveIdentityMatrix(n)`: Returns an $n \times n$ identity matrix.
- `NaiveZeroMatrix(m, n=None)`: Returns an $m \times n$ zero matrix.
- `NaiveOneMatrix(m, n=None)`: Returns an $m \times n$ matrix filled with ones.

## ⚠️ Note on Complexity
As the name "Naive" suggests, this library is designed for **educational purposes or small-scale exact calculations**. It is not optimized for high-performance large-scale computing like NumPy.

## 📄 License
[MIT License](LICENSE)
