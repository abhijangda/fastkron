# PyFastKron API

PyFastKron provides modules for NumPy and PyTorch. The NumPy module, `FastKronNumpy` supports x86 backend. The PyTorch backend, `FastKronTorch` supports both x86 and CUDA backends.

Import modules as:

```
import pyfastkron.fastkronnumpy
import pyfastkron.fastkrontorch
```

## GeKMM

```
def gekmm(x, fs, alpha=1.0, beta=0.0, y=None, trX = False, trF = False)
```
Perform Generalized Kronecker Matrix-Matrix Multiplication (GeKMM).

$Z = \alpha ~ X \times \left( F^1 \otimes F^2 \otimes \dots F^N \right) + \beta Y$

Supports 32-bit floats or 64-bit double.

Can throw following exceptions:
* '': 

* **Parameters**
    * `x` is a 2-D array
    * `fs` is a list of 2-D arrays
    * `alpha` and `beta` are constants
    * `y` is a 2-D array
    * `trX` if true performs transpose of `x` before computing GeKMM
    * `trF` if true performs transpose of each of `fs` before computing GeKMM

* **Returns**
    Returns output as a 2-D array.

## Example

```
import pyfastkron.fastkronnumpy as fk

x = np.ones((10, 8**5), dtype=np.float32)
fs = [np.ones((8,8), dtype=np.float32) for i in range(5)]
y = fk.gekmm(x, fs)
```