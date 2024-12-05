# PyFastKron API

PyFastKron provides modules for NumPy and PyTorch. The NumPy module, `FastKronNumpy` supports x86 backend. The PyTorch module, `FastKronTorch` supports both x86 and CUDA backends.

## FastKronNumpy

Import modules as:

```
import numpy as np
import pyfastkron.fastkronnumpy
```

Functions:
```
def gekmm(fs : List[np.ndarray], x : np.ndarray,
          alpha : float = 1.0, beta : float = 0.0,
          y : Optional[np.ndarray] = None) -> np.ndarray

def gemkm(x : np.ndarray, fs : List[np.ndarray],
          alpha : float = 1.0, beta : float = 0.0,
          y : Optional[np.ndarray] = None)
```
Perform Generalized Kronecker Matrix-Matrix Multiplication (GeKMM):

$Z = \alpha ~ X \times \left( F^1 \otimes F^2 \otimes \dots F^N \right) + \beta Y$

or Generalized Matrix Kronecker-Matrix Multiplication (GeMKM):

$Z = \alpha ~ \left( F^1 \otimes F^2 \otimes \dots F^N \right) \times X + \beta Y$

Both functions support dimension numpy dimension broadcasting semantics.

* **Parameters**
    * `x` is an np array
    * `fs` is a list of np arrays
    * `alpha` and `beta` are constants
    * `y` is an np array

* **Returns**
    Returns output as an np array.

### Example

```
import pyfastkron.fastkronnumpy as fk

x = np.ones((10, 8**5), dtype=np.float32)
fs = [np.ones((8,8), dtype=np.float32) for i in range(5)]
y = fk.gekmm(x, fs)
```

## FastKronTorch

Import modules as:

```
import numpy as np
import pyfastkron.fastkronnumpy
```

Functions:

```
def gekmm(fs : List[np.ndarray], x : np.ndarray,
          alpha : float = 1.0, beta : float = 0.0,
          y : Optional[np.ndarray] = None) -> np.ndarray

def gemkm(x : np.ndarray, fs : List[np.ndarray],
          alpha : float = 1.0, beta : float = 0.0,
          y : Optional[np.ndarray] = None)
```
Perform Generalized Kronecker Matrix-Matrix Multiplication (GeKMM):

$Z = \alpha ~ X \times \left( F^1 \otimes F^2 \otimes \dots F^N \right) + \beta Y$

or Generalized Matrix Kronecker-Matrix Multiplication (GeMKM):

$Z = \alpha ~ \left( F^1 \otimes F^2 \otimes \dots F^N \right) \times X + \beta Y$

Both functions support dimension numpy dimension broadcasting semantics (see below for more information).

Both functions are implemented as torch.autograd.Function and hence support gradient calculation.

* **Parameters**
    * `x` is an np array
    * `fs` is a list of np arrays
    * `alpha` and `beta` are constants
    * `y` is an np array

* **Returns**
    Returns output as an np array.

### Example

```
import pyfastkron.fastkrontorch as fk

x = torch.ones((10, 8**5), dtype=np.float32)
fs = [torch.ones((8,8), dtype=np.float32) for i in range(5)]
y = fk.gekmm(x, fs)
```

## GeMKM Broadcast Semantics
TODO Relook at these

The behavior depends on the dimensionality of the tensors as follows:

* If $X$ and all $F$s are 1-dimensional, the dot product (scalar) is returned.
* If $X$ and atleast one $F$ is 2-dimensional, the matrix kronecker-matrix product is returned.
* If $X$ is 1-dimensional and atleast one $F$ is 2-dimensional, then a 1 is prepended to dimensions of $X$, a 1 is appended to dimensions of $F$, and after MKM the added dimensions are removed.
* If $X$ is 2-dimensional and all $F$s are 1-dimensional, the matrix kronecker-vector product is returned.
* If $X$ and atleast one $F$ are at least 1-dimensional and at least $X$ or one of $F$ is N-dimensional (where N > 2), then a batched MKM is returned. If $X$ is 1-dimensional, a 1 is prepended to its dimension for the purpose of the batched MKM and removed after. If any of $F$s argument is 1-dimensional, a 1 is appended to its dimension for the purpose of the batched MKM and removed after. The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be broadcastable). However, all $F$s must have same batch dimensions.

## GeKMM Broadcast Semantics

The behavior depends on the dimensionality of the tensors as follows:

* If all $F$s and $X$ are 1-dimensional, the dot product (scalar) is returned.
* If atleast one $F$ and $X$ is 2-dimensional, the kronecker-matrix matrix product is returned.
* If atleast one $F$ is 2-dimensional and $X$ is 1-dimensional, then a 1 is preppended to dimensions of $F$, a 1 is appended to dimensions of $X$, and after KMM the added dimensions are removed.
* If all $F$s are 1-dimensional and $X$ is 2-dimensional, then 1 is preppended to dimensions of $F$ and the kronecker-matrix matrix product is returned and added dimensions are removed.
* If atleast one $F$ and $X$ are at least 1-dimensional or one of $F$ is N-dimensional (where N > 2), then a batched KMM is returned. If $X$ is 1-dimensional, a 1 is appended to its dimension for the purpose of the batched KMM and removed after. If any of $F$s argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the batched KMM and removed after. The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be broadcastable). However, all $F$s must have same batch dimensions.