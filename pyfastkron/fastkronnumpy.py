from .fastkronbase import fastkronX86, FastKronBase

import platform

try:
  import numpy as np
except:
  pass

class FastKronNumpy(FastKronBase):
  def __init__(self):
    super().__init__(True, False)

  def tensor_data_ptr(self, tensor):
    if tensor is None: return 0
    return tensor.ctypes.data

  def supportedDevice(self, x):
    return True

  def supportedTypes(self, x, fs):
    return x.dtype in [np.float32, np.double]

  def trLastTwoDims(self, x):
    axes = list(range(len(x.shape) - 2)) + [len(x.shape) - 1, len(x.shape) - 2]
    return x.transpose(axes)

  def device_type(self, x):
    return "cpu"

  def gemkm(self, x, fs, z, alpha, beta, y, temp1, temp2,
            trX = False, trF = False):

    fn = None
    stridedBatchedFn = None

    if x.dtype == np.float32:
      fn = fastkronX86.libFastKron.sgemkm
      stridedBatchedFn = fastkronX86.libFastKron.sgemkmStridedBatched
    elif x.dtype == np.double:
      fn = fastkronX86.libFastKron.dgemkm
      stridedBatchedFn = fastkronX86.libFastKron.dgemkmStridedBatched

    if temp1 is None:
      raise ValueError("Operand temp1 must be valid 2D Tensor")

    if z is None:
      raise ValueError("Operand z must be valid 2D Tensor")

    if y is not None and self.tensor_data_ptr(z) == self.tensor_data_ptr(y):
      if temp2 is None:
        raise ValueError("Operand temp2 must be a valid Tensor when z == y")

    super().xgemm(fastkronX86, FastKronBase.MMTypeMKM, fn, stridedBatchedFn,
                   x, fs, z, alpha, beta, y, temp1, temp2, trX, trF)

  def gekmm(self, fs, x, z, alpha, beta, y, temp1, temp2,
            trX = False, trF = False):

    fn = None
    stridedBatchedFn = None

    if x.dtype == np.float32:
      fn = fastkronX86.libFastKron.sgekmm
      stridedBatchedFn = fastkronX86.libFastKron.sgekmmStridedBatched
    elif x.dtype == np.double:
      fn = fastkronX86.libFastKron.dgekmm
      stridedBatchedFn = fastkronX86.libFastKron.dgekmmStridedBatched

    if temp1 is None:
      raise ValueError("Operand temp1 must be valid 2D Tensor")

    if z is None:
      raise ValueError("Operand z must be valid 2D Tensor")

    if y is not None and self.tensor_data_ptr(z) == self.tensor_data_ptr(y):
      if temp2 is None:
        raise ValueError("Operand temp2 must be a valid Tensor when z == y")

    super().xgemm(fastkronX86, FastKronBase.MMTypeKMM, fn, stridedBatchedFn,
                   x, fs, z, alpha, beta, y, temp1, temp2, trX, trF)

__fastkronnumpy = FastKronNumpy()

def gemkm(x, fs, alpha=1.0, beta=0.0, y=None, trX = False, trF = False):
  '''
  Perform Generalized Matrix Kronecker-Matrix Multiplication :
  
  $Z = a X * (F^1 (*) F^2 (*) ... F^N) + b Y$

  Parameters
  ----------
  x  : 2D numpy array
  fs : A list of 2D numpy array
  alpha and beta: constants
  y  : 2D numpy array 
  trX: Transpose x before computing GeKMM
  trF: Transpose each element of fs before computing GeKMM

  Returns
  -------
  z : 2D numpy array
  '''

  if type(x) is not np.ndarray:
    raise ValueError("Input 'x' should be a ndarray")
  if type(fs) is not list:
    raise ValueError("Input 'fs' should be a list of np.ndarray")
  for i,f in enumerate(fs):
    if type(f) is not np.ndarray:
      raise ValueError(f"Input fs[{i}] should be a ndarray")

  orig_xshape = x.shape

  x, fs = __fastkronnumpy.reshapeInput(x, fs, trX, trF)

  if not __fastkronnumpy.isSupported(x, fs):
    z = __fastkronnumpy.shuffleGeKMM(np, x, fs, alpha, beta, y, trX, trF)
  else:
    rs, ts = __fastkronnumpy.gekmmSizes(FastKronBase.MMTypeMKM, x, fs, trX=trX, trF=trF)
    temp1 = np.ndarray(ts, dtype=x.dtype)
    temp2 = np.ndarray(ts, dtype=x.dtype) if rs != ts else None
    z = np.ndarray(shape=rs, dtype=x.dtype)
    __fastkronnumpy.gemkm(x, fs, z, alpha, beta, y, temp1, temp2, trX, trF)
    z = z.reshape(rs)

  return z

def gekmm(fs, x, alpha=1.0, beta=0.0, y=None, trX = False, trF = False):
  '''
  Perform Generalized Kronecker-Matrix Matrix Multiplication :
  
  $Z = a (F^1 (*) F^2 (*) ... F^N) * X + b Y$

  Parameters
  ----------
  x  : 2D numpy array
  fs : A list of 2D numpy array
  alpha and beta: constants
  y  : 2D numpy array 
  trX: Transpose x before computing GeKMM
  trF: Transpose each element of fs before computing GeKMM

  Returns
  -------
  z : 2D numpy array
  '''

  if type(x) is not np.ndarray:
    raise ValueError("Input 'x' should be a ndarray")
  if type(fs) is not list:
    raise ValueError("Input 'fs' should be a list of np.ndarray")
  for i,f in enumerate(fs):
    if type(f) is not np.ndarray:
      raise ValueError(f"Input fs[{i}] should be a ndarray")

  orig_xshape = x.shape

  x, fs = __fastkronnumpy.reshapeInput(x, fs, trX, trF)

  if not __fastkronnumpy.isSupported(x, fs):
    z = __fastkronnumpy.shuffleGeKMM(np, x, fs, alpha, beta, y, trX, trF)
  else:
    rs, ts = __fastkronnumpy.gekmmSizes(FastKronBase.MMTypeKMM, x, fs, trX=trX, trF=trF)
    temp1 = np.ndarray(ts, dtype=x.dtype)
    temp2 = np.ndarray(ts, dtype=x.dtype) if rs != ts else None
    z = np.ndarray(shape=rs, dtype=x.dtype)
    __fastkronnumpy.gekmm(fs, x, z, alpha, beta, y, temp1, temp2, trX, trF)
    z = z.reshape(rs)

  return z