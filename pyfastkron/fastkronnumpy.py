from .fastkronbase import FastKronBase
from . import FastKron

try:
  import numpy as np
except:
  pass

class FastKronNumpy(FastKronBase):
  def __init__(self):
    super().__init__(True, False)

  def tensor_data_ptr(self, tensor):
    return tensor.ctypes.data

  def supportedDevice(self, x):
    return True

  def supportedTypes(self, x, fs):
    return x.dtype in [np.float32, np.double]

  def trLastTwoDims(self, x, dim1, dim2):
    return x.transpose(list(range(len(x.shape) - 2)) + [dim1, dim2,])

  def gekmm(self, x, fs, y, alpha, beta, z, temp1, temp2,
            trX = False, trF = False):

    fn = None
    if x.dtype == np.float32:
      fn = FastKron.sgekmm
    elif x.dtype == np.double:
      fn = FastKron.dgekmm

    self.xgekmm(fn, self.backend("cpu"), x, fs, y, alpha, beta, z, temp1, temp2, trX, trF)

__fastkronnumpy = FastKronNumpy()

def gekmm(x, fs, alpha=1.0, beta=0.0, y=None, trX = False, trF = False):
  '''
  Perform Generalized Kronecker-Matrix Multiplication:
  
  $Z = \alpha ~ X \times \left( F^1 \otimes F^2 \otimes \dots F^N \right) + \beta Y$

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

  x,fs = __fastkronnumpy.reshapeInput(x, fs, trX, trF)

  if not __fastkronnumpy.isSupported(x, fs):
    z = __fastkronnumpy.shuffleGeKMM(np, x, fs, alpha, beta, y, trX, trF)
  else:
    rs, ts = __fastkronnumpy.gekmmSizes(x, fs, trX=trX, trF=trF)
    temp1 = np.ndarray(ts, dtype=x.dtype)
    if not trX:
      z = np.ndarray((x.shape[0], rs//x.shape[0]), dtype=x.dtype)
    else:
      z = np.ndarray((x.shape[1], rs//x.shape[1]), dtype=x.dtype)
    __fastkronnumpy.gekmm(x, fs, z, alpha, beta, y, temp1, None, trX, trF)

  if len(orig_xshape) != 2:
    z = z.reshape(list(orig_xshape[:-1]) + [z.shape[-1]])

  return z
