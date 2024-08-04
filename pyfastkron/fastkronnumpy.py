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

  def gekmm(self, x, fs, y, alpha, beta, z, temp1, temp2,
            trX = False, trF = False):

    fn = None
    if x.dtype == np.float32:
      fn = FastKron.sgekmm
    elif x.dtype == np.int32:
      fn = FastKron.igekmm
    elif x.dtype == np.double:
      fn = FastKron.dgekmm

    self.xgekmm(fn, self.backend("cpu"), x, fs, y, alpha, beta, z, temp1, temp2, trX, trF)

__fastkronnumpy = FastKronNumpy()

def gekmm(x, fs, alpha=1.0, beta=0.0, z=None, trX = False, trF = False):
  rs, ts = __fastkronnumpy.gekmmSizes(x, fs, trX=trX, trF=trF)
  temp1 = np.zeros(ts, dtype=x.dtype)
  if not trX:
    y = np.zeros((x.shape[0], rs//x.shape[0]), dtype=x.dtype)
  else:
    y = np.zeros((x.shape[1], rs//x.shape[1]), dtype=x.dtype)
  __fastkronnumpy.gekmm(x, fs, y, alpha, beta, z, temp1, None, trX, trF)
  return y
