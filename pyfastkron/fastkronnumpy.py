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

  def check(self, x, fs, y):
    self.checkShapeAndTypes(x, fs, y)

  def gekmm(self, x, fs, y, alpha, beta, z, temp, 
            trX = False, trF = False):

    self.check(x, fs, y)

    fn = None
    if x.dtype == np.float32:
      fn = FastKron.sgekmm
    elif x.dtype == np.int32:
      fn = FastKron.igekmm
    elif x.dtype == np.double:
      fn = FastKron.dgekmm

    self.xgekmm(fn, self.backend("cpu"), x, fs, y, alpha, beta, z, temp, trX, trF)