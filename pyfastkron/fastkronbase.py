from functools import reduce
from . import FastKron

def product(values):
  return reduce((lambda a, b: a * b), values)

class FastKronBase:
  def hasBackend(backends, enumBackend) :
    return (backends & int(enumBackend)) == int(enumBackend)

  def __init__(self, x86, cuda):
    self.backends = FastKron.backends()
    self.handle = FastKron.init()

    self.x86 = x86 and FastKronBase.hasBackend(self.backends, FastKron.Backend.X86)
    if self.x86:
      FastKron.initX86(self.handle)

    self.cuda = cuda and FastKronBase.hasBackend(self.backends, FastKron.Backend.CUDA)

  def __del__(self):
    FastKron.destroy(self.handle)
    self.handle = self.backends = self.x86 = self.cuda = None

  def hasCUDA(self):
    return self.cuda
  
  def hasX86(self):
    return self.x86

  def tensor_data_ptr(self, tensor):
    raise NotImplementedError()

  def ps(self, fsshape):
    return [f[0] for f in fsshape]
  
  def qs(self, fsshape):
    return [f[1] for f in fsshape]
  
  def fptrs(self, fs):
    return [self.tensor_data_ptr(f) for f in fs]

  def matrixShape(self, m, tr):
    if tr == False:
      return (m.shape[0], m.shape[1])
    else:
      return (m.shape[1], m.shape[0])

  def checkShapeAndTypes(self, x, fs, y, trX, trF):
    # Only operate on 2-dims matrices
    assert len(x.shape) == 2
    assert len(fs[0].shape) == 2

    xshape = self.matrixShape(x, trX)
    fsshape = [self.matrixShape(f, trF) for f in fs]

    assert xshape[1] == product(self.ps(fsshape))
    
    assert x.dtype    == fs[0].dtype
    assert len(set([f.dtype for f in fs])) == 1

    if y is not None:
      yshape = self.matrixShape(y, False)
      assert xshape[0] == yshape[0]
      assert yshape[1] == product(self.qs(fsshape))
      assert x.dtype   == y.dtype
  
  def backend(self, device_type):
    if device_type == "cpu":
      return FastKron.Backend.X86
    if device_type == "cuda":
      return FastKron.Backend.CUDA

  def gekmmSizes(self, x, fs, trX = False, trF = False):
    self.checkShapeAndTypes(x, fs, None, trX, trF)

    xshape = self.matrixShape(x, trX)
    fsshape = [self.matrixShape(f, trF) for f in fs]

    return FastKron.gekmmSizes(self.handle, xshape[0], len(fs), self.ps(fsshape), self.qs(fsshape))

  def xgekmm(self, fngekmm, backend, x, fs, y, alpha, beta, z, temp1, temp2, trX = False, trF = False):
    # Are pointers valid?
    assert temp1 is not None
    assert y is not None

    if z is not None and self.tensor_data_ptr(z) == self.tensor_data_ptr(y):
      assert temp2 is not None

    self.checkShapeAndTypes(x, fs, y, trX, trF)

    xshape = self.matrixShape(x, trX)
    fsshape = [self.matrixShape(f, trF) for f in fs]

    fngekmm(self.handle, backend, xshape[0], len(fs), self.ps(fsshape), self.qs(fsshape),
            self.tensor_data_ptr(x), FastKron.Op.N if not trX else FastKron.Op.T,
            self.fptrs(fs), FastKron.Op.N if not trF else FastKron.Op.T,
            self.tensor_data_ptr(y),
            alpha, beta, 0 if z is None else self.tensor_data_ptr(z), 
            self.tensor_data_ptr(temp1), 0 if temp2 is None else self.tensor_data_ptr(temp2))