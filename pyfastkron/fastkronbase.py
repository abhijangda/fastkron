from functools import reduce
from . import FastKron

def product(values):
  return reduce((lambda a, b: a * b), values)

class FastKronBase:
  def hasBackend(backends, enumBackend) :
    return (backends & int(enumBackend)) == int(enumBackend)

  def __init__(self, x86, cuda):
    self.handle = None
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

  def ps(self, fs):
    return [f.shape[0] for f in fs]
  
  def qs(self, fs):
    return [f.shape[1] for f in fs]
  
  def fptrs(self, fs):
    return [self.tensor_data_ptr(f) for f in fs]

  def checkShapeAndTypes(self, x, fs, y):
    if x.shape[1] != product(self.ps(fs)):
      assert False

    assert x.shape[1] == product(self.ps(fs))
    
    assert x.dtype    == fs[0].dtype
    assert len(set([f.dtype for f in fs])) == 1

    if y is not None:
      assert x.shape[0] == y.shape[0]
      assert y.shape[1] == product(self.qs(fs))
      assert x.dtype    == y.dtype
  
  def backend(self, device_type):
    if device_type == "cpu":
      return FastKron.Backend.X86
    if device_type == "cuda":
      return FastKron.Backend.CUDA

  def gekmmSizes(self, x, fs):
    self.checkShapeAndTypes(x, fs, None)
    return FastKron.gekmmSizes(self.handle, x.shape[0], len(fs), self.ps(fs), self.qs(fs))

  def xgekmm(self, fngekmm, backend, x, fs, y, alpha, beta, z, temp1, temp2, trX = False, trF = False):
    #Are pointers valid?
    assert temp1 is not None
    assert y is not None

    if z is not None and self.tensor_data_ptr(z) == self.tensor_data_ptr(y):
      assert temp2 is not None

    fngekmm(self.handle, backend, x.shape[0], len(fs), self.ps(fs), self.qs(fs),
            self.tensor_data_ptr(x), FastKron.Op.N if not trX else FastKron.Op.T,
            self.fptrs(fs), FastKron.Op.N if not trF else FastKron.Op.T,
            self.tensor_data_ptr(y),
            alpha, beta, 0 if z is None else self.tensor_data_ptr(z), 
            self.tensor_data_ptr(temp1), 0 if temp2 is None else self.tensor_data_ptr(temp2))