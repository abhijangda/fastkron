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

  def checkShapeAndTypes(self, x, fs, y, z, trX, trF):
    # Only operate on 2-dims matrices
    xshape = self.matrixShape(x, trX)
    fsshape = [self.matrixShape(f, trF) for f in fs]

    if xshape[1] != product(self.ps(fsshape)):
      raise ValueError(f"Input operand x has a mismatch with its dimension 1 ('{xshape[1]}') with dimension 0 of kronecker product of fs ('{product(self.ps(fsshape))}')")
    
    if x.dtype != fs[0].dtype:
      raise ValueError(f"Operand types mismatches {x.dtype} != {fs[0].dtype}")
    
    if len(set([f.dtype for f in fs])) != 1:
      raise ValueError(f"Type of Kronecker factors do not match. Found {len(set([f.dtype for f in fs]))} different types")

    if z is not None:
      zshape = self.matrixShape(z, False)
      assert xshape[0] == zshape[0]
      assert zshape[1] == product(self.qs(fsshape))
      assert x.dtype   == z.dtype
    
    if y is not None:
      yshape = self.matrixShape(y, False)
      if yshape[0] == xshape[0] or yshape[1] == product(self.qs(fsshape)):
        raise ValueError(f"Input operand 'y' shape ('{yshape}') mismatch with '({xshape[0], product(self.qs(fsshape))})'")
      assert x.dtype   == y.dtype

  def backend(self, device_type):
    if device_type == "cpu":
      return FastKron.Backend.X86
    if device_type == "cuda":
      return FastKron.Backend.CUDA
    raise RuntimeError(f"Invalid device {device_type}")

  def gekmmSizes(self, x, fs, trX = False, trF = False):
    self.checkShapeAndTypes(x, fs, None, None, trX, trF)

    xshape = self.matrixShape(x, trX)
    fsshape = [self.matrixShape(f, trF) for f in fs]

    return FastKron.gekmmSizes(self.handle, xshape[0], len(fs), self.ps(fsshape), self.qs(fsshape))

  def xgekmm(self, fngekmm, backend, x, fs, z, alpha, beta, y, temp1, temp2, trX = False, trF = False):
    if temp1 is None:
      raise ValueError("Operand temp1 must be valid 2D Tensor")

    if z is None:
      raise ValueError("Operand z must be valid 2D Tensor")

    if y is not None and self.tensor_data_ptr(z) == self.tensor_data_ptr(y):
      if temp2 is None:
        raise ValueError("Operand temp2 must be a valid Tensor when z == y")

    self.checkShapeAndTypes(x, fs, z, y, trX, trF)

    xshape = self.matrixShape(x, trX)
    fsshape = [self.matrixShape(f, trF) for f in fs]

    fngekmm(self.handle, backend, xshape[0], len(fs), self.ps(fsshape), self.qs(fsshape),
            self.tensor_data_ptr(x), FastKron.Op.N if not trX else FastKron.Op.T,
            self.fptrs(fs), FastKron.Op.N if not trF else FastKron.Op.T,
            self.tensor_data_ptr(z),
            alpha, beta, 0 if y is None else self.tensor_data_ptr(y), 
            self.tensor_data_ptr(temp1), 0 if temp2 is None else self.tensor_data_ptr(temp2))