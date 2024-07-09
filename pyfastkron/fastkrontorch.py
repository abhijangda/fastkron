from functools import reduce
from . import FastKron

def product(values):
  return reduce((lambda a, b: a * b), values)

try:
  import torch
except:
  pass

class FastKronTorch:
  def __init__(self):
    self.handle = None
    self.backends = FastKron.backends()
    self.handle = FastKron.init()

    if torch.cuda.is_available() and FastKronTorch.hasBackend(self.backends, FastKron.Backend.CUDA):
      FastKron.initCUDA(self.handle, [torch.cuda.current_stream().cuda_stream], 1, 1, 1, 1)
    if FastKronTorch.hasBackend(self.backends, FastKron.Backend.X86):
      FastKron.initX86(self.handle)

  def hasBackend(backends, enumBackend) :
    return (backends & int(enumBackend)) == int(enumBackend)

  def ps(self, fs):
    return [f.shape[0] for f in fs]
  
  def qs(self, fs):
    return [f.shape[1] for f in fs]

  def fptrs(self, fs):
    return [f.data_ptr() for f in fs]

  def _check(self, x, fs, y, stream):
    assert x.shape[1] == product(self.ps(fs))
    
    assert x.dtype    == fs[0].dtype
    assert len(set([f.dtype for f in fs])) == 1
    
    assert x.device   == fs[0].device
    assert len(set([f.device for f in fs])) == 1
    if x.device.type == "cuda" and stream is not None:
      assert stream.device == x.device

    if y is not None:
      assert x.shape[0] == y.shape[0]
      assert y.shape[1] == product(self.qs(fs))
      assert x.dtype    == y.dtype
      assert x.device   == y.device

    if x.device.type == 'cpu':
      assert FastKronTorch.hasBackend(self.backends, FastKron.Backend.X86)
    
    if x.device.type == 'cuda':
      assert FastKronTorch.hasBackend(self.backends, FastKron.Backend.CUDA)

  def _backendForDevice(self, device):
    if device.type == "cpu":
      return FastKron.Backend.X86
    if device.type == "cuda":
      return FastKron.Backend.CUDA

  def gekmmSizes(self, x, fs):
    self._check(x, fs, None, None)
    return FastKron.gekmmSizes(self.handle, x.shape[0], len(fs), self.ps(fs), self.qs(fs))

  def gekmm(self, x, fs, y, alpha, beta, z, temp, 
            trX = False, trF = False, stream = None):
    if x.device.type == "cuda" and stream is None:
      stream = torch.cuda.current_stream()

    self._check(x, fs, y, stream)

    fn = None
    if x.dtype == torch.float:
      fn = FastKron.sgekmm
    elif x.dtype == torch.int:
      fn = FastKron.igekmm
    elif x.dtype == torch.double:
      fn = FastKron.dgekmm

    fn(self.handle, self._backendForDevice(x.device), x.shape[0], len(fs), self.ps(fs), self.qs(fs), 
       x.data_ptr(), FastKron.Op.N if not trX else FastKron.Op.T,
       self.fptrs(fs), FastKron.Op.N if not trF else FastKron.Op.T,
       y.data_ptr(),
       alpha, beta, 0 if z is None else z.data_ptr(), 
       temp.data_ptr(), 0)