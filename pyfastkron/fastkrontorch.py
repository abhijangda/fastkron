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
    self.pyfastkron = None
    backend = 0
    if torch.cuda.is_available():
      backend = FastKron.Backend.CUDA
      self.pyfastkron = FastKron.init(backend)
      print(torch.cuda.current_stream().cuda_stream)
      FastKron.initCUDA(self.pyfastkron, [torch.cuda.current_stream().cuda_stream], 1, 1, 1, 1)
  
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
    if stream is not None:
      assert stream.device == x.device

    if y is not None:
      assert x.shape[0] == y.shape[0]
      assert y.shape[1] == product(self.qs(fs))
      assert x.dtype    == y.dtype
      assert x.device   == y.device

  def gekmmSizes(self, x, fs):
    self._check(x, fs, None, None)
    return FastKron.gekmmSizes(self.pyfastkron, x.shape[0], len(fs), self.ps(fs), self.qs(fs))

  def gekmm(self, x, fs, y, alpha, beta, z, temp, 
            trX = False, trF = False, stream = None):
    if stream is None:
      stream = torch.cuda.current_stream()

    self._check(x, fs, y, stream)

    fn = None
    if x.dtype == torch.float:
      fn = FastKron.sgekmm
    elif x.dtype == torch.int:
      fn = FastKron.igekmm
    elif x.dtype == torch.double:
      fn = FastKron.dgekmm

    backend = None
    if x.is_cuda:
      backend = FastKron.Backend.CUDA
  
    fn(self.pyfastkron, backend, x.shape[0], len(fs), self.ps(fs), self.qs(fs), 
       x.data_ptr(), FastKron.Op.N if not trX else FastKron.Op.T,
       self.fptrs(fs), FastKron.Op.N if not trF else FastKron.Op.T,
       y.data_ptr(),
       alpha, beta, 0 if z is None else z.data_ptr(), 
       temp.data_ptr(), 0)