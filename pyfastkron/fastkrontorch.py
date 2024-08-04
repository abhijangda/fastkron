from .fastkronbase import FastKronBase
from . import FastKron

try:
  import torch
except:
  pass

class FastKronTorch(FastKronBase):
  def __init__(self):
    super().__init__(True, torch.cuda.is_available())
    if self.cuda:
      FastKron.initCUDA(self.handle, [torch.cuda.current_stream().cuda_stream])

  def tensor_data_ptr(self, tensor):
    return tensor.data_ptr()

  def check(self, x, fs, y, stream):
    assert x.device  == fs[0].device

    if x.device.type == "cuda" and stream is not None:
      assert stream.device == x.device

    if y is not None:
      assert x.device == y.device

  def gekmm(self, x, fs, y, alpha, beta, z, temp1, temp2, 
            trX = False, trF = False, stream = None):
    if x.device.type == "cuda" and stream is None:
      stream = torch.cuda.current_stream()

    self.check(x, fs, y, stream)

    fn = None
    if x.dtype == torch.float:
      fn = FastKron.sgekmm
    elif x.dtype == torch.int:
      fn = FastKron.igekmm
    elif x.dtype == torch.double:
      fn = FastKron.dgekmm

    self.xgekmm(fn, self.backend(x.device.type), x, fs, y, alpha, beta, z, temp1, temp2, trX, trF)
  
__fastkrontorch = FastKronTorch()

def gekmm(x, fs, alpha=1.0, beta=0.0, z=None, trX = False, trF = False):
  rs, ts = __fastkrontorch.gekmmSizes(x, fs, trX=trX, trF=trF)
  temp1 = torch.zeros(ts, dtype=x.dtype, device=x.device)
  if not trX:
    y = x.new_empty((x.shape[0], rs//x.shape[0]))
  else:
    y = x.new_empty((x.shape[1], rs//x.shape[1]))
  __fastkrontorch.gekmm(x, fs, y, alpha, beta, z, temp1, None, trX, trF)
  return y
