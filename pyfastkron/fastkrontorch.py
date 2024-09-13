from .fastkronbase import FastKronBase, product
import platform

if platform.system() == "Linux" and platform.processor() == "x86_64":
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

  def supportedDevice(self, x):
    return (x.device.type == 'cuda' and torch.version.hip == None) or (x.device.type == 'cpu')

  def check(self, x, fs, y, z, stream):
    devices = [x.device] + [f.device for f in fs]
    if y is not None:
      devices += [y.device]
    if z is not None:
      devices += [z.device]

    devices = set(devices)

    if len(devices) != 1:
      raise RuntimeError(f'Expected all tensors to be on the same device, but found {len(devices)} devices: {[str(d) for d in devices]}')

    if x.device.type == "cuda" and stream is not None:
      if stream.device != x.device:
        raise RuntimeError(f"Expected stream to be on same device as tensors, but found {stream.device} and {x.device} are different")

  def supportedTypes(self, x, fs):
    return x.dtype in [torch.float32, torch.float64]

  def trLastTwoDims(self, x, dim1, dim2):
    return x.transpose(dim1, dim2)

  def gekmm(self, x, fs, y, alpha, beta, z, temp1, temp2, 
            trX = False, trF = False, stream = None):

    if x.device.type == "cuda" and stream is None:
      stream = torch.cuda.current_stream()

    self.check(x, fs, y, z, stream)

    fn = None
    if x.dtype == torch.float:
      fn = FastKron.sgekmm
    elif x.dtype == torch.double:
      fn = FastKron.dgekmm

    self.xgekmm(fn, self.backend(x.device.type), x, fs, y, alpha, beta, z, temp1, temp2, trX, trF)
  
__fastkrontorch = FastKronTorch()

def gekmm(x, fs, alpha=1.0, beta=0.0, y=None, trX = False, trF = False):
  '''
  Perform Generalized Kronecker-Matrix Multiplication:
  
  $Z = \alpha ~ X \times \left( F^1 \otimes F^2 \otimes \dots F^N \right) + \beta Y$

  Parameters
  ----------
  x  : 2D torch tensor 
  fs : A list of 2D torch tensor
  alpha and beta: constants
  y  : 2D torch tensor
  trX: Transpose x before computing GeKMM
  trF: Transpose each element of fs before computing GeKMM

  Returns
  -------
  z : 2D torch tensor
  '''

  if type(x) is not torch.Tensor:
    raise ValueError("Input 'x' should be a Tensor")
  if type(fs) is not list:
    raise ValueError("Input 'fs' should be a list of Tensor")
  for i,f in enumerate(fs):
    if type(f) is not torch.Tensor:
      raise ValueError(f"Input fs[{i}] should be a Tensor")
  if y != None and (type(y) is not torch.Tensor or y.ndim != 2):
    raise ValueError(f"Input 'y' should be a 2D Tensor")

  orig_xshape = x.shape

  x,fs = __fastkrontorch.reshapeInput(x, fs, trX, trF)

  if not __fastkrontorch.isSupported(x, fs):
    z = __fastkrontorch.shuffleGeKMM(torch, x, fs, alpha, beta, y, trX, trF)
  else:
    rs, ts = __fastkrontorch.gekmmSizes(x, fs, trX=trX, trF=trF)
    temp1 = x.new_empty(ts)
    temp2 = x.new_empty(ts) if rs != ts else None

    if not trX:
      z = x.new_empty((x.shape[0], rs//x.shape[0]))
    else:
      z = x.new_empty((x.shape[1], rs//x.shape[1]))

    __fastkrontorch.gekmm(x, fs, z, alpha, beta, y, temp1, temp2, trX, trF)
  
  if len(orig_xshape) != 2:
    z = z.reshape((list(orig_xshape[:-1]) + [z.shape[-1]]))

  return z
