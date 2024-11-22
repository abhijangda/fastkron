from .fastkronbase import fastkronX86, fastkronCUDA, FastKronBase, product

import platform
from typing import List

try:
  import torch
except:
  pass

class FastKronTorch(FastKronBase):
  def __init__(self):
    cuda = False
    try:
      cuda = torch.cuda.is_available()
    except:
      pass

    super().__init__(True, cuda)

  def tensor_data_ptr(self, tensor):
    if tensor is None: return 0
    return tensor.data_ptr()

  def supportedDevice(self, x):
    return (x.device.type == 'cuda' and torch.version.hip == None) or \
           (x.device.type == 'cpu')

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

  def trLastTwoDims(self, mmtype, x):
    if mmtype == FastKronBase.MMTypeMKM:
      return x.transpose(-1, -2)
    elif mmtype == FastKronBase.MMTypeKMM:
      return x.transpose(-2, -3)

  def device_type(self, x):
    return x.device.type 

  def handle(self, x):
    if self.device_type(x) == "cpu":
      return fastkronX86
    elif self.device_type(x) == "cuda":
      return fastkronCUDA

  def asContiguousTensor(self, x, forceContiguous=False):
    if forceContiguous: return False, x.contiguous()
    if x.is_contiguous(): return False, x
    if x.ndim > 1 and x.stride()[-2] == 1 and x.stride()[-1] == x.shape[-2]: return True, x
    return False, x.contiguous()

  def stride(self, x):
    return x.stride()

  def gemkm(self, requires_grad, x, fs, 
            y = None, alpha = 1, beta = 0, stream = None):
    if type(x) is not torch.Tensor:
      raise ValueError("Input 'x' should be a Tensor")
    if type(fs) is not list and type(fs) is not tuple:
      raise ValueError("Input 'fs' should be a list of Tensor")
    for i,f in enumerate(fs):
      if type(f) is not torch.Tensor:
        raise ValueError(f"Input fs[{i}] should be a Tensor")
    if y != None and type(y) is not torch.Tensor:
      raise ValueError(f"Input 'y' should be a 2D Tensor")

    trX,x, trF,fs = fastkrontorch.reshapeInput(x, fs)

    if x.device.type == "cuda" and stream is None:
      stream = torch.cuda.current_stream()

    self.check(x, fs, y, None, stream)

    fn = None
    stridedBatchedFn = None
    
    if x.dtype == torch.float:
      fn = self.handle(x).libFastKron.sgemkm
      stridedBatchedFn = self.handle(x).libFastKron.sgemkmStridedBatched
    elif x.dtype == torch.double:
      fn = self.handle(x).libFastKron.dgemkm
      stridedBatchedFn = self.handle(x).libFastKron.dgemkmStridedBatched

    zs = []
    print(102, x.shape)
    if requires_grad or not self.isSupported(x, fs):
      z, zs = self.shuffleGeMM(requires_grad, torch, FastKronBase.MMTypeMKM, x, fs, alpha, beta, y, trX, trF)
    else:
      rs, ts = self.gekmmSizes(FastKronBase.MMTypeMKM, x, fs, trX=trX, trF=trF)
      z = x.new_empty(rs)
      temp1 = x.new_empty(ts)
      temp2 = x.new_empty(ts) if rs != ts else None
      super().xgemm(self.handle(x), FastKronBase.MMTypeMKM, fn, stridedBatchedFn, x, fs, y, alpha, beta, z, temp1, temp2, trX, trF)
      z = z.reshape(rs)
    print(112, z.shape)
    return z, zs
  
  def mkmBackward(self, grad_z, x, fs, zs):
    trX,x, trF, fs = fastkrontorch.reshapeInput(x, fs)
    grad_fs = []
    grad_zs = [grad_z]
    zbatchShape = zs[-1].shape[:-2]
    zs = tuple(reversed(zs[:-1])) + (x,)

    for z,f in zip(zs, fs):
      prev_grad_z = grad_zs[-1]
      orig_fshape = f.shape
      fp = self.p(FastKronBase.MMTypeMKM, f, trF)
      fq = self.q(FastKronBase.MMTypeMKM, f, trF)

      prev_grad_z = prev_grad_z.reshape((prev_grad_z.shape[:-2] + (prev_grad_z.shape[-2],)) + (fq, prev_grad_z.shape[-1]//fq))
      prev_grad_z = prev_grad_z.transpose(-1,-2)
      prev_grad_z = prev_grad_z.reshape(prev_grad_z.shape[:-3] + (prev_grad_z.shape[-3] * prev_grad_z.shape[-2], prev_grad_z.shape[-1]))

      #Backward pass for z
      if x.data_ptr() != z.data_ptr() or x.requires_grad:
        grad_z = torch.matmul(prev_grad_z, (f if trF else f.mT))
        grad_z = grad_z.reshape(zbatchShape + z.shape[-2:])
        grad_zs += [grad_z]

      #Backward pass for f
      orig_zshape = z.shape
      z = z.reshape(z.shape[:-2] + ((z.shape[-2] * z.shape[-1])//fp, fp))
      trZ = z.mT
      grad_f = trZ @ prev_grad_z
      grad_fs += [grad_f]
      z = z.reshape(orig_zshape)

    return (grad_zs[-1] if x.requires_grad else None,) + tuple(grad_fs)

  def gekmm(self, fs, x, y, alpha, beta, z, temp1, temp2, 
            trX = False, trF = False, stream = None):

    if x.device.type == "cuda" and stream is None:
      stream = torch.cuda.current_stream()

    self.check(x, fs, y, z, stream)

    fn = None
    stridedBatchedFn = None
    
    if x.dtype == torch.float:
      fn = self.handle(x).libFastKron.sgekmm
      stridedBatchedFn = self.handle(x).libFastKron.sgekmmStridedBatched
    elif x.dtype == torch.double:
      fn = self.handle(x).libFastKron.dgekmm
      stridedBatchedFn = self.handle(x).libFastKron.dgekmmStridedBatched

    super().xgemm(self.handle(x), FastKronBase.MMTypeKMM, fn, stridedBatchedFn, x, fs, y, alpha, beta, z, temp1, temp2, trX, trF)

fastkrontorch = FastKronTorch()

class GeMKM(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x: torch.Tensor, *fs) -> torch.Tensor:
    print(140, x.shape, [f.shape for f in fs])
    z, zs = fastkrontorch.gemkm(True, x, fs)
    ctx.save_for_backward(x, *fs, *zs)
    ctx.num_facs = len(fs)
    return z

  @staticmethod
  def backward(ctx, grad_z):
    num_facs = ctx.num_facs
    x = ctx.saved_tensors[0]
    fs = ctx.saved_tensors[1:num_facs + 1]
    zs = ctx.saved_tensors[num_facs+1:]
    return fastkrontorch.mkmBackward(grad_z, x, fs, zs)

 

class GeKMM(torch.autograd.Function):
  @staticmethod
  def forward(ctx, fs: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    if type(x) is not torch.Tensor:
      raise ValueError("Input 'x' should be a Tensor")
    if type(fs) is not list:
      raise ValueError("Input 'fs' should be a list of Tensor")
    for i,f in enumerate(fs):
      if type(f) is not torch.Tensor:
        raise ValueError(f"Input fs[{i}] should be a Tensor")
    # if y != None and type(y) is not torch.Tensor:
    #   raise ValueError(f"Input 'y' should be a 2D Tensor")

    trX,x, trF,fs = fastkrontorch.reshapeInput(x, fs)

    alpha = 1
    beta = 0
    y = None

    if torch.is_grad_enabled() or not fastkrontorch.isSupported(x, fs):
      z = fastkrontorch.shuffleGeMM(torch, FastKronBase.MMTypeKMM, x, fs, alpha, beta, 
                                      y, trX, trF)
    else:
      rs, ts = fastkrontorch.gekmmSizes(FastKronBase.MMTypeKMM, x, fs, trX=trX, trF=trF)
      temp1 = x.new_empty(ts)
      temp2 = x.new_empty(ts) if rs != ts else None
      z = x.new_empty(size=rs, dtype=x.dtype)
      fastkrontorch.gekmm(fs, x, z, alpha, beta, y, temp1, temp2, trX, trF)
      z = z.reshape(rs)

    return z

  @staticmethod
  def backend(ctx, grad_z):
    return None, None

def gemkm(x : torch.Tensor, fs : List[torch.Tensor]) -> torch.Tensor:
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
  print(226, x.requires_grad, type(fs), type(fs[0]))
  if torch.is_grad_enabled():
    return GeMKM.apply(x, *fs)
  else:
    return fastkrontorch.gemkm(torch.is_grad_enabled(), x, fs)[0]

def gekmm(fs, x, alpha=1.0, beta=0.0, y=None):
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
  return GeKMM.apply(fs, x)
  