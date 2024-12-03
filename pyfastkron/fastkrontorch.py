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
    if type(tensor) is list or type(tensor) is tuple:
      return [t.data_ptr() for t in tensor]
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

    if requires_grad:
      if alpha != 1: raise ValueError(f"When requires_grad is True alpha should be 1 but is {alpha}")
      if beta != 0: raise ValueError(f"When requires_grad is True beta should be 0 but is {beta}")
      if y != None: raise ValueError(f"When requires_grad is True y should be None")

    if y != None and type(y) is not torch.Tensor:
      raise ValueError(f"Input 'y' should be a 2D Tensor")
    
    is_vec = x.ndim == 1

    trX,x, trF,fs = fastkrontorch.reshapeInput(FastKronBase.MMTypeMKM, x, fs)

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
    rs, ts = self.gekmmSizes(FastKronBase.MMTypeMKM, x, fs, trX=trX, trF=trF)
    z = x.new_empty(rs)
    temp1 = x.new_empty(ts)
    temp2 = x.new_empty(ts) if rs != ts else None
    super().xgemm(self.handle(x), FastKronBase.MMTypeMKM, fn, stridedBatchedFn, x, fs, z, alpha, beta, y, [temp1, temp2], trX, trF)
    z = z.reshape(rs)

    if is_vec and z.ndim > 1:
      z = z.squeeze()

    return z, zs
  
  def mkmForward(self, requires_grad, x, fs, stream = None):
    if type(x) is not torch.Tensor:
      raise ValueError("Input 'x' should be a Tensor")
    if type(fs) is not list and type(fs) is not tuple:
      raise ValueError("Input 'fs' should be a list of Tensor")
    for i,f in enumerate(fs):
      if type(f) is not torch.Tensor:
        raise ValueError(f"Input fs[{i}] should be a Tensor but is {type(fs[i])}")
  
    is_vec = x.ndim == 1
    alpha = 1
    beta = 0

    trX,x, trF,fs = fastkrontorch.reshapeInput(FastKronBase.MMTypeMKM, x, fs)

    if x.device.type == "cuda" and stream is None:
      stream = torch.cuda.current_stream()

    self.check(x, fs, None, None, stream)

    fn = None
    stridedBatchedFn = None
    if x.dtype == torch.float:
      fn = self.handle(x).libFastKron.smkmForward
      stridedBatchedFn = self.handle(x).libFastKron.smkmForwardStridedBatched
    elif x.dtype == torch.double:
      fn = self.handle(x).libFastKron.dmkmForward
      stridedBatchedFn = self.handle(x).libFastKron.dmkmForwardStridedBatched

    zs = []
    rs, ts = self.gekmmSizes(FastKronBase.MMTypeMKM,\
                             x, fs, trX=trX, trF=trF, intermediates=True)

    z = x.new_empty(rs)
    zs = [x.new_empty(s) for s in ts]
    super().xgemm(self.handle(x), FastKronBase.MMTypeMKM, fn, stridedBatchedFn, x, fs, z, 1.0, 0.0, None, zs, trX, trF, writeIntermediates=True)
    z = z.reshape(rs)
    zs = [inter.reshape(s) for inter,s in zip(zs,ts)]
    if is_vec and z.ndim > 1:
      z = z.squeeze()

    return z, zs

  def mkmBackward(self, grad_z, x, fs, zs):
    trX,x, trF, fs = fastkrontorch.reshapeInput(FastKronBase.MMTypeMKM, x, fs)
    return self.__mkmBackward(grad_z, x, fs, zs, trX, trF, x.requires_grad, [f.requires_grad for f in fs])

  def __mkmBackward(self, grad_z, x, fs, zs, trX, trF, x_requires_grad, fs_requires_grad):
    is_vec = grad_z.ndim == 1
    if is_vec:
      grad_z = grad_z.unsqueeze(0)

    grad_fs = []
    grad_zs = [grad_z]
    zbatchShape = grad_z.shape[:-2]
    zs = tuple(zs) + (x,)
    for z,f,f_requires_grad in zip(zs, fs, fs_requires_grad):
      prev_grad_z = grad_zs[-1]
      orig_fshape = f.shape
      fp = self.p(FastKronBase.MMTypeMKM, f, trF)
      fq = self.q(FastKronBase.MMTypeMKM, f, trF)
      #[m,k] -> [m,q,k/q]
      prev_grad_z = prev_grad_z.reshape((prev_grad_z.shape[:-2] + (prev_grad_z.shape[-2],)) + \
                                       (fq, prev_grad_z.shape[-1]//fq))

      #->[m,k/q,q]
      prev_grad_z = prev_grad_z.transpose(-1,-2)

      #->[m*k/q,q]
      prev_grad_z = prev_grad_z.reshape(prev_grad_z.shape[:-3] + \
                                        (prev_grad_z.shape[-3] * prev_grad_z.shape[-2],\
                                        prev_grad_z.shape[-1]))
      #Backward pass for z
      if x.data_ptr() != z.data_ptr() or x_requires_grad:
        grad_z = torch.matmul(prev_grad_z, f.mT)
        grad_z = grad_z.reshape(zbatchShape + z.shape[-2:])
        grad_zs += [grad_z]

      #Backward pass for f
      if f_requires_grad:
        orig_zshape = z.shape
        z = z.reshape(z.shape[:-2] + ((z.shape[-2] * z.shape[-1])//fp, fp))
        trZ = z.mT
        grad_f = trZ @ prev_grad_z
        grad_fs += [grad_f]
        z = z.reshape(orig_zshape)
      else:
        grad_fs += [None]

    return (grad_zs[-1] if x_requires_grad else None,) + tuple(grad_fs)

  def gekmm(self, requires_grad, fs, x,
            y = None, alpha = 1, beta = 0, 
            stream = None):
    if type(x) is not torch.Tensor:
      raise ValueError("Input 'x' should be a Tensor")
    if type(fs) is not list and type(fs) is not tuple:
      raise ValueError("Input 'fs' should be a list of Tensor")
    for i,f in enumerate(fs):
      if type(f) is not torch.Tensor:
        raise ValueError(f"Input fs[{i}] should be a Tensor")

    if requires_grad:
      if alpha != 1: raise ValueError(f"When requires_grad is True alpha should be 1 but is {alpha}")
      if beta != 0: raise ValueError(f"When requires_grad is True beta should be 0 but is {beta}")
      if y != None: raise ValueError(f"When requires_grad is True y should be None")

    if y != None and type(y) is not torch.Tensor:
      raise ValueError(f"Input 'y' should be a 2D Tensor")

    if x.device.type == "cuda" and stream is None:
      stream = torch.cuda.current_stream()

    self.check(x, fs, y, None, stream)

    is_vec = x.ndim == 1

    trX,x, trF,fs = fastkrontorch.reshapeInput(FastKronBase.MMTypeKMM, x, fs)
    fn = None
    stridedBatchedFn = None
    
    if x.dtype == torch.float:
      fn = self.handle(x).libFastKron.sgekmm
      stridedBatchedFn = self.handle(x).libFastKron.sgekmmStridedBatched
    elif x.dtype == torch.double:
      fn = self.handle(x).libFastKron.dgekmm
      stridedBatchedFn = self.handle(x).libFastKron.dgekmmStridedBatched

    zs = []
    rs, ts = fastkrontorch.gekmmSizes(FastKronBase.MMTypeKMM, x, fs, trX=trX, trF=trF)
    temp1 = x.new_empty(ts)
    temp2 = x.new_empty(ts) if rs != ts else None
    z = x.new_empty(size=rs, dtype=x.dtype)
    super().xgemm(self.handle(x), FastKronBase.MMTypeKMM, fn, stridedBatchedFn,\
                  x, fs, z, alpha, beta, y, [temp1, temp2], trX, trF)
    z = z.reshape(rs)
    
    if is_vec and z.ndim > 1:
      z = z.squeeze()

    return z, zs

  def kmmForward(self, requires_grad, fs, x, stream = None):
    if type(x) is not torch.Tensor:
      raise ValueError("Input 'x' should be a Tensor")
    if type(fs) is not list and type(fs) is not tuple:
      raise ValueError("Input 'fs' should be a list of Tensor")
    for i,f in enumerate(fs):
      if type(f) is not torch.Tensor:
        raise ValueError(f"Input fs[{i}] should be a Tensor")

    if x.device.type == "cuda" and stream is None:
      stream = torch.cuda.current_stream()

    self.check(x, fs, None, None, stream)

    is_vec = x.ndim == 1
    alpha = 1
    beta = 0

    trX,x, trF,fs = fastkrontorch.reshapeInput(FastKronBase.MMTypeKMM, x, fs)
    fn = None
    stridedBatchedFn = None
    
    if x.dtype == torch.float:
      fn = self.handle(x).libFastKron.skmmForward
      stridedBatchedFn = self.handle(x).libFastKron.skmmForwardStridedBatched
    elif x.dtype == torch.double:
      fn = self.handle(x).libFastKron.dkmmForward
      stridedBatchedFn = self.handle(x).libFastKron.dkmmForwardStridedBatched

    zs = []
    rs, ts = self.gekmmSizes(FastKronBase.MMTypeKMM, x, fs, trX=trX, trF=trF, intermediates=True)
    zs = [x.new_empty(s) for s in ts]
    z = x.new_empty(size=rs)
    super().xgemm(self.handle(x), FastKronBase.MMTypeKMM, fn, stridedBatchedFn,\
                  x, fs, z, 1.0, 0.0, None, zs, trX, trF, writeIntermediates=True)
    z = z.reshape(rs)
    zs = [inter.reshape(s) for inter,s in zip(zs,ts)]
    if is_vec and z.ndim > 1:
      z = z.squeeze()

    return z, zs

  def kmmBackward(self, grad_z, x, fs, zs):
    trX,x, trF, fs = fastkrontorch.reshapeInput(FastKronBase.MMTypeKMM, x, fs)
    grad_fs = []
    grad_zs = [grad_z]
    zbatchShape = zs[0].shape[:-2]
    zs = tuple(z.mT for z in zs)

    grads = self.__mkmBackward(grad_z.mT, x.mT, [f.mT for f in fs], zs, trX, trF,
                               x.requires_grad, [f.requires_grad for f in fs])
    grad_x = grads[0]
    grad_fs = grads[1:]

    return (grad_x.mT if grad_x is not None else None, ) + \
           tuple(g.mT if g is not None else None for g in grad_fs)

  def shuffleGeMM(self, mmtype, x, fs, 
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
    
    is_vec = x.ndim == 1

    trX,x, trF,fs = self.reshapeInput(mmtype, x, fs)

    if x.device.type == "cuda" and stream is None:
      stream = torch.cuda.current_stream()

    self.check(x, fs, y, None, stream)

    z = super().shuffleGeMM(False, torch, mmtype, x, fs, alpha, beta, y, trX, trF)
    
    if is_vec and z.ndim > 1:
      z = z.squeeze()

    return z
  
  def shuffleGeMKM(self, x, fs, y = None, alpha = 1, beta = 0, stream = None):
    return self.shuffleGeMM(FastKronBase.MMTypeMKM, x, fs, y, alpha, beta, stream)
  
  def shuffleGeKMM(self, fs, x, y = None, alpha = 1, beta = 0, stream = None):
    return self.shuffleGeMM(FastKronBase.MMTypeKMM, x, fs, y, alpha, beta, stream)
  

fastkrontorch = FastKronTorch()

class GeMKM(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, *fs) -> torch.Tensor:
    z, zs = fastkrontorch.mkmForward(True, x, fs)
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
  def forward(ctx, x, *fs) -> torch.Tensor:
    z, zs = fastkrontorch.kmmForward(True, fs, x)
    ctx.save_for_backward(x, *fs, *zs)
    ctx.num_facs = len(fs)
    return z

  @staticmethod
  def backward(ctx, grad_z):
    num_facs = ctx.num_facs
    x = ctx.saved_tensors[0]
    fs = ctx.saved_tensors[1:num_facs + 1]
    zs = ctx.saved_tensors[num_facs+1:]
    return fastkrontorch.kmmBackward(grad_z, x, fs, zs)

def gemkm(x : torch.Tensor, fs : List[torch.Tensor],
          alpha=1.0, beta=0.0, y=None) -> torch.Tensor:
  '''
  Perform Generalized Kronecker-Matrix Multiplication:
  
  $Z = \alpha ~ X \times \left( F^1 \otimes F^2 \otimes \dots F^N \right) + \beta Y$

  Parameters
  ----------
  x  : 2D torch tensor 
  fs : A list of 2D torch tensor
  alpha and beta: constants
  y  : 2D torch tensor
  trX: Transpose   x before computing GeKMM
  trF: Transpose each element of fs before computing GeKMM

  Returns
  -------
  z : 2D torch tensor
  '''
  if not fastkrontorch.isSupported(x, fs):
    return fastkrontorch.shuffleGeMKM(x, fs, y, alpha, beta)
  
  requires_grad = [f.requires_grad for f in fs if f.requires_grad]
  if len(requires_grad) > 0: requires_grad = True
  else: requires_grad = x.requires_grad
    
  if requires_grad:
    return GeMKM.apply(x, *tuple(fs))
  else:
    return fastkrontorch.gemkm(False, x, fs, y, alpha, beta)[0]

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
  if not fastkrontorch.isSupported(x, fs):
    return fastkrontorch.shuffleGeKMM(fs, x, y, alpha, beta)

  requires_grad = [f.requires_grad for f in fs if f.requires_grad]
  if len(requires_grad) > 0: requires_grad = True
  else: requires_grad = x.requires_grad

  if requires_grad:
    return GeKMM.apply(x, *tuple(fs))
  else:
    return fastkrontorch.gekmm(False, fs, x, y, alpha, beta)[0]
  