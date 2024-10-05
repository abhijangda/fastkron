from functools import reduce
import platform

from .fastkronhandle import FastKronHandle

if platform.system() == "Linux":
  if platform.machine() == "x86_64" or platform.machine() == "AMD64":
    from . import FastKronX86
    fastkronX86 = FastKronHandle("x86", FastKronX86)
  else:
    fastkronX86 = None

  fastkronCUDA = None
  try:
    import torch
    if torch.cuda.is_available():
      if float(torch.version.cuda) >= 12.0:
        from . import FastKronCUDA
        fastkronCUDA = FastKronHandle("cuda", FastKronCUDA)
  except:
    fastkronCUDA = None

def product(values):
  return reduce((lambda a, b: a * b), values)

class FastKronBase:
  def __init__(self, x86, cuda):
    self.x86 = x86 and fastkronX86 != None
    self.cuda = cuda and fastkronCUDA != None
  
  def hasX86(self):
    return self.x86
  
  def hasCUDA(self):
    return self.cuda

  def supportedSystem(self):
    return platform.system() == "Linux"
  
  def supportedProcessor(self):
    return platform.processor() == "x86_64" or platform.machine() == "AMD64"

  def isSupported(self, x, fs):
    return self.supportedSystem()     and self.supportedProcessor() and \
           self.supportedTypes(x, fs) and self.supportedDevice(x)

  def supportedDevice(self, x):
    raise NotImplementedError()

  def supportedTypes(self, x, fs):
    raise NotImplementedError()

  def tensor_data_ptr(self, tensor):
    raise NotImplementedError()

  
  def p(self, f, trF):
    if trF:
      return f.shape[-1]
    else:
      return f.shape[-2]
  
  def q(self, f, trF):
    if trF:
      return f.shape[-2]
    else:
      return f.shape[-1]

  def ps(self, fs, trF):
    if trF:
      return [f.shape[-1] for f in fs]
    else:
      return [f.shape[-2] for f in fs]
  
  def qs(self, fs, trF):
    if trF:
      return [f.shape[-2] for f in fs]
    else:
      return [f.shape[-1] for f in fs]

  def m(self, x, trX):
    if trX:
      return x.shape[-1]
    else:
      return x.shape[-2]
  
  def k(self, x, trX):
    if trX:
      return x.shape[-2]
    else:
      return x.shape[-1]

  def fptrs(self, fs):
    return [self.tensor_data_ptr(f) for f in fs]

  def xfsShape(self, xshape, fshapes, trX, trF):
    newxshape = None
    newfshapes = []

    if trX == False:
      if len(xshape) == 1:
        newxshape = (1, xshape[0])

    elif trX:
      if len(xshape) == 1:
        newxshape = (xshape[0], 1)

    for fs in fshapes:
      newfshapes += [fshapes]
      if trF == False:
        if len(fs) == 1:
          newfshapes[-1] = [(fs[0], 1)]
      elif trF:
        if len(fs) == 1:
          newfshapes[-1] = [(1, fs[0])]

    return newxshape, newfshapes

  def reshapeInput(self, x, fs, trX, trF):
    if trX:
      if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1))
    else:
      if len(x.shape) == 1:
        x = x.reshape((1,x.shape[0]))
    
    for f in fs:
      if trF:
        if f.ndim == 1:
          f = f.reshape((1, fs[0]))
      else:
        if f.ndim == 1:
          f = f.reshape((fs[0], 1))

    return x, fs

  def batchedDims(self, x, fs, trX, trF, addPadding=True):
    xbatch = x.shape[:-2]
    prefbatch = fs[0].shape[:-2]

    if addPadding == True:
      if len(xbatch) < len(prefbatch):
        xbatch = tuple([1] * (len(prefbatch) - len(xbatch))) + xbatch
      elif len(xbatch) > len(prefbatch):
        prefbatch = tuple([1] * (len(xbatch) - len(prefbatch))) + prefbatch

    return xbatch, prefbatch

  def broadcastShape(self, xbatch, prefbatch, postfbatch):
    finalShape = ()
    xbatch = tuple(reversed(xbatch))
    prefbatch = tuple(reversed(prefbatch))
    xi = 0
    prefi = 0
    while xi < len(xbatch) and prefi < len(prefbatch):
      if xbatch[xi] == prefbatch[prefi]:
        finalShape += (xbatch[xi],)
      elif prefbatch[prefi] != 1:
        finalShape += (prefbatch[prefi],)
      elif xbatch[xi] != 1:
        finalShape += (xbatch[xi],)
      else:
        raise ValueError(f"Expected value of dimension {len(xbatch) - xi} of x and f do not match: {xbatch[xi]} != {fbatch[fi]}")
      xi += 1
      prefi += 1

    return tuple(reversed(finalShape))

  def gekmmSizes(self, x, fs, trX = False, trF = False):
    self.checkShapeAndTypes(x, fs, None, None, trX, trF)

    device_type = self.device_type(x)
    xbatch, prefbatch = self.batchedDims(x, fs, trX, trF)
    rs, ts = None, None

    if self.supportedSystem() and self.supportedProcessor():
      if device_type == 'cpu':
        if fastkronX86 != None:
          rs, ts = fastkronX86.gekmmSizes((self.m(x, trX), self.k(x, trX)), self.ps(fs, trF), self.qs(fs, trF))
        else:
          raise ValueError(f"Device type '{device_type}' not supported")
      elif device_type == 'cuda':
        rs, ts = fastkronCUDA.gekmmSizes((self.m(x, trX), self.k(x, trX)), self.ps(fs, trF), self.qs(fs, trF))
    else:
      rs, ts = (self.matrixShape(x, trX)[0] * product(self.qs([self.matrixShape(f, trF) for f in fs]))), -1

    zbroadcastshape = self.broadcastShape(xbatch, prefbatch, ())
    return self.broadcastShape(xbatch, prefbatch, ()) + (self.m(x, trX), rs//self.m(x, trX)), \
          (1 if len(zbroadcastshape) == 0 else product(zbroadcastshape))*ts


  def checkShapeAndTypes(self, x, fs, z, y, trX, trF):
    # Only operate on 2-dims matrices
    if self.k(x, trX) != product(self.ps(fs, trF)):
      raise ValueError(f"Input operand x has a mismatch with its dimension 1 ('{self.k(x, trX)}') with dimension 0 of kronecker product of fs ('{product(self.ps(fs, trF))}')")
    
    if x.dtype != fs[0].dtype:
      raise ValueError(f"Operand types mismatches {x.dtype} != {fs[0].dtype}")
    
    if len(set([f.dtype for f in fs])) != 1:
      raise ValueError(f"Type of Kronecker factors do not match. Found {len(set([f.dtype for f in fs]))} different types")

    for i,f in enumerate(fs):
      if fs[0].shape[:-2] != f.shape[:-2]:
        raise ValueError(f"Batched dims of factors do not match: {fs[0].shape[:-2]} != {f.shape[:-2]}")

    xbatch, prefbatch = self.batchedDims(x, fs, trX, trF)
    finalShape = self.broadcastShape(xbatch, prefbatch, ())

    if y is not None:
      if yshape[-2:] != (self.m(x, trX), self.k(x, trX)):
        raise ValueError(f"")
      if yshape[1] != product(self.qs(fs, trF)):
        raise ValueError(f"")
      assert x.dtype   == z.dtype
    
    if z is not None:
      if z.shape[-2:] != (self.m(x, trX), product(self.qs(fs, trF))):
        raise ValueError(f"Output operand 'z' shape ('{z.shape}') mismatch with '({self.m(x, trX), product(self.qs(fs, trF))})'")
      if z.shape[:-2] != finalShape:
        raise ValueError(f"Output operand batched dimensions do not match with broadcasted dimensions ('{z.shape[:-2]}' != {finalShape})'")
      assert x.dtype == z.dtype
  
  def trLastTwoDims(self, x):
    raise NotImplementedError()

  def device_type(self, x):
    raise NotImplementedError()

  def xgekmm(self, handle, fn, x, fs, z, alpha, beta, y, temp1, temp2,
            trX = False, trF = False):

    self.checkShapeAndTypes(x, fs, z, y, trX, trF)

    orig_xshape = x.shape
    orig_fshape = []
    for f in fs:
      orig_fshape += [f.shape]

    xbatch, fbatch = self.batchedDims(x, fs, trX, trF, addPadding = False)
    if len(fbatch) == 0 and (len(xbatch) == 0 or not trX):
      m = self.m(x, trX)
      if len(xbatch) > 0:
        m = m * product(xbatch)

      handle.xgekmm(fn, m, len(fs), self.ps(fs, trF), self.qs(fs, trF),
                    self.tensor_data_ptr(x), 
                    self.fptrs(fs),
                    self.tensor_data_ptr(z), alpha, beta, 
                    self.tensor_data_ptr(y),
                    self.tensor_data_ptr(temp1), 
                    self.tensor_data_ptr(temp2), 
                    trX, trF)
    else:
      xbatch, fbatch = self.batchedDims(x, fs, trX, trF, addPadding = True)
      zindices = z.shape[:-2]
      z = z.reshape((product(z.shape[:-2]),)+z.shape[-2:])
      x = x.reshape((product(xbatch),)+x.shape[-2:])
      for i in range(len(fs)):
        fs[i] = fs[i].reshape((product(fbatch),)+fs[i].shape[-2:])

      for batchLinearIdxZ in range(product(z.shape[:-2])):
        batchLinearIdxX = 0
        batchLinearIdxF = 0
        tmpx = tmpf = batchLinearIdxZ
        xDimProds = product(xbatch)
        zDimProds = product(zindices)
        fDimProds = product(fbatch)

        for dim,size in enumerate(zindices):
          batchLinearIdxX = batchLinearIdxX*xDimProds + (0 if xbatch[dim] == 1 else tmpx // (zDimProds//zindices[dim]))
          batchLinearIdxF = batchLinearIdxF*fDimProds + (0 if fbatch[dim] == 1 else tmpf // (zDimProds//zindices[dim]))

          tmpx = tmpx%(zDimProds//zindices[dim])
          tmpf = tmpf%(zDimProds//zindices[dim])

          xDimProds = xDimProds//xbatch[dim]
          fDimProds = fDimProds//fbatch[dim]
          zDimProds = zDimProds//zindices[dim]

        batchLinearIdxX += tmpx
        batchLinearIdxF += tmpf

        m = self.m(x, trX)
        handle.xgekmm(fn, m, len(fs), self.ps(fs, trF), self.qs(fs, trF),
                      self.tensor_data_ptr(x[batchLinearIdxX, :]), 
                      self.fptrs([f[batchLinearIdxF,:] for f in fs]),
                      self.tensor_data_ptr(z[batchLinearIdxZ, :]), alpha, beta, 
                      self.tensor_data_ptr(y),
                      self.tensor_data_ptr(temp1), 
                      self.tensor_data_ptr(temp2), 
                      trX, trF)
    
    x.reshape(orig_xshape)
    for i in range(len(fs)):
      fs[i] = fs[i].reshape(orig_fshape[i])

  def shuffleGeKMM(self, framework, x, fs, alpha = None, beta = None, y = None, trX = False, trF = False):
    self.checkShapeAndTypes(x, fs, y, None, trX, trF)

    rs, _ = self.gekmmSizes(x, fs, trX=trX, trF=trF)
    m,  k = self.m(x, trX), self.k(x, trX)

    if trX:
      x = self.trLastTwoDims(x)

    z = x
    l = rs[-1]

    for i,f in enumerate(reversed(fs)):
      if trF: f = self.trLastTwoDims(f)
      z = z.reshape(z.shape[:-2] + (m * k//self.p(f, False), self.p(f, False)))
      z = framework.matmul(z, f)
      z = z.reshape(z.shape[:-2] + (m, k//self.p(f, False), self.q(f, False)))
      z = self.trLastTwoDims(z)
      z = z.reshape(z.shape[:-3] + (m*k//self.p(f, False), self.q(f, False)))
      k = (k//self.p(f, False)) * self.q(f, False)

    if alpha != None:
      z = alpha * (z.reshape(z.shape[:-2] + (m, l)))
    if beta != None and y != None:
      z += beta * y
    return z