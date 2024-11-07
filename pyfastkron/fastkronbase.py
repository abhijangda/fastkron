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
  MMTypeMKM = 1
  MMTypeKMM = 2

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

  
  def p(self, mmtype, f, trF):
    if mmtype == FastKronBase.MMTypeMKM:
      if trF:
        return f.shape[-1]
      else:
        return f.shape[-2]
    elif mmtype == FastKronBase.MMTypeKMM:
      if trF:
        return f.shape[-2]
      else:
        return f.shape[-1]
  
  def q(self, mmtype, f, trF):
    if mmtype == FastKronBase.MMTypeMKM:
      if trF:
        return f.shape[-2]
      else:
        return f.shape[-1]
    elif mmtype == FastKronBase.MMTypeKMM:
      if trF:
        return f.shape[-1]
      else:
        return f.shape[-2]

  def ps(self, mmtype, fs, trF):
    return [self.p(mmtype, f, trF) for f in fs]

  def qs(self, mmtype, fs, trF):
    return [self.q(mmtype, f, trF) for f in fs]

  def m(self, mmtype, x, trX):
    if mmtype == FastKronBase.MMTypeMKM:
      if trX:
        return x.shape[-1]
      else:
        return x.shape[-2]
    elif mmtype == FastKronBase.MMTypeKMM:
      if trX:
        return x.shape[-2]
      else:
        return x.shape[-1]
  
  def k(self, mmtype, x, trX):
    if mmtype == FastKronBase.MMTypeMKM:
      if trX:
        return x.shape[-2]
      else:
        return x.shape[-1]
    elif mmtype == FastKronBase.MMTypeKMM:
      if trX:
        return x.shape[-1]
      else:
        return x.shape[-2]

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

  def batchedDims(self, x1, x2, addPadding=True):
    x1batch = x1.shape[:-2]
    x2batch = x2.shape[:-2]

    if addPadding == True:
      if len(x1batch) < len(x2batch):
        x1batch = tuple([1] * (len(x2batch) - len(x1batch))) + x1batch
      elif len(x1batch) > len(x2batch):
        x2batch = tuple([1] * (len(x1batch) - len(x2batch))) + x2batch

    return x1batch, x2batch

  def broadcastShape(self, shape1, shape2):
    #This function requires both shapes to be of same length
    finalShape = ()

    for s1, s2 in zip(reversed(shape1), reversed(shape2)):
      if s1 == s2:
        finalShape += (s1,)
      elif s1 != 1:
        finalShape += (s1,)
      elif s2 != 1:
        finalShape += (s2,)
      else:
        raise ValueError(f"Shape {shape1} of x and {shape2} of f are not broadcastable")

    return tuple(reversed(finalShape))

  def gekmmSizes(self, mmtype, x, fs, trX = False, trF = False):
    self.checkShapeAndTypes(mmtype, x, fs, None, None, trX, trF)

    device_type = self.device_type(x)
    xbatch, fbatch = self.batchedDims(x, fs[0], addPadding=True)
    rs, ts = None, None

    if self.supportedSystem() and self.supportedProcessor():
      if device_type == 'cpu':
        if fastkronX86 != None:
          rs, ts = fastkronX86.gekmmSizes((self.m(mmtype, x, trX), self.k(mmtype, x, trX)), 
                                          self.ps(mmtype,  fs, trF),
                                          self.qs(mmtype, fs, trF))
        else:
          raise ValueError(f"Device type '{device_type}' not supported")
      elif device_type == 'cuda':
        rs, ts = fastkronCUDA.gekmmSizes((self.m(mmtype, x, trX), self.k(mmtype, x, trX)),
                                          self.ps(mmtype, fs, trF),
                                          self.qs(mmtype, fs, trF))
    else:
      rs, ts = (self.matrixShape(x, trX)[0] * product(self.qs([self.matrixShape(f, trF) for f in fs]))), -1

    zbroadcastshape = self.broadcastShape(xbatch, fbatch)
    zshape = []
    if mmtype == FastKronBase.MMTypeMKM:
      zshape = (self.m(mmtype, x, trX), rs//self.m(mmtype, x, trX))
    elif mmtype == FastKronBase.MMTypeKMM:
      zshape = (rs//self.m(mmtype, x, trX), self.m(mmtype, x, trX))

    return self.broadcastShape(xbatch, fbatch) + zshape, \
          (1 if len(zbroadcastshape) == 0 else product(zbroadcastshape))*ts

  def checkShapeAndTypes(self, mmtype, x, fs, z, y, trX, trF):
    # Only operate on 2-dims matrices
    if self.k(mmtype, x, trX) != product(self.ps(mmtype, fs, trF)):
      raise ValueError(f"Input operand x has a mismatch with its dimension 1 ('{self.k(mmtype, x, trX)}') with dimension 0 of kronecker product of fs ('{product(self.ps(mmtype, fs, trF))}')")
    
    if x.dtype != fs[0].dtype:
      raise ValueError(f"Operand types mismatches {x.dtype} != {fs[0].dtype}")
    
    if len(set([f.dtype for f in fs])) != 1:
      raise ValueError(f"Type of Kronecker factors do not match. Found {len(set([f.dtype for f in fs]))} different types")

    for i,f in enumerate(fs):
      if fs[0].shape[:-2] != f.shape[:-2]:
        raise ValueError(f"Batched dims of factors do not match: {fs[0].shape[:-2]} != {f.shape[:-2]}")

    xbatch, fbatch = self.batchedDims(x, fs[0])
    finalShape = self.broadcastShape(xbatch, fbatch)

    if z is not None:
      expectedZshape = (self.m(mmtype, x, trX), product(self.qs(mmtype, fs, trF)))
      if mmtype == FastKronBase.MMTypeKMM:
        expectedZshape = (expectedZshape[1], expectedZshape[0])

      if z.shape[-2:] != expectedZshape:
        raise ValueError(f"Output operand 'z' shape '{z.shape}' mismatch with '{expectedZshape}'")
      if z.shape[:-2] != finalShape:
        raise ValueError(f"Output operand batched dimensions do not match with broadcasted dimensions ('{z.shape[:-2]}' != {finalShape})'")
      assert x.dtype == z.dtype

    if y is not None and z is not None:
      if self.broadcastShape(*self.batchedDims(y, z)) != z.shape[:-2]:
        raise ValueError(f"Input operand 'y' shape {y.shape} cannot be broadcasted to output 'z' shape {z.shape}")
      assert y.dtype   == z.dtype

  def trLastTwoDims(self, x):
    raise NotImplementedError()

  def device_type(self, x):
    raise NotImplementedError()

  def xgemm(self, handle, mmtype, fn, stridedBatchedFn, x, fs, z, alpha, beta, y, temp1, temp2,
            trX = False, trF = False):

    self.checkShapeAndTypes(mmtype, x, fs, z, y, trX, trF)

    orig_xshape = x.shape
    orig_fshape = []
    for f in fs:
      orig_fshape += [f.shape]
    orig_yshape = y.shape if y is not None else None

    xbatch, fbatch = self.batchedDims(x, fs[0], addPadding = False)
    zbatch = z.shape[:-2]
    if y is not None:
      ybatch, _ = self.batchedDims(y, z, addPadding=True)
    else:
      ybatch = zbatch

    if (mmtype == FastKronBase.MMTypeMKM and len(fbatch) == 0 and (len(xbatch) == 0 or not trX) and ybatch == zbatch) or \
       (mmtype == FastKronBase.MMTypeKMM and len(fbatch) == 0 and len(xbatch) == 0 and len(zbatch) == 0):
      m = self.m(mmtype, x, trX)
      if len(xbatch) > 0:
        m = m * product(xbatch)

      handlefn = handle.xgemkm if mmtype == FastKronBase.MMTypeMKM else handle.xgekmm

      handlefn(fn, m, len(fs), self.ps(mmtype, fs, trF), self.qs(mmtype, fs, trF),
                    self.tensor_data_ptr(x), 
                    self.fptrs(fs),
                    self.tensor_data_ptr(z), alpha, beta, 
                    self.tensor_data_ptr(y),
                    self.tensor_data_ptr(temp1), 
                    self.tensor_data_ptr(temp2), 
                    trX, trF)
    else:
      xbatch, fbatch = self.batchedDims(x, fs[0], addPadding = True)
      if y is not None:
        ybatch, _ = self.batchedDims(y, z, addPadding=True)
      else:
        ybatch = zbatch

      z = z.reshape((product(z.shape[:-2]),)+z.shape[-2:])
      x = x.reshape((product(xbatch),)+x.shape[-2:])
      for i in range(len(fs)):
        fs[i] = fs[i].reshape((product(fbatch),)+fs[i].shape[-2:])

      if y is not None:
        y = y.reshape((product(ybatch),) + y.shape[-2:])

      #TODO: Compress batched dimensions into the three cases of the below 
      #loop for better performance

      #Compute each batch of Z using stridedbatched 
      batchLinearIdxZ = 0
      MaxLinearIdxZ = product(z.shape[:-2])
      while batchLinearIdxZ < MaxLinearIdxZ:
        #Go through each linear index of batch dimensions of Z
        xidx = fidx = zidx = yidx = None
        strideX = strideF = strideZ = strideY = 0
        strideZ = product(z.shape[-2:])
        zidx = yidx = batchLinearIdxZ
        
        xidx = 0
        fidx = 0
        yidx = 0
        tmpx = tmpf = tmpy = batchLinearIdxZ
        xDimProds = 1
        fDimProds = 1
        yDimProds = 1

        #Find linear index of x and fs for z
        for zdim, xdim, fdim, ydim in zip(reversed(zbatch), reversed(xbatch), reversed(fbatch), reversed(ybatch)):
          xidx += (tmpx % xdim) * xDimProds
          xDimProds = xDimProds * xdim
          tmpx = tmpx // zdim

          fidx += (tmpf % fdim) * fDimProds
          fDimProds = fDimProds * fdim
          tmpf = tmpf // zdim

          yidx += (tmpy % ydim) * yDimProds
          yDimProds = yDimProds * ydim
          tmpy = tmpy // zdim

        batchCount = zbatch[-1]

        dim = -1
        if xbatch[dim] > 1 and fbatch[dim] == 1:
          #Batched X with same Fs
          strideX = product(x.shape[-2:])
          strideF = [0 for f in fs]
        elif xbatch[dim] == fbatch[dim]:
          #Each X with corresponding Fs
          strideX = product(x.shape[-2:])
          strideF = [product(f.shape[-2:]) for f in fs]
        elif xbatch[dim] == 1 and fbatch[dim] > 1:
          #Same X with batched Fs
          strideX = 0
          strideF = [product(f.shape[-2:]) for f in fs]
        
        if zbatch[dim] == ybatch[dim]:
          strideY = strideZ
        elif zbatch[dim] > 1 and ybatch[dim] == 1:
          strideY = 0
        else:
          raise ValueError(f"Output 'z' {z.shape} cannot be broadcastable to 'y' {y.shape}")
        
        m = self.m(mmtype, x, trX)
        #Apply StridedBatched on the last dimension
        print(367, strideF)
        handlefn = handle.xgemkmStridedBatched if mmtype == FastKronBase.MMTypeMKM else handle.xgekmmStridedBatched
        handlefn(stridedBatchedFn, m, len(fs), self.ps(mmtype, fs, trF), self.qs(mmtype, fs, trF),
                self.tensor_data_ptr(x[xidx, :]), strideX,
                self.fptrs([f[fidx,:] for f in fs]), strideF,
                batchCount, self.tensor_data_ptr(z[zidx, :]), strideZ, alpha, beta, 
                self.tensor_data_ptr(y[yidx, :]), strideY,
                self.tensor_data_ptr(temp1), 
                self.tensor_data_ptr(temp2), 
                trX, trF)
        
        batchLinearIdxZ += zbatch[dim]

      x.reshape(orig_xshape)
      for i in range(len(fs)):
        fs[i] = fs[i].reshape(orig_fshape[i])
      if y is not None:
        y = y.reshape(orig_yshape)

  def shuffleGeKMM(self, framework, x, fs, alpha = None, beta = None, y = None, trX = False, trF = False):
    self.checkShapeAndTypes(x, fs, None, y, trX, trF)

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
    if beta != None and y is not None:
      z += beta * y
    return z