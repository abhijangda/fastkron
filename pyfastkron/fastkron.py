from functools import reduce
import ctypes

#TODO: Make these as fields of PyFastKronWrapper
def to_ctype_array(elems, ctype_t):
  return (ctype_t * len(elems))(*elems)

def product(values):
  return reduce((lambda a, b: a * b), values)

class FastKronOp:
  def __init__(self, op):
    self.op = op

FastKronOpT = FastKronOp(2)
FastKronOpN = FastKronOp(1)

class FastKronBackend:
    def __init__(self, backend):
      self.backend = backend

FastKronBackendX86 = FastKronBackend(1)
FastKronBackendARM = FastKronBackend(2)
FastKronBackendCUDA = FastKronBackend(3)
FastKronBackendHIP = FastKronBackend(4)

class PyFastKronWrapper:
  def __init__(self, cuda_stream):
    self.libKron = ctypes.CDLL("libFastKron.so")

    cppHandleTy = ctypes.c_ulong

    self.initFn = self.libKron.fastKronInit
    self.initFn.argtypes = [ctypes.POINTER(cppHandleTy), ctypes.c_int]
    self.initFn.restype = ctypes.c_int

    self.cudaInitFn = self.libKron.fastKronInitCUDA
    self.cudaInitFn.argtypes = [cppHandleTy, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    self.cudaInitFn.restype = ctypes.c_int
    
    self.destroyFn = self.libKron.fastKronDestroy
    self.destroyFn.argtypes = [cppHandleTy]
    self.destroyFn.restype = ctypes.c_int

    self.gekmmSizesFn = self.libKron.gekmmSizes
    self.gekmmSizesFn.argtypes = [cppHandleTy, ctypes.c_uint, ctypes.c_uint,
                                  ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
                                  ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]

    self.cpp_handle = ctypes.c_ulong(0)
    if self.initFn(ctypes.byref(self.cpp_handle), FastKronBackendCUDA.backend) != 0:
      print("error 1")
      return
    
    print(55, cuda_stream.cuda_stream, ctypes.c_void_p(cuda_stream.cuda_stream))
    if self.cudaInitFn(self.cpp_handle, ctypes.c_void_p(cuda_stream.cuda_stream), 1, 1, 1, -1) != 0:
      print("error 2")
      return

    self.sgekmmFn = self.libKron.sgekmm
    self.sgekmmFn.argtypes = [cppHandleTy, ctypes.c_uint, ctypes.c_uint,
                              ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
                              ctypes.c_void_p, ctypes.c_uint,
                              ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint,
                              ctypes.c_void_p,
                              ctypes.c_float, ctypes.c_float, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_void_p, 
                              ctypes.c_void_p]

    self.sgekmmTuneFn = self.libKron.sgekmmTune
    self.sgekmmTuneFn.argtypes = [cppHandleTy, ctypes.c_uint, ctypes.c_uint,
                                  ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
                                  ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
    self.sgekmmTuneFn.restype = ctypes.c_uint

  def sgekmmSizes(self, m, ps, qs):
    assert len(ps) == len(qs)

    resultSize = ctypes.c_size_t(0)
    tempSize   = ctypes.c_size_t(0)
    self.gekmmSizesFn(self.cpp_handle, ctypes.c_uint(m), ctypes.c_uint(len(ps)), 
                      to_ctype_array(ps, ctypes.c_uint), to_ctype_array(qs, ctypes.c_uint),
                      ctypes.byref(resultSize), ctypes.byref(tempSize))
    return resultSize.value//4, tempSize.value//4
  
  def sgekmm(self, m, ps, qs, x, opX, fs, opFs, y, alpha, beta, z, t1, t2, stream):
    assert len(ps) == len(qs)
    assert len(ps) == len(fs)

    return self.sgekmmFn(self.cpp_handle, m, len(ps),
                         to_ctype_array(ps, ctypes.c_uint), to_ctype_array(qs, ctypes.c_uint), 
                         ctypes.c_void_p(x), opX.op, 
                         to_ctype_array([ctypes.c_void_p(f) for f in fs], ctypes.c_void_p), opFs.op,
                         ctypes.c_void_p(y), ctypes.c_float(alpha), ctypes.c_float(beta),
                         ctypes.c_void_p(z), ctypes.c_void_p(t1), ctypes.c_void_p(t2), stream)

  def sgekmmTune(self, m, ps, qs, opX, opFs, stream):
    assert len(ps) == len(qs)
    assert len(ps) == len(fs)

    return self.sgekmmTuneFn(self.cpp_handle, m, len(ps),
                             to_ctype_array(ps, ctypes.c_uint),
                             to_ctype_array(qs, ctypes.c_uint),
                             opX.op, opFs.op, stream)

  def __del__(self):
    self.destroyFn(self.cpp_handle)
    self.cpp_handle = ctypes.c_ulong(0)

try:
  import torch
except:
  pass

class FastKronTorch:
  def __init__(self):
    self.pyfastkron = PyFastKronWrapper(torch.cuda.current_stream())
  
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
    fn = None
    if x.dtype == torch.float:
      fn = self.pyfastkron.sgekmmSizes
    elif x.dtype == torch.double:
      fn = self.pyfastkron.dgekmmSizes
    elif x.dtype == torch.int:
      fn = self.pyfastkron.igekmmSizes

    return fn(x.shape[0], self.ps(fs), self.qs(fs))

  def gekmmTune(self, x, fs, y,
                trX = False, trF = False, stream = None):
    if stream is None:
      stream = torch.cuda.current_stream()

    self._check(x, fs, y, stream)

    fn = None
    if x.dtype == torch.float:
      fn = self.pyfastkron.sgekmmTune
    elif x.dtype == torch.int:
      fn = self.pyfastkron.igekmmTune
    elif x.dtype == torch.double:
      fn = self.pyfastkron.dgekmmTune

    fn(x.shape[0], self.ps(fs), self.qs(fs),
       FastKronOpN if not trX else FastKronOpT,
       FastKronOpN if not trF else FastKronOpT, 
       ctypes.c_void_p(stream.cuda_stream))

  def gekmm(self, x, fs, y, alpha, beta, z, temp, 
            trX = False, trF = False, stream = None):
    if stream is None:
      stream = torch.cuda.current_stream()

    self._check(x, fs, y, stream)

    fn = None
    if x.dtype == torch.float:
      fn = self.pyfastkron.sgekmm
    elif x.dtype == torch.int:
      fn = self.pyfastkron.igekmm
    elif x.dtype == torch.double:
      fn = self.pyfastkron.dgekmm

    fn(x.shape[0], self.ps(fs), self.qs(fs), 
       x.data_ptr(), FastKronOpN if not trX else FastKronOpT, 
       self.fptrs(fs), FastKronOpN if not trF else FastKronOpT,
       y.data_ptr(),
       alpha, beta, None if z is None else z.data_ptr(), 
       temp.data_ptr(), None,
       ctypes.c_void_p(stream.cuda_stream))
  
if __name__ == "__main__":
  import torch
  fastKron = FastKronTorch()
  M = 10
  N = 5
  Ps = [8] * N
  Qs = [8] * N
  
  x = torch.ones((M, reduce((lambda a, b: a * b), Ps)), dtype=torch.float32).cuda()
  y = torch.zeros((M, reduce((lambda a, b: a * b), Qs)), dtype=torch.float32).cuda()
  fs = [torch.ones((Ps[0], Qs[0]), dtype=torch.float32).cuda() for i in range(0, N)]

  rs, ts = fastKron.gekmmSizes(x, fs)
  fastKron.gekmmTune(x, fs, y)

  t1 = torch.zeros(rs, dtype=torch.float32).cuda()

  fastKron.gekmm(x, fs, y, 1.0, 0.0, None, t1)
  print(y)