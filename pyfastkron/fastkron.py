from functools import reduce
import ctypes

def to_ctype_array(elems, ctype_t):
  return (ctype_t * len(elems))(*elems)

def product(values):
  reduce((lambda a, b: a * b), values)

class FastKronOp:
  def __init__(self, op):
    self.op = op

FastKronOpT = FastKronOp(2)
FastKronOpN = FastKronOp(1)

class PyFastKronWrapper:
  def __init__(self):
    self.libKron = ctypes.CDLL("libFastKron.so")

    cppHandleTy = ctypes.c_ulong

    self.initFn = self.libKron.fastKronInit
    self.initFn.argtypes = [ctypes.POINTER(cppHandleTy), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    self.initFn.restype = ctypes.c_int
    
    self.destroyFn = self.libKron.fastKronDestroy
    self.destroyFn.argtypes = [cppHandleTy]
    self.destroyFn.restype = ctypes.c_int

    self.gekmmSizesFn = self.libKron.gekmmSizes
    self.gekmmSizesFn.argtypes = [cppHandleTy, ctypes.c_uint, ctypes.c_uint,
                                  ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint),
                                  ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]

    self.cpp_handle = ctypes.c_ulong(0)
    if self.initFn(ctypes.byref(self.cpp_handle), 1, -1, -1, -1) != 0:
      print("error")

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

  def gekmmSizes(self, m, n, ps, qs):
    assert len(ps) == len(qs)
    assert len(ps) == n

    resultSize = ctypes.c_size_t(0)
    tempSize   = ctypes.c_size_t(0)
    self.gekmmSizesFn(self.cpp_handle, ctypes.c_uint(m), ctypes.c_uint(n), 
                      to_ctype_array(ps, ctypes.c_uint), to_ctype_array(qs, ctypes.c_uint),
                      ctypes.byref(resultSize), ctypes.byref(tempSize))
    return resultSize.value, tempSize.value
  
  def sgekmm(self, m, n, ps, qs, x, opX, fs, opFs, y, alpha, beta, z, t1, t2, stream):
    assert len(ps) == len(qs)
    assert len(ps) == n
    assert len(fs) == n

    return self.sgekmmFn(self.cpp_handle, m, n,
                         to_ctype_array(ps, ctypes.c_uint), to_ctype_array(qs, ctypes.c_uint), 
                         ctypes.c_void_p(x), opX.op, 
                         to_ctype_array([ctypes.c_void_p(f) for f in fs], ctypes.c_void_p), opFs.op,
                         ctypes.c_void_p(y), ctypes.c_float(alpha), ctypes.c_float(beta),
                         ctypes.c_void_p(z), ctypes.c_void_p(t1), ctypes.c_void_p(t2), stream)

  def sgekmmTune(self, m, n, ps, qs, opX, opFs, stream):
    return self.sgekmmTuneFn(self.cpp_handle, m, n,
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
    self.pyfastkron = PyFastKronWrapper()
  
  def ps(self, fs):
    return [f.shape[0] for f in fs]
  
  def qs(self, fs):
    return [f.shape[0] for f in fs]

  def fptrs(self, fs):
    return [f.data_ptr() for f in fs]

  def _check(self, x, fs, y):
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == product(self.ps(fs))
    assert y.shape[1] == product(self.qs(fs))
    assert x.dtype    == y.dtype
    assert x.dtype    == fs[0].dtype
    assert len(set([f.dtype for f in fs])) == 1

  def gekmmSizes(self, x, fs):
    self._check(x, fs)
    return self.pyfastkron.gekmmSizes(x.shape[0], x.shape[1], self.ps(fs), self.qs(fs))

  def gekmmTune(self, x, fs, y, stream = torch.cuda.current_stream()):
    self._check(x, fs, y)
    fn = None
    if x.dtype == torch.float:
      fn = self.pyfastkron.sgekmmTune
    elif x.dtype == torch.int:
      fn = self.pyfastkron.igekmmTune
    #TODO
    fn(x.shape[0], len(fs), self.ps(fs), self.qs(fs),
        FastKronOpN, FastKronOpN, 
        ctypes.c_void_p(stream.cuda_stream))

  def gekmm(self, x, fs, y, alpha, beta, z, temp, stream = torch.cuda.current_stream()):
    self._check(x, fs, y)
    fn = None
    if x.dtype == torch.float:
      fn = self.pyfastkron.sgekmm
    elif x.dtype == torch.int:
      fn = self.pyfastkron.igekmm
    
    fn(x.shape[0], len(fs), self.ps(fs), self.qs(fs), 
        x.data_ptr(), FastKronOpN, self.fptrs(fs), FastKronOpN, 
        y, alpha, beta, z.data_ptr(), t1,
        ctypes.c_void_p(stream.cuda_stream))

if __name__ == "__main__":
  import torch
  fastKron = FastKronTorch()
  M = 10
  N = 10
  Ps = [2] * N
  Qs = [2] * N
  
  x = torch.ones((M, reduce((lambda a, b: a * b), Ps)), dtype=torch.float32).cuda()
  y = torch.zeros((M, reduce((lambda a, b: a * b), Qs)), dtype=torch.float32).cuda()
  fs = [torch.ones((Ps[0], Qs[0]), dtype=torch.float32).cuda() for i in range(0, N)]

  rs, ts = fastKron.gekmmSizes(x, fs)
  fastKron.sgekmmTune(x, fs, y)

  t1 = torch.ones((M, reduce((lambda a, b: a * b), Ps)), dtype=torch.float32).cuda()

  handle.sgekmm(M, N, Ps, Qs, x.data_ptr(), FastKronOpN, [f.data_ptr() for f in fs], FastKronOpN,
                y.data_ptr(), 1.0, 0.0, None, t1.data_ptr(), None, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
  print(y)