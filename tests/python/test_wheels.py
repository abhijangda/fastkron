from functools import reduce
import numpy as np

import pyfastkron.fastkronnumpy as fk

def product(values):
  return reduce((lambda a, b: a * b), values)

def transpose(m):
  axis = tuple(range(len(m.shape[:-2]))) + \
         (len(m.shape) - 1, len(m.shape) - 2)
  return m.transpose(axis)

def reference(mmtype, x, fs):
  batchKron = fs[0].shape[:-2]
  if len(batchKron) == 0:
    outputKron = fs[0]
    for m in fs[1:]:
        outputKron = np.kron(outputKron, m)
  else:
    batchDims = product(batchKron)
    for i,f in enumerate(fs):
      fs[i] = fs[i].reshape((batchDims,) + f.shape[-2:])

    output = fs[0]
    for f in fs[1:]:
      prev = output
      output = np.ndarray(shape=(batchDims, prev.shape[-2] * f.shape[-2], prev.shape[-1] * f.shape[-1]),
                          dtype=f.dtype)
      for b in range(batchDims):
        output[b:] = np.kron(prev[b,:], f[b,:])
    outputKron = output.reshape(batchKron + (output.shape[-2], output.shape[-1]))

  if mmtype == "mkm":
    return np.matmul(x, outputKron)
  elif mmtype == "kmm":
    return np.matmul(outputKron, x)

def run(mmtype, m, n, p, q, dtype, device, trX, trF,
        high=5, batchDimX=[], batchDimFPre=[], batchDimZ=[]):
  #Using integer values instead of real numbers because
  #floating point is not associative
  if mmtype == "mkm":
    xshape = [m, p**n] if not trX else [p**n, m]
  elif mmtype == "kmm":
    xshape = [p**n, m] if not trX else [m, p**n]

  if m == 1:
    if trX:
      xshape = [xshape[0],]
    else:
      xshape = [xshape[1],]

  xshape = list(batchDimX) + xshape

  if mmtype == "mkm":
    fshape = [p, q] if not trF else [q, p]
  elif mmtype == "kmm":
    fshape = [q, p] if not trF else [p, q]

  if q == 1:
    if trF:
      fshape = [fshape[1],]
    else:
      fshape = [fshape[0],]

  fshape = list(batchDimFPre) + fshape

  zshape = list(batchDimZ)
  if mmtype == "mkm":
    zshape += [m,q**n]
  elif mmtype == "kmm":
    zshape += [q**n,m]

  x = np.random.randint(0,   high=high,size=xshape).astype(dtype)
  fs = [np.random.randint(0, high=high,size=fshape).astype(dtype)\
        for i in range(n)]
  z = np.random.randint(0,   high=high,size=zshape).astype(dtype)
  if trX:
    x = transpose(x)
  if trF:
    fs = [transpose(f) for f in fs]

  alpha = 3.0
  beta = 0.0

  if mmtype == "mkm":
    y = fk.gemkm(x, fs, alpha, beta, z)
  elif mmtype == "kmm":
    y = fk.gekmm(fs, x, alpha, beta, z)

  ref = alpha * reference(mmtype, x, fs) + beta * z
  val = np.isclose(y, ref, rtol=1e-04).all().item()
  print(52)
  assert val

def device_tests(device):
  for mmtype in ["mkm", "kmm"]:
    run(mmtype, 8, 3, 8, 8, np.float32, device, False, False)
    run(mmtype, 16, 3, 32, 32, np.float32, device, False, False)
    run(mmtype, 16, 2, 128, 127, np.float32, device, True, False)

    run(mmtype, 10, 5, 6, 6, np.float32, device, True, False)

    # #double
    run(mmtype, 11, 10, 3, 3, np.double, device, False, True)
    run(mmtype, 200, 2, 32, 32, np.double, device, True, True)

    #float16
    run(mmtype, 102, 4, 8, 8, np.float16, device, False, False, high=2)

def test_cpu():
  device_tests("cpu")

if __name__ == "__main__":
  test_cpu()