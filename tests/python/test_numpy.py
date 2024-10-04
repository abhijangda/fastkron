from functools import reduce
import numpy as np

import pyfastkron.fastkronnumpy as fk

def product(values):
  return reduce((lambda a, b: a * b), values)

def reference(x, fs, trX, trF):
  # if trX:
  #   x = x.transpose()
  # if trF:
  #   fs = [f.T for f in fs]
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

  return np.matmul(x, outputKron)

def run(m, n, p, q, dtype, device, trX, trF, high=5, batchDimX=[], batchDimFPre=[], batchDimFPost=[]):
  #Using integer values instead of real numbers because 
  #floating point is not associative
  xshape = [m, p**n] if not trX else [p**n, m]
  if m == 1:
    if trX:
      xshape = [xshape[0],]
    else:
      xshape = [xshape[1],]

  if not trX:
    xshape = batchDimX + xshape

  fshape = [p, q] if not trF else [q, p]
  if q == 1:
    if trF:
      fshape = [fshape[1],]
    else:
      fshape = [fshape[0],]
  
  fshape = batchDimFPre + fshape + batchDimFPost 

  x = np.random.randint(0, high=high,size=xshape).astype(dtype)
  fs = [np.random.randint(0, high=high,size=fshape).astype(dtype)\
        for i in range(n)]

  y = fk.gekmm(x, fs, 1.0, 0.0, None, trX=trX, trF=trF)
  # y = reference_2(x, fs, trX, trF)
  for i,f in enumerate(fs):
    fs[i] = fs[i].reshape(fshape)

  ref = reference(x, fs, trX, trF)
  val = np.isclose(y, ref, rtol=1e-04).all().item()
  print(52)
  assert val

def device_tests(device):
  run(128, 5, 8, 8, np.float32, device, False, False, batchDimX=[2,], batchDimFPre=[])
  run(128, 5, 8, 8, np.float32, device, False, False, batchDimX=[2,3], batchDimFPre=[2,3])
  run(128, 5, 8, 8, np.float32, device, False, False, batchDimX=[2,1,], batchDimFPre=[3,])
  run(128, 5, 8, 8, np.float32, device, False, False, batchDimX=[2,1,], batchDimFPre=[2,4,])
  run(128, 5, 8, 8, np.float32, device, False, False, batchDimX=[2,], batchDimFPre=[3,2,])

  # run(128, 5, 8, 8, np.float32, device, True, True, batchDimX=[2,], batchDimFPre=[])
  # run(128, 5, 8, 8, np.float32, device, True, True, batchDimX=[2,1,], batchDimFPre=[3,])
  # run(128, 5, 8, 8, np.float32, device, True, True, batchDimX=[2,1,], batchDimFPre=[2,4,])
  # run(128, 5, 8, 8, np.float32, device, True, True, batchDimX=[2,], batchDimFPre=[3,2,])
  # run(1024, 5, 8, 8, np.float32, device, False, False, batchDimX=[], batchDimFPre=[2,], batchDimFPost=[3,])
  # run(1024, 5, 8, 8, np.float32, device, False, False, batchDimX=[2,1], batchDimFPre=[2,], batchDimFPost=[3,])
  # return
  run(128, 5, 8, 8, np.float32, device, False, False)
  # run(10, 5, 6, 6, np.float32, device, True, False)

  #3-D x
  run(10, 5, 6, 6, np.float32, device, False, False, m1=2, q1=1)

  #double
  run(11, 10, 3, 3, np.double, device, False, True)
  run(200, 2, 32, 32, np.double, device, True, True)

  #float16
  run(102, 4, 8, 8, np.float16, device, False, False, high=2)
  run(10, 3, 16, 8, np.float16, device, True, False, high=2)

def test_cpu():
  device_tests("cpu")

if __name__ == "__main__":
  test_cpu()