from functools import reduce
import numpy as np

import pyfastkron.fastkronnumpy as fk

def reference(x, fs, trX, trF):
  if trX:
    x = x.T
  if trF:
    fs = [f.T for f in fs]
  outputKron = fs[0]
  for m in fs[1:]:
      outputKron = np.kron(outputKron, m)
  return np.matmul(x, outputKron)

def run(m, n, p, q, dtype, device, trX, trF):
  #Using integer values instead of real numbers because 
  #floating point is not associative
  xshape = (m, p**n) if not trX else (p**n, m)
  fsshape = (p, q) if not trF else (q, p)
  x = np.random.randint(0, high=5,size=xshape).astype(dtype)
  fs = [np.random.randint(0, high=5,size=fsshape).astype(dtype) for i in range(n)]

  y = fk.gekmm(x, fs, 1.0, 0.0, None, trX=trX, trF=trF)

  ref = reference(x, fs, trX, trF)
  val = np.isclose(y, ref, rtol=1e-04).all().item()

  assert val

def device_tests(device):
  run(1024, 5, 8, 8, np.float32, device, False, False)
  run(10, 5, 6, 6, np.float32, device, True, False)
  run(11, 10, 3, 3, np.double, device, False, True)
  run(200, 2, 32, 32, np.double, device, True, True)

def test_cpu():
  if fk.__fastkronnumpy.hasX86():
    device_tests("cpu")

if __name__ == "__main__":
  test_cpu()