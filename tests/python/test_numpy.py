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

def run(m, n, p, q, dtype, device, trX, trF, high=5):
  #Using integer values instead of real numbers because 
  #floating point is not associative
  xshape = (m, p**n) if not trX else (p**n, m)
  fsshape = (p, q) if not trF else (q, p)
  x = np.random.randint(0, high=high,size=xshape).astype(dtype)
  fs = [np.random.randint(0, high=high,size=fsshape).astype(dtype) for i in range(n)]

  y = fk.gekmm(x, fs, 1.0, 0.0, None, trX=trX, trF=trF)

  ref = reference(x, fs, trX, trF)
  val = np.isclose(y, ref, rtol=1e-04).all().item()

  assert val

def device_tests(device):
  run(1024, 5, 8, 8, np.float32, device, False, False)
  run(10, 5, 6, 6, np.float32, device, True, False)
  run(11, 10, 3, 3, np.double, device, False, True)
  run(200, 2, 32, 32, np.double, device, True, True)
  run(102, 4, 8, 8, np.float16, device, False, False, high=2)
  run(10, 3, 16, 8, np.float16, device, True, False, high=2)

def test_cpu():
  if fk.__fastkronnumpy.hasX86():
    device_tests("cpu")

if __name__ == "__main__":
  test_cpu()