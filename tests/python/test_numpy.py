from functools import reduce
import numpy as np

from pyfastkron import FastKronNumpy

fastKron = FastKronNumpy()

def reference(x, fs):
  outputKron = fs[0]
  for m in fs[1:]:
      outputKron = np.kron(outputKron, m)
  return np.matmul(x, outputKron)

def run(m, n, p, q, dtype, device):
  #Using integer values instead of real numbers because 
  #floating point is not associative
  x = np.random.randint(0, high=5,size=(m, p**n)).astype(dtype)
  y = np.zeros((m, q**n)).astype(dtype)
  fs = [np.random.randint(0, high=5,size=(p, q)).astype(dtype) for i in range(n)]
  rs, ts = fastKron.gekmmSizes(x, fs)
  t = np.zeros(rs, dtype=dtype)

  fastKron.gekmm(x, fs, y, 1.0, 0.0, None, t, None)

  ref = reference(x, fs)
  val = np.isclose(y, ref, rtol=1e-04).all().item()

  assert val

def device_tests(device):
  run(1024, 5, 8, 8, np.float32, device)
  run(10, 5, 6, 6, np.float32, device)
  run(11, 10, 3, 3, np.double, device)
  run(200, 2, 32, 32, np.double, device)

def test_cpu():
  if fastKron.hasX86():
    device_tests("cpu")

if __name__ == "__main__":
  test_cpu()