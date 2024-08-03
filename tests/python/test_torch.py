from functools import reduce
import torch

import pyfastkron.fastkrontorch as fk

def reference(input, kronmats):
  outputKron = kronmats[0]
  for m in kronmats[1:]:
      outputKron = torch.kron(outputKron, m)
  return torch.matmul(input, outputKron)

def run(m, n, p, q, dtype, device):
  #Using integer values instead of real numbers because 
  #floating point is not associative
  x = torch.randint(high=5,size=(m, p**n), dtype=dtype).to(device)
  fs = [torch.randint(high=5,size=(p, q), dtype=dtype).to(device) for i in range(n)]

  y = fk.gekmm(x, fs, 1.0, 0.0, None)

  ref = reference(x, fs)
  val = torch.isclose(y, ref, rtol=1e-04).all().item()

  assert val

def device_tests(device):
  run(1024, 5, 8, 8, torch.float32, device)
  run(10, 5, 6, 6, torch.float32, device)
  run(11, 10, 3, 3, torch.double, device)
  run(200, 2, 32, 32, torch.double, device)

def test_cuda():
  if fk.__fastkrontorch.hasCUDA():
    device_tests("cuda")

def test_cpu():
  if fk.__fastkrontorch.hasX86():
    device_tests("cpu")

if __name__ == "__main__":
  test_cuda()
  test_cpu()