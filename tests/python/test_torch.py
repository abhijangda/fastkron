from functools import reduce
import torch

import pyfastkron.fastkrontorch as fk

def reference(x, fs, trX, trF):
  if trX:
    x = x.T
  if trF:
    fs = [f.T for f in fs]
  outputKron = fs[0]
  for m in fs[1:]:
      outputKron = torch.kron(outputKron, m)
  return torch.matmul(x, outputKron)

def run(m, n, p, q, dtype, device, trX, trF, high=5):
  #Using integer values instead of real numbers because 
  #floating point is not associative
  xshape = (m, p**n) if not trX else (p**n, m)
  fsshape = (p, q) if not trF else (q, p)
  x = torch.randint(high=high,size=xshape, dtype=dtype).to(device)
  fs = [torch.randint(high=high,size=fsshape, dtype=dtype).to(device) for i in range(n)]

  y = fk.gekmm(x, fs, 1.0, 0.0, None, trX=trX, trF=trF)

  ref = reference(x, fs, trX, trF)
  val = torch.isclose(y, ref, rtol=1e-04).all().item()
  if not val:
    print (y)
    print (ref)
  assert val

def device_tests(device):
  run(1024, 5, 8, 8, torch.float32, device, False, False)
  run(10, 5, 6, 6, torch.float32, device, True, False)
  run(11, 10, 3, 3, torch.double, device, False, True)
  run(200, 2, 32, 32, torch.double, device, True, True)
  run(102, 4, 8, 8, torch.float16, device, False, False, high=2)
  run(10, 3, 16, 8, torch.float16, device, True, False, high=2)

def test_cuda():
  if fk.__fastkrontorch.hasCUDA():
    device_tests("cuda")

def test_cpu():
  if fk.__fastkrontorch.hasX86():
    device_tests("cpu")

if __name__ == "__main__":
  test_cuda()
  test_cpu()