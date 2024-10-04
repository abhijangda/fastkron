from functools import reduce
import torch

import pyfastkron.fastkrontorch as fk


def product(values):
  return reduce((lambda a, b: a * b), values)

def reference(x, fs, trX, trF):
  if trX:
    x = x.T
  if trF:
    fs = [f.T for f in fs]
  outputKron = fs[0]
  for m in fs[1:]:
      outputKron = torch.kron(outputKron, m)
  return torch.matmul(x, outputKron)

def run(m, n, p, q, dtype, device, trX, trF, high=5, m1=1, q1=1):
  #Using integer values instead of real numbers because 
  #floating point is not associative
  xshape = (m, p**n) if not trX else (p**n, m)
  if m == 1:
    if trX:
      xshape = (xshape[0],)
    else:
      xshape = (xshape[1],)

  if m1 != 1 and not trX:
    xshape = (m1,) + xshape
 
  fshape = (p, q) if not trF else (q, p)
  if q == 1:
    if trF:
      fshape = (fshape[1],)
    else:
      fshape = (fshape[0],)

  # if q1 != 1 and not trF:
  #   fshape = (q1,) + fshape

  x = torch.randint(high=high,size=xshape, dtype=dtype).to(device)
  fs = [torch.randint(high=high,size=fshape, dtype=dtype).to(device) for i in range(n)]

  y = fk.gekmm(x, fs, 1.0, 0.0, None, trX=trX, trF=trF)

  ref = reference(x, fs, trX, trF)
  val = torch.isclose(y, ref, rtol=1e-04).all().item()

  assert val

def run_2(m, n, ps, qs, dtype, device, trX, trF, high=5, m1=1, q1=1):
  #Using integer values instead of real numbers because 
  #floating point is not associative
  xshape = (m, product(ps)) if not trX else (product(ps), m)
  if m == 1:
    if trX:
      xshape = (xshape[0],)
    else:
      xshape = (xshape[1],)

  if m1 != 1 and not trX:
    xshape = (m1,) + xshape
  assert (trF is False)
  # if q1 != 1 and not trF:
  #   fshape = (q1,) + fshape

  x = torch.randint(high=high,size=xshape, dtype=dtype).to(device)
  fs = [0,0,0]
  fs[0] = torch.tensor([[ 2.0000,  0.0000,  0.0000],
        [ 0.0000,  1.7321,  0.0000],
        [ 1.0000, -0.5774,  1.2910]], dtype=dtype)
  fs[1] = torch.tensor([[ 2.0000,  0.0000,  0.0000],
        [ 0.0000,  1.7321,  0.0000],
        [ 1.0000, -0.5774,  1.2910]])
  # fs = [torch.randint(high=high,size=(ps[i], qs[i]), dtype=dtype).to(device) for i in range(n)]

  y = fk.gekmm(x, fs, 1.0, 0.0, None, trX=trX, trF=trF)

  ref = reference(x, fs, trX, trF)
  val = torch.isclose(y, ref, rtol=1e-04).all().item()
  print(val)
  assert val

def device_tests(device):
  run_2(24, 3, [3,2,4],[3,2,4], torch.float32, device, False, False)
  return
  run(1024, 5, 8, 8, torch.float32, device, False, False)
  run(10, 5, 6, 6, torch.float32, device, True, False)
  run(10, 5, 6, 6, torch.float32, device, True, False)

  run(10, 3, 32, 8, torch.float32, device, False, False)
  run(10, 3, 32, 8, torch.float32, device, True, True)

  run(10, 3, 32, 1, torch.float32, device, False, False)

  #3-D x
  run(10, 5, 6, 6, torch.float32, device, False, False, m1=2, q1=1)
  
  #Double
  run(11, 10, 3, 3, torch.double, device, False, True)
  run(200, 2, 32, 32, torch.double, device, True, True)
  
  #Float16
  run(102, 4, 8, 8, torch.float16, device, False, False, high=2)
  run(10, 3, 16, 8, torch.float16, device, True, False, high=2)

def test_cuda():
  if fk.__fastkrontorch.hasCUDA():
    device_tests("cuda")

def test_cpu():
  if fk.__fastkrontorch.hasX86():
    device_tests("cpu")

if __name__ == "__main__":
  # test_cuda()
  test_cpu()