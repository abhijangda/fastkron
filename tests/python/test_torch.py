from functools import reduce
import torch

import pyfastkron.fastkrontorch as fk

def product(values):
  return reduce((lambda a, b: a * b), values)

def transpose(m):
  axis = tuple(range(len(m.shape[:-2]))) + \
         (len(m.shape) - 1, len(m.shape) - 2)
  return torch.transpose(m, -2, -1)


def reference(mmtype, x, fs, trX, trF, device):
  if trX:
    x = transpose(x)
  if trF:
    fs = [transpose(f) for f in fs]

  batchKron = fs[0].shape[:-2]
  if len(batchKron) == 0:
    outputKron = fs[0]
    for m in fs[1:]:
        outputKron = torch.kron(outputKron, m)
  else:
    batchDims = product(batchKron)
    for i,f in enumerate(fs):
      fs[i] = fs[i].reshape((batchDims,) + f.shape[-2:])

    output = fs[0]
    for f in fs[1:]:
      prev = output
      s = (batchDims, prev.shape[-2] * f.shape[-2], prev.shape[-1] * f.shape[-1])
      output = torch.zeros((batchDims, prev.shape[-2] * f.shape[-2], prev.shape[-1] * f.shape[-1]),
                           dtype=f.dtype).to(device)
      for b in range(batchDims):
        output[b,:,:] = torch.kron(prev.contiguous()[b,:,:], f.contiguous()[b,:,:])
    outputKron = output.reshape(batchKron + (output.shape[-2], output.shape[-1]))

  if mmtype == "mkm":
    return torch.matmul(x, outputKron)
  elif mmtype == "kmm":
    return torch.matmul(outputKron, x)

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
  
  x = torch.randint(0, high=high,size=xshape, dtype=dtype).to(device)
  fs = [torch.randint(0, high=high,size=fshape, dtype=dtype).to(device)\
        for i in range(n)]
  z = torch.randint(0,high=high, size=zshape, dtype=dtype).to(device)

  alpha = 1.0
  beta = 2.0

  if mmtype == "mkm":
    y = fk.gemkm(x, fs, alpha, beta, z, trX=trX, trF=trF)
  elif mmtype == "kmm":
    y = fk.gekmm(fs, x, alpha, beta, z, trX=trX, trF=trF)

  ref = alpha * reference(mmtype, x, fs, trX, trF, device) + beta * z
  val = torch.isclose(y, ref, rtol=1e-04).all().item()
  print(52)
  assert val

def device_tests(device):
  for mmtype in ["kmm"]: #"mkm", 
    # run(mmtype, 16, 5, 8, 8, torch.float32, device, False, False)
    # run(mmtype, 10, 5, 6, 6, torch.float32, device, True, False)

    # run(mmtype, 16, 5, 8, 8, torch.float32, device, False, False, batchDimX=[2,], batchDimFPre=[], batchDimZ=[2,])
    # run(mmtype, 32, 5, 8, 8, torch.float32, device, False, False, batchDimX=[2,3], batchDimFPre=[2,3])
    # run(mmtype, 8, 5, 8, 8, torch.float32, device, False, False, batchDimX=[2,1,], batchDimFPre=[3,])
    run(mmtype, 1, 5, 8, 8, torch.float32, device, False, False, batchDimX=[2,1,], batchDimFPre=[2,4,])
    run(mmtype, 32, 4, 8, 8, torch.float32, device, False, False, batchDimX=[3,3,1,], batchDimFPre=[3,1,4,])
    run(mmtype, 24, 4, 8, 8, torch.float32, device, False, False, batchDimX=[2,], batchDimFPre=[3,2,])

    run(mmtype, 16, 4, 8, 8, torch.float32, device, False, False, batchDimX=[2,], batchDimFPre=[3,2,], batchDimZ=[3,1])

    run(mmtype, 16, 5, 8, 8, torch.float32, device, True, True, batchDimX=[2,], batchDimFPre=[])
    run(mmtype, 32, 5, 8, 8, torch.float32, device, True, True, batchDimX=[2,1,], batchDimFPre=[3,])
    run(mmtype,13, 5, 8, 8, torch.float32, device, True, True, batchDimX=[2,1,], batchDimFPre=[2,4,])
    run(mmtype, 29, 5, 8, 8, torch.float32, device, True, True, batchDimX=[2,], batchDimFPre=[3,2,])

    #double
    run(mmtype, 11, 10, 3, 3, torch.double, device, False, True)
    run(mmtype, 200, 2, 32, 32, torch.double, device, True, True)

    run(mmtype, 128, 5, 8, 8, torch.double, device, True, True, batchDimX=[2,1,], batchDimFPre=[2,4,])

    continue
    #float16
    run(102, 4, 8, 8, torch.float16, device, False, False, high=2)
    run(102, 4, 8, 8, torch.float16, device, False, False, high=2, batchDimX=[2,], batchDimFPre=[])
    run(102, 4, 8, 8, torch.float16, device, False, False, high=2, batchDimX=[2,1,], batchDimFPre=[3,])
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