from functools import reduce
import torch
import torch.autograd

import pyfastkron.fastkrontorch as fk

def product(values):
  return reduce((lambda a, b: a * b), values)

def transpose(m):
  axis = tuple(range(len(m.shape[:-2]))) + \
         (len(m.shape) - 1, len(m.shape) - 2)
  return m.mT #torch.transpose(m, -2, -1)

def reference(mmtype, x, fs, device):
  batchKron = fs[0].shape[:-2]
  if len(batchKron) == 0:
    outputKron = fs[0]
    for m in fs[1:]:
        outputKron = torch.kron(outputKron, m)
  else:
    batchDims = product(batchKron)
    fs = [f.reshape((batchDims,) + f.shape[-2:]) for f in fs]

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

def run(mmtype, m, n, ps, qs, dtype, device, trX, trF,
        high=5, batchDimX=[], batchDimFPre=[], batchDimZ=[],
        gradcheck=False):

  if type(ps) is int:
    ps = [ps]

  if len(ps) == 1:
    ps = [ps[0]]*n

  if type(qs) is int:
    qs = [qs]

  if len(qs) == 1:
    qs = [qs[0]]*n

  #Using integer values instead of real numbers because
  #floating point is not associative
  if mmtype == "mkm":
    xshape = [m, product(ps)] if not trX else [product(ps), m]
  elif mmtype == "kmm":
    xshape = [product(ps), m] if not trX else [m, product(ps)]

  xshape = list(batchDimX) + xshape

  if mmtype == "mkm":
    fshape = [[ps[i], qs[i]] if not trF else [qs[i], ps[i]] for i in range(n)]
  elif mmtype == "kmm":
    fshape = [[qs[i], ps[i]] if not trF else [ps[i], qs[i]] for i in range(n)]

  fshape = [list(batchDimFPre) + fshape[i] for i in range(n)]

  zshape = list(batchDimZ)

  if mmtype == "mkm":
    zshape += [m,product(qs)]
  elif mmtype == "kmm":
    zshape += [product(qs),m]

  x = torch.randint(0, high=high,size=xshape, dtype=dtype).to(device)
  fs = [torch.randint(0, high=high,size=fshape[i], dtype=dtype).to(device)\
        for i in range(n)]
  z = torch.randint(0,high=high, size=zshape, dtype=dtype).to(device)

  if trX:
    x = transpose(x)
  if trF:
    fs = [transpose(f) for f in fs]

  fs = tuple(fs)

  if not gradcheck:
    alpha = 3.0
    beta = 1.0
    if mmtype == "mkm":
      y = fk.gemkm(x, fs, alpha, beta, z)
    elif mmtype == "kmm":
      y = fk.gekmm(fs, x, alpha, beta, z)

    if x.device.type == "cuda":
      torch.cuda.synchronize()
    ref = alpha * reference(mmtype, x, fs, device)
    if z != None:
      ref += beta * z

    val = torch.isclose(y, ref).all().item()
    print(101)
    assert val
  else:
    x.requires_grad = True
    for f in fs:
      f.requires_grad = True
    if mmtype == "kmm":
      torch.autograd.gradcheck(fk.KMM.apply, (x,*fs), eps=1e-5, atol=1e-4)
    elif mmtype == "mkm":
      torch.autograd.gradcheck(fk.MKM.apply, (x,*fs), eps=1e-5, atol=1e-4)
    print(116)

def device_tests(device):
  with torch.no_grad():
    for mmtype in ["mkm", "kmm"]:
      run(mmtype, 16, 5, 8, 8, torch.float32, device, False, False)
      run(mmtype, 10, 5, 6, 6, torch.float32, device, True, False)

      run(mmtype, 16, 5, 8, 8, torch.float32, device, False, False, batchDimX=[2,], batchDimFPre=[], batchDimZ=[2,])
      run(mmtype, 32, 5, 8, 8, torch.float32, device, False, False, batchDimX=[2,3], batchDimFPre=[2,3])
      run(mmtype, 8, 5, 8, 8, torch.float32, device, False, False, batchDimX=[2,1,], batchDimFPre=[3,])
      run(mmtype, 2, 5, 8, 8, torch.float32, device, False, False, batchDimX=[2,1,], batchDimFPre=[2,4,])
      run(mmtype, 32, 4, 8, 8, torch.float32, device, False, False, batchDimX=[3,3,1,], batchDimFPre=[3,1,4,])
      run(mmtype, 24, 4, 8, 8, torch.float32, device, False, False, batchDimX=[2,], batchDimFPre=[3,2,])

      run(mmtype, 16, 4, 8, 8, torch.float32, device, False, False, batchDimX=[2,], batchDimFPre=[3,2,], batchDimZ=[3,1])

      run(mmtype, 16, 4, 16, 8, torch.float32, device, True, True, batchDimX=[2,], batchDimFPre=[])
      run(mmtype, 32, 5, 8, 8, torch.float32, device, True, True, batchDimX=[2,1,], batchDimFPre=[3,])
      run(mmtype, 13, 5, 8, 8, torch.float32, device, True, True, batchDimX=[2,1,], batchDimFPre=[2,4,])
      run(mmtype, 19, 3, 8, 32, torch.float32, device, True, True, batchDimX=[2,], batchDimFPre=[3,2,])

      #double
      run(mmtype, 11, 10, 3, 3, torch.double, device, False, True)
      run(mmtype, 200, 2, 32, 32, torch.double, device, True, True)

      run(mmtype, 128, 5, 8, 8, torch.double, device, True, True, batchDimX=[2,1,], batchDimFPre=[2,4,])

      #float16
      run(mmtype, 102, 4, 8, 8, torch.float16, device, False, False, high=2)
      run(mmtype, 102, 4, 8, 8, torch.float16, device, False, False, high=2, batchDimX=[2,], batchDimFPre=[])
      run(mmtype, 102, 4, 8, 8, torch.float16, device, False, False, high=2, batchDimX=[2,1,], batchDimFPre=[3,])
      run(mmtype, 10, 3, 16, 8, torch.float16, device, True, False, high=2)

  for mmtype in ["mkm", "kmm"]:
    run(mmtype, 5, 4, 6, 6, torch.double, device, False, True, batchDimX=[1,], batchDimFPre=[2,], gradcheck=True)
    run(mmtype, 5, 4, 4, 6, torch.double, device, True, True, batchDimX=[1,], batchDimFPre=[2,], gradcheck=True)

def test_cuda():
  if torch.cuda.is_available():
    device_tests("cuda")

def test_cpu():
    device_tests("cpu")

if __name__ == "__main__":
  test_cuda()
  test_cpu()