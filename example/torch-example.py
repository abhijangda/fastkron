from functools import reduce

try:
  import torch
  import pyfastkron.fastkrontorch as fk

  M = 10
  N = 5
  Ps = [8] * N
  Qs = [8] * N

  # CUDA example in PyTorch
  x = torch.ones((M, reduce((lambda a, b: a * b), Ps)), dtype=torch.float32).cuda()
  y = torch.zeros((M, reduce((lambda a, b: a * b), Qs)), dtype=torch.float32).cuda()
  fs = [torch.ones((Ps[0], Qs[0]), dtype=torch.float32).cuda() for i in range(0, N)]

  y = fk.gekmm(x, fs)

  print(y)
except:
  pass

try:
  import numpy as np
  import pyfastkron.fastkronnumpy as fk

  M = 10
  N = 5
  Ps = [8] * N
  Qs = [8] * N

  # Numpy example
  x = np.ones((M, reduce((lambda a, b: a * b), Ps)), dtype=np.double)
  fs = [np.ones((Ps[0], Qs[0]), dtype=np.double) for i in range(0, N)]
  z = np.ones((M, reduce((lambda a, b: a * b), Qs)), dtype=np.double)

  y = fk.gekmm(x, fs, alpha=2.0, beta=1.0, z=z)

  print(y)
except:
  pass