from functools import reduce

# try:
if True:
  # CUDA example in PyTorch

  import torch
  import pyfastkron.fastkrontorch as fk

  M = 10
  N = 5
  Ps = [8] * N
  Qs = [8] * N

  # KMM of X with F[0], F[1], ... F[N]
  x = torch.ones((M, reduce((lambda a, b: a * b), Ps)), dtype=torch.float16).cuda()
  fs = [torch.ones((Ps[0], Qs[0]), dtype=torch.float16).cuda() for i in range(0, N)]

  z = fk.gekmm(x, fs)

  print(z)

  # KMM of X.T with F[0] ... F[N]
  M = 10
  N = 3
  Ps = [16] * N
  Qs = [32] * N

  x = torch.ones((reduce((lambda a, b: a * b), Ps), M), dtype=torch.float32)
  fs = [torch.ones((Ps[0], Qs[0]), dtype=torch.float32) for i in range(0, N)]

  z = fk.gekmm(x, fs, trX = True)

  print(z)

# except:
#   pass

try:
  # Numpy example

  import numpy as np
  import pyfastkron.fastkronnumpy as fk

  M = 10
  N = 5
  Ps = [8] * N
  Qs = [8] * N

  # KMM of X with F[0] ... F[N]

  x = np.ones((M, reduce((lambda a, b: a * b), Ps)), dtype=np.double)
  fs = [np.ones((Ps[0], Qs[0]), dtype=np.double) for i in range(0, N)]
  y = np.ones((M, reduce((lambda a, b: a * b), Qs)), dtype=np.double)

  z = fk.gekmm(x, fs, alpha=2.0, beta=1.0, y=y)

  print(z)

  # KMM of X with F.T[0] ... F.T[N]
  M = 10
  N = 3
  Ps = [16] * N
  Qs = [32] * N

  x = np.ones((M, reduce((lambda a, b: a * b), Ps)), dtype=np.double)
  fs = [np.ones((Qs[0], Ps[0]), dtype=np.double) for i in range(0, N)]

  z = fk.gekmm(x, fs, alpha=2.0, beta= 1.0, y=None, trF = True)

  print(z)
except:
  pass