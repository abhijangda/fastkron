from functools import reduce
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

y = fk.gekmm(x, fs, 1.0, 0.0, None)

print(y)