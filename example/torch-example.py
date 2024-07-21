from functools import reduce
import torch

from pyfastkron import FastKronTorch

fastKron = FastKronTorch()
M = 10
N = 5
Ps = [8] * N
Qs = [8] * N

# CUDA example in PyTorch
x = torch.ones((M, reduce((lambda a, b: a * b), Ps)), dtype=torch.float32).cuda()
y = torch.zeros((M, reduce((lambda a, b: a * b), Qs)), dtype=torch.float32).cuda()
fs = [torch.ones((Ps[0], Qs[0]), dtype=torch.float32).cuda() for i in range(0, N)]

rs, ts = fastKron.gekmmSizes(x, fs)

t1 = torch.zeros(rs, dtype=torch.float32).cuda()

fastKron.gekmm(x, fs, y, 1.0, 0.0, None, t1)

print(y)

# X86 example in PyTorch

x = x.cpu()
y = y.cpu()
fs = [f.cpu() for f in fs]

t1 = t1.cpu()

fastKron.gekmm(x, fs, y, 1.0, 0.0, None, t1)

print(y)