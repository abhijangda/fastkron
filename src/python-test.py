import torch

M = 1024
N = 4
P = 16
Q = 16

def baseline(input, kronmats):
    outputKron = kronmats[0]
    for m in kronmats[1:]:
        outputKron = torch.kron(outputKron, m)
    return torch.matmul(input, outputKron)

X = torch.ones((M, P**4), dtype = torch.float).cuda()
Fs = []
for n in range(N):
  Fs.append(torch.ones((P, Q), dtype = torch.float).cuda())

Y = baseline(X, Fs)

from fastkron import PyFastKron

p = PyFastKron()

