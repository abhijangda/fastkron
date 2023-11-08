import torch

M = 1024
N = 2
P = 128
Q = 128

def baseline(input, kronmats):
    outputKron = kronmats[0]
    for m in kronmats[1:]:
        outputKron = torch.kron(outputKron, m)
    return torch.matmul(input, outputKron)

X = torch.ones((M, P**N), dtype = torch.float).cuda()
Fs = []
for n in range(N):
  Fs.append(torch.ones((P, Q), dtype = torch.float).cuda())

# Y = baseline(X, Fs)

from fastkron import PyFastKron

p = PyFastKron()
(rs, ts) = p.resultTempSizes(X, Fs)
Y1 = torch.zeros(rs, dtype = torch.float).cuda()
T1 = torch.zeros(ts, dtype = torch.float).cuda()
T2 = torch.zeros(ts, dtype = torch.float).cuda()

p.kmm(X, Fs, Y1, T1, T2)
print(Y1[0])