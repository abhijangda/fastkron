import PyFastKronWrapper
import torch
from functools import reduce

print(dir(PyFastKronWrapper.init))
i = PyFastKronWrapper.init(8)
print(i)
print(PyFastKronWrapper.gekmmSizes.__doc__)

PyFastKronWrapper.initCUDA(i, None, 1, 1, 1, 1)

M = 10
N = 5
Ps = [8] * N
Qs = [8] * N

r, _ = PyFastKronWrapper.gekmmSizes(i, M, N, Ps, Qs)

x = torch.ones((M, reduce((lambda a, b: a * b), Ps)), dtype=torch.float32).cuda()
y = torch.zeros((M, reduce((lambda a, b: a * b), Qs)), dtype=torch.float32).cuda()
fs = [torch.ones((Ps[0], Qs[0]), dtype=torch.float32).cuda() for i in range(0, N)]

z = torch.ones(r, dtype=torch.float32).cuda()

PyFastKronWrapper.sgekmm(i, PyFastKronWrapper.Backend.CUDA, M, N, Ps, Qs, x.data_ptr(), PyFastKronWrapper.Op.N, [f.data_ptr() for f in fs], PyFastKronWrapper.Op.N, y.data_ptr(), 1.0, 0.0, 0, z.data_ptr(), 0)

print(y)