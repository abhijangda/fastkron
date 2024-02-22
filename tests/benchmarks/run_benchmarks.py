import os
import subprocess
import re
import shutil
import math
from functools import reduce
import time
import torch

backend = 'CUDA'

def run_command(command):
  (s, o) = subprocess.getstatusoutput(command)
  if s != 0:
    print (f"Running {command}\n", o)
    assert False
  return o

class Shape:
  def __init__(self, m, n, p, q):
    self.m = m
    self.n = n
    self.ps = [p for i in range(0, n)]
    self.qs = [q for i in range(0, n)]
    self.k = reduce((lambda a, b: a * b), self.ps)

  def flops(self):
    ops = 0
    k = self.k
    for p,q in zip(reversed(self.ps),reversed(self.qs)):
      k = (k/p)*q
      ops += k * p
    return 2 * self.m * ops

  def __repr__(self):
    return f"{self.m}_{self.ps[0]}x{self.qs[0]}^{self.n}"
  def __str__(self):
    return repr(self)
  def __eq__(self, other):
    return repr(self) == repr(other)

class GPyTorchEval:
  def __init__(self):
    pass

  def run_single_gpu(self, shape):
    r = self._run_kron(shape)
    torch.cuda.empty_cache()
    return r

  def _run_kron(self, shape):
    import gpytorch as gp
    import torch
    factors = []
    for p,q in zip(shape.ps, shape.qs):
      f = torch.ones(p, q, dtype=float)
      if backend == 'CUDA':
        f = f.cuda()
      factors += [f] 
    x = torch.ones(shape.m, shape.k, dtype=float)
    if backend == 'CUDA':
      x = x.cuda()
    kp = gp.lazy.KroneckerProductLazyTensor(*factors)
    def run_case(r):
        t1 = time.time()
        for i in range(r):
            y = x @ kp
        torch.cuda.synchronize()
        t2 = time.time()
        return (t2-t1)*1000/r
    
    run_case(10)
    t = run_case(20)
    flops = shape.flops()/(t/1e3)
    return (flops/1e9,)
  
class FastKronEval:
  def __init__(self):
    pass

  def setup_cmake(self):
    d = os.getcwd()
    if os.path.exists('build/'):
      shutil.rmtree('build/')
    os.mkdir('build/')
    os.chdir('build/')
    backend_flags = '-DNVCC_GENCODE_FLAGS="-gencode arch=compute_80,code=sm_80" -DENABLE_CUDA=ON' if backend == 'CUDA' else ""
    run_command('cmake .. ' + backend_flags)
    os.chdir(d)

  def gen_kernels(self, shape, opX, opF, distKernels):
    run_command("python3 src/gen_tuner_kernels.py -distinct-factors " + \
                str(shape.n) + " " + " ".join([f"{pq[0]},{pq[1]}" for pq in zip(shape.ps, shape.qs)]) + \
                " -opX " + opX + " -opF " + opF + \
                (" -dist-kernels " if distKernels else ""))

  def build_kron(self):
    run_command("cd build && make benchmark -j")

  def run_fastkron(self, shape, GM, GK, LocalKrons, opX, opF):
    kron = f"cd build && ./tests/benchmarks/benchmark -m {shape.m} -n {shape.n} -p {shape.ps[0]} -q {shape.qs[0]} -r 100 -w 10 -t float --tune --opx {opX} --opf {opF}"
    if GM * GK != 1:
      kron += f" --gpus {GM*GK} --GM {GM} --GK {GK} --gpuLocalKrons {LocalKrons}"
    kron += " --backend CUDA" if backend == 'CUDA' else ''

    o = run_command(kron + " --fuse")
    fused = re.findall(r"GFLOPS\: ([\d\.]+)", o)[0]
    fusedtime = re.findall(r"Time: ([\d\.]+) ms", o)[0]
    if shape.ps[0] <= 32:
      o = run_command(kron)
      wofuse = re.findall(r"GFLOPS\: ([\d\.]+)", o)[0]
      wofusetime = re.findall(r"Time: ([\d\.]+) ms", o)[0]
    else:
      wofuse = fused
      wofusetime = fusedtime

    if GM*GK == 1:
      return (shape, wofuse, fused)
    else:
      return (shape, GM, GK, wofuse, fused)

  def run_single_gpu(self, shape, opX, opF):
    self.gen_kernels(shape, opX, opF, False)
    self.setup_cmake()
    self.build_kron()
    return self.run_fastkron(shape, 1, 1, 1, opX, opF)

def run_single_gpu_nn():
  M = 1024
  cases = [
           Shape(M, 5, 8, 8),     Shape(M, 6, 8, 8),
           Shape(M, 4, 16, 16),   Shape(M, 5, 16, 16),
           Shape(M, 3, 32, 32),   Shape(M, 4, 32, 32),
           Shape(M, 2, 64, 64),   Shape(M, 3, 64, 64),
           Shape(M, 2, 128, 128), 
           Shape(320, 3, 128, 128)]

  M = 16
  cases += [Shape(M, 8, 8, 8),
           Shape(M, 6, 16, 16),
           Shape(M, 5, 32, 32),
           Shape(M, 4, 64, 64),
          #  Shape(M, 3, 128, 128)
           ]

  for shape in cases:
    fk = FastKronEval().run_single_gpu(shape, "N", "N")
    gp = GPyTorchEval().run_single_gpu(shape)
    print(" & ".join((str(p) for p in (fk + gp))))

def run_single_gpu_nt():
  M = 1024
  cases = [Shape(M, 6, 8, 8), Shape(M, 4, 32, 32),
           Shape(M, 3, 64, 64), Shape(320, 3, 128, 128)]

  M = 16
  cases += [Shape(M, 8, 8, 8),
           Shape(M, 6, 16, 16),
           Shape(M, 5, 32, 32),
           Shape(M, 4, 64, 64),
          #  Shape(M, 3, 128, 128)
           ]

  for shape in cases:
    fk = FastKronEval().run_single_gpu(shape,"N", "T")
    print(" & ".join((str(p) for p in (fk))))
  
def run_single_gpu_tt():
  M = 1024
  cases = [Shape(M, 6, 8, 8), Shape(M, 4, 32, 32),
           Shape(M, 3, 64, 64), Shape(320, 3, 128, 128)]

  M = 16
  cases += [Shape(M, 8, 8, 8),
           Shape(M, 6, 16, 16),
           Shape(M, 5, 32, 32),
           Shape(M, 4, 64, 64),
          #  Shape(M, 3, 128, 128)
           ]

  for shape in cases:
    fk = FastKronEval().run_single_gpu(shape,"T", "T")
    print(" & ".join((str(p) for p in (fk))))

def multi_gpu(scaling):
  cases = []
  M_64 = 128
  cases += [Shape(M_64, 4, 64, 64)]
  M_128 = 8
  cases += [Shape(M_128, 4, 128, 128)]
  
  # run_command("make gen-multi-gpu-tests-kernel")

  for shape in cases:
    GMs = [1, 2, 2, 4, 4]
    GKs = [1, 1, 2, 2, 4]
    fk = FastKronEval()
    fk.gen_kernels(shape, "N", "N", True)
    fk.setup_cmake()
    fk.build_kron()
    for j,gpus in enumerate([1, 2, 4, 8]):
      gm = GMs[j]
      gk = GKs[j]
      shapeGM = Shape(shape.m * (gpus if scaling == "weak" else 1), shape.n, shape.ps[0], shape.qs[0])
      LocalKrons = shapeGM.n if gk == 1 else shapeGM.n - 2
      r = fk.run_fastkron(shapeGM, gm, gk, LocalKrons)
      print(" & ".join((str(p) for p in r)))

if False:
  print("------- Single GPU NN-------")
  print(" & ".join(("M_PxQ^N", "FastKron-wo-fuse", "FastKron", "GPyTorch")))
  run_single_gpu_nn()

  print("------- Single GPU NT-------")
  print(" & ".join(("M_PxQ^N", "FastKron-wo-fuse", "FastKron")))
  run_single_gpu_nt()

  print("------- Single GPU TT-------")
  print(" & ".join(("M_PxQ^N", "FastKron-wo-fuse", "FastKron")))
  run_single_gpu_tt()

  print("------- Multi GPU Weak Scaling --------")
  print(" & ".join(("M_PxQ^N", "GM", "GK", "FastKron-wo-fuse", "FastKron")))
  multi_gpu("weak")

  print("------- Multi GPU Strong Scaling --------")
  print(" & ".join(("M_PxQ^N", "GM", "GK", "FastKron-wo-fuse", "FastKron")))
  multi_gpu("strong")

print("------ Multi CPU NN------")
backend = "x86"
gp = GPyTorchEval().run_single_gpu(Shape(1024, 3, 64, 64))
print(gp)