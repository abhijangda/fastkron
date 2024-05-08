import os
import subprocess
import re
import shutil
import math
from functools import reduce
import time
import torch
import sys
import argparse

def run_command(command):
  (s, o) = subprocess.getstatusoutput(command)
  if s != 0:
    print (f"Running {command}\n", o)
    assert False
  return o

def total_gpu_memory():
  o = run_command("nvidia-smi -d MEMORY -q -i 0")
  mems = re.findall(r'Total\s*\:\s*(\d+)', o)
  mems = [int(m) for m in mems]
  return max(mems)

TuningModes = ['FullTune', 'FastTune', 'NoTune']

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
  def __init__(self, backend, elemtype):
    self.backend = backend
    if elemtype == "float":
      self.elemtype = torch.float
    elif elemtype == "double":
      self.elemtype = torch.double
    elif elemtype == "half":
      self.elemtype = torch.half
    if self.backend == "x86" and "OMP_NUM_THREADS" in os.environ:
      torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

  def run_single_gpu(self, shape):
    r = self._run_kron(shape)
    torch.cuda.empty_cache()
    return r

  def _run_kron(self, shape):
    import gpytorch as gp
    import torch
    factors = []
    for p,q in zip(shape.ps, shape.qs):
      f = torch.ones(p, q, dtype=self.elemtype)
      if self.backend == 'cuda':
        f = f.cuda()
      factors += [f] 
    x = torch.ones(shape.m, shape.k, dtype=self.elemtype)
    if self.backend == 'cuda':
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
    t = min(run_case(5), run_case(5), run_case(5), run_case(5), run_case(5))
    flops = shape.flops()/(t/1e3)
    return (flops/1e9,)
  
class FastKronEval:
  def __init__(self, backend, mode, elemtype):
    self.backend = backend
    self.tuningmode = mode
    self.built = False
    self.elemtype = elemtype
  
  def setup_cmake(self):
    if self.built == True:
      return
    d = os.getcwd()
    if os.path.exists('build/'):
      shutil.rmtree('build/')
    os.mkdir('build/')
    os.chdir('build/')
    if self.backend == "cuda":
      backend_flags = '-DCMAKE_CUDA_FLAGS="-Xptxas -v -O3" -DENABLE_CUDA=ON'
    elif self.backend == "x86":
      backend_flags = "-DENABLE_X86=ON"
    if self.tuningmode == "FullTune":
      backend_flags += " -DFULL_TUNE=ON"
    run_command('cmake .. ' + backend_flags)
    os.chdir(d)

  def gen_kernels(self, shape, opX, opF, distKernels):
    if self.tuningmode == 'FullTune':
      run_command("python3 src/gen_tuner_kernels.py -distinct-factors " + \
                  str(shape.n) + " " + " ".join([f"{pq[0]},{pq[1]}" for pq in zip(shape.ps, shape.qs)]) + \
                  " -opX " + opX + " -opF " + opF + \
                  (" -dist-kernels " if distKernels else "") + \
                  " -backend " + self.backend + " -types " + self.elemtype)
    elif self.tuningmode == 'FastTune' or self.tuningmode == 'NoTune':
      run_command("cd build/ && make gen-single-gpu-kernels")

  def build_kron(self):
    run_command(f"cd build && make benchmark_{self.backend} -j")

  def run_fastkron(self, shape, GM, GK, LocalKrons, opX, opF):
    kron = f"cd build && {'TUNE=0' if self.tuningmode=='NoTune' else ''} ./tests/benchmarks/benchmark_{self.backend} -m {shape.m} -n {shape.n} -p {shape.ps[0]} -q {shape.qs[0]} -r 10 -w {50 if self.tuningmode=='NoTune' else 20} -t {self.elemtype} --tune --opx {opX} --opf {opF}"
    if GM * GK != 1:
      kron += f" --gpus {GM*GK} --GM {GM} --GK {GK} --gpuLocalKrons {LocalKrons}"
    kron += " --backend " + self.backend

    o = run_command(kron + " --fuse")
    fused = re.findall(r"GFLOPS\: ([\d\.]+)", o)[0]
    fusedtime = re.findall(r"Time: ([\d\.]+) ms", o)[0]
    if shape.ps[0] <= 32 and shape.ps[0] == shape.qs[0]:
      o = run_command(kron)
      wofuse = re.findall(r"GFLOPS\: ([\d\.]+)", o)[0]
      wofusetime = re.findall(r"Time: ([\d\.]+) ms", o)[0]
    else:
      wofuse = fused
      wofusetime = fusedtime

    if GM*GK == 1:
      return (shape, float(wofuse), float(fused))
    else:
      return (shape, GM, GK, wofuse, fused)

  def run_single_gpu(self, shape, opX, opF):
    if self.built == False:
      self.gen_kernels(shape, opX, opF, False)
      self.setup_cmake()
      self.build_kron()
      if self.tuningmode == 'FastTune' or self.tuningmode == 'NoTune':
        self.built = True
    return self.run_fastkron(shape, 1, 1, 1, opX, opF)

def benchmark_single_gpu(device, opX, opF, mode, elemtype, dataset):
  print(f"------- Single {device.upper()} {elemtype.upper()} {mode} {opX}{opF} -------")
  device = device.lower()
  cases = []
  if dataset == "large":
    if device == "cuda":
      if total_gpu_memory() <= 16*1024:
        M = 256
        M2 = 128
      else:
        M = 1024
        M2 = 320
    else:
      M = 256
      M2 = 128

    cases = [
            Shape(M, 5, 8, 8),     Shape(M, 6, 8, 8),
            Shape(M, 4, 16, 16),   Shape(M, 5, 16, 16),
            Shape(M, 3, 32, 32),   Shape(M, 4, 32, 32),
            Shape(M, 2, 64, 64),   Shape(M, 3, 64, 64),
            Shape(M, 2, 128, 128), 
            Shape(M2, 3, 128, 128)]

    M = 16
    cases += [Shape(M, 8, 8, 8),
            Shape(M, 6, 16, 16),
            Shape(M, 5, 32, 32),
            Shape(M, 4, 64, 64),
            #  Shape(M, 3, 128, 128)
            ]
  elif dataset == "full":
    MAX_SIZE = 256 * 1024 * 1024 if device == "x86" else 1024*1024*1024
    factor = 2 if elemtype == "double" else 1
    for p in [2,4,8,16,32,64,128]:
      for q in [2,4,8,16,32,64,128]:
        for n in range(1,13 if device == "x86" else 20):
          for m in [1,4,16,64,256] + ([] if device == "x86" else [1024]):
            if m*(p**n) > MAX_SIZE//factor or m*(q**n) > MAX_SIZE//factor: # or p**n < 64 or q**n < 64:
              continue
            cases += [Shape(m, n, p, q)]

  fkeval = FastKronEval(device, mode, elemtype)
  fkeval.setup_cmake()
  for shape in cases:
    try:
      fk = fkeval.run_single_gpu(shape, opX, opF)
    except:
      fk = (shape, 1, 1)
    try:
      gp = GPyTorchEval(device, elemtype).run_single_gpu(shape)
    except:
      gp = (1, 1)
    print(str(fk[0]), " & ", " & ".join(("%.3f"%p) for p in (fk[1:] + gp + (fk[-1]/gp[-1],))))

def run_nn(device, mode, elemtype, dataset):
  benchmark_single_gpu(device, "N", "N", mode, elemtype, dataset)

def run_nt(device, mode):
  benchmark_single_gpu(device, "N", "T", mode, elemtype, dataset)

def run_tt(device, mode, elemtype, dataset):
  benchmark_single_gpu(device, "T", "T", mode, elemtype, dataset)

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


# print(" & ".join(("M_PxQ^N", "FastKron-wo-fuse", "FastKron", "GPyTorch")))
# #run_nn("cuda", tuningmode)

# print("------- Single GPU NT-------")
# print(" & ".join(("M_PxQ^N", "FastKron-wo-fuse", "FastKron")))
# # run_nt("cuda", tuningmode)

# print("------- Single GPU TT-------")
# print(" & ".join(("M_PxQ^N", "FastKron-wo-fuse", "FastKron")))
# run_tt("cuda", tuningmode)

# if False:
#   print("------- Multi GPU Weak Scaling --------")
#   print(" & ".join(("M_PxQ^N", "GM", "GK", "FastKron-wo-fuse", "FastKron")))
#   multi_gpu("weak")

#   print("------- Multi GPU Strong Scaling --------")
#   print(" & ".join(("M_PxQ^N", "GM", "GK", "FastKron-wo-fuse", "FastKron")))
#   multi_gpu("strong")

# # print("------ x86 NN------")
# # run_nn("x86")

# # print("------ x86 NT------")
# # run_nt("x86")

# print("------ x86 TT------")
# run_tt("x86")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-backends'    , required=True, type=str, nargs="+")
  parser.add_argument('-types'       , required=True, type=str, nargs="+")
  parser.add_argument("-tune-modes"  , required=True, type=str, nargs="+")
  parser.add_argument("-dataset"     , required=True, type=str)
  args = parser.parse_args()
  
  assert args.dataset in ["large", "full"]

  for backend in args.backends:
    for elemtype in args.types:
      for mode in args.tune_modes:
        assert backend in ["cuda", "x86", "hip"]
        assert elemtype in ["float", "int", "double"]
        assert mode in TuningModes

        run_nn(backend, mode, elemtype, args.dataset)
        #run_tt(backend, mode, elemtype, args.dataset)
