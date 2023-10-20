import os
import subprocess
import re

def run_command(command):
  (s, o) = subprocess.getstatusoutput(command)
  if s != 0:
    print (f"Running {command}\n", o)
    assert False
  return o

def gen_kernels(shape):
  run_command("python3 src/gen_tuner_kernels.py -distinct-factors " + \
              str(shape.n) + " " + " ".join([f"{pq[0]},{pq[1]}" for pq in zip(shape.ps, shape.qs)]))

def build_kron():
  run_command("make kron -j")

def run_kron(shape):
  kron = f"./kron -m {shape.m} -n {shape.n} -p {shape.ps[0]} -q {shape.qs[0]} -r 20 -w 10 -t float --tune"
  o = run_command(kron)
  wofuse = re.findall(r"GFLOPS\: ([\d\.]+)", o)[0]
  o = run_command(kron + " --fuse")
  fused = re.findall(r"GFLOPS\: ([\d\.]+)", o)[0]
  
  print(shape, wofuse, fused)

class Shape:
  def __init__(self, m, n, p, q):
    self.m = m
    self.n = n
    self.ps = [p for i in range(0, n)]
    self.qs = [q for i in range(0, n)]

  def __repr__(self):
    return f"{self.m}_{self.ps[0]}x{self.qs[0]}^{self.n}"

M = 1024

cases = [Shape(M, 5, 8, 8),     Shape(M, 6, 8, 8),
         Shape(M, 4, 16, 16),   Shape(M, 5, 16, 16),
         Shape(M, 3, 32, 32),   Shape(M, 4, 32, 32),
         Shape(M, 2, 64, 64),   Shape(M, 3, 64, 64),
         Shape(M, 2, 128, 128), Shape(M, 3, 128, 128)]

for shape in cases:
  gen_kernels(shape)
  build_kron()
  run_kron(shape)