import argparse
import subprocess
import re
import sys

def run_command(command):
  (s, o) = subprocess.getstatusoutput(command)
  if s != 0:
    print (f"Running {command}\n", o)
  return o

def tune(m, n, p, q, opX, opF, backend):
  o = run_command(f'../build/tests/benchmarks/benchmark_cuda -m {m} -n {n} -p {p} -q {q} -r {10} -w {10} -t float --tune --backend {backend} --fuse')
  o = o[o.find('Minimum Time'):]
  kernels = re.findall(r'\d+\s(.+)\sruns\sfor', o)
  kernels = list(set(kernels))
  gflops = re.findall(r'GFLOPS: (\d+\.\d+)', o)
  print(f"{m}x{p**n}*({p}x{q}^{n})",kernels, gflops)

  return ["{"+f'Matrix({m},{p**n}), \"{kernels[0]}\"'+"},"]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-backend'           , type=str)
  parser.add_argument('-opX'               , required=True , type=str)
  parser.add_argument('-opF'               , required=True , type=str)
  
  args = parser.parse_args()

  assert args.opX in ["N", "T"]
  assert args.opF in ["N", "T"]

  run_command(f'cd ../build/ && make benchmark_{args.backend} -j')

  maplines = ""
  indent = 1
  for p,q in zip([64,128],[64,128]): #2,4,8,16,32
    maplines += "  " * indent + "{\n"
    indent += 1
    maplines += "  " * indent + f"Factor({p},{q})"+",{\n"
    indent += 1
    for n in range(2, 10):
      if p**n > 1024*1024 or q**n > 1024*1024 or q**n < 64 or p**n < 64:
        continue
      for m in [1024] : #1,2,4,8,16,32,64,128,256,512,
        maplines += "  " * indent + "".join(tune(m, n, p, q, args.opX, args.opF, args.backend)) + "\n"
    indent -= 1
    maplines += "  " * indent + "}\n"
    indent -= 1
    maplines += "  " * indent + "},\n"


  print(maplines)