import argparse
import subprocess
import re
import sys

def run_command(command):
  (s, o) = subprocess.getstatusoutput(command)
  if s != 0:
    print (f"Running {command}\n", o)
  return o

def tune(ms, n, p, q, opX, opF, backend):
  # run_command(f'python gen_tuner_kernels.py -backend {backend} -same-factors {n} {p},{q} -opX {opX} -opF {opF}')
  # run_command(f'cd ../build/ && make benchmark_{backend} -j')
  for m in ms:
    o = run_command(f'../build/tests/benchmarks/benchmark_cuda -m {m} -n {n} -p {p} -q {q} -r {10} -w {10} -t float --tune --backend {backend} --fuse')
    o = o[o.find('Minimum Time'):]
    kernels = re.findall(r'\d+\s(.+)\sruns\sfor', o)
    kernels = set(kernels)
    gflops = re.findall(r'GFLOPS: (\d+\.\d+)', o)
    print(f"{m}x{p**n}*({p}x{q}^{n})",kernels, gflops)

def parse_same_factors(case):
  n = int(case[0])
  assert len(case[1:]) == 1
  split = case[1].split(',')
  p = int(split[0]) #[int(split[0]) for i in range(n)]
  q = int(split[1]) #[int(split[1]) for i in range(n)]
  
  # k = compute_k(ps, qs)
  return (n, p, q)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-distinct-factors'  , required=False, type=str,  action='append',     nargs="+")
  parser.add_argument('-same-factors'      , required=False, type=str,  action='append',     nargs="+")
  parser.add_argument('-m'                 , required=True, type=int, action='append', nargs="+")
  parser.add_argument('-opX'               , required=True , type=str)
  parser.add_argument('-opF'               , required=True , type=str)
  parser.add_argument('-num-kernels'       , required=False, type=int,   default=10000)
  parser.add_argument('-backend'           , type=str)
  args = parser.parse_args()
  parsed_cases = []

  if args.same_factors is not None:  
    for case in args.same_factors:
      try:
        parsed_cases += [parse_same_factors(case)]
      except Exception as e:
        print(f"Invalid case: {case}")
        print(e)
        sys.exit(0)
  else:
    for n in range(2, 4):
      for p,q in zip([32,64,128], [32,64,128]):
          parsed_cases += [(n, p, q)]
    print(len(parsed_cases), parsed_cases)

  if args.backend is None or args.backend.lower() not in ['cuda', 'x86', 'hip', 'arm']:
    print(f"Invalid backend: {args.backend}")
    sys.exit(0)

  print("Print and tune kernels for ", parsed_cases)
  assert args.opX in ["N", "T"]
  assert args.opF in ["N", "T"]

  for case in parsed_cases:
    ms = list(args.m[0])
    if case[0] == 1024:
      ms = [1,2,4,8,16, 32, 64, 128, 256,512,1024,2048,4096]
    # print (m, case)
    tune(ms, case[0], case[1], case[2], args.opX, args.opF, args.backend)