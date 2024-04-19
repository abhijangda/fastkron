import argparse
import subprocess
import re
import sys

def run_command(command):
  (s, o) = subprocess.getstatusoutput(command)
  if s != 0:
    print (f"Running {command}\n", o)
  return o

def tune(ms, n, p, q, opX, opF, fuse, backend, elemtype):
  # run_command(f'python gen_tuner_kernels.py -backend {backend} -same-factors {n} {p},{q} -opX {opX} -opF {opF}')
  # run_command(f'cd ../build/ && make benchmark_{backend} -j')
  # for m in ms:
    o = run_command(f'OMP_NUM_THREADS=64 taskset -c 0-64 ../build/tests/benchmarks/benchmark_{backend} -m {m} -n {n} -p {p} -q {q} -r {10} -w {10} -t {elemtype} --tune --backend {backend} {"--fuse" if fuse else ""}')
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
  if args.backend is None or args.backend.lower() not in ['cuda', 'x86', 'hip', 'arm']:
    print(f"Invalid backend: {args.backend}")
    sys.exit(0)

  print("Print and tune kernels for ", parsed_cases)
  assert args.opX in ["N", "T"]
  assert args.opF in ["N", "T"]

  # run_command(f'python ./gen_tuner_kernels.py -backend {args.backend} -same-factors 3 128,128 -same-factors 3 64,64 -same-factors 4 32,32 -same-factors 5 16,16 -same-factors 6 8,8 -same-factors 10 4,4 -same-factors 20 2,2 -opX N -opF N -types {"double"}')
  # run_command(f'cd ../build/ && make benchmark_{args.backend} -j')

  for p in [2,4,8,16,32,64,128]:
    for q in [2,4,8,16,32,64,128]:
      if p != q:
        continue
<<<<<<< HEAD
      for n in range(1,13):
=======
      for n in range(1,16):
>>>>>>> 709dc55 (OpF = T for cpu general kernel)
        for m in [1,4,16,64,256]:
          if m*(p**n) > 1024*1024*1024 or m*(q**n) > 1024*1024*1024 or p**n < 64 or q**n < 64:
            continue
          # if p <= 32 and q <= 32:
          #   continue
<<<<<<< HEAD
          if p <= 32:
            tune(m, n, p, q, args.opX, args.opF, False, args.backend, "float")
=======
          tune(m, n, p, q, args.opX, args.opF, False, args.backend, "float")
>>>>>>> 709dc55 (OpF = T for cpu general kernel)
          tune(m, n, p, q, args.opX, args.opF, True, args.backend, "float")