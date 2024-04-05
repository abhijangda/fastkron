import argparse
import subprocess
import re
import sys

def run_command(command):
  (s, o) = subprocess.getstatusoutput(command)
  if s != 0:
    print (f"Running {command}\n", o)
  return o

def tune(m, n, p, q, opX, opF, backend, fuse):
  o = run_command(f'../build/tests/benchmarks/benchmark_cuda -m {m} -n {n} -p {p} -q {q} -r {10} -w {10} -t float --tune --backend {backend} {"--fuse" if fuse else ""}')
  o = o[o.find('Minimum Time'):]
  
  kernelSeries = re.findall(r'\s*\[(\d+), (\d+)\] = (\d+) (.+) runs', o)
  allKernelsExec = []
  for kernelExec in kernelSeries:
    start,end,k,kernel = kernelExec
    allKernelsExec += [(int(end) - int(start) + 1, int(k), kernel)]

  allKernelsExec = list(set(allKernelsExec))
  gflops = re.findall(r'GFLOPS: (\d+\.\d+)', o)
  print(f"{m}x{p**n}*({p}x{q}^{n})",allKernelsExec, gflops)

  return allKernelsExec #["{"+f'Matrix({m},{p**n}), \"{kernels[0]}\"'+"},"]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-backend'           , type=str)
  parser.add_argument('-opX'               , required=True , type=str)
  parser.add_argument('-opF'               , required=True , type=str)
  
  args = parser.parse_args()

  assert args.opX in ["N", "T"]
  assert args.opF in ["N", "T"]
  run_command(f'python ./gen_tuner_kernels.py -backend {args.backend} -same-factors 2 128,128 -same-factors 2 64,64 -same-factors 3 32,32 -same-factors 5 16,16 -same-factors 6 8,8 -same-factors 10 4,4 -same-factors 20 2,2 -opX N -opF N -match-configs-file kernels/best-kernels/a100-kernels')
  run_command(f'cd ../build/ && make benchmark_{args.backend} -j')

  shapeToKernel = {}

  for p in [4,8,16,32,64,128]:
    for q in [4,8,16,32,64,128]:
      for n in range(1,20):
        for m in [1,4,16,64,256,1024]:
          if m*(p**n) > 2*1024*1024*1024 or m*(q**n) > 2*1024*1024*1024:
            continue
          if (p!=q):
            continue
          for canfuse in [False, True]:
            if canfuse and (p != q or p > 32):
              continue
            allKernelsExec = tune(m, n, p, q, args.opX, args.opF, args.backend, canfuse)
            for kernelExec in allKernelsExec:
              key = f"Factor({p},{q}),{kernelExec[0]}"
              if key not in shapeToKernel:
                shapeToKernel[key] = []
              if len(shapeToKernel[key]) > 0 and (shapeToKernel[key][-1][2] == kernelExec[2] and shapeToKernel[key][-1][1] <= kernelExec[1] and shapeToKernel[key][-1][0] <= m):
                continue
              shapeToKernel[key] += [(m, kernelExec[1], kernelExec[2])]
  
  maplines = ""
  indent = 1
  
  for k,vs in shapeToKernel.items():
    maplines += "  " * indent + "{\n"
    indent += 1
    maplines += "  " * indent + "{"+k+"}," + " {\n"
    indent += 1
    for v in vs:
      maplines += "  " * indent + "{" + f'Matrix({v[0]}, {v[1]}), "{v[2]}"' + "}" + ",\n"
    indent -= 1
    maplines += "  " * indent + "}\n"
    indent -= 1
    maplines += "  " * indent + "},\n"


  print(maplines)