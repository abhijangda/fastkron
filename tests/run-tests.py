#!/usr/bin/python
import os
import shutil
import subprocess
import sys

def execute(command):
  print(f"Executing {command}")
  (s, o) = subprocess.getstatusoutput(command)
  if s != 0:
    print(f'Error in executing "{command}"')
    print(o)
    assert False
  return o

backend = sys.argv[1].lower()
single_or_multi = sys.argv[2].lower()

all_backends = ['cuda', 'x86']# 'hip', 'arm']

assert backend in (all_backends + ['all'])
assert single_or_multi in ['single', 'multi', 'all']

if backend == 'all':
  backends = all_backends
else:
  backends = [backend]

if single_or_multi == 'all':
  single_or_multi = ['single', 'multi']
else:
  single_or_multi = [single_or_multi]

test_cases = {k : {'single':{}, 'multi': {}} for k in all_backends}

test_cases['cuda']['single'] = {'gen-single-gpu-kernels' : ['single-gpu-cuda-NN', 'single-gpu-cuda-TT']}
test_cases['cuda']['multi'] = {
  'gen-multi-cuda-tests-kernel'         : ['DIST_COMM=NCCL multi-cuda-no-fusion-tests',
                                           'DIST_COMM=P2P multi-cuda-no-fusion-tests'],
  'gen-multi-cuda-tuner-kernels'        : ['multi-cuda-tuner-tests'],
  'gen-multi-cuda-no-fusion-non-square-tests-kernel' : ['DIST_COMM=P2P multi-cuda-no-fusion-non-square-tests',
                                                        'DIST_COMM=NCCL multi-cuda-no-fusion-non-square-tests'],
  'gen-multi-cuda-distinct-shapes'      : ['DIST_COMM=P2P multi-cuda-distinct-shapes',
                                           'DIST_COMM=NCCL multi-cuda-distinct-shapes']
}

test_cases['x86']['single'] = {'gen-x86-kernels' : ['x86-cpu-NN', 'x86-cpu-TT']}

if os.path.exists("build/"):
  shutil.rmtree("build/")

if not os.path.exists("build/"):
  os.mkdir("build/")

os.chdir("build/")
cmake = ""
for b in backends:
  cmake += f"-DENABLE_{b.upper()}=ON "

if 'cuda' in backends:
  cmake += '-DCMAKE_CUDA_FLAGS="-Xptxas -v -O3"'

execute(f'cmake .. {cmake}')

for mode in single_or_multi:
  if mode == 'single':
    for backend in backends:
      for case in test_cases[backend][mode]:
        execute(f'make {case}')

    execute(f'make -j')

    for backend in backends:
      for case in test_cases[backend][mode]:
        for run in test_cases[backend][mode][case]:
            output = execute(f'make {run if " " not in run else run.split(" ")[1]} -j')
            output = execute((f"TUNE=0 tests/{backend}/"+run) if ' ' not in run else run.replace(' ', f' tests/{backend}/'))
            if 'FAILED' in output:
              print(output)
  if mode == 'multi':
    for case in test_cases['cuda'][mode]:
      execute(f'make {case}')
      execute(f'make -j')
      for run in test_cases['cuda'][mode][case]:
        output = execute(f'make {run if " " not in run else run.split(" ")[1]} -j')
        output = execute((f"tests/{backend}/"+run) if ' ' not in run else run.replace(' ', f' tests/{backend}/'))
        if 'FAILED' in output:
          print(output)