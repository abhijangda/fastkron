#!/usr/bin/python
import os
import shutil
import subprocess

def execute(command):
    print(f"Executing {command}")
    (s, o) = subprocess.getstatusoutput(command)
    if s != 0:
        print(f'Error in executing "{command}"')
        print(o)
        assert False
    return o

gen_test_kernels = {
                    # 'gen-single-cuda-kernels'               : ['single-cuda-no-fusion-tests', 'single-cuda-fusion-tests'],
                    # 'gen-single-cuda-non-square-TT-kernels' : ['single-cuda-non-square-TT-tests'],
                    # 'gen-tuner-kernels'                    : ['single-cuda-tuner-tests'],
                    # 'gen-non-square-kernels'    : ['single-cuda-non-square-tests'],
                    # 'gen-single-cuda-distinct-shapes'       : ['single-cuda-distinct-shapes'],
                    # 'gen-single-cuda-odd-shapes'            : ['single-cuda-odd-shapes'],

                    'gen-multi-cuda-tests-kernel'         : ['DIST_COMM=NCCL multi-cuda-no-fusion-tests', 'DIST_COMM=P2P multi-cuda-no-fusion-tests'],
                    # 'gen-multi-cuda-tuner-kernels'        : ['multi-cuda-tuner-tests'],
                    # 'gen-multi-cuda-no-fusion-non-square-tests-kernel' : ['DIST_COMM=P2P multi-cuda-no-fusion-non-square-tests', 'DIST_COMM=NCCL multi-cuda-no-fusion-non-square-tests'],
                    # 'gen-multi-cuda-distinct-shapes'      : ['DIST_COMM=P2P multi-cuda-distinct-shapes', 'DIST_COMM=NCCL multi-cuda-distinct-shapes']
                  }

sorted_keys = sorted(list(gen_test_kernels.keys()))

if os.path.exists("build/"):
    shutil.rmtree("build/")

if not os.path.exists("build/"):
    os.mkdir("build/")

os.chdir("build/")
execute("cmake .. -DENABLE_CUDA=ON")

for gen in sorted_keys:
    print(f"========= Running {gen} =========")
    execute(f'make {gen}')
    for run in gen_test_kernels[gen]:
        output = execute(f'make {run if " " not in run else run.split(" ")[1]} -j')
        output = execute(("tests/gpu/"+run) if ' ' not in run else run.replace(' ', ' tests/gpu/'))
        if 'FAILED' in output:
            print(output)
        