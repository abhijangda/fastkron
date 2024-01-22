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
                    'gen-single-gpu-kernels'            : ['single-gpu-no-fusion-tests', 'single-gpu-fusion-tests'],
                    'gen-single-gpu-NT-kernels'         : ['single-gpu-NT-tests'],
                    'gen-tuner-kernels'                 : ['single-gpu-tuner-tests'],
                    'gen-non-square-tuner-test-kernels' : ['single-gpu-non-square-tuner-tests'],
                    'gen-single-gpu-distinct-shapes'    : ['single-gpu-distinct-shapes'],
                    'gen-single-gpu-odd-shapes'         : ['single-gpu-odd-shapes'],
                    # 'gen-multi-gpu-tests-kernel'        : ['DIST_COMM=NCCL multi-gpu-no-fusion-tests', 'DIST_COMM=P2P multi-gpu-no-fusion-tests'],
                    # 'gen-multi-gpu-tuner-kernels'       : ['multi-gpu-tuner-tests'],
                    # 'gen-multi-gpu-no-fusion-non-square-tests-kernel' : ['DIST_COMM=P2P multi-gpu-no-fusion-non-square-tests', 'DIST_COMM=NCCL multi-gpu-no-fusion-non-square-tests'],
                    # 'gen-multi-gpu-distinct-shapes'     : ['DIST_COMM=P2P multi-gpu-distinct-shapes', 'DIST_COMM=NCCL multi-gpu-distinct-shapes']
                  }

sorted_keys = sorted(list(gen_test_kernels.keys()))

if os.path.exists("build/"):
    shutil.rmtree("build/")

if not os.path.exists("build/"):
    os.mkdir("build/")

os.chdir("build/")
execute("cmake ..")

for gen in sorted_keys:
    print(f"========= Running {gen} =========")
    execute(f'make {gen}')
    for run in gen_test_kernels[gen]:
        output = execute(f'make {run if " " not in run else run.split(" ")[1]} -j')
        output = execute(("./"+run) if ' ' not in run else run.replace(' ', ' ./'))
        if 'FAILED' in output:
            print(output)
        