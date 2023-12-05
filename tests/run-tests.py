#!/usr/bin/python

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
                    'gen-single-gpu-kernels'            : ['run-single-gpu-no-fusion-tests', 'run-single-gpu-fusion-tests'],
                    'gen-tuner-kernels'                 : ['run-single-gpu-tuner-tests'],
                    'gen-non-square-tuner-test-kernels' : ['run-single-gpu-non-square-tuner-tests'],
                    'gen-single-gpu-distinct-shapes'    : ['run-single-gpu-distinct-shapes'],
                    'gen-single-gpu-odd-shapes'         : ['run-single-gpu-odd-shapes'],
                    'gen-multi-gpu-tests-kernel'        : ['run-multi-gpu-nccl-no-fusion-tests', 'run-multi-gpu-p2p-no-fusion-tests'],
                    'gen-multi-gpu-tuner-kernels'       : ['run-multi-gpu-tuner-tests'],
                    'gen-multi-gpu-no-fusion-non-square-tests-kernel' : ['run-p2p-multi-gpu-no-fusion-non-square-tests', 'run-nccl-multi-gpu-no-fusion-non-square-tests'],
                    'gen-multi-gpu-distinct-shapes'     : ['run-p2p-multi-gpu-distinct-shapes', 'run-nccl-multi-gpu-distinct-shapes']
                    }

sorted_keys = sorted(list(gen_test_kernels.keys()))

for gen in sorted_keys:
    execute(f'make {gen}')
    for run in gen_test_kernels[gen]:
        print(f"========= Running {run[len('run-'):]} =========")
        output = execute(f'make {run} -j')
        if 'FAILED' in output:
            print(output)
