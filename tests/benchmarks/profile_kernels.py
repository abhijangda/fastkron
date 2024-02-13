import subprocess
import re
import os
import sys
import copy

ENV_VAR = "PYTHONPATH=/home/parasail/.local/lib/python3.8/site-packages/:$PYTHONPATH LD_LIBRARY_PATH=%s:"%(os.getcwd())
NVPROF_BIN="/usr/local/cuda/bin/nvprof"
RequiredMetrics = ["shared_load_transactions_per_request", "gld_transactions_per_request", "global_load_requests"]
NVPROF_FLAGS="--metrics "+",".join(RequiredMetrics)

NVPROF_COMMAND=" ".join([NVPROF_BIN, NVPROF_FLAGS])

npoints = 320
cases = {4:(7, 10), 8:(3,6), 16:(2,5), 32: (2,4)}

FASTKRON_BIN = "./kron"
FASTKRON_FLAGS = "-b 320 -f %d -s %d -t float -r 1 -w 0"
GPYTORCH_BIN = "python3 kronecker-model.py"
GPYTORCH_FLAGS = "320 %d %d 1" 
MetricValues = {
    "fastkron": {m : 0 for m in RequiredMetrics},
    "cublas": {m : 0 for m in RequiredMetrics},
    "transpose": {m : 0 for m in RequiredMetrics},
}
metric_values = []

def parseMetricLine(line):
    # return re.findall(r'(\d+)\s+([\_\w\d\-]+)\s+([\w+\s\d\(\)\-]+)\s+(.+?)\s+(.+?)\s+(.+)',line)
    line = line.split("  ")
    new_line = []
    for m in line:
        if m != "":
            new_line += [m.strip()]
    return new_line

def parseMetric(metrics_value, nvprofOutput):
    o = nvprofOutput[nvprofOutput.find("Metric result:"):]
    lines = list(re.findall(r'.+', o))
    l = 0
    while l < len(lines):
        line = lines[l]
        if "Kernel:" in line:
            kernel_name = ""
            if "kronGemmKernel" in line:
                kernel_name = "fastkron"
            elif "gemm" in line:
                kernel_name = "cublas"
            elif "at" in line and "native" in line and "elementwise_kernel" in line:
                kernel_name = "transpose"
            else:
                print(line)
                sys.exit(0)
            print(kernel_name)
            l += 1
            line = lines[l]
            while l < len(lines) and "Kernel:" not in lines[l]:
                line = lines[l]
                parsed = parseMetricLine(line)
                if parsed[1] in RequiredMetrics:
                    if parsed[1]=="global_load_requests":
                        metrics_value[kernel_name][parsed[1]] += float(parsed[5])*int(parsed[0])
                    else:
                        metrics_value[kernel_name][parsed[1]] = max(float(parsed[5]), metrics_value[kernel_name][parsed[1]]) 
                l += 1
        if l < len(lines) and "Kernel:" not in lines[l]:
            l += 1

for g in cases:
    for d in range(cases[g][0], cases[g][1]+1):
        metric_value = copy.deepcopy(MetricValues)
        metric_values += [(g,d,metric_value)]
        if True:
            fastkron_command = FASTKRON_BIN + " " + FASTKRON_FLAGS%(d, g)
            command = "sudo " + ENV_VAR + " " + NVPROF_COMMAND + " " + fastkron_command
            print(command)
            (s, o) = subprocess.getstatusoutput(command)
            if s != 0:
                print(o)
            else:
                parseMetric(metric_value, o)

        if True:
            gpytorch_command = GPYTORCH_BIN + " " + GPYTORCH_FLAGS%(d, g)
            command = "sudo " + ENV_VAR + " " + NVPROF_COMMAND + " " + gpytorch_command
            print(command)
            (s, o) = subprocess.getstatusoutput(command)
            if s != 0:
                print(o)
            else:
                parseMetric(metric_value, o)

firstRow = []
print(metric_values)
kernels = list(MetricValues.keys())
metrics = RequiredMetrics
for kernel in kernels:
    for metric in RequiredMetrics:
        firstRow += [kernel+"-"+metric]

print("&".join(["g", "d"] + firstRow))

for metric_value in metric_values:
    row = []
    for kernel in kernels:
        for metric in RequiredMetrics:
            row += [str(metric_value[2][kernel][metric])]
    print("&".join([str(metric_value[0]), str(metric_value[1])]+row))