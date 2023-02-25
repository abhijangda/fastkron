import subprocess
import re

results = []

Ds = {
    2: [5, 18],
    4: [4, 10],
    8: [2, 6],
    16: [2, 5],
    32: [2, 5],
    64: [2, 3]
}
grids = [4,8,16,32,64]
for n in [1024]:
    for trace_samples in [255]:
        for grid in grids:
            for d in range(Ds[grid][0], Ds[grid][1] + 1):
                command = "python3 KISSGP.py %d %d %d %d"%(n, d, grid, trace_samples)
                print(command)
                s, o = subprocess.getstatusoutput(command)
                if s == 0:
                    total = re.findall(r'Total time\s*([\d\.]+)', o)[0]
                    kron = re.findall(r'Kron Time so far: \s*([\d\.]+)', o)
                    kron = kron[-1]
                    print(kron, total)
                    results += [(n, grid, d, trace_samples, kron, total)]
                else:
                    print (s)
                    results += [(n, grid, d, trace_samples, -1, -1)]

print("N & d & grid & TraceSamples & KronTime & TotalTime")
for r in results:
    print(", ".join([str(o) for o in r]))