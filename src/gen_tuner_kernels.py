import argparse
import math 
import sys
import os
import shutil
import functools

print(os.getcwd())
assert 'src' in os.listdir('.')
kernel_dir = "src/device/kron-kernels/"

#Device limits
MAX_SHARED_MEM = 48 * 1024

def slurp(file):
  with open(file, "r") as f:
    return f.read()

def empty_dir(dir):
  for filename in os.listdir(dir):
    file_path = os.path.join(dir, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

def element_size(elem_type : str) -> int:
  if elem_type.lower() == "float":
    return 4

class KronMatMulShape:
  def __init__(self, m, k, n, p, q):
    self.m = m
    self.k = k
    self.n = n
    self.p = p
    self.q = q

class KernelConfig:  
  def __init__(self, shape : KronMatMulShape, kron_rows : int, kron_cols : int, 
               tileQ : int, tileP : int, tileM: int, 
               rowModTileIsZero : int, cRegRows: int, cRegCols: int, kEqVar: int,
               FusedKernel : int, dist: int, elemType : str):
    self.shape = shape
    self.num_threads = ((shape.k//shape.p)//cRegRows) * (tileQ//cRegCols)
    self.kron_rows = kron_rows
    self.kron_cols = kron_cols
    self.tileQ = tileQ
    self.tileP = tileP
    self.tileM = tileM
    self.rowModTileIsZero = rowModTileIsZero
    self.cRegRows = cRegRows
    self.cRegCols = cRegCols
    self.kEqVar = kEqVar
    self.fused_kernels = FusedKernel
    self.dist = dist
    self.elemType = "float"

    """Compute several constants of kernel
       Corresponds to line numbers in kernel.cuh
    """
    #Line 67
    self.wsz = (self.shape.k//self.shape.p)//self.cRegRows 
    self.shared_mem_usage = (self.tileM * self.shape.k + self.tileP * self.tileQ)*element_size(elemType)

  def __repr__(self):
    return f"{self.num_threads}, {self.shape.q}, {self.shape.p}, {self.tileQ}, {self.tileM}, {self.shape.k}, {self.cRegRows}, {self.cRegCols}, {self.fused_kernels}, {self.elemType}, {self.rowModTileIsZero}, {self.kEqVar}, {self.dist}"

  def kernelname(self):
    return repr(self).replace(", ", "_")

  def filename(self):
    return f"kernel_{self.kernelname()}.cu"

  def hostFuncName(self):
    return f"host_{self.kernelname()}"

  def hostFuncDecl(self):
    return f"void {self.hostFuncName()}(KernelParams<float, {self.fused_kernels}> params, FusedParams<float, {self.fused_kernels}> fusedParams, DistributedParams<float> distParams, dim3 grid, dim3 block, cudaStream_t stream)"

  def templateDecl(self):
    return f"float, float4, {self.num_threads}, RowParallelismTy::Low, {self.tileM}, {self.rowModTileIsZero}, {self.shape.k}, {self.shape.q}, {self.shape.p}, {self.tileQ}, {self.kEqVar}, 1, {self.cRegRows}, {self.cRegCols}, {self.tileP}, {self.fused_kernels}, {self.dist}"
  
  def kernelDecl(self):
    return f"kronGemmKernel<{self.templateDecl()}>"

  def __eq__(self, other):
    return repr(self) == repr(other)

  def kernelInfo(self):
    return "KernelInfo{"+\
            f"(void*){self.hostFuncName()},"+\
            repr(self).replace("float", "ElementType::Float") + "}"

  def isValid(self):
    return self.wsz > 0 and \
           self.shape.k > self.tileP and \
           self.num_threads >= 32 and self.num_threads <= 1024 and \
           self.shared_mem_usage <= MAX_SHARED_MEM and \
           self.cRegRows in [1, 2, 4] and \
           (self.rowModTileIsZero == 1 or (self.rowModTileIsZero == 0 and self.tileM > 1)) and \
           (self.fused_kernels == 1 or (self.fused_kernels > 1 and self.shape.p == self.tileP and self.shape.q == self.tileQ)) and \
           self.kEqVar in [0, 1] and self.dist in [0, 1] 
          #  and "128, 64, 64, 64, 2, 4096, 2, 16, 1, float, 1, 0" in repr(self)

  def __hash__(self):
    return hash(self.__repr__())

def all_sliced_mults(m, k, n, ps, qs):
  sliced_mults = []
  prevTmpK = k
  for i in range(n):
    f = n - i - 1
    sliced_mult = (m, prevTmpK, ps[f], qs[f])
    prevTmpK = (prevTmpK//ps[f])*qs[f]
    sliced_mults += [sliced_mult]
  sliced_mults = set(sliced_mults)
  return list(sliced_mults)

def generate_kernel_decls(cases, useFusion, useDistKernels, numKernels, onlySpecificConfigs):
  if not os.path.exists(kernel_dir):
    os.mkdir(kernel_dir)

  empty_dir(kernel_dir)
  configs = {}

  for (m, k, n, ps, qs) in cases:
    allSameShapes = len(set(ps + qs)) == 1
    __configs = []  
    for (_, _, p, q) in all_sliced_mults(m, k, n, ps, qs):
      TilePs = [min(p, 32)]
      TileQs = [2**i for i in range(1, max(2, int(math.log2(q)))+1)]
      TileKs = [2**i for i in range(1, max(2, int(math.log2(k)))+1)]
      TileMs = [1, 2]
      CRows = [2**i for i in range(0, max(0, int(math.log2(p)))+1)]
      CCols = [2**i for i in range(0, max(0, int(math.log2(q)))+1)]

      shape = KronMatMulShape(m, k, n, p, q)
      for tM in TileMs:
        for tQ in TileQs:
          for tK in TileKs:
            for regRows in CRows:
              for regCols in CCols:
                for tP in TilePs:
                  for rowModTileIsZero in [0, 1]:
                    for kEqVar in [0]:
                      fusedCases = range(1, int(math.log(tK, tP))+1) if allSameShapes and useFusion else [1]
                      for numFusedKerns in fusedCases:
                        distKernels = [0, 1] if useDistKernels else [0]
                        for dist in distKernels: 
                          __configs += [KernelConfig(KronMatMulShape(m, tK, n, p, q), 
                                                                  p, q, tQ, tP, tM, 
                                        rowModTileIsZero, regRows, regCols, kEqVar,
                                        numFusedKerns, dist, "Float")]
    configs[str([m, k, n, ps, qs])] = __configs

  print("Generated configs: ", ";".join([str(k) + "-> %d"%len(configs[k]) for k in configs]))
  
  #Filter only valid configs
  validConfigs = {}
  for k in configs:
    validConfigs[k] = []
    for config in configs[k]:
      if config.isValid():
        validConfigs[k] += [config]
  
  print("Valid configs", sum([len(validConfigs[k]) for k in validConfigs]))

  uniqueConfigs = {k : list(set(validConfigs[k])) for k in validConfigs}

  print("Unique configs", sum([len(uniqueConfigs[k]) for k in uniqueConfigs]))

  combinedConfigs = []
  for k in uniqueConfigs:
    configs = uniqueConfigs[k]
    configs = configs[:min(len(configs), numKernels)]
    if onlySpecificConfigs != []:
        __configs = []
        for config in configs:
          for specificConfig in onlySpecificConfigs:
            if specificConfig.replace(' ', '') in repr(config).replace(' ', ''):
              __configs += [config]
              break
        configs = __configs
    combinedConfigs += configs
  
  combinedConfigs = list(set(combinedConfigs))
  print("Generating", len(combinedConfigs), "configs")
  if len(combinedConfigs) == 0:
    return

  for config in combinedConfigs:
    #Write host function for each config
    kernel_filename = os.path.join(kernel_dir, config.filename())
    with open(kernel_filename, "w") as f:
      kernel_file_template = "\n".join(['#include "../kernel.cuh"',
                                        "",
                                        config.hostFuncDecl()+"{",
                                        f"  {config.kernelDecl()}<<<grid, block, 0, stream>>>(params, fusedParams, distParams);",
                                        "}"]);
      f.write(kernel_file_template)

  #declare KernelInfo for each config
  host_decls = '#include "../device_functions.cuh"\n'
  for config in combinedConfigs:
    host_decls += config.hostFuncDecl() + ";\n"
  host_decls += "\n"
  
  kernel_infos = "#define ALL_KERNELS \\\n"
  for config in combinedConfigs:
    kernel_infos += config.kernelInfo() + ",\\\n"
  
  kernel_infos = kernel_infos[:kernel_infos.rfind(",")]
  kernel_infos += "\n"
  with open(os.path.join(kernel_dir, "kernel_decl.inc"), "w") as f:
    #Remove last comma and backslash
    f.write(host_decls)
    f.write(kernel_infos)
  
  make_device_kernels = "DEVICE_KERNELS="
  for config in combinedConfigs:
    make_device_kernels += os.path.join(kernel_dir, config.filename().replace(".cu",".o")) + " "

  with open(os.path.join(kernel_dir, "make_device_kernels"), "w") as f:
    f.write(make_device_kernels)

def parse_distinct_factors(case):
  m = 2
  n = int(case[0])
  ps = [int(p) for p in case[1:n+1]]
  qs = [int(q) for q in case[n+1: ]]
  assert len(ps) == n
  assert len(qs) == n
  k = 1
  for p in ps:
    assert p >= 1
    k = p * k
  for q in qs:
    assert q >= 1

  return (m, k, n, ps, qs)

def parse_same_factors(case):
  m = 2
  n = int(case[0])
  ps = [int(case[1]) for i in range(n)]
  qs = [int(case[2]) for i in range(n)]
  assert len(ps) == n
  assert len(qs) == n
  k = 1
  for p in ps:
    assert p >= 1
    k = p * k
  for q in qs:
    assert q >= 1

  return (m, k, n, ps, qs)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-distinct-factors', required=False, nargs="+", action='append', type=int)
  parser.add_argument('-same-factors', required=False, nargs="+", action='append', type=int)
  parser.add_argument('-no-fuse', required=False, action='store_true')
  parser.add_argument('-num-kernels', required=False, type=int, default=10000)
  parser.add_argument('-dist-kernels', required=False, action='store_true', default=False)
  parser.add_argument('-match-configs', nargs="+", action='append', type=str)
  parser.add_argument('-match-configs-file', required=False, type=str)

  args = parser.parse_args()
  parsed_cases = []
  
  if args.distinct_factors is not None:
    for case in args.distinct_factors:
      try:
        parsed_cases += [parse_distinct_factors(case)]
      except Exception as e:
        print(f"Invalid case: {case}")
        print(e)
        sys.exit(0)

  if args.same_factors is not None:  
    for case in args.same_factors:
      try:
        parsed_cases += [parse_same_factors(case)]
      except Exception as e:
        print(f"Invalid case: {case}")
        print(e)
        sys.exit(0)
  
  print("Generating kernels for ")
  print(parsed_cases)
  assert (args.match_configs == None and args.match_configs_file == None) or \
         (args.match_configs != None and args.match_configs_file == None) or \
         (args.match_configs == None and args.match_configs_file != None)

  match_configs = args.match_configs[0] if args.match_configs != None else []
  if args.match_configs_file != None:
    contents = slurp(args.match_configs_file)
    match_configs = contents.split('\n')
    print(match_configs)
  generate_kernel_decls(parsed_cases, not args.no_fuse, args.dist_kernels, args.num_kernels, match_configs)