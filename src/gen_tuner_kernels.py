import argparse
import math 
import sys
import os
import shutil
import functools

kernel_dir = os.path.join(os.path.dirname(__file__), "device/kron-kernels/")

#Device limits
#Volta
MAX_SHARED_MEM = 96 * 1024
#Ampere
MAX_SHARED_MEM = 164 * 1024
#Normal
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

def factors(n):
  return list(set(functools.reduce(list.__add__, 
              ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))

def isPowerOfTwo(x):
    return (x and (not(x & (x - 1))) )

class KronMatMulShape:
  def __init__(self, m, k, n, p, q):
    self.m = m
    self.k = k
    self.n = n
    self.p = p
    self.q = q

  def __repr__(self):
    return f"{self.m}x{self.k}, {self.p}x{self.q}, {self.n}"

WARP_SIZE=32

class KernelConfig:  
  def __init__(self, shape : KronMatMulShape, kron_rows : int, kron_cols : int, 
               tileQ : int, tileP : int, tileM: int,
               cRegRows: int, cRegCols: int,
               FusedKernel : int, dist: int, elemType : str, aalign: int, kalign: int,
               allPowersOf2: int, opX : bool, opF : bool):
    self.shape = shape
    self.num_threads = ((shape.k//shape.p)//cRegRows) * (tileQ//cRegCols)
    self.kron_rows = kron_rows
    self.kron_cols = kron_cols
    self.tileQ = tileQ
    self.tileP = tileP
    self.tileM = tileM
    self.cRegRows = cRegRows
    self.cRegCols = cRegCols
    self.opX = opX
    self.opF = opF
    self.fused_kernels = FusedKernel
    assert self.fused_kernels > 0
    self.dist = dist
    self.elemType = "float"
    self.aalign = aalign
    self.kalign = kalign

    """Compute several constants of kernel
       Corresponds to line numbers in kernel.cuh
    """
    self.wsz = (self.shape.k//self.shape.p)//self.cRegRows 
    self.shared_mem_usage = (self.tileM * ((self.shape.k/self.shape.p)*self.tileP) + self.tileP * self.tileQ)*element_size(elemType)

  def threads(self):
    if self.num_threads%WARP_SIZE != 0:
      return (self.num_threads//WARP_SIZE + 1)*WARP_SIZE
    return self.num_threads
  
  def __repr__(self):
    return f"{self.threads()}, {self.shape.q}, {self.shape.p}, {self.tileQ}, {self.shape.k}, {self.tileM}, {self.fused_kernels}, {self.dist}, {self.cRegRows}, {self.cRegCols}, {self.elemType}, {self.aalign}, {self.kalign}, {self.opX}, {self.opF}"

  def kernelname(self):
    return repr(self).replace(", ", "_")

  def filename(self):
    return f"kernel_{self.kernelname()}.cu"

  def hostFuncName(self):
    return f"invoke{self.kernelname()}"

  def hostFuncDecl(self):
    return f"void {self.hostFuncName()}(KernelParams<{self.fused_kernels}> params, FusedParams<{self.fused_kernels}> fusedParams, DistributedParams distParams, EpilogueParams epilogueParams, dim3 grid, dim3 block, uint32_t sharedSize, cudaStream_t stream)"

  def templateDecl(self):
    return f"float, float2, float4, {self.threads()}, {self.shape.q}, {self.shape.p}, {self.tileP}, {self.tileQ}, {self.shape.k}, {self.tileM}, {self.fused_kernels}, {self.dist}, {self.cRegRows}, {self.cRegCols}, 1, {self.aalign}, {self.kalign}, fastKronOp_{self.opX}, fastKronOp_{self.opF}"
  
  def kernelDecl(self):
    return f"kronGemmKernel<{self.templateDecl()}>"

  def __eq__(self, other):
    return repr(self) == repr(other)

  def kernelInfo(self):
    #TODO: should be same as tempelDecl, hostFuncDecl, and __repr__
    constructor = f"{self.threads()}, {self.shape.q}, {self.shape.p}, {self.tileP}, {self.tileQ}, {self.shape.k}, {self.tileM}, {self.fused_kernels}, {self.dist}, {self.cRegRows}, {self.cRegCols}, {self.elemType}, {self.aalign}, {self.kalign}, fastKronOp_{self.opX}, fastKronOp_{self.opF}"
    return "KernelInfo{"+\
            f"(void*){self.hostFuncName()},"+\
            f"get{self.kernelname()},"+\
            constructor.replace("float", "ElementType::Float") + "}"

  def getKernelFuncName(self):
    return f"get{self.kernelname()}"

  def getKernelFuncDecl(self):
    return f"void* {self.getKernelFuncName()}()"

  def isValid(self):
    return self.wsz > 0 and \
           self.shape.k % self.shape.p == 0 and \
           self.num_threads >= 64 and self.threads() <= 1024 and \
           self.shared_mem_usage <= MAX_SHARED_MEM and \
           self.cRegRows in [1, 2, 4] and \
           (self.fused_kernels == 1 or (self.fused_kernels > 1 and self.shape.p == self.tileP and self.shape.q == self.tileQ)) and \
           self.dist in [0, 1] and \
           self.cRegCols <= 32 and \
           self.tileM * self.cRegRows * self.cRegCols <= 64

  def __hash__(self):
    return hash(repr(self))

def all_sliced_mults(m, k, n, opX, ps, qs):
  sliced_mults = []
  prevTmpK = k
  for i in range(n):
    f = n - i - 1
    sliced_mult = (m, prevTmpK, opX if i == 0 else "N", ps[f], qs[f])
    prevTmpK = (prevTmpK//ps[f])*qs[f]
    sliced_mults += [sliced_mult]
  sliced_mults = set(sliced_mults)
  return list(sliced_mults)

def xalignment(m, cols, op):
  if op == "T":
    #TODO: Return Alignment based on TileM and M
    return 1 #max([a for a in [1, 2, 4] if m % a == 0])
  else:
    return max([a for a in [1, 2, 4] if cols % a == 0])

def falignment(cols):
  return max([a for a in [1, 2, 4] if cols % a == 0])

def generate_kernel_decls(cases, opX, opF, useFusion, useDistKernels, numKernels, onlySpecificConfigs):
  if not os.path.exists(kernel_dir):
    os.mkdir(kernel_dir)

  empty_dir(kernel_dir)
  configs = {}
  
  for (m, k, n, ps, qs) in cases:
    allSameShapes = len(set(ps + qs)) == 1# and isPowerOfTwo(ps[0])
    for (_, currK, opx, p, q) in all_sliced_mults(m, k, n, opX, ps, qs):
      TilePs = [min(p, 32)] + [i for i in factors(p) if i > 32]
      TileQs = factors(q) #[2**i for i in range(1, max(2, int(math.log2(q)))+1)]
      k_factors = factors(currK)
      TileKs = [f for f in k_factors if f % p == 0]
      TileMs = [1, 2] #[2 ** i for i in range(0, int(math.log2(m)))]

      shape = KronMatMulShape(m, currK, n, p, q)
      if shape not in configs:
        configs[shape] = []
      __configs = []  
      for tM in TileMs:
        for tQ in TileQs:
          for tK in TileKs:
            if tK < p:
              continue
            CRows = factors(tK//p)
            CCols = factors(tQ)
            aalign = xalignment(tM, tK, opX)
            kronalign = falignment(tQ)
            for regRows in CRows:
              for regCols in CCols:
                for tP in TilePs:
                  for kEqVar in [0]:
                    fusedCases = range(1, int(math.log(tK, p))+1) if allSameShapes and useFusion else [1]
                    for numFusedKerns in fusedCases:
                      distKernels = [0, 1] if useDistKernels else [0]
                      for dist in distKernels: 
                        __configs += [KernelConfig(KronMatMulShape(m, tK, n, p, q), 
                                                                   p, q, tQ, tP, tM, 
                                      regRows, regCols,
                                      numFusedKerns, dist, "Float", aalign, kronalign, allSameShapes,
                                      opx, opF)]
      configs[shape] += __configs

  print("Generated configs:\n" + "\n".join([str(k) + "-> %d"%len(configs[k]) for k in configs]))
  
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
                                        config.getKernelFuncDecl()+"{",
                                        f"  return (void*)&{config.kernelDecl()};",
                                        "}",
                                        config.hostFuncDecl()+"{",
                                        f"  {config.kernelDecl()}<<<grid, block, sharedSize, stream>>>(params, fusedParams, distParams, epilogueParams);",
                                        "}"]);
      f.write(kernel_file_template)

  #declare KernelInfo for each config
  host_decls = ''
  for config in combinedConfigs:
    host_decls += config.hostFuncDecl() + ";\n" + config.getKernelFuncDecl() + ";\n"
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
  
  kernels_cmake = "set(CUDA_KERNELS "
  for config in combinedConfigs:
    kernels_cmake += os.path.join(kernel_dir, config.filename()) + "\n"

  with open(os.path.join(kernel_dir, "kernels.cmake"), "w") as f:
    f.write(kernels_cmake + ")")

def compute_k(ps, qs):
  k = 1
  for p in ps:
    k = k * p

  return k

def parse_distinct_factors(case):
  m = 2
  n = int(case[0])
  assert len(case[1:]) == n
  ps = []
  qs = []
  
  for pq in case[1:]:
    split = pq.split(',')
    ps += [int(split[0])]
    qs += [int(split[1])]
  
  k = compute_k(ps, qs)
  return (m, k, n, ps, qs)

def parse_same_factors(case):
  m = 2
  n = int(case[0])
  assert len(case[1:]) == 1
  split = case[1].split(',')
  ps = [int(split[0]) for i in range(n)]
  qs = [int(split[1]) for i in range(n)]
  
  k = compute_k(ps, qs)
  return (m, k, n, ps, qs)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-distinct-factors',   required=False , type=str,  action='append',     nargs="+")
  parser.add_argument('-same-factors',       required=False , type=str,  action='append',     nargs="+")
  parser.add_argument('-opX',                required=True , type=str)
  parser.add_argument('-opF',                required=True , type=str)
  parser.add_argument('-num-kernels',        required=False, type=int,                       default=10000)
  parser.add_argument('-no-fuse',            required=False, action='store_true')
  parser.add_argument('-dist-kernels',       required=False, action='store_true', default=False)
  parser.add_argument('-match-configs',      required=False, type=str,  action='append',     nargs="+")
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
  
  print("Generating kernels for ", parsed_cases)
  assert args.opX in ["N", "T"]
  assert args.opF in ["N", "T"]
  assert (args.match_configs == None and args.match_configs_file == None) or \
         (args.match_configs != None and args.match_configs_file == None) or \
         (args.match_configs == None and args.match_configs_file != None)

  match_configs = args.match_configs[0] if args.match_configs != None else []
  if args.match_configs_file != None:
    contents = slurp(args.match_configs_file)
    match_configs = contents.split('\n')
  generate_kernel_decls(parsed_cases, args.opX, args.opF, not args.no_fuse, args.dist_kernels,
                        args.num_kernels, match_configs)
