import argparse
import math 
import sys
import os
import shutil
import functools

kernel_dir = os.path.join(os.path.dirname(__file__), "kernels/")

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
    self.l = k
    if type(p) == list:
      for i in range(len(p)):
        self.l = (self.l//self.p[i]) * self.q[i]

  def __repr__(self):
    return f"{self.m}x{self.k}, {self.p}x{self.q}, {self.n}"

  def __str__(self):
    return repr(self)

  def __hash__(self):
    return hash(repr(self))

  def __eq__(self, other):
    return self.m == other.m and self.k == other.k and self.n == other.n and self.p == other.p and self.q == other.q

WARP_SIZE=32

class Kernel:
  def __init__(self, shape : KronMatMulShape, problem : KronMatMulShape, kron_rows : int, kron_cols : int, tileQ : int, tileP : int, tileM : int, 
               FusedKernel : int, dist: int, elemType : str, rk : int, rq : int, allPowersOf2: int, opX : str, opF : str):
    self.shape = shape
    self.kron_rows = kron_rows
    self.kron_cols = kron_cols
    self.tileQ = tileQ
    self.tileP = tileP
    self.tileM = tileM
    self.problem = problem
    self.opX = opX
    self.opF = opF
    self.fused_kernels = FusedKernel
    assert self.fused_kernels > 0
    self.elemType = "float"
    self.dist = dist
    self.rk = rk
    self.rq = rq

  def kernelname(self):
    return repr(self).replace(", ", "_")

  def __eq__(self, other):
    return repr(self) == repr(other)

  def getKernelFuncName(self):
    return f"get{self.kernelname()}"

  def getKernelFuncDecl(self):
    return f"void* {self.getKernelFuncName()}()"

  def hostFuncName(self):
    return f"invoke{self.kernelname()}"

  def __hash__(self):
    return hash(repr(self))

class CPUKernel(Kernel):
  def __init__(self, shape : KronMatMulShape, problem : KronMatMulShape, kron_rows : int, kron_cols : int,
               tileQ : int, tileP : int, tileM: int, rk: int, rq: int,
               FusedKernel : int, dist: int, elemType : str, aalign: int, kalign: int, allPowersOf2: int, opX : str, opF : str):
    super().__init__(shape, problem, kron_rows, kron_cols, tileQ, tileP, tileM, FusedKernel, dist, elemType, rk, rq, allPowersOf2, opX, opF)
    self.aalign = aalign
    self.kalign = kalign

  def __repr__(self):
    return f"{self.shape.q}, {self.shape.p}, {self.tileP}, {self.tileQ}, {self.shape.k}, {self.tileM}, {self.fused_kernels}, {self.rk}, {self.rq}, {self.dist}, {self.elemType}, {self.opX}, {self.opF}"
  
  def filename(self):
    return f"{self.kernelname()}.cpp"
  
  def templateDecl(self):
    return f"float, float2, float4, {self.shape.p}, {self.shape.q}, {self.tileP}, {self.tileQ}, {self.shape.k}, {self.tileM}, {self.fused_kernels}, {self.rk}, {self.rq}, {self.aalign}, {self.kalign}, fastKronOp_{self.opX}, fastKronOp_{self.opF}"

  def kernelDecl(self):
    return f"cpuKernel<{self.templateDecl()}>"

  def hostFuncDecl(self):
    return f"void {self.hostFuncName()}(KernelParams<{self.fused_kernels}> params, FusedParams<{self.fused_kernels}> fusedParams, DistributedParams distParams, EpilogueParams epilogueParams)"

  def hostInvokeFile(self):
    return "\n".join(['#include "../../kernel.h"', "",
                      self.getKernelFuncDecl()+"{",
                      f"  return (void*)&{self.kernelDecl()};",
                      "}",
                      self.hostFuncDecl()+"{",
                      f"  {self.kernelDecl()}(params, fusedParams, distParams, epilogueParams);",
                      "}"])

  def kernelInfo(self):
    constructor = f"{self.shape.q}, {self.shape.p}, {self.tileP}, {self.tileQ}, {self.shape.k}, {self.tileM}, {self.fused_kernels}, {self.dist}, {self.rk}, {self.rq}, {self.elemType}, fastKronOp_{self.opX}, fastKronOp_{self.opF}"
    return "CPUKernel{"+\
            f"(void*){self.hostFuncName()},"+\
            f"get{self.kernelname()},"+\
            constructor.replace("float", "ElementType::Float") + "}"

  def isValid(self):
    AVXLen = 8
    #After transposing of slices, TileX has element of each slice in contiguous order.
    #So, number of slices should be multiple of vector
    cond = (((self.opX == "T" or not isPowerOfTwo(self.problem.k) or not isPowerOfTwo(self.problem.l)) and (self.shape.k // self.shape.p) % 8 != 0 and self.shape.k % self.rk == 0) or \
            (self.aalign == 8 and self.rk % AVXLen == 0))
    if isPowerOfTwo(self.shape.p) and isPowerOfTwo(self.shape.q) and self.shape.p >= 4 and self.shape.q >= 4:
      #15 YMM Registers.
      cond = cond and self.rk == min(16, self.shape.k//self.shape.p) and self.rq == min(4, self.tileQ)
    # print(self, cond, self.shape.k, self.shape.p, self.rk, self.problem.k, isPowerOfTwo(self.problem.k), (self.shape.k // self.shape.p) % 8 != 0, self.shape.k % self.rk == 0)
    return cond and self.shape.k * self.tileM <= 16*1024 and \
           self.shape.k % self.shape.p == 0 and \
           self.tileM * (self.shape.k//self.shape.p) * self.tileQ * 4 <= 1*1024*1024 and \
           self.rk/AVXLen < 8 and \
            (self.fused_kernels == 1 or \
              (self.fused_kernels > 1 and self.fused_kernels <= 6 and self.shape.p == self.tileP and \
              #Next fused intermediate must have atleast AVXLen slices to make sure
              #Transpose X->TileX loads contiguous AVXLen first elements of slices of same P
               self.shape.q == self.tileQ and (self.shape.k//(self.shape.p**self.fused_kernels)) >= AVXLen) \
            ) and \
           self.dist in [0, 1] and \
           self.rq <= AVXLen 
          #  and \
          #  self.rq > 1 and self.shape.k >= 8192 and self.rk > 8

class CUDAKernel(Kernel):
  def __init__(self, shape : KronMatMulShape, problem : KronMatMulShape, kron_rows : int, kron_cols : int, 
               tileQ : int, tileP : int, tileM: int,
               cRegRows: int, cRegCols: int,
               FusedKernel : int, dist: int, elemType : str, aalign: int, kalign: int,
               allPowersOf2: int, opX : str, opF : str):
    aalign = min(4, aalign)
    kalign = min(4, kalign)
    super().__init__(shape, problem, kron_rows, kron_cols, tileQ, tileP, tileM, FusedKernel, dist, elemType, cRegRows, cRegCols, allPowersOf2, opX, opF)
    self.num_threads = ((shape.k//shape.p)//cRegRows) * (tileQ//cRegCols)
    self.tileQ = tileQ
    self.tileP = tileP
    self.tileM = tileM
    self.aalign = aalign
    self.kalign = kalign

    """Compute several constants of kernel
       Corresponds to line numbers in kernel.cuh
    """
    self.wsz = (self.shape.k//self.shape.p)//self.rk 
    self.shared_mem_usage = (self.tileM * ((self.shape.k/self.shape.p)*self.tileP) + self.tileP * self.tileQ)*element_size(elemType)

  def threads(self):
    if self.num_threads%WARP_SIZE != 0:
      return (self.num_threads//WARP_SIZE + 1)*WARP_SIZE
    return self.num_threads
  
  def __repr__(self):
    return f"{self.threads()}, {self.shape.q}, {self.shape.p}, {self.tileQ}, {self.shape.k}, {self.tileM}, {self.fused_kernels}, {self.dist}, {self.rk}, {self.rq}, {self.elemType}, {self.aalign}, {self.kalign}, {self.opX}, {self.opF}"

  def kernelname(self):
    return f"cuda_{super().kernelname()}"

  def filename(self):
    return f"{self.kernelname()}.cu"

  def hostFuncDecl(self):
    return f"void {self.hostFuncName()}(KernelParams<{self.fused_kernels}> params, FusedParams<{self.fused_kernels}> fusedParams, DistributedParams distParams, EpilogueParams epilogueParams, dim3 grid, dim3 block, uint32_t sharedSize, cudaStream_t stream)"

  def templateDecl(self):
    return f"float, float2, float4, {self.threads()}, {self.shape.q}, {self.shape.p}, {self.tileP}, {self.tileQ}, {self.shape.k}, {self.tileM}, {self.fused_kernels}, {self.dist}, {self.rk}, {self.rq}, 1, {self.aalign}, {self.kalign}, fastKronOp_{self.opX}, fastKronOp_{self.opF}"
  
  def kernelDecl(self):
    return f"cudaKernel<{self.templateDecl()}>"

  def hostInvokeFile(self):
    return "\n".join(['#include "../kernel.cuh"', "",
                      self.getKernelFuncDecl()+"{",
                      f"  return (void*)&{self.kernelDecl()};",
                      "}",
                      self.hostFuncDecl()+"{",
                      f"  {self.kernelDecl()}<<<grid, block, sharedSize, stream>>>(params, fusedParams, distParams, epilogueParams);",
                      "}"])

  def kernelInfo(self):
    #TODO: should be same as tempelDecl, hostFuncDecl, and __repr__
    constructor = f"{self.threads()}, {self.shape.q}, {self.shape.p}, {self.tileP}, {self.tileQ}, {self.shape.k}, {self.tileM}, {self.fused_kernels}, {self.dist}, {self.rk}, {self.rq}, {self.elemType}, {self.aalign}, {self.kalign}, fastKronOp_{self.opX}, fastKronOp_{self.opF}"
    return "CUDAKernel{"+\
            f"(void*){self.hostFuncName()},"+\
            f"get{self.kernelname()},"+\
            constructor.replace("float", "ElementType::Float") + "}"

  def isValid(self):
    return self.wsz > 0 and \
           self.shape.k % self.shape.p == 0 and \
           self.num_threads >= 64 and self.threads() <= 1024 and \
           self.shared_mem_usage <= MAX_SHARED_MEM and \
           self.rk in [1, 2, 4] and \
           (self.fused_kernels == 1 or (self.fused_kernels > 1 and self.fused_kernels <= 6 and self.shape.p == self.tileP and self.shape.q == self.tileQ)) and \
           self.dist in [0, 1] and \
           self.rq <= 32 and \
           self.tileM * self.rk * self.rq <= 64
  
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
    return max([a for a in [1, 2, 4, 8] if cols % a == 0])

def falignment(cols):
  return max([a for a in [1, 2, 4, 8] if cols % a == 0])

def generate_kernel_decls(cases, opX, opF, useFusion, useDistKernels, numKernels, onlySpecificConfigs, backend):
  global kernel_dir

  if not os.path.exists(kernel_dir):
    os.mkdir(kernel_dir)

  if backend == 'cuda':
    kernel_dir = os.path.join(kernel_dir, 'cuda/kron-kernels')
  elif backend == 'x86':
    kernel_dir = os.path.join(kernel_dir, 'cpu/x86/kron-kernels')

  empty_dir(kernel_dir)
  configs = {}

  for (m, k, n, ps, qs) in cases:
    allSameShapes = len(set(ps + qs)) == 1# and isPowerOfTwo(ps[0])
    for (_, currK, opx, p, q) in all_sliced_mults(m, k, n, opX, ps, qs):
      MinTile = 32 if backend == 'x86' else 32
      TilePs = [min(p, MinTile)] + [i for i in factors(p) if i > MinTile]
      TileQs = factors(q) #[2**i for i in range(1, max(2, int(math.log2(q)))+1)]
      k_factors = factors(currK)
      TileKs = [f for f in k_factors if f % p == 0]
      TileMs = [1, 2] #[2 ** i for i in range(0, int(math.log2(m)))]

      for tM in TileMs:
        for tQ in TileQs:
          for tK in TileKs:
            if tK < p:
              continue
            CRows = factors(tK//p)
            CCols = factors(tQ)
            for regRows in CRows:
              for regCols in CCols:
                for tP in TilePs:
                  fusedCases = range(1, int(math.log(tK, p))+1) if allSameShapes and useFusion else [1]
                  for numFusedKerns in fusedCases:
                    aalign = xalignment(tM, tK, opx)
                    kronalign = falignment(tQ)
                    shape = KronMatMulShape(m, tK, numFusedKerns, p, q)
                    if shape not in configs:
                      configs[shape] = []
                    __configs = []
                    if backend == 'cuda':
                      for kEqVar in [0]:
                          distKernels = [0, 1] if useDistKernels else [0]
                          for dist in distKernels: 
                            __configs += [CUDAKernel(KronMatMulShape(m, tK, n, p, q), 
                                                     KronMatMulShape(m, k, n, ps, qs),
                                                    p, q, tQ, tP, tM, regRows, regCols,
                                                    numFusedKerns, dist, "Float", aalign, kronalign, allSameShapes,
                                                    opx, opF)]
                    elif backend == 'x86':
                      dist = 0
                      __configs += [CPUKernel(KronMatMulShape(m, tK, n, p, q),
                                              KronMatMulShape(m, k, n, ps, qs),
                                              p, q, tQ, tP, tM, regRows, regCols, numFusedKerns, 
                                              dist, "Float", aalign, kronalign, allSameShapes, opx, opF)]

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

    if onlySpecificConfigs != []:
        __configs = []
        for config in configs:
          for specificConfig in onlySpecificConfigs:
            if specificConfig.replace(' ', '') in repr(config).replace(' ', ''):
              __configs += [config]
              break

        configs = __configs
    configs = configs[:min(len(configs), numKernels)]
    combinedConfigs += configs
  
  combinedConfigs = list(set(combinedConfigs))
  print("Generating", len(combinedConfigs), "configs")
  if len(combinedConfigs) == 0:
    return

  for config in combinedConfigs:
    #Write host function for each config
    kernel_filename = os.path.join(kernel_dir, config.filename())
    with open(kernel_filename, "w") as f:
      f.write(config.hostInvokeFile())

  #declare KernelInfo for each config
  host_decls = ''
  for config in combinedConfigs:
    host_decls += config.hostFuncDecl() + ";\n" + config.getKernelFuncDecl() + ";\n"
  host_decls += "\n"
  
  kernel_infos = f"#define ALL_{backend.upper()}_KERNELS \\\n"
  for config in combinedConfigs:
    kernel_infos += config.kernelInfo() + ",\\\n"
  
  kernel_infos = kernel_infos[:kernel_infos.rfind(",")]
  kernel_infos += "\n"
  with open(os.path.join(kernel_dir, "kernel_decl.inc"), "w") as f:
    #Remove last comma and backslash
    f.write(host_decls)
    f.write(kernel_infos)
  
  kernels_cmake = f"set({backend.upper()}_KERNELS "
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
  parser.add_argument('-distinct-factors'  , required=False, type=str,  action='append',     nargs="+")
  parser.add_argument('-same-factors'      , required=False, type=str,  action='append',     nargs="+")
  parser.add_argument('-opX'               , required=True , type=str)
  parser.add_argument('-opF'               , required=True , type=str)
  parser.add_argument('-num-kernels'       , required=False, type=int,   default=10000)
  parser.add_argument('-no-fuse'           , required=False, action='store_true')
  parser.add_argument('-dist-kernels'      , required=False, action='store_true', default=False)
  parser.add_argument('-match-configs'     , required=False, type=str,  action='append',     nargs="+")
  parser.add_argument('-match-configs-file', required=False, type=str)
  parser.add_argument('-backend'           , type=str)
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
  
  if args.backend is None or args.backend.lower() not in ['cuda', 'x86', 'rocm', 'arm']:
    print(f"Invalid backend: {args.backend}")
    sys.exit(0)

  print("Generating kernels for ", parsed_cases)
  assert args.opX in ["N", "T"]
  assert args.opF in ["N", "T"]
  if args.match_configs != None:
    assert type(args.match_configs) == list and len(args.match_configs) == 1
  assert (args.match_configs == None and args.match_configs_file == None) or \
         (args.match_configs != None and args.match_configs_file == None) or \
         (args.match_configs == None and args.match_configs_file != None)

  match_configs = args.match_configs[0] if args.match_configs != None else []

  if args.match_configs_file != None and match_configs != []:
    contents = slurp(args.match_configs_file)
    match_configs = contents.split('\n')
  generate_kernel_decls(parsed_cases, args.opX, args.opF, not args.no_fuse, args.dist_kernels,
                        args.num_kernels, match_configs, args.backend)
