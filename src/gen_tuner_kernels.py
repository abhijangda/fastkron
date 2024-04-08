import argparse
import math 
import sys
import os
import shutil
import functools

all_kernels_dir = os.path.join(os.path.dirname(__file__), "kernels/")

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
  if not os.path.exists(kernel_dir):
    os.makedirs(kernel_dir, exist_ok=True)
    return

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
  if elem_type.lower() == "int":
    return 4
  if elem_type.lower() == "double":
    return 8

def vec_type(elem_type : str, len : int) -> str:
  elem_type = elem_type.lower()
  assert len in [1, 2, 4]
  if len == 1:
    return elem_type
  return f"{elem_type}{len}"

def vector_lens(elem_type: str) -> list:
  if element_size(elem_type) == 4:
    return [1, 2, 4]
  elif element_size(elem_type) == 8:
    return [1, 2]

def elem_type_to_fastkron_type(elem_type: str) -> str:
  if elem_type.lower() == "float":
    return "FastKronType::FastKronFloat"
  if elem_type.lower() == "int":
    return "FastKronType::FastKronInt"
  if elem_type.lower() == "double":
    return "FastKronType::FastKronDouble"
  if elem_type.lower() == "float":
    return "FastKronType::FastKronFloat"
  assert false, f"{elem_type} not in FastKron"

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
               FusedKernel : int, dist: int, elemType : str, opt_level : int, rm : int, rk : int, rq : int, allPowersOf2: int, opX : str, opF : str):
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
    self.elemType = elemType
    self.dist = dist
    self.rk = rk
    self.rq = rq
    self.rm = rm
    self.opt_level = opt_level

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

  def constructorArgs(self):
    return f"(void*){self.hostFuncName()}, Factor({self.shape.p}, {self.shape.q}), Factor({self.tileP}, {self.tileQ}), Matrix({self.tileM}, {self.shape.k}), {self.fused_kernels}, {self.dist}, {self.rm}, {self.rk}, {self.rq}, {elem_type_to_fastkron_type(self.elemType)}, {self.opt_level}, fastKronOp_{self.opX}, fastKronOp_{self.opF}"

class CPUKernel(Kernel):
  def __init__(self, shape : KronMatMulShape, problem : KronMatMulShape, kron_rows : int, kron_cols : int,
               tileQ : int, tileP : int, tileM: int, rk: int, rq: int,
               FusedKernel : int, dist: int, elemType : str, opt_level : int, aalign: int, kalign: int, allPowersOf2: int, opX : str, opF : str):
    super().__init__(shape, problem, kron_rows, kron_cols, tileQ, tileP, tileM, FusedKernel, dist, elemType, opt_level, rk, rq, allPowersOf2, opX, opF)
    self.aalign = aalign
    self.kalign = kalign

  def __repr__(self):
    return f"{self.shape.q}, {self.shape.p}, {self.tileP}, {self.tileQ}, {self.shape.k}, {self.tileM}, {self.fused_kernels}, {self.rk}, {self.rq}, {self.dist}, {self.elemType}, {self.opX}, {self.opF}"
  
  def filename(self):
    return f"{self.kernelname()}.cpp"
  
  def templateDecl(self):
    return f"{self.elemType}, {self.shape.p}, {self.shape.q}, {self.tileP}, {self.tileQ}, {self.shape.k}, {self.tileM}, {self.fused_kernels}, {self.rk}, {self.rq}, {self.aalign}, {self.kalign}, fastKronOp_{self.opX}, fastKronOp_{self.opF}"

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
    return "CPUKernel{" + self.constructorArgs() + "}"

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

class GPUKernel(Kernel):
  def __init__(self, gpu_type : str, shape : KronMatMulShape, problem : KronMatMulShape, kron_rows : int, kron_cols : int, 
               tileQ : int, tileP : int, tileM: int,
               regM: int, cRegRows: int, cRegCols: int,
               FusedKernel : int, dist: int, elemType : str, opt_level : int, aalign: int, kalign: int,
               allPowersOf2: int, opX : str, opF : str):
    aalign = min(4, aalign)
    kalign = min(4, kalign)
    super().__init__(shape, problem, kron_rows, kron_cols, tileQ, tileP, tileM, FusedKernel, dist, elemType, opt_level, regM, cRegRows, cRegCols, allPowersOf2, opX, opF)
    self.num_threads = (tileM//regM) * ((shape.k//shape.p)//cRegRows) * (tileQ//cRegCols)
    self.tileQ = tileQ
    self.tileP = tileP
    self.tileM = tileM
    self.aalign = aalign
    self.kalign = kalign
    self.gpu_type = gpu_type

    assert gpu_type in ["cuda", "hip"]

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
    return f"{self.threads()}_{self.shape.p}x{self.shape.q}_{self.tileP}x{self.tileQ}_{self.fused_kernels}_{self.tileM}x{self.shape.k}_{self.rm}x{self.rk}x{self.rq}_{self.opX}{self.opF}_{self.dist}_{self.opt_level}_{self.elemType}_{self.aalign}_{self.kalign}"

  def kernelname(self):
    return f"{self.gpu_type}_{super().kernelname()}"

  def filename(self):
    return f"{self.kernelname()}.{'cu' if self.gpu_type == 'cuda' else 'hip'}"

  def hostFuncDecl(self):
    return f"void {self.hostFuncName()}(KernelParams<{self.fused_kernels}> params, FusedParams<{self.fused_kernels}> fusedParams, DistributedParams distParams, EpilogueParams epilogueParams, dim3 grid, dim3 block, uint32_t sharedSize, {self.gpu_type}Stream_t stream)"

  def templateDecl(self):
    #TODO: repr and this should be same
    return f"{self.elemType}, {vec_type(self.elemType, 2)}, {vec_type(self.elemType, 4)}, {self.threads()}, {self.shape.q}, {self.shape.p}, {self.tileP}, {self.tileQ}, {self.shape.k}, {self.tileM}, {self.fused_kernels}, {self.dist}, {self.rm}, {self.rk}, {self.rq}, {self.opt_level}, {self.aalign}, {self.kalign}, fastKronOp_{self.opX}, fastKronOp_{self.opF}"

  def kernelDecl(self):
    return f"cudaKernel<{self.templateDecl()}>"

  def hostInvokeFile(self):
    return "\n".join(['#include "kernels/cuda/kernel.cuh"', "",
                      self.getKernelFuncDecl()+"{",
                      f"  return (void*)&{self.kernelDecl()};",
                      "}",
                      self.hostFuncDecl()+"{",
                      f"  {self.kernelDecl()}<<<grid, block, sharedSize, stream>>>(params, fusedParams, distParams, epilogueParams);",
                      "}"])

  def kernelInfo(self):
    #TODO: should be same as tempelDecl, hostFuncDecl, and __repr__
    return f"{self.gpu_type.upper()}Kernel{{"+\
            self.constructorArgs() + ","+\
            f"get{self.kernelname()}, {self.threads()}, " +\
            f"{self.aalign}, {self.kalign}" + "}"

  def isValid(self):
    return self.wsz > 0 and \
           self.shape.k % self.shape.p == 0 and \
           self.num_threads >= 64 and self.threads() <= 1024 and \
           self.shared_mem_usage <= MAX_SHARED_MEM and \
           self.rk in [1, 2, 4] and \
           (self.fused_kernels == 1 or (self.fused_kernels > 1 and self.fused_kernels <= 6 and self.shape.p == self.tileP and self.shape.q == self.tileQ and self.opt_level == 3)) and \
           self.dist in [0, 1] and \
           self.rq <= 32 and \
           self.tileM * self.rk * self.rq <= 64 and self.opt_level == 3
  
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

def xalignment(m, cols, op, elem_type):
  if op == "T":
    return 1 #max([a for a in [1, 2, 4] if m % a == 0])
  else:
    return max([a for a in vector_lens(elem_type) if cols % a == 0])

def falignment(cols, elem_type):
  return max([a for a in vector_lens(elem_type) if cols % a == 0])

def generate_kernel_decls(cases, opXs, opFs, types, useFusion, useDistKernels, numKernels, onlySpecificConfigs, backend):
  if not os.path.exists(all_kernels_dir):
    os.mkdir(all_kernels_dir)

  if backend == 'cuda' or backend == "hip":
    kernel_dir = os.path.join(all_kernels_dir, f'{backend}/kron-kernels')
  elif backend == 'x86':
    kernel_dir = os.path.join(all_kernels_dir, 'cpu/x86/kron-kernels')

  empty_dir(kernel_dir)
  configs = {}
  for opX in opXs:
    for opF in opFs:
      for elem_type in types:
        for (m, k, n, ps, qs) in cases:
          allSameShapes = len(set(ps + qs)) == 1# and isPowerOfTwo(ps[0])
          for (_, currK, opx, p, q) in all_sliced_mults(m, k, n, opX, ps, qs):
            MinTile = 32 if backend == 'x86' else 32
            TilePs = [min(p, MinTile)] + [i for i in factors(p) if i > MinTile]
            TileQs = factors(q) #[2**i for i in range(1, max(2, int(math.log2(q)))+1)]
            k_factors = factors(currK)
            TileKs = [f for f in k_factors if f % p == 0]
            TileMs = [1,2,4,8] if opx == "T" else [1,2] #[2 ** i for i in range(0, int(math.log2(m)))]

            for tM in TileMs:
              for tQ in TileQs:
                for tK in TileKs:
                  if tK < p:
                    continue
                  CRows = factors(tK//p)
                  CCols = factors(tQ)
                  RegMs = factors(tM)
                  for regRows in CRows:
                    for regCols in CCols:
                      for regM in RegMs:
                        for tP in TilePs:
                          fusedCases = range(1, int(math.log(tK, p))+1) if allSameShapes and useFusion else [1]
                          for numFusedKerns in fusedCases:
                            aalign = xalignment(tM, tK, opx, elem_type)
                            kronalign = falignment(tQ, elem_type)
                            shape = KronMatMulShape(m, tK, numFusedKerns, p, q)
                            if shape not in configs:
                              configs[shape] = []
                            __configs = []
                            for opt_level in range(0, 4):
                              if opt_level <= 1 or aalign == 1:
                                new_aalign = aalign
                              elif opt_level == 2:
                                new_aalign = min(aalign, kronalign)
                              else:
                                new_aalign = aalign
  
                              if backend in ['cuda', 'hip']:
                                    distKernels = [0, 1] if useDistKernels else [0]
                                    for dist in distKernels: 
                                      __configs += [GPUKernel(backend, KronMatMulShape(m, tK, n, p, q), 
                                                              KronMatMulShape(m, k, n, ps, qs),
                                                              p, q, tQ, tP, tM, regM, regRows, regCols,
                                                              numFusedKerns, dist, elem_type, opt_level, new_aalign, 1 if (opt_level <= 1) else kronalign, allSameShapes,
                                                              opx, opF)]
                              elif backend == 'x86':
                                dist = 0
                                __configs += [CPUKernel(KronMatMulShape(m, tK, n, p, q),
                                                        KronMatMulShape(m, k, n, ps, qs),
                                                        p, q, tQ, tP, tM, regM, regRows, regCols, numFusedKerns, 
                                                        dist, elem_type, opt_level, aalign, kronalign, allSameShapes, opx, opF)]

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
  parser.add_argument('-opX'               , required=True , type=str,  nargs="+")
  parser.add_argument('-opF'               , required=True , type=str,  nargs="+")
  parser.add_argument('-num-kernels'       , required=False, type=int,   default=10000)
  parser.add_argument('-no-fuse'           , required=False, action='store_true')
  parser.add_argument('-dist-kernels'      , required=False, action='store_true', default=False)
  parser.add_argument('-match-configs'     , required=False, type=str,  action='append',     nargs="+")
  parser.add_argument('-match-configs-file', required=False, type=str)
  parser.add_argument('-backend'           , required=True,  type=str)
  parser.add_argument('-types'              , required=True,  type=str, nargs="+")
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
  
  if args.backend is None or args.backend.lower() not in ['cuda', 'x86', 'hip', 'arm']:
    print(f"Invalid backend: {args.backend}")
    sys.exit(0)

  print("Generating kernels for ", parsed_cases)
  for opX in args.opX:
    assert opX in ["N", "T"]
  for opF in args.opF:
    assert opF in ["N", "T"]
  for t in args.types:
    assert t in ["float", "int", "double", "half"]

  if args.match_configs != None:
    assert type(args.match_configs) == list and len(args.match_configs) == 1
  assert (args.match_configs == None and args.match_configs_file == None) or \
         (args.match_configs != None and args.match_configs_file == None) or \
         (args.match_configs == None and args.match_configs_file != None)

  match_configs = args.match_configs[0] if args.match_configs != None else []

  if args.match_configs_file != None and match_configs == []:
    contents = slurp(args.match_configs_file)
    contents = contents.split('\n')
    for line in contents:
      if line.strip() != "":
        match_configs += [line]
  
  generate_kernel_decls(parsed_cases, args.opX, args.opF, args.types, not args.no_fuse, args.dist_kernels,
                        args.num_kernels, match_configs, args.backend)
