import argparse
import math 
import sys
import os
import functools

print(os.getcwd())
assert 'src' in os.listdir('.')
kernel_filename = "src/kernel_decl.inc"

#Device limits
MAX_SHARED_MEM = 48 * 1024

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
               FusedKernel : int, elemType : str):
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
    self.elemType = elemType

    """Compute several constants of kernel
       Corresponds to line numbers in kernel.cuh
    """
    #Line 67
    self.wsz = (self.shape.k//self.shape.p)//self.cRegRows 
    self.shared_mem_usage = (self.tileM * self.shape.k + self.tileP * self.tileQ)*element_size(elemType)

  def __repr__(self):
    return f"({self.shape.k}, {self.num_threads}, {self.kron_rows}, {self.kron_cols}, {self.tileQ}, {self.tileP}, {self.tileM}, {self.cRegRows}, {self.cRegCols}, {self.fused_kernels}, {self.elemType}, {self.rowModTileIsZero}, {self.kEqVar})"

  def __eq__(self, other):
    return repr(self) == repr(other)

  def code(self):
    return "KernelInfo{"+\
            f"(void*)kronGemmKernel<T, VecT, {self.num_threads}, RowParallelismTy::Low, {self.tileM}, {self.rowModTileIsZero}, {self.shape.k}, {self.shape.q}, {self.shape.p}, {self.tileQ}, {self.kEqVar}, 1, {self.cRegRows}, {self.cRegCols}, {self.tileP}, {self.fused_kernels}>,"+\
            f"{self.num_threads}, {self.shape.q}, {self.shape.p}, {self.tileQ}, {self.tileM}, {self.shape.k}, {self.cRegRows}, {self.cRegCols}, {self.fused_kernels}, ElemType, {self.rowModTileIsZero}, {self.kEqVar}"+ "}"

  def isValid(self):
    return self.wsz > 0 and \
           self.shape.k > self.tileP and \
           self.num_threads >= 32 and self.num_threads <= 1024 and \
           self.shared_mem_usage <= MAX_SHARED_MEM and \
           self.cRegRows in [1, 2, 4] and \
           (self.rowModTileIsZero == 1 or (self.rowModTileIsZero == 0 and self.tileM > 1)) and \
           (self.fused_kernels == 1 or (self.fused_kernels > 1 and self.shape.p == self.tileP and self.shape.q == self.tileQ)) and \
           self.kEqVar in [0, 1]

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

def generate_kernel_decls(cases, useFusion, numKernels):
  configs = {}

  for (m, k, n, ps, qs) in cases:
    allSameShapes = len(set(ps + qs)) == 1
    __configs = []  
    for (_, _, p, q) in all_sliced_mults(m, k, n, ps, qs):
      TilePs = [min(p, 32)]
      TileQs = [2**i for i in range(2, max(2, int(math.log2(q)))+1)]
      TileKs = [2**i for i in range(2, max(2, int(math.log2(k)))+1)]
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
                        __configs += [KernelConfig(KronMatMulShape(m, tK, n, p, q), 
                                                                p, q, tQ, tP, tM, 
                                    rowModTileIsZero, regRows, regCols, kEqVar,
                                    numFusedKerns, "Float")]
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

  contents = f"#define MAX_K {shape.k}\n"
  contents += f"#define MIN_K {shape.k}\n"
  contents += f"#define MIN_KP_K {shape.p}\n"
  contents += f"#define MAX_KP_K {shape.p}\n"
  contents += "#define KERNEL_DECL(T, VecT, ElemType) \\\n"
  combinedConfigs = []
  for k in uniqueConfigs:
    configs = uniqueConfigs[k]
    configs = configs[:min(len(configs), numKernels)]
    combinedConfigs += configs
  
  combinedConfigs = list(set(combinedConfigs))
  for config in combinedConfigs:
    contents += config.code() + ",\\\n"
  contents = contents[:contents.rfind(",")]
  contents += "\n"
  with open(kernel_filename, "w") as f:
    #Remove last comma and backslash
    f.write(contents)

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

  #TODO: args should be like below:
  # distinct-shapes: No need of m and k. specify size of factor.
  # same-shapes: All factor of same shape 
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
  generate_kernel_decls(parsed_cases, not args.no_fuse, args.num_kernels)