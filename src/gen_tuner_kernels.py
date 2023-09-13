import argparse
import math 
import sys
import os
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
    self.num_threads = ((shape.k//tileP)//cRegRows) * (tileQ//cRegCols)
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
    return f"({self.num_threads}, {self.kron_rows}, {self.kron_cols}, {self.tileQ}, {self.tileP}, {self.tileM}, {self.cRegRows}, {self.cRegCols}, {self.fused_kernels}, {self.elemType})"

  def code(self):
    return "KernelInfo{"+\
            f"(void*)kronGemmKernel<T, VecT, {self.num_threads}, RowParallelismTy::Low, {self.tileM}, {self.rowModTileIsZero}, {self.shape.k}, {self.shape.p}, {self.shape.q}, {self.tileQ}, {self.kEqVar}, 1, {self.cRegRows}, {self.cRegCols}, {self.tileP}, {self.fused_kernels}>,"+\
            f"{self.num_threads}, {self.shape.p}, {self.shape.q}, {self.tileQ}, {self.tileM}, {self.shape.k}, {self.cRegRows}, {self.cRegCols}, {self.fused_kernels}, ElemType, {self.rowModTileIsZero}, {self.kEqVar}"+ "}"

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
    return hash(repr(self))

def generate_kernel_decls(ms, ks, ns, ps, qs):
  configs = []

  for m, k, n, p, q in zip(ms, ks, ns, ps, qs): 
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
                    for numFusedKerns in range(1, int(math.log(tK, tP))+1):
                      configs += [KernelConfig(KronMatMulShape(m, tK, n, p, q), 
                                                               p, q, tQ, tP, tM, 
                                  rowModTileIsZero, regRows, regCols, kEqVar,
                                  numFusedKerns, "Float")]

  print("Generated ", len(configs), " configs")
  
  #Filter only valid configs
  validConfigs = []
  for config in configs:
    if config.isValid():
      validConfigs += [config]
  
  print("Valid configs", len(validConfigs))

  uniqueConfigs = list(set(validConfigs))

  print("Unique configs", len(uniqueConfigs))

  contents = f"#define MAX_K {shape.k}\n"
  contents += f"#define MIN_K {shape.k}\n"
  contents += f"#define MIN_KP_K {shape.p}\n"
  contents += f"#define MAX_KP_K {shape.p}\n"
  contents += "#define KERNEL_DECL(T, VecT, ElemType) \\\n"
  for config in uniqueConfigs:
    contents += config.code() + ",\\\n"
  contents = contents[:contents.rfind(",")]
  contents += "\n"
  with open(kernel_filename, "w") as f:
    #Remove last comma and backslash
    f.write(contents)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', required=True, nargs="+", type=int)
  parser.add_argument('-k', required=True, nargs="+", type=int)
  parser.add_argument('-n', required=True, nargs="+", type=int)
  parser.add_argument('-p', required=True, nargs="+", type=int)
  parser.add_argument('-q', required=True, nargs="+", type=int)
  
  args = parser.parse_args()
  ms = len(args.m)
  try:
    assert ms == len(args.k)
    assert ms == len(args.n)
    assert ms == len(args.p)
    assert ms == len(args.q)
  except:
    print(f"Invalid no. of ms: {ms}, ns: {len(args.n)}, ps: {len(args.p)}, qs: {len(args.q)}")
    sys.exit(0)

  for (m, k, n, p, q) in zip(args.m, args.k, args.n, args.p, args.q):
    try:
      assert m >= 1 and k >= 1 and p >= 1 and n >= 1 and q >= 1
      assert k == p**n
      assert q > 1
    except:
      print(f"Invalid parameter m: {m}, k: {k}, n: {n}, p: {p}, q: {q}")
      sys.exit(0)

  generate_kernel_decls(args.m, args.k, args.n, args.p, args.q)