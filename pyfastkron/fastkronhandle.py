class FastKronHandle:
  def hasBackend(backends, enumBackend) :
    return (backends & int(enumBackend)) == int(enumBackend)

  def __init__(self, backend, libFastKron):
      self.libFastKron = libFastKron
      self.backends = libFastKron.backends()
      self.handle = libFastKron.init()
      self.backend = None

      if backend.lower() == 'x86':
        if FastKronHandle.hasBackend(self.backends, libFastKron.Backend.X86):
          libFastKron.initX86(self.handle)
          self.backend = libFastKron.Backend.X86
      elif backend.lower() == 'cuda':
        if FastKronHandle.hasBackend(self.backends, libFastKron.Backend.CUDA):
          import torch
          libFastKron.initCUDA(self.handle, [torch.cuda.current_stream().cuda_stream])
          self.backend = libFastKron.Backend.CUDA
      else:
        assert "Invalid backend", backed

  def __del__(self):
    if self.handle is not None:
      self.libFastKron.destroy(self.handle)
      self.handle = self.backends = self.x86 = self.cuda = None

  def version(self):
    return self.libFastKron.version()

  def backend(self, device_type):
    if device_type == "cpu":
      return self.libFastKron.Backend.X86
    if device_type == "cuda":
      return self.libFastKron.Backend.CUDA
    raise RuntimeError(f"Invalid device {device_type}")

  def gekmmSizes(self, xshape, ps, qs):
    return self.libFastKron.gekmmSizes(self.handle, xshape[0], len(ps), ps, qs)

  def xgemkm(self, fn, m, n, ps, qs, x, fs, z, alpha, beta, y, 
             temp1, temp2, trX = False, trF = False):
    fn(self.handle, self.backend, m, n, ps, qs,
       x, self.libFastKron.Op.N if not trX else self.libFastKron.Op.T,
       fs, self.libFastKron.Op.N if not trF else self.libFastKron.Op.T,
       z,
       alpha, beta, y, 
       temp1, temp2)

  #TODO: Change argument order according to cublas API see comment in pywapper.cpp
  def xgemkmStridedBatched(self, fn, m, n, ps, qs, x, strideX, fs, strideFs,
                           batchCount, z, strideZ, alpha, beta, y, strideY,
                           temp1, temp2, trX = False, trF = False):
    fn(self.handle, self.backend, m, n, ps, qs,
       x, self.libFastKron.Op.N if not trX else self.libFastKron.Op.T, strideX,
       fs, self.libFastKron.Op.N if not trF else self.libFastKron.Op.T, strideFs,
       z, strideZ,
       alpha, beta, batchCount, y, strideY,
       temp1, temp2)
  
  def xgekmm(self, fn, m, n, ps, qs, x, fs, z, alpha, beta, y, 
             temp1, temp2, trX = False, trF = False):
    fn(self.handle, self.backend, n, qs, ps, m,
       fs, self.libFastKron.Op.N if not trF else self.libFastKron.Op.T,
       x, self.libFastKron.Op.N if not trX else self.libFastKron.Op.T,
       z,
       alpha, beta, y, 
       temp1, temp2)

  #TODO: Change argument order according to cublas API see comment in pywapper.cpp
  def xgekmmStridedBatched(self, fn, m, n, ps, qs, x, strideX, fs, strideFs,
                           batchCount, z, strideZ, alpha, beta, y, strideY,
                           temp1, temp2, trX = False, trF = False):
    fn(self.handle, self.backend, n, qs, ps, m,
       fs, self.libFastKron.Op.N if not trF else self.libFastKron.Op.T, strideFs,
       x, self.libFastKron.Op.N if not trX else self.libFastKron.Op.T, strideX,
       z, strideZ,
       alpha, beta, batchCount, y, strideY,
       temp1, temp2)