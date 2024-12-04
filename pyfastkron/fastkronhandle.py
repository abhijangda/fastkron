class FastKronHandle:
    def hasBackend(backends, enumBackend):
        return (backends & int(enumBackend)) == int(enumBackend)

    def __init__(self, backend, libFastKron):
        self.libFastKron = libFastKron
        self.backends = libFastKron.backends()
        self.handle = libFastKron.init()
        self.backend = None

        if backend.lower() == 'x86':
            if FastKronHandle.hasBackend(self.backends,
                                         libFastKron.Backend.X86):
                libFastKron.initX86(self.handle)
                self.backend = libFastKron.Backend.X86
        elif backend.lower() == 'cuda':
            if FastKronHandle.hasBackend(self.backends,
                                         libFastKron.Backend.CUDA):
                import torch
                streams = [torch.cuda.current_stream().cuda_stream]
                libFastKron.initCUDA(self.handle, streams)
                self.backend = libFastKron.Backend.CUDA
        else:
            assert "Invalid backend", backend

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
        return self.libFastKron.gekmmSizes(self.handle, xshape[0],
                                           len(ps), ps, qs)

    def gekmmSizesForward(self, xshape, ps, qs):
        return self.libFastKron.gekmmSizesForward(self.handle, xshape[0],
                                                  len(ps), ps, qs)

    def xgemkm(self, fn, m, n, ps, qs, x, fs, z, alpha, beta, y,
               temp1, temp2, trX=False, trF=False):
        fn(self.handle, self.backend, m, n, ps, qs,
           x, self.fastKronOp(trX),
           fs, self.fastKronOp(trF),
           z,
           alpha, beta, y,
           temp1, temp2)

    def fastKronOp(self, tr):
        return self.libFastKron.Op.N if not tr else self.libFastKron.Op.T

    # TODO: Change argument order according to cublas API see comment in pywapper.cpp
    def xgemkmStridedBatched(self, fn, m, n, ps, qs, x, strideX, fs, strideFs,
                             batchCount, z, strideZ, alpha, beta, y, strideY,
                             temp1, temp2, trX=False, trF=False):
        fn(self.handle, self.backend, m, n, ps, qs,
           x, self.fastKronOp(trX), strideX,
           fs, self.fastKronOp(trF), strideFs,
           z, strideZ,
           alpha, beta, batchCount, y, strideY,
           temp1, temp2)

    def xgekmm(self, fn, m, n, ps, qs, x, fs, z, alpha, beta, y,
               temp1, temp2, trX=False, trF=False):
        fn(self.handle, self.backend, n, qs, ps, m,
           fs, self.fastKronOp(trF),
           x, self.fastKronOp(trX),
           z,
           alpha, beta, y,
           temp1, temp2)

    def xgekmmStridedBatched(self, fn, m, n, ps, qs, x, strideX, fs, strideFs,
                             batchCount, z, strideZ, alpha, beta, y, strideY,
                             temp1, temp2, trX=False, trF=False):
        fn(self.handle, self.backend, n, qs, ps, m,
           fs, self.fastKronOp(trF), strideFs,
           x, self.fastKronOp(trX), strideX,
           z, strideZ,
           alpha, beta, batchCount, y, strideY,
           temp1, temp2)

    def xmkmForward(self, fn, m, n, ps, qs, x, fs, z,
                    intermediates, trX=False, trF=False):
        fn(self.handle, self.backend, m, n, ps, qs,
           x, self.fastKronOp(trX), fs, self.fastKronOp(trF),
           z, intermediates)

    def xmkmForwardStridedBatched(self, fn, m, n, ps, qs, x, strideX,
                                  fs, strideFs, batchCount, z, strideZ,
                                  intermediates, strideIntermediates,
                                  trX=False, trF=False):
        fn(self.handle, self.backend, m, n, ps, qs,
            x, self.fastKronOp(trX), strideX,
            fs, self.fastKronOp(trF), strideFs,
            z, strideZ, batchCount, intermediates, strideIntermediates)

    def xkmmForward(self, fn, m, n, ps, qs, x, fs, z,
                    intermediates, trX=False, trF=False):
        fn(self.handle, self.backend, n, qs, ps, m,
           fs, self.fastKronOp(trF), x, self.fastKronOp(trX), z, intermediates)

    def xkmmForwardStridedBatched(self, fn, m, n, ps, qs, x, strideX,
                                  fs, strideFs, batchCount, z, strideZ,
                                  intermediates, strideIntermediates,
                                  trX=False, trF=False):
        fn(self.handle, self.backend, n, qs, ps, m,
           fs, self.fastKronOp(trF), strideFs, x, self.fastKronOp(trX),
           strideX, z, strideZ, batchCount, intermediates,
           strideIntermediates)
