from functools import reduce
import platform

from .fastkronhandle import FastKronHandle
fastkronX86 = None
fastkronCUDA = None

if platform.system() == "Linux":
    if platform.machine() == "x86_64" or platform.machine() == "AMD64":
        try:
            from . import FastKronX86
            fastkronX86 = FastKronHandle("x86", FastKronX86)
        except:
            pass

    try:
        import torch
        if torch.cuda.is_available():
            if float(torch.version.cuda) >= 11.0:
                from . import FastKronCUDA
                fastkronCUDA = FastKronHandle("cuda", FastKronCUDA)
    except:
        pass


def product(values):
    return reduce((lambda a, b: a * b), values)


class FastKronBase:
    MMTypeMKM = 1
    MMTypeKMM = 2

    def __init__(self, x86, cuda):
        self.x86 = x86 and fastkronX86 is not None
        self.cuda = cuda and fastkronCUDA is not None

    def hasX86(self):
        return self.x86

    def hasCUDA(self):
        return self.cuda

    def supportedSystem(self):
        return platform.system() == "Linux" and \
               (self.x86 == True or self.cuda == True)

    def supportedProcessor(self):
        return (platform.machine() == "x86_64" or
                platform.machine() == "AMD64") and \
                self.x86 == True

    def isSupported(self, x, fs):
        return self.supportedSystem() and self.supportedProcessor() and \
               self.supportedTypes(x, fs) and self.supportedDevice(x)

    def supportedDevice(self, x):
        raise NotImplementedError()

    def supportedTypes(self, x, fs):
        raise NotImplementedError()

    def tensor_data_ptr(self, tensor):
        raise NotImplementedError()

    def asContiguousTensor(self, x, forceContiguous=False):
        raise NotImplementedError()

    def stride(self, x):
        raise NotImplementedError()

    def p(self, mmtype, f, trF):
        if mmtype == FastKronBase.MMTypeMKM:
            return f.shape[-2]
        elif mmtype == FastKronBase.MMTypeKMM:
            return f.shape[-1]

    def q(self, mmtype, f, trF):
        if mmtype == FastKronBase.MMTypeMKM:
            return f.shape[-1]
        elif mmtype == FastKronBase.MMTypeKMM:
            return f.shape[-2]

    def ps(self, mmtype, fs, trF):
        return [self.p(mmtype, f, trF) for f in fs]

    def qs(self, mmtype, fs, trF):
        return [self.q(mmtype, f, trF) for f in fs]

    def m(self, mmtype, x, trX):
        if mmtype == FastKronBase.MMTypeMKM:
            return x.shape[-2]
        elif mmtype == FastKronBase.MMTypeKMM:
            return x.shape[-1]

    def k(self, mmtype, x, trX):
        if mmtype == FastKronBase.MMTypeMKM:
            return x.shape[-1]
        elif mmtype == FastKronBase.MMTypeKMM:
            return x.shape[-2]

    def fptrs(self, fs):
        return [self.tensor_data_ptr(f) for f in fs]

    def reshapeInput(self, mmtype, x, fs):
        trX, x = self.asContiguousTensor(x)

        if mmtype == FastKronBase.MMTypeMKM:
            if trX:
                if x.ndim == 1:
                    x = x.reshape((x.shape[0], 1))
            else:
                if x.ndim == 1:
                    x = x.reshape((1, x.shape[0]))
        elif mmtype == FastKronBase.MMTypeKMM:
            if trX:
                if len(x.shape) == 1:
                    x = x.reshape((1, x.shape[0]))
            else:
                if len(x.shape) == 1:
                    x = x.reshape((x.shape[0], 1))

        trFs = []

        for f in fs:
            trF, f = self.asContiguousTensor(f)
            trFs += [trF]

        if len(set(trFs)) > 1:
            # When factors have different values of trF then
            # make all factors contiguous and set trF to False
            newFs = []
            for i, f in enumerate(fs):
                newFs += [self.asContiguousTensor(f, forceContiguous=True)[1]]
                trFs[i] = False

            fs = newFs

        for i, f in enumerate(fs):
            trF = trFs[i]
            if trF:
                if f.ndim == 1:
                    f = f.reshape((1, fs[0]))
            else:
                if f.ndim == 1:
                    f = f.reshape((fs[0], 1))

        return trX, x, trFs[0], fs

    def batchedDims(self, x1, x2, addPadding=True):
        x1batch = x1.shape[:-2]
        x2batch = x2.shape[:-2]

        if addPadding is True:
            if len(x1batch) < len(x2batch):
                x1batch = tuple([1] * (len(x2batch) - len(x1batch))) + x1batch
            elif len(x1batch) > len(x2batch):
                x2batch = tuple([1] * (len(x1batch) - len(x2batch))) + x2batch

        return x1batch, x2batch

    def broadcastShape(self, shape1, shape2):
        # This function requires both shapes to be of same length
        finalShape = ()

        for s1, s2 in zip(reversed(shape1), reversed(shape2)):
            if s1 == s2:
                finalShape += (s1,)
            elif s1 == 1:
                finalShape += (s2,)
            elif s2 == 1:
                finalShape += (s1,)
            else:
                raise ValueError(f"Shape {shape1} of x and {shape2}"
                                 "of f are not broadcastable")

        return tuple(reversed(finalShape))

    def gekmmSizes(self, mmtype, x, fs,
                   trX=False, trF=False, intermediates=False):
        self.checkShapeAndTypes(mmtype, x, fs, None, None, trX, trF)

        device_type = self.device_type(x)
        xbatch, fbatch = self.batchedDims(x, fs[0], addPadding=True)
        rs, ts = None, None

        if self.supportedSystem() and self.supportedProcessor():
            fn = None
            if device_type == 'cpu':
                if fastkronX86 is not None:
                    fn = fastkronX86.gekmmSizes if not intermediates else\
                         fastkronX86.gekmmSizesForward
                else:
                    raise ValueError(f"Device type '{device_type}'"
                                     "not supported")
            elif device_type == 'cuda':
                fn = fastkronCUDA.gekmmSizes if not intermediates else\
                     fastkronCUDA.gekmmSizesForward

            rs, ts = fn((self.m(mmtype, x, trX), self.k(mmtype, x, trX)),
                        self.ps(mmtype, fs, trF), self.qs(mmtype, fs, trF))
        else:
            rsN = product(self.qs(mmtype, fs, trF))
            rs, ts = (self.m(mmtype, x, trX) * rsN), -1

        zbroadcastshape = self.broadcastShape(xbatch, fbatch)
        zshape = []
        if mmtype == FastKronBase.MMTypeMKM:
            zshape = (self.m(mmtype, x, trX), rs//self.m(mmtype, x, trX))
        elif mmtype == FastKronBase.MMTypeKMM:
            zshape = (rs//self.m(mmtype, x, trX), self.m(mmtype, x, trX))

        if not intermediates:
            return zbroadcastshape + zshape, \
                (1 if len(zbroadcastshape) == 0 else
                 product(zbroadcastshape))*ts
        else:
            intermediateShapes = []
            for s in ts:
                if mmtype == FastKronBase.MMTypeMKM:
                    m = self.m(mmtype, x, trX)
                    shape = zbroadcastshape + (m, s//m)
                elif mmtype == FastKronBase.MMTypeKMM:
                    m = self.m(mmtype, x, trX)
                    shape = zbroadcastshape + (s//m, m)

                intermediateShapes += [shape]

            return zbroadcastshape + zshape, intermediateShapes

    def checkShapeAndTypes(self, mmtype, x, fs, z, y, trX, trF):
        # Only operate on 2-dims matrices
        if self.k(mmtype, x, trX) != product(self.ps(mmtype, fs, trF)):
            raise ValueError("Input operand x has a mismatch with its "
                             f"dimension 1 ('{self.k(mmtype, x, trX)}') "
                             "with dimension 0 of kronecker product of fs "
                             f"('{product(self.ps(mmtype, fs, trF))}')")

        if x.dtype != fs[0].dtype:
            raise ValueError("Operand types mismatches "
                             f"{x.dtype} != {fs[0].dtype}")

        if len(set([f.dtype for f in fs])) != 1:
            raise ValueError("Type of Kronecker factors do not match."
                             f"Found {len(set([f.dtype for f in fs]))}"
                             "different types")

        for i, f in enumerate(fs):
            if fs[0].shape[:-2] != f.shape[:-2]:
                raise ValueError("Batched dims of factors do not match:"
                                 f"{fs[0].shape[:-2]} != {f.shape[:-2]}")

        xbatch, fbatch = self.batchedDims(x, fs[0])
        finalShape = self.broadcastShape(xbatch, fbatch)

        if z is not None:
            expectedZshape = (self.m(mmtype, x, trX),
                              product(self.qs(mmtype, fs, trF)))
            if mmtype == FastKronBase.MMTypeKMM:
                expectedZshape = (expectedZshape[1], expectedZshape[0])

            if z.shape[-2:] != expectedZshape:
                raise ValueError(f"Output operand 'z' shape '{z.shape}'"
                                 f"mismatch with '{expectedZshape}'")
            if z.shape[:-2] != finalShape:
                raise ValueError("Output operand batched dimensions do not"
                                 "match with broadcasted dimensions"
                                 f"('{z.shape[:-2]}' != {finalShape})'")
            assert x.dtype == z.dtype

        if y is not None and z is not None:
            if self.broadcastShape(*self.batchedDims(y, z)) != z.shape[:-2]:
                raise ValueError(f"Input operand 'y' shape {y.shape}"
                                 "cannot be broadcasted to output"
                                 f"'z' shape {z.shape}")
            assert y.dtype == z.dtype

    def trLastTwoDims(self, mmtype, x):
        raise NotImplementedError()

    def device_type(self, x):
        raise NotImplementedError()

    def xgemm(self, handle, mmtype, fn, stridedBatchedFn,
              x, fs, z, alpha, beta, y, intermediates,
              trX=False, trF=False, writeIntermediates=False):

        self.checkShapeAndTypes(mmtype, x, fs, z, y, trX, trF)

        if beta != 0.0 and y is None:
            raise ValueError(f"When beta != 0 {beta} then"
                             "y should not be None")

        orig_xshape = x.shape
        orig_fshape = []
        for f in fs:
            orig_fshape += [f.shape]
        orig_yshape = y.shape if y is not None else None

        xbatch, fbatch = self.batchedDims(x, fs[0], addPadding=False)
        zbatch = z.shape[:-2]
        if y is not None:
            ybatch, _ = self.batchedDims(y, z, addPadding=True)
        else:
            ybatch = zbatch
        ismkm = mmtype == FastKronBase.MMTypeMKM
        iskmm = mmtype == FastKronBase.MMTypeKMM
        if (ismkm and len(fbatch) == 0 and
            (len(xbatch) == 0 or not trX) and ybatch == zbatch) or \
           (iskmm and len(fbatch) == 0 and
                len(xbatch) == 0 and len(zbatch) == 0):
            m = self.m(mmtype, x, trX)
            if len(xbatch) > 0:
                m = m * product(xbatch)

            if not writeIntermediates:
                handlefn = handle.xgemkm if ismkm else handle.xgekmm

                handlefn(fn, m, len(fs),
                         self.ps(mmtype, fs, trF), self.qs(mmtype, fs, trF),
                         self.tensor_data_ptr(x),
                         self.fptrs(fs),
                         self.tensor_data_ptr(z), alpha, beta,
                         self.tensor_data_ptr(y),
                         self.tensor_data_ptr(intermediates[0]),
                         self.tensor_data_ptr(intermediates[1]),
                         trX, trF)
            else:
                handlefn = handle.xmkmForward if ismkm else handle.xkmmForward

                handlefn(fn, m, len(fs),
                         self.ps(mmtype, fs, trF), self.qs(mmtype, fs, trF),
                         self.tensor_data_ptr(x),
                         self.fptrs(fs),
                         self.tensor_data_ptr(z),
                         self.tensor_data_ptr(intermediates),
                         trX, trF)
        else:
            xbatch, fbatch = self.batchedDims(x, fs[0], addPadding=True)
            if y is not None:
                ybatch, _ = self.batchedDims(y, z, addPadding=True)
            else:
                ybatch = zbatch

            z = z.reshape((product(z.shape[:-2]),)+z.shape[-2:])
            x = x.reshape((product(xbatch),)+x.shape[-2:])
            if writeIntermediates:
                batchShape = (product(z.shape[:-2]),)
                intermediates = [i.reshape(batchShape + i.shape[-2:])
                                 for i in intermediates]
            fs = list(fs)

            for i in range(len(fs)):
                fs[i] = fs[i].reshape((product(fbatch),)+fs[i].shape[-2:])
            # After above reshape a tensor made by slicing a parent tensor,
            # like [0:M:N] will force the reshape to make the tensor
            # contiguous, so recompute trX and trF
            trX, x, trF, fs = self.reshapeInput(mmtype, x, fs)

            if y is not None:
                y = y.reshape((product(ybatch),) + y.shape[-2:])

            # TODO: Compress batched dimensions into the three cases
            # of the below loop for better performance

            # Compute each batch of Z using stridedbatched
            batchLinearIdxZ = 0
            MaxLinearIdxZ = product(z.shape[:-2])
            while batchLinearIdxZ < MaxLinearIdxZ:
                # Go through each linear index of batch dimensions of Z
                xidx = fidx = zidx = yidx = None
                strideX = strideF = strideZ = strideY = 0
                strideZ = self.stride(z)[0]
                if writeIntermediates:
                    strideIntermediates = [self.stride(i)[0]
                                           for i in intermediates]
                zidx = yidx = batchLinearIdxZ

                xidx = 0
                fidx = 0
                yidx = 0
                tmpx = tmpf = tmpy = batchLinearIdxZ
                xDimProds = 1
                fDimProds = 1
                yDimProds = 1

                # Find linear index of x and fs for z
                for zdim, xdim, fdim, ydim in zip(reversed(zbatch),
                                                  reversed(xbatch),
                                                  reversed(fbatch),
                                                  reversed(ybatch)):
                    xidx += (tmpx % xdim) * xDimProds
                    xDimProds = xDimProds * xdim
                    tmpx = tmpx // zdim

                    fidx += (tmpf % fdim) * fDimProds
                    fDimProds = fDimProds * fdim
                    tmpf = tmpf // zdim

                    yidx += (tmpy % ydim) * yDimProds
                    yDimProds = yDimProds * ydim
                    tmpy = tmpy // zdim

                batchCount = zbatch[-1]

                dim = -1
                if xbatch[dim] > 1 and fbatch[dim] == 1:
                    # Batched X with same Fs
                    strideX = self.stride(x)[0]
                    strideF = [0 for f in fs]
                elif xbatch[dim] == fbatch[dim]:
                    # Each X with corresponding Fs
                    strideX = self.stride(x)[0]
                    strideF = [self.stride(f)[0] for f in fs]
                elif xbatch[dim] == 1 and fbatch[dim] > 1:
                    # Same X with batched Fs
                    strideX = 0
                    strideF = [self.stride(f)[0] for f in fs]

                if zbatch[dim] == ybatch[dim]:
                    strideY = strideZ
                elif zbatch[dim] > 1 and ybatch[dim] == 1:
                    strideY = 0
                else:
                    raise ValueError(f"Output 'z' {z.shape} cannot be"
                                     f"broadcastable to 'y' {y.shape}")

                m = self.m(mmtype, x, trX)
                if not writeIntermediates:
                    if mmtype == FastKronBase.MMTypeMKM:
                        handlefn = handle.xgemkmStridedBatched
                    else:
                        handlefn = handle.xgekmmStridedBatched
                    if y is not None:
                        yptr = self.tensor_data_ptr(y[yidx, :])
                    else:
                        yptr = 0
                    # Apply StridedBatched on the last dimension
                    handlefn(stridedBatchedFn, m, len(fs),
                             self.ps(mmtype, fs, trF),
                             self.qs(mmtype, fs, trF),
                             self.tensor_data_ptr(x[xidx, :]), strideX,
                             self.fptrs([f[fidx, :] for f in fs]), strideF,
                             batchCount,
                             self.tensor_data_ptr(z[zidx, :]), strideZ,
                             alpha, beta,
                             yptr, strideY,
                             self.tensor_data_ptr(intermediates[0]),
                             self.tensor_data_ptr(intermediates[1]),
                             trX, trF)
                else:
                    if mmtype == FastKronBase.MMTypeMKM:
                        handlefn = handle.xmkmForwardStridedBatched
                    else:
                        handlefn = handle.xkmmForwardStridedBatched
                    interSlice = [i[zidx, :] for i in intermediates]

                    # Apply StridedBatched on the last dimension
                    handlefn(stridedBatchedFn, m, len(fs),
                             self.ps(mmtype, fs, trF),
                             self.qs(mmtype, fs, trF),
                             self.tensor_data_ptr(x[xidx, :]), strideX,
                             self.fptrs([f[fidx, :] for f in fs]), strideF,
                             batchCount,
                             self.tensor_data_ptr(z[zidx, :]), strideZ,
                             self.tensor_data_ptr(interSlice),
                             strideIntermediates,
                             trX, trF)

                batchLinearIdxZ += zbatch[dim]

            x.reshape(orig_xshape)
            for i in range(len(fs)):
                fs[i] = fs[i].reshape(orig_fshape[i])
            if y is not None:
                y = y.reshape(orig_yshape)

    def shuffleGeMM(self, requires_grad, framework, mmtype, x, fs,
                    alpha=None, beta=None,
                    y=None, trX=False, trF=False):
        self.checkShapeAndTypes(mmtype, x, fs, None, y, trX, trF)

        rs, _ = self.gekmmSizes(mmtype, x, fs, trX=trX, trF=trF)
        m,  k = self.m(mmtype, x, trX), self.k(mmtype, x, trX)
        z = x
        zs = []

        enumerator = enumerate(fs) if mmtype == FastKronBase.MMTypeKMM else\
            enumerate(reversed(fs))
        for i, f in enumerator:
            fp = self.p(mmtype, f, False)
            fq = self.q(mmtype, f, False)
            if mmtype == FastKronBase.MMTypeMKM:
                z = z.reshape(z.shape[:-2] + (m * k//fp, fp))
                z = framework.matmul(z, f)
                z = z.reshape(z.shape[:-2] + (m, k//fp, fq))
                z = self.trLastTwoDims(mmtype, z)
                zshape = (m*k//fp, fq)
                grad_zshape = (m, (k//fp) * fq)
            elif mmtype == FastKronBase.MMTypeKMM:
                z = z.reshape(z.shape[:-2] + (fp, m * k//fp))
                z = framework.matmul(f, z)
                z = z.reshape(z.shape[:-2] + (fq, k//fp, m))
                z = self.trLastTwoDims(mmtype, z)
                zshape = (fq, m*k//fp)
                grad_zshape = (fq * (k//fp), m)
            zbatchShape = z.shape[:-3]
            z = z.reshape(z.shape[:-3] + zshape)
            if requires_grad:
                zs += [z.reshape(zbatchShape + grad_zshape)]
            k = (k//fp) * fq
        z = z.reshape(rs)
        if alpha is not None and alpha != 1:
            z = alpha * (z.reshape(rs))
        if beta is not None and beta != 0 and y is not None:
            z += beta * y

        return z
