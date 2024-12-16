from typing import List, Optional, Union, Tuple
from .fastkronbase import fastkronX86, fastkronCUDA, FastKronBase
import torch


class FastKronTorch(FastKronBase):
    def __init__(self):
        cuda = torch.cuda.is_available()

        super().__init__(True, cuda)

    def tensor_data_ptr(self, tensor: torch.Tensor) -> int:
        if tensor is None:
            return 0
        if type(tensor) is list or type(tensor) is tuple:
            return [t.data_ptr() for t in tensor]
        return tensor.data_ptr()

    def supportedDevice(self, x: torch.Tensor) -> str:
        return (x.device.type == 'cuda' and torch.version.hip is None and \
                fastkronCUDA is not None) or \
               (x.device.type == 'cpu') and fastkronX86 is not None

    def check(self, x: torch.Tensor, fs: List[torch.Tensor],
              y: Optional[torch.Tensor] = None,
              z: Optional[torch.Tensor] = None,
              stream: Optional[torch.cuda.Stream] = None):
        devices = [x.device] + [f.device for f in fs]
        if y is not None:
            devices += [y.device]
        if z is not None:
            devices += [z.device]

        devices = set(devices)

        if len(devices) != 1:
            raise RuntimeError('Expected all tensors to be on the same device,'
                               f'but found {len(devices)} devices:'
                               f'{[str(d) for d in devices]}')

        if x.device.type == "cuda" and stream is not None:
            if stream.device != x.device:
                raise RuntimeError(f"Expected stream to be on same device as"
                                   f"tensors, but found {stream.device} and"
                                   f"{x.device} are different")

    def supportedTypes(self, x: torch.Tensor, fs: List[torch.Tensor]) -> bool:
        return x.dtype in [torch.float32, torch.float64]

    def trLastTwoDims(self, mmtype: int,
                      x: torch.Tensor) -> torch.Tensor:
        if mmtype == FastKronBase.MMTypeMKM:
            return x.transpose(-1, -2)
        elif mmtype == FastKronBase.MMTypeKMM:
            return x.transpose(-2, -3)

    def device_type(self, x: torch.Tensor) -> str:
        return x.device.type

    def handle(self, x: torch.Tensor):
        if self.device_type(x) == "cpu":
            return fastkronX86
        elif self.device_type(x) == "cuda":
            return fastkronCUDA

    def asContiguousTensor(self, x: torch.Tensor,
                           forceContiguous: bool = False) -> \
            Tuple[bool, torch.Tensor]:
        if forceContiguous:
            return False, x.contiguous()
        if x.is_contiguous():
            return False, x
        if x.ndim > 1 and x.stride()[-2] == 1 and \
           x.stride()[-1] == x.shape[-2]:
            return True, x
        return False, x.contiguous()

    def stride(self, x: torch.Tensor) -> int:
        return x.stride()

    def gemkm(self, requires_grad: bool,
              x: torch.Tensor, fs: List[torch.Tensor],
              alpha: float = 1.0, beta: float = 0.0,
              y: Optional[torch.Tensor] = None,
              stream: Optional[torch.cuda.Stream] = None) -> torch.Tensor:
        if type(x) is not torch.Tensor:
            raise ValueError("Input 'x' should be a Tensor")
        if type(fs) is not list and type(fs) is not tuple:
            raise ValueError("Input 'fs' should be a list of Tensor")
        for i, f in enumerate(fs):
            if type(f) is not torch.Tensor:
                raise ValueError(f"Input fs[{i}] should be a Tensor")

        if requires_grad:
            if alpha != 1:
                raise ValueError("When requires_grad is True "
                                 f"alpha should be 1 but is {alpha}")
            if beta != 0:
                raise ValueError("When requires_grad is True "
                                 f"beta should be 0 but is {beta}")
            if y is not None:
                raise ValueError("When requires_grad is True "
                                 "y should be None")

        if y is not None and type(y) is not torch.Tensor:
            raise ValueError("Input 'y' should be a Tensor")

        is_vec = x.ndim == 1

        trX, x, trF, fs = self.reshapeInput(FastKronBase.MMTypeMKM, x, fs)
        if x.device.type == "cuda" and stream is None:
            stream = torch.cuda.current_stream()

        self.check(x, fs, y, None, stream)

        fn = None
        stridedBatchedFn = None
        libFastKron = self.handle(x).libFastKron
        if requires_grad:
            if x.dtype == torch.float:
                fn = libFastKron.smkmForward
                stridedBatchedFn = libFastKron.smkmForwardStridedBatched
            elif x.dtype == torch.double:
                fn = libFastKron.dmkmForward
                stridedBatchedFn = libFastKron.dmkmForwardStridedBatched
        else:
            if x.dtype == torch.float:
                fn = libFastKron.sgemkm
                stridedBatchedFn = libFastKron.sgemkmStridedBatched
            elif x.dtype == torch.double:
                fn = libFastKron.dgemkm
                stridedBatchedFn = libFastKron.dgemkmStridedBatched

        zs = []
        rs, ts = self.gekmmSizes(FastKronBase.MMTypeMKM, x, fs,
                                 trX=trX, trF=trF,
                                 intermediates=requires_grad)
        z = x.new_empty(rs)
        if requires_grad:
            zs = [x.new_empty(s) for s in ts]
        else:
            temp1 = x.new_empty(ts)
            temp2 = x.new_empty(ts) if rs != ts else None
            zs = [temp1, temp2]

        super().xgemm(self.handle(x), FastKronBase.MMTypeMKM,
                      fn, stridedBatchedFn,
                      x, fs, z, alpha, beta, y, zs,
                      trX, trF, writeIntermediates=requires_grad)
        z = z.reshape(rs)

        if requires_grad:
            zs = [inter.reshape(s) for inter, s in zip(zs, ts)]

        if is_vec and z.ndim > 1:
            z = z.squeeze()

        return z, zs

    def mkmBackward(self, grad_z: torch.Tensor,
                    x: torch.Tensor, x_requires_grad: bool,
                    fs: List[torch.Tensor], fs_requires_grad: List[bool],
                    zs: List[torch.Tensor]) -> torch.Tensor:
        trX, x, trF, fs = self.reshapeInput(FastKronBase.MMTypeMKM, x, fs)
        return self.__mkmBackward(grad_z, x, fs, zs,
                                  trX, trF,
                                  x.requires_grad,
                                  fs_requires_grad)

    def __mkmBackward(self, grad_z: torch.Tensor,
                      x: torch.Tensor, fs: List[torch.Tensor],
                      zs: List[torch.Tensor],
                      trX: bool, trF: bool,
                      x_requires_grad: bool, fs_requires_grad: List[bool]) ->\
            Tuple[Optional[torch.Tensor]]:
        is_vec = grad_z.ndim == 1
        if is_vec:
            grad_z = grad_z.unsqueeze(0)

        grad_fs = []
        grad_zs = [grad_z]
        zbatchShape = grad_z.shape[:-2]
        zs = tuple(zs) + (x,)
        for z, f, f_requires_grad in zip(zs, fs, fs_requires_grad):
            prev_grad_z = grad_zs[-1]

            fp = self.p(FastKronBase.MMTypeMKM, f, trF)
            fq = self.q(FastKronBase.MMTypeMKM, f, trF)

            # [m,k] -> [m,q,k/q]
            new_shape = (prev_grad_z.shape[:-2] + (prev_grad_z.shape[-2],)) + \
                (fq, prev_grad_z.shape[-1]//fq)
            prev_grad_z = prev_grad_z.reshape(new_shape)

            # ->[m,k/q,q]
            prev_grad_z = prev_grad_z.transpose(-1, -2)

            # ->[m*k/q,q]
            new_shape = prev_grad_z.shape[:-3] + \
                (prev_grad_z.shape[-3] * prev_grad_z.shape[-2],
                 prev_grad_z.shape[-1])
            prev_grad_z = prev_grad_z.reshape(new_shape)

            # Backward pass for z
            if x.data_ptr() != z.data_ptr() or x_requires_grad:
                grad_z = torch.matmul(prev_grad_z, f.mT)
                grad_z = grad_z.reshape(zbatchShape + z.shape[-2:])
                grad_zs += [grad_z]

            # Backward pass for f
            if f_requires_grad:
                orig_zshape = z.shape
                z = z.reshape(z.shape[:-2] +
                              ((z.shape[-2] * z.shape[-1])//fp, fp))
                trZ = z.mT
                grad_f = trZ @ prev_grad_z
                grad_fs += [grad_f]
                z = z.reshape(orig_zshape)
            else:
                grad_fs += [None]

        return (grad_zs[-1] if x_requires_grad else None,) + tuple(grad_fs)

    def gekmm(self, requires_grad: bool,
              fs: List[torch.Tensor], x: torch.Tensor,
              alpha: float = 1.0, beta: float = 0.0,
              y: Optional[torch.Tensor] = None,
              stream: Optional[torch.cuda.Stream] = None) -> \
            Tuple[torch.Tensor]:
        if type(x) is not torch.Tensor:
            raise ValueError("Input 'x' should be a Tensor")
        if type(fs) is not list and type(fs) is not tuple:
            raise ValueError("Input 'fs' should be a list of Tensor")
        for i, f in enumerate(fs):
            if type(f) is not torch.Tensor:
                raise ValueError(f"Input fs[{i}] should be a Tensor")

        if requires_grad:
            if alpha != 1:
                raise ValueError("When requires_grad is True"
                                 f"alpha should be 1 but is {alpha}")
            if beta != 0:
                raise ValueError("When requires_grad is True"
                                 f"beta should be 0 but is {beta}")
            if y is not None:
                raise ValueError("When requires_grad is True"
                                 "y should be None")

        if y is not None and type(y) is not torch.Tensor:
            raise ValueError("Input 'y' should be a 2D Tensor")

        if x.device.type == "cuda" and stream is None:
            stream = torch.cuda.current_stream()

        self.check(x, fs, y, None, stream)

        is_vec = x.ndim == 1

        trX, x, trF, fs = self.reshapeInput(FastKronBase.MMTypeKMM, x, fs)
        fn = None
        stridedBatchedFn = None

        libFastKron = self.handle(x).libFastKron
        if requires_grad:
            if x.dtype == torch.float:
                fn = libFastKron.skmmForward
                stridedBatchedFn = libFastKron.skmmForwardStridedBatched
            elif x.dtype == torch.double:
                fn = libFastKron.dkmmForward
                stridedBatchedFn = libFastKron.dkmmForwardStridedBatched
        else:
            if x.dtype == torch.float:
                fn = libFastKron.sgekmm
                stridedBatchedFn = libFastKron.sgekmmStridedBatched
            elif x.dtype == torch.double:
                fn = libFastKron.dgekmm
                stridedBatchedFn = libFastKron.dgekmmStridedBatched

        rs, ts = self.gekmmSizes(FastKronBase.MMTypeKMM, x, fs,
                                 trX=trX, trF=trF, intermediates=requires_grad)
        z = x.new_empty(size=rs)

        if requires_grad:
            zs = [x.new_empty(s) for s in ts]
        else:
            temp1 = x.new_empty(ts)
            temp2 = x.new_empty(ts) if rs != ts else None
            zs = [temp1, temp2]

        super().xgemm(self.handle(x), FastKronBase.MMTypeKMM,
                      fn, stridedBatchedFn,
                      x, fs, z, alpha, beta, y, zs, trX, trF,
                      writeIntermediates=requires_grad)
        if requires_grad:
            zs = [inter.reshape(s) for inter, s in zip(zs, ts)]

        z = z.reshape(rs)

        if is_vec and z.ndim > 1:
            z = z.squeeze()

        return z, zs

    def kmmBackward(self, grad_z: torch.Tensor,
                    x: torch.Tensor, x_requires_grad: bool,
                    fs: List[torch.Tensor], fs_requires_grad: List[bool],
                    zs: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        trX, x, trF, fs = self.reshapeInput(FastKronBase.MMTypeKMM, x, fs)
        zs = tuple(z.mT for z in zs)

        grads = self.__mkmBackward(grad_z.mT, x.mT, [f.mT for f in fs], zs,
                                   trX, trF,
                                   x_requires_grad, fs_requires_grad)
        grad_x = grads[0]
        grad_fs = grads[1:]

        return (grad_x.mT if grad_x is not None else None, ) + \
            tuple(g.mT if g is not None else None for g in grad_fs)

    def shuffleGeMM(self, mmtype: int,
                    x: torch.Tensor, fs: List[torch.Tensor],
                    alpha: float = 1.0, beta: float = 0.0,
                    y: Optional[torch.Tensor] = None) -> torch.Tensor:
        if type(x) is not torch.Tensor:
            raise ValueError("Input 'x' should be a Tensor")
        if type(fs) is not list and type(fs) is not tuple:
            raise ValueError("Input 'fs' should be a list of Tensor")
        for i, f in enumerate(fs):
            if type(f) is not torch.Tensor:
                raise ValueError(f"Input fs[{i}] should be a Tensor")
        if y is not None and type(y) is not torch.Tensor:
            raise ValueError("Input 'y' should be a Tensor")

        is_vec = x.ndim == 1

        trX, x, trF, fs = self.reshapeInput(mmtype, x, fs)

        self.check(x, fs, y, None, None)

        z = super().shuffleGeMM(False, torch, mmtype, x, fs,
                                alpha, beta, y, trX, trF)

        if is_vec and z.ndim > 1:
            z = z.squeeze()

        return z

    def shuffleGeMKM(self, x: torch.Tensor, fs: List[torch.Tensor],
                     alpha: float = 1.0, beta: float = 0.0,
                     y: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.shuffleGeMM(FastKronBase.MMTypeMKM,
                                x, fs, alpha, beta, y)

    def shuffleGeKMM(self, fs: List[torch.Tensor], x: torch.Tensor,
                     alpha: float = 1.0, beta: float = 0.0,
                     y: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.shuffleGeMM(FastKronBase.MMTypeKMM,
                                x, fs, alpha, beta, y)


fastkrontorch = FastKronTorch()


class MKM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, *fs: torch.Tensor) -> torch.Tensor:
        z, zs = fastkrontorch.gemkm(True, x, fs)
        ctx.save_for_backward(x, *fs, *zs)
        ctx.num_facs = len(fs)
        return z

    @staticmethod
    def backward(ctx, grad_z: torch.Tensor):
        num_facs = ctx.num_facs
        x = ctx.saved_tensors[0]
        x_requires_grad = ctx.needs_input_grad[0]
        fs = ctx.saved_tensors[1:num_facs + 1]
        fs_requires_grad = ctx.needs_input_grad[1:num_facs + 1]
        zs = ctx.saved_tensors[num_facs+1:]
        return fastkrontorch.mkmBackward(grad_z, x, x_requires_grad,
                                         fs, fs_requires_grad, zs)


class KMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, *fs: torch.Tensor) -> torch.Tensor:
        z, zs = fastkrontorch.gekmm(True, fs, x)
        ctx.save_for_backward(x, *fs, *zs)
        ctx.num_facs = len(fs)
        return z

    @staticmethod
    def backward(ctx, grad_z: torch.Tensor):
        num_facs = ctx.num_facs
        x = ctx.saved_tensors[0]
        x_requires_grad = ctx.needs_input_grad[0]
        fs = ctx.saved_tensors[1:num_facs + 1]
        fs_requires_grad = ctx.needs_input_grad[1:num_facs + 1]
        zs = ctx.saved_tensors[num_facs+1:]
        return fastkrontorch.kmmBackward(grad_z, x, x_requires_grad,
                                         fs, fs_requires_grad, zs)


def gemkm(x: torch.Tensor, fs: List[torch.Tensor],
          alpha: float = 1.0, beta: float = 0.0,
          y: Optional[torch.Tensor] = None) -> torch.Tensor:
    '''
    Perform Generalized Kronecker-Matrix Multiplication:

    $Z = alpha X * ( F^1 (*) F^2 (*) ... F^N ) + beta Y$

    Parameters
    ----------
    x : 2D torch tensor
    fs: A list of 2D torch tensor
    alpha and beta: constants
    y : 2D torch tensor
    trX: Transpose   x before computing GeKMM
    trF: Transpose each element of fs before computing GeKMM

    Returns
    -------
    z: 2D torch tensor
    '''
    if not fastkrontorch.isSupported(x, fs):
        return fastkrontorch.shuffleGeMKM(x, fs, alpha, beta, y)

    requires_grad = [f.requires_grad for f in fs if f.requires_grad]
    requires_grad = True if len(requires_grad) > 0 else x.requires_grad

    if requires_grad:
        return MKM.apply(x, *tuple(fs))
    else:
        return fastkrontorch.gemkm(False, x, fs, alpha, beta, y)[0]


def gekmm(fs: List[torch.Tensor], x: torch.Tensor,
          alpha: float = 1.0, beta: float = 0.0,
          y: Optional[torch.Tensor] = None) -> torch.Tensor:
    '''
    Perform Generalized Kronecker-Matrix Multiplication:

    $Z = alpha ( F^1 (*) F^2 (*) ... F^N ) * X + beta Y$

    Parameters
    ----------
    x : torch tensor
    fs: A list of torch tensor
    alpha and beta: constants
    y : torch tensor

    Returns
    -------
    z: torch tensor
    '''
    if not fastkrontorch.isSupported(x, fs):
        return fastkrontorch.shuffleGeKMM(fs, x, alpha, beta, y)

    requires_grad = [f.requires_grad for f in fs if f.requires_grad]
    requires_grad = True if len(requires_grad) > 0 else x.requires_grad

    if requires_grad:
        return KMM.apply(x, *tuple(fs))
    else:
        return fastkrontorch.gekmm(False, fs, x, alpha, beta, y)[0]
