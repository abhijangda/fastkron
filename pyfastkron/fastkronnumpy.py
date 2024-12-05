from .fastkronbase import fastkronX86, FastKronBase
import numpy as np

from typing import List, Optional, Union, Tuple


class FastKronNumpy(FastKronBase):
    def __init__(self):
        super().__init__(True, False)

    def tensor_data_ptr(self, tensor: np.ndarray) -> int:
        if tensor is None:
            return 0
        return tensor.ctypes.data

    def supportedDevice(self, x: np.ndarray) -> bool:
        return fastkronX86 is not None

    def supportedTypes(self, x: np.ndarray, fs: List[np.ndarray]) -> bool:
        return x.dtype in [np.float32, np.double]

    def trLastTwoDims(self, mmtype: int,
                      x: np.ndarray) -> np.ndarray:
        if mmtype == FastKronBase.MMTypeMKM:
            axes = list(range(len(x.shape) - 2)) +\
                [len(x.shape) - 1, len(x.shape) - 2]
        elif mmtype == FastKronBase.MMTypeKMM:
            axes = list(range(len(x.shape) - 3)) + \
                [len(x.shape) - 2, len(x.shape) - 3, len(x.shape) - 1]
        return x.transpose(axes)

    def asContiguousTensor(self, x,
                           forceContiguous=False) -> Tuple[bool, np.ndarray]:
        if forceContiguous:
            return False, x.ascontiguousarray()
        if x.data.c_contiguous:
            return False, x
        strides = self.stride(x)
        if x.ndim > 1 and strides[-2] == 1 and strides[-1] == x.shape[-2] * 1:
            return True, x
        return False, x.ascontiguousarray()

    def stride(self, x: np.ndarray) -> List[int]:
        return [s//x.dtype.itemsize for s in x.strides]

    def device_type(self, x: np.ndarray) -> str:
        return "cpu"

    def gemkm(self, x: np.ndarray, fs: List[np.ndarray],
              alpha: float = 1, beta: float = 0,
              y: Optional[np.ndarray] = None) -> np.ndarray:
        if type(x) is not np.ndarray:
            raise ValueError("Input 'x' should be a ndarray")
        if type(fs) is not list:
            raise ValueError("Input 'fs' should be a list of np.ndarray")
        for i, f in enumerate(fs):
            if type(f) is not np.ndarray:
                raise ValueError(f"Input fs[{i}] should be a ndarray")

        trX, x, trF, fs = self.reshapeInput(FastKronBase.MMTypeMKM, x, fs)

        fn = None
        stridedBatchedFn = None
        libFastKron = fastkronX86.libFastKron
        if x.dtype == np.float32:
            fn = libFastKron.sgemkm
            stridedBatchedFn = libFastKron.sgemkmStridedBatched
        elif x.dtype == np.double:
            fn = libFastKron.dgemkm
            stridedBatchedFn = libFastKron.dgemkmStridedBatched

        rs, ts = self.gekmmSizes(FastKronBase.MMTypeMKM,
                                 x, fs, trX=trX, trF=trF)
        temp1 = np.ndarray(ts, dtype=x.dtype)
        requires_temp2 = rs != ts or y is not None
        temp2 = np.ndarray(ts, dtype=x.dtype) if requires_temp2 else None
        z = np.ndarray(shape=rs, dtype=x.dtype)

        super().xgemm(fastkronX86, FastKronBase.MMTypeMKM,
                      fn, stridedBatchedFn,
                      x, fs, z, alpha, beta, y, [temp1, temp2], trX, trF)

        z = z.reshape(rs)
        return z

    def gekmm(self, fs: List[np.ndarray], x: np.ndarray,
              alpha: float = 1, beta: float = 0,
              y: Optional[np.ndarray] = None) -> np.ndarray:

        if type(x) is not np.ndarray:
            raise ValueError("Input 'x' should be a ndarray")
        if type(fs) is not list:
            raise ValueError("Input 'fs' should be a list of np.ndarray")
        for i, f in enumerate(fs):
            if type(f) is not np.ndarray:
                raise ValueError(f"Input fs[{i}] should be a ndarray")

        fn = None
        stridedBatchedFn = None
        libFastKron = fastkronX86.libFastKron
        if x.dtype == np.float32:
            fn = libFastKron.sgekmm
            stridedBatchedFn = libFastKron.sgekmmStridedBatched
        elif x.dtype == np.double:
            fn = libFastKron.dgekmm
            stridedBatchedFn = libFastKron.dgekmmStridedBatched

        trX, x, trF, fs = self.reshapeInput(FastKronBase.MMTypeMKM, x, fs)

        rs, ts = self.gekmmSizes(FastKronBase.MMTypeKMM,
                                 x, fs, trX=trX, trF=trF)
        temp1 = np.ndarray(ts, dtype=x.dtype)
        requires_temp2 = rs != ts or y is not None
        temp2 = np.ndarray(ts, dtype=x.dtype) if requires_temp2 else None
        z = np.ndarray(shape=rs, dtype=x.dtype)
        super().xgemm(fastkronX86, FastKronBase.MMTypeKMM,
                      fn, stridedBatchedFn,
                      x, fs, z, alpha, beta, y, [temp1, temp2], trX, trF)
        z = z.reshape(rs)

        return z

    def shuffleGeMM(self, mmtype: int,
                    x: np.ndarray, fs: List[np.ndarray],
                    alpha: float = 1, beta: float = 0,
                    y: Optional[np.ndarray] = None) -> np.ndarray:
        if type(x) is not np.ndarray:
            raise ValueError("Input 'x' should be a ndarray")
        if type(fs) is not list:
            raise ValueError("Input 'fs' should be a list of np.ndarray")
        for i, f in enumerate(fs):
            if type(f) is not np.ndarray:
                raise ValueError(f"Input fs[{i}] should be a ndarray")
        if y is not None and type(y) is not np.ndarray:
            raise ValueError("Input 'y' should be a ndarray")

        is_vec = x.ndim == 1

        trX, x, trF, fs = self.reshapeInput(mmtype, x, fs)

        z = super().shuffleGeMM(False, np, mmtype, x, fs,
                                alpha, beta, y, trX, trF)

        if is_vec and z.ndim > 1:
            z = z.squeeze()

        return z

    def shuffleGeMKM(self, x: np.ndarray, fs: List[np.ndarray],
                     alpha: float = 1, beta: float = 0,
                     y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.shuffleGeMM(FastKronBase.MMTypeMKM, x, fs, alpha, beta, y)

    def shuffleGeKMM(self, fs: List[np.ndarray], x: np.ndarray,
                     alpha: float = 1, beta: float = 0,
                     y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.shuffleGeMM(FastKronBase.MMTypeKMM, x, fs, alpha, beta, y)


fastkronnumpy = FastKronNumpy()


def gemkm(x: np.ndarray, fs: List[np.ndarray],
          alpha: float = 1.0, beta: float = 0.0,
          y: Optional[np.ndarray] = None) -> np.ndarray:
    '''
    Perform Generalized Matrix Kronecker-Matrix Multiplication :

    $Z = a X * (F^1 (*) F^2 (*) ... F^N) + b Y$

    Parameters
    ----------
    x : numpy array
    fs: A list of numpy array
    alpha and beta: constants
    y : numpy array

    Returns
    -------
    z: numpy array
    '''

    if not fastkronnumpy.isSupported(x, fs):
        return fastkronnumpy.shuffleGeMKM(x, fs, alpha, beta, y)

    return fastkronnumpy.gemkm(x, fs, alpha, beta, y)


def gekmm(fs: List[np.ndarray], x: np.ndarray,
          alpha: float = 1.0, beta: float = 0.0,
          y: Optional[np.ndarray] = None) -> np.ndarray:
    '''
    Perform Generalized Kronecker-Matrix Matrix Multiplication :

    $Z = a (F^1 (*) F^2 (*) ... F^N) * X + b Y$

    Parameters
    ----------
    x : numpy array
    fs: A list of numpy array
    alpha and beta: constants
    y : numpy array

    Returns
    -------
    z: numpy array
    '''

    if not fastkronnumpy.isSupported(x, fs):
        return fastkronnumpy.shuffleGeKMM(fs, x, alpha, beta, y)

    return fastkronnumpy.gekmm(fs, x, alpha, beta, y)
