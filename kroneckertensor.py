import fire
from enum import Enum
from timeit import default_timer as timer

import torch
import tensorly
import gpytorch as gp


class KronTensorType(Enum):
    GPYTORCH = 0
    PROPOSED = 1


def get_device(device=0):
    return f"cuda:{device}" if (device >= 0 and torch.cuda.is_available()) else "cpu"


def get_dtype(dtype='float32'):
    return torch.float32 if dtype == 'float32' else torch.float64


def timekrnonmvm(krontensor, batch_size=1, dtype='float32', device=0):
    vector = torch.rand(krontensor.shape[0], batch_size)
    vector = vector.to(device=get_device(device), dtype=get_dtype(dtype))

    t_start = timer()
    out = krontensor.matmul(vector)
    return timer() - t_start


def main(
        tensortype: int = 1,
        dimsize: int =5,
        ndim: int =3,
        batch_size: int = 2,
        dtype: str = 'float32'
):
    tensortype = KronTensorType(tensortype)

    kron_factors = [torch.rand(dimsize, dimsize) for _ in range(ndim)]

    if tensortype == KronTensorType.GPYTORCH:
        krontensor = gp.lazy.KroneckerProductLazyTensor(*kron_factors)
    else:
        raise NotImplementedError

    mvm_time = timekrnonmvm(krontensor=krontensor, batch_size=batch_size, dtype=dtype)

    print("Time taken: ", mvm_time)


if __name__ == '__main__':
    fire.Fire(main)
