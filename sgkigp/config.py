import os
import random
import torch
import gpytorch
import numpy as np
from enum import Enum

DEFAULT_DTYPE = 'float64'
PRJ_PATH = os.path.expanduser('~/sgkigp/')
WANDB_PATH = os.path.expanduser('~/sgkigp/wandb/')


DEFAULT_INDEX_DTYPE = torch.long


def np2torch(nparray, device, dtype):
    return torch.from_numpy(nparray).to(dtype=dtype, device=device)


def get_device(device=0):
    device = f"cuda:{device}" if (device >= 0 and torch.cuda.is_available()) else "cpu"
    return device


def dtype(default=DEFAULT_DTYPE, use_torch=False):

    if default == 'float32':
        return torch.float32 if use_torch else np.float32
    elif default == 'float64':
        return torch.float64 if use_torch else np.float64
    else:
        raise NotImplementedError


def set_seeds(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_gpytorch_settings(
        cg_tol=1e-2,
        eval_cg_tolerance=1e-4,
        max_cg_iters=100,
        num_trace_samples=5,
        max_cholesky_size=-1,   # Bye default Cholesky is turned off
    ):

    gpytorch.settings.cg_tolerance._set_value(cg_tol)
    gpytorch.settings.eval_cg_tolerance._set_value(eval_cg_tolerance)
    gpytorch.settings.max_cg_iterations._set_value(max_cg_iters)
    gpytorch.settings.num_trace_samples._set_value(num_trace_samples)
    gpytorch.settings.max_cholesky_size._set_value(max_cholesky_size)


class MethodName(Enum):
    SKI = 0
    SPARSE = 1
    SGPR = 2
    SKIP = 3


class KernelsType(Enum):
    RBFKERNEL = 0
    MATTERNHALF = 1
    MATTERNONEANDHALF = 2
    MATTERNTWOANDHALF = 3
    SOBOLEVONE = 4
    SOBOLEVTWO = 5
    SOBOLEVTHREE = 6
    SpectralMixtureOneDim = 7


class SobolevKernType(Enum):
    SOBOLEV1 = 0
    SOBOLEV2 = 1
    SOBOLEV3 = 2


class SgShifted(Enum):
    ZERO = 0
    ONE = 1
    TWO = 2


class InterpType(Enum):
    LINEAR = 0
    CUBIC = 1
    SIMPLEX = 2
    SHIFSG = 3


class SgBasisType(Enum):
    NAIVE = 0
    MODIFIED = 1
    CONSSTART = 2
    BOUNDSTART = 3
    ANOVA = 4


class MatmulAlgo(Enum):

    ITERATIVE = 1
    RECURSIVE = 2
    NAIVE = 3
    TOEPLITZ = 4


class SpareGridInfo(object):
    def __init__(self, grid_level=1, basis_type=0, interp_type=0):
        self.grid_level = grid_level
        self.basis_type = SgBasisType(basis_type)
        self.interp_type = InterpType(interp_type)
