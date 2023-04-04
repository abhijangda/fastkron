import torch
import gpytorch
import numpy as np

from gpytorch.lazy import delazify, ToeplitzLazyTensor

import sgkigp.config as config
from sgkigp.utils import get_in_out_orders
from sgkigp.interp.sparse.sglocations import get_sg_points_1d

from sgkigp.config import SgBasisType, MatmulAlgo


def process_algo(algo_type=MatmulAlgo.ITERATIVE):
    return None if algo_type == MatmulAlgo.RECURSIVE else torch.Tensor(1)


def process_length_scales(ls, ndim, device, dtype):
    return [torch.Tensor([np.sqrt(ls[i])]).view(1, 1).type(dtype).to(device) for i in range(ndim)]


def compute_covars(grid_level,
                   ndim,
                   umin,
                   umax,
                   ls=None,
                   base_kernel=None,
                   basis=SgBasisType.NAIVE,
                   device=config.get_device(),
                   dtype=config.dtype(use_torch=True),
                   use_toeplitz=True):

    # Checking length scales
    assert len(umin) == len(umax) == ndim
    if ls is not None:
        assert len(ls) == ndim
        ls = process_length_scales(ls, ndim, device=device, dtype=dtype)
    else:
        assert len(base_kernel) == ndim

    # Setting up kernels
    base_kernel_dim = []
    for i in range(ndim):
        if base_kernel is None:
            _base_kernel = gpytorch.kernels.RBFKernel()
            _base_kernel.lengthscale = ls[i]
        else:
            _base_kernel = base_kernel[i]
        _base_kernel.to(device=device, dtype=dtype)
        base_kernel_dim += _base_kernel,

    if use_toeplitz:
        covars = []
        sorting_orders = {}

        for i in range(ndim):
            covars_dim = []

            for level in range(grid_level + 1):
                locations = get_sg_points_1d(level, basis=basis, umin=umin[i], umax=umax[i])

                if level not in sorting_orders.keys():
                    sorting_orders[level] = get_in_out_orders(points=locations)

                # this step sort location which turns the kernel matrix into Toeplitz
                locations = torch.from_numpy(locations[sorting_orders[level][0]]).to(device=device, dtype=dtype)

                # edge- case
                if len(locations) == 1:
                    kernel_column = delazify(base_kernel_dim[i](locations, locations))
                    if use_toeplitz:
                        covars_dim_level = ToeplitzLazyTensor(kernel_column).squeeze(dim=0)
                        covars_dim += covars_dim_level,
                    else:
                        covars_dim += kernel_column.reshape(-1, 1),
                    continue
                kernel_column = delazify(base_kernel_dim[i](locations[0].reshape(-1, 1), locations)).squeeze()

                covars_dim_level = ToeplitzLazyTensor(kernel_column)

                if not use_toeplitz:
                    covars_dim_level = ToeplitzLazyTensor(kernel_column).evaluate()

                covars_dim += covars_dim_level,

            covars += covars_dim,
        return covars, sorting_orders

    covars = []
    for i in range(ndim):
        locations = get_sg_points_1d(grid_level, basis=basis, umin=umin[i], umax=umax[i])
        locations = torch.from_numpy(locations).to(device=device, dtype=dtype)
        covars += delazify(base_kernel_dim[i](locations)),
    return torch.stack(covars), None

