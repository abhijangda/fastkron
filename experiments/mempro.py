import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm

import sgkigp.config as config
from sgkigp.config import SgBasisType, InterpType, MethodName

from gpytorch.lazy import DiagLazyTensor
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.grid_interpolation_kernel import GridInterpolationKernel

from sgkigp.interp.sparse.construct import G
from sgkigp.interp.sparse.weights import compute_phi
from sgkigp.interp.sginterp import SparseInterpolation
from sgkigp.kernels.gridinterpkernel import ModifiedGridInterpolationKernel

# TODO: resolve load_dataset issue ....

from pytorch_memlab import profile
from pytorch_memlab import MemReporter


_cache_ = {
  'memory_allocated': 0,
  'max_memory_allocated': 0,
  'memory_reserved': 0,
  'max_memory_reserved': 0,
}


def _get_memory_info(info_name, unit, rval=False):

    tab = '\t'
    if info_name == 'memory_allocated':
        current_value = torch.cuda.memory.memory_allocated()
    elif info_name == 'max_memory_allocated':
        current_value = torch.cuda.memory.max_memory_allocated()
    elif info_name == 'memory_reserved':
        tab = '\t\t'
        current_value = torch.cuda.memory.memory_reserved()
    elif info_name == 'max_memory_reserved':
        current_value = torch.cuda.memory.max_memory_reserved()
    else:
        raise ValueError()

    divisor = 1
    if unit.lower() == 'kb':
        divisor = 1024
    elif unit.lower() == 'mb':
        divisor = 1024*1024
    elif unit.lower() == 'gb':
        divisor = 1024*1024*1024
    else:
        raise ValueError()

    diff_value = current_value - _cache_[info_name]
    _cache_[info_name] = current_value

    if rval:
        return current_value/divisor
    else:
        return f"{info_name}: \t {current_value} ({current_value/divisor:.3f} {unit.upper()})" \
           f"\t diff_{info_name}: {diff_value} ({diff_value/divisor:.3f} {unit.upper()})"


def print_memory_info(unit='kb'):

    print(_get_memory_info('memory_allocated', unit))
    print(_get_memory_info('max_memory_allocated', unit))
    print(_get_memory_info('memory_reserved', unit))
    print(_get_memory_info('max_memory_reserved', unit))
    print('')


def get_memory_info(unit='kb'):
    #return {key:_get_memory_info(key, unit, True) for key in['memory_allocated', 'memory_reserved']}
    return {key: _get_memory_info(key, unit, True) for key in ['memory_allocated', 'memory_reserved']}


def main(ndim: int = 4, grid_level: int = 4,
         interp_type: int = 0, log_path: str = '', method: int = 0
         ):

    device = config.get_device()
    dtype = config.dtype('float32', use_torch=True)

    method = MethodName(method)
    log_path = os.path.expanduser(log_path)

    train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(ndim, device=device, dtype=dtype)

    umin = [0.0]*ndim
    umax = [1.0]*ndim
    comb = True
    interp_type = InterpType(interp_type)
    basis = SgBasisType.NAIVE  # mem is independent of this option

    if method == MethodName.SPARSE:
        profile_sparse_grid(umin, umax, ndim, comb, basis,
                            device, dtype, train_x, interp_type, grid_level, log_path)
    elif method == MethodName.SKI:
        profile_ski_matrix(train_x, grid_level, device, dtype, log_path)
    else:
        raise NotImplementedError


def profile_ski_matrix(train_x, grid_level, device, dtype, log_path):

    log_path = log_path + "_ski_"

    # Setting about same size as Sparse GRID
    ndim = train_x.size(-1)
    grid_size_dim = max(int(np.floor(G(grid_level, ndim) ** (1 / ndim))), 5)
    grid_size = [grid_size_dim] * ndim
    umin = torch.min(train_x, 0).values.cpu().numpy()
    umax = torch.max(train_x, 0).values.cpu().numpy()
    grid_bounds = tuple((umin[i], umax[i]) for i in range(ndim))

    # reporter = MemReporter()
    # print("First report ...")
    # reporter.report(verbose=True)
    #
    # print("Grid: ", grid_size)

    if False:
        grid_interp_kernel = ModifiedGridInterpolationKernel(
            base_kernel=RBFKernel(),
            grid_size=grid_size,
            num_dims=ndim,
            grid_bounds=grid_bounds
        ).to(device=device, dtype=dtype)

    else:
        grid_interp_kernel = GridInterpolationKernel(
            base_kernel=RBFKernel(),
            grid_size=grid_size,
            num_dims=ndim,
            grid_bounds=grid_bounds
        ).to(device=device, dtype=dtype)


    # print("Report ... 4")
    # reporter.report(verbose=True)

    npoints = train_x.shape[0]
    rhs = torch.randn(npoints, device=device, dtype=dtype)
    diag = torch.ones(npoints, device=device, dtype=dtype)
    grid_interp_kernel = grid_interp_kernel.forward(train_x, train_x) + DiagLazyTensor(diag)

    # print("Report ... 5")
    # reporter.report(verbose=True)

    #grid_interp_kernel.requires_grad = False
    # with torch.no_grad():
    #     solve_out = grid_interp_kernel._solve(rhs, preconditioner=None)

    print_memory_info(unit='gb')


def make_sparse_tensor(index_tensor, value_tensor, n_rows, n_cols):

    # Make the sparse tensor
    type_name = value_tensor.type().split(".")[-1]  # e.g. FloatTensor
    interp_size = torch.Size((n_rows, n_cols))
    if index_tensor.is_cuda:
        cls = getattr(torch.cuda.sparse, type_name)
    else:
        cls = getattr(torch.sparse, type_name)
    return cls(index_tensor, value_tensor, interp_size)


def profile_sparse_grid(umin, umax, ndim, comb, basis, device, dtype, train_x, interp_type, grid_level, log_path):

    grid_bounds = SparseInterpolation().compute_grid_bounds(umin=umin, umax=umax, ndim=ndim)
    sub_grids = SparseInterpolation().create_sparse_grid(
        grid_level=grid_level,
        ndim=ndim,
        grid_bounds=grid_bounds,
        comb=comb,
        basis=basis,
        device=device,
        dtype=dtype
    )

    num_subgrids = len(sub_grids)
    coefficients_row_sparsity = np.zeros((train_x.shape[0]))
    for sub_grid in tqdm(sub_grids):
        coefficients = SparseInterpolation().sg_get_basis(x_target=train_x, grid=sub_grid,
                                                          interp_type=interp_type,
                                                          basis=basis, comb=comb, dtype=dtype),

        unique, counts = np.unique(coefficients[-1]._indices().cpu().numpy()[0, :], return_counts=True)
        coefficients_row_sparsity += counts

    raw_mean_sparsity = np.mean(coefficients_row_sparsity)

    print("Raw sparsity mean: ", raw_mean_sparsity)

    stats = {
        'sg': num_subgrids,
        'sparsity': raw_mean_sparsity
    }
    stats.update(get_memory_info(unit='gb'))
    with open(log_path + "_stats.yaml", 'w') as outfile:
        json.dump(stats, outfile)

    Phi = compute_phi(X=train_x, grid_level=grid_level, ndim=ndim, umin=umin, umax=umax, comb=comb,
                      interp_type=interp_type,
                      basis=basis, use_torch=True, dtype=dtype, device=device)
    stats = {
        'sg': num_subgrids,
        'sparsity': raw_mean_sparsity
    }
    stats.update(get_memory_info(unit='gb'))

    with open(log_path + "_stats.yaml", 'w') as outfile:
        json.dump(stats, outfile)

    print("stats: ", stats)
    print_memory_info(unit='gb')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--grid_level', type=int, default=2)
    parser.add_argument('--ndim', type=int, default=2)
    parser.add_argument('--interp_type', type=int, default=0)
    parser.add_argument('--log', type=str, default='')
    parser.add_argument('--method', type=int, default=2)
    args = parser.parse_args()

    main(ndim=args.ndim, grid_level=args.grid_level,
         interp_type=args.interp_type, log_path=args.log, method=args.method)

