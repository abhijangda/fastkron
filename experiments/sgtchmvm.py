import fire
import torch
import numpy as np

from matplotlib import pyplot as plt

import sgkigp.config as config
from sgkigp.config import SgBasisType

from sgkigp.algos.sgnpmvm import get_kernel_matrix
from sgkigp.kernels.sgcovars import compute_covars
from sgkigp.algos.sgmatmuliterative import _sg_kernel_matmul_efficient

from sgkigp.interp.sparse.construct import sparse_grid_size_table
from sgkigp.interp.sparse.sgindices import compute_LI, compute_LI_pairs


def main(use_toeplitz=-1):

    # configuration set up
    USE_TOEPLITZ = True if use_toeplitz >= 0 else False
    print("Using Toeplitz: ", USE_TOEPLITZ)
    basis_type = SgBasisType.NAIVE
    DEFAULT_DTYPE = 'float64'
    grid_level = 5
    ndim = 6
    ls = [0.878, 0.8, 0.6, 0.5, 0.4, 0.2][:ndim]
    max_nvecs = 10

    assert len(ls) == ndim
    ls = [np.float64(_ls) for _ls in ls]
    K = ls
    sg_table, rg_sizes = sparse_grid_size_table(grid_level, ndim, basis=basis_type)
    sg_1d_sizes = np.cumsum(rg_sizes)
    sg_size = sg_table[-1][-1]
    nvecs = sg_size
    nvecs = np.min([max_nvecs, nvecs])
    X = np.random.rand(sg_size, nvecs)
    dtype = config.dtype(DEFAULT_DTYPE, use_torch=True)
    device = config.get_device()
    rhs = torch.from_numpy(X).to(dtype=dtype, device=device)

    # Compute indices
    LIs = compute_LI_pairs(grid_level, ndim, comb=False, basis=basis_type)
    LI = compute_LI(grid_level, ndim, basis=basis_type)
    kmat = get_kernel_matrix(LI, K, basis=basis_type)

    print("GL: " + str(grid_level) + ", n dimension: " + str(ndim))
    print("|GL|: " + str(LI.shape[0]) + ", n vectors: " + str(nvecs))

    # computing MVM

    covars, sorting_orders = compute_covars(grid_level, ndim, ls=ls,
                            umin=[0.0] * ndim, umax=[1.0] * ndim,
                            dtype=dtype, device=device,
                            use_toeplitz=USE_TOEPLITZ, basis=basis_type)

    desired = torch.from_numpy(np.matmul(kmat, X)).to(device=device, dtype=dtype)

    from sgkigp.algos.sgmatmuliterative import split_and_map, multiply_3d_splits, club_3d_tensors
    from sgkigp.utils import get_in_out_orders, torch_einsum_2d_times_3d, covar_matmul
    from line_profiler import LineProfiler
    lp = LineProfiler()

    lp.add_function(split_and_map)
    lp.add_function(multiply_3d_splits)
    lp.add_function(club_3d_tensors)
    lp.add_function(get_in_out_orders)
    lp.add_function(torch_einsum_2d_times_3d)
    lp.add_function(covar_matmul)

    lp_wrapper = lp( _sg_kernel_matmul_efficient)
    actual = lp_wrapper(covars, LIs, grid_level, ndim, rhs, sg_table, rg_sizes,
                        use_toeplitz=USE_TOEPLITZ, sorting_orders=sorting_orders)
    lp.print_stats()

    print("Shapes: ", desired.shape, actual.shape)
    print("Sum: ", torch.sum(desired - actual)/desired.numel())

    try:
        stats = torch.cuda.memory_stats()
        peak_bytes_requirement = stats["allocated_bytes.all.peak"]
        print('Peak memory (in mb): ', peak_bytes_requirement/1e6)
    except:
        print("Done!")

    # plt.figure(figsize=(10, 6))
    # plt.imshow(torch.abs(desired).detach().numpy())
    # plt.colorbar()
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.imshow(torch.abs(actual- desired).detach().numpy())
    # plt.colorbar()
    # plt.show()

    # print(torch.abs(actual).detach().numpy())
    # print(torch.abs(desired).detach().numpy())


if __name__ == '__main__':
    fire.Fire(main)
