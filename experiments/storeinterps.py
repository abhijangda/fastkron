import os
import tqdm
import torch
import argparse
import gpytorch as gp

import numpy as np
from sgkigp.interp.sparse.construct import G as SGsize

rg_size = lambda grid_level, ndim: np.prod([max(int(np.floor(SGsize(grid_level, ndim) ** (1 / ndim))), 5)]*ndim)
import sgkigp.config as config


def main(ndim: int = 4, grid_level: int = 4,
         interp_type: int = 0, log_path: str = '', method: int = 0,
         seed = 1337, device: int = 0, dtype: str = 'float32', func_val = 3):

    # Set-seeds and device
    config.set_seeds(seed=seed)
    device = config.get_device(device=device)
    dtype = config.dtype(dtype, use_torch=True)

    # Preparing data iterations
    if func_val > 0:  # synthetic dataset
        datapath = os.path.expanduser("~/sgkigp/data/")
        datapath = datapath + 'f' + str(func_val) + "_ndim" + str(ndim) + ".pt"
        train_x, train_y, val_x, val_y, test_x, test_y = torch.load(datapath)

    else: # real dataset
        raise NotImplementedError

    # Move to device and dtype
    train_x = train_x.to(device=device, dtype=dtype)
    train_y = train_y.to(device=device, dtype=dtype)
    val_x = val_x.to(device=device, dtype=dtype)
    val_y = val_y.to(device=device, dtype=dtype)
    test_x = test_x.to(device=device, dtype=dtype)
    test_y = test_y.to(device=device, dtype=dtype)

    train_x = torch.stack([train_x, val_x, test_x])

    # Setting about same size as Sparse GRID
    boundary_slack = 0.01
    ndim = train_x.size(-1)
    grid_size_dim = max(int(np.floor(SGsize(grid_level, ndim) ** (1 / ndim))), 5)
    grid_size = [grid_size_dim] * ndim
    umin = torch.min(train_x, 0).values.cpu().numpy() - boundary_slack
    umax = torch.max(train_x, 0).values.cpu().numpy() + boundary_slack
    grid_bounds = tuple((umin[i], umax[i]) for i in range(ndim))

    kernel = gp.kernels.GridInterpolationKernel(
            gp.kernels.RBFKernel(), grid_size=grid_size, grid_bounds=grid_bounds,
        )


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
