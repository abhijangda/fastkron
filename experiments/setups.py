import os
import torch
import numpy as np

from sgkigp.interp.sparse.construct import G as SGsize

from sgkigp.config import MethodName, InterpType, SgShifted
from experiments.models import SKIModel_base, SKIModel_bypass
from experiments.models import SGPRModel
from experiments.models import SparseGridGpModel
from experiments.models import SKIPModel


def load_func_dataset(ndim, device, dtype, func_val=3):
    datapath = os.path.expanduser("~/sgkigp/data/")
    datapath = datapath + 'f' + str(func_val) + "_ndim" + str(ndim) + ".pt"
    train_x, train_y, val_x, val_y, test_x, test_y = torch.load(datapath)

    put_on_device = lambda x: x.to(device=device, dtype=dtype)
    return (put_on_device(dd) for dd in [train_x, train_y, val_x, val_y, test_x, test_y])


def setup_models(method, train_x, train_y, n_inducing,
                 kernel_type, min_noise, device, dtype,
                 val_x, test_x, boundary_slack, grid_level, interp_type, basis_type,
                 bypass_covar, use_modified=False, grid_size_dim=-1,
                 sg_shifted=SgShifted.ZERO):

    if method == MethodName.SGPR:
        return setup_sgpr(train_x, train_y=train_y, n_inducing=n_inducing,
                          kernel_type=kernel_type, min_noise=min_noise, device=device, dtype=dtype)

    elif method == MethodName.SPARSE:
        return setup_sparse_grid(train_x, val_x, test_x, boundary_slack, train_y, grid_level,
                                 interp_type, basis_type, kernel_type, min_noise, device, dtype, sg_shifted)

    elif method == MethodName.SKI:
        return setup_ski(train_x, boundary_slack, train_y, grid_level,
                         kernel_type, min_noise, device, dtype,
                         bypass_covar=bypass_covar, use_modified=use_modified,
                         interp_type=interp_type, grid_size_dim=grid_size_dim)
        # train_x, boundary_slack, train_y, grid_level, kernel_type,
        # min_noise, device, dtype, interp_type = InterpType.CUBIC,
        # bypass_covar = False, use_modified = False

    elif method == MethodName.SKIP:
        return setup_skip(train_x,  train_y,  kernel_type, min_noise, device, dtype)
    else:
        raise NotImplementedError


def setup_sgpr(train_x, train_y, n_inducing, kernel_type, min_noise, device, dtype):
    return SGPRModel(train_x, train_y, n_inducing=n_inducing,
                     kernel_type=kernel_type, min_noise=min_noise).to(device=device, dtype=dtype)


def setup_sparse_grid(train_x, val_x, test_x, boundary_slack, train_y, grid_level,
                      interp_type, basis_type, kernel_type, min_noise, device, dtype, sg_shifted):

    try:
        all_x = torch.vstack([train_x, val_x, test_x])
        umin = torch.min(all_x, 0).values.numpy() - boundary_slack
        umax = torch.max(all_x, 0).values.numpy() + boundary_slack
    except TypeError:
        umin = torch.min(train_x, 0).values.cpu().numpy() - boundary_slack
        umax = torch.max(train_x, 0).values.cpu().numpy() + boundary_slack

    print("Umin: ", umin)
    print("Umax: ", umax)
    model = SparseGridGpModel(
        train_x, train_y,
        grid_level=grid_level,
        umin=umin,
        umax=umax,
        interp_type=interp_type,
        basis_type=basis_type,
        kernel_type=kernel_type,
        sg_shifted=sg_shifted,
        min_noise=min_noise,
        dtype=dtype
    ).to(device=device, dtype=dtype)

    return model


def setup_skip(train_x,  train_y,  kernel_type, min_noise, device, dtype):
    return SKIPModel(
        train_x, train_y,
        kernel_type=kernel_type,
        grid_size=100,
        min_noise=min_noise
    ).to(device=device, dtype=dtype)


def setup_ski(train_x, boundary_slack, train_y, grid_level, kernel_type,
              min_noise, device, dtype, interp_type = InterpType.CUBIC,
              bypass_covar=False, use_modified=False, grid_size_dim=-1):

    # Setting about same size as Sparse GRID

    ndim = train_x.size(-1)
    if grid_size_dim <= 0:
        grid_size_dim = int(np.floor(SGsize(grid_level, ndim) ** (1 / ndim))) + 1
    grid_size = [grid_size_dim]*ndim
    umin = torch.min(train_x, 0).values.cpu().numpy() - boundary_slack
    umax = torch.max(train_x, 0).values.cpu().numpy() + boundary_slack
    grid_bounds = tuple((umin[i], umax[i]) for i in range(ndim))

    SKIModel = SKIModel_bypass if bypass_covar else SKIModel_base
    print("Grid: ", grid_size)
    return SKIModel(
        train_x, train_y,
        grid_size=grid_size,
        grid_bounds=grid_bounds,
        kernel_type=kernel_type,
        min_noise=min_noise,
        use_modified=use_modified,
        interp_type=interp_type,
    ).to(device=device, dtype=dtype)

