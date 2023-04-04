# Credits: part of the code in this file is adapted from: https://github.com/activatedgeek/simplex-gp/
import os
import fire
import json
import torch
from enum import Enum
from tqdm.auto import tqdm
import numpy as np
import gpytorch as gp

from timeit import default_timer as timer

import sgkigp.config as config
from sgkigp.config import SgBasisType, InterpType, MethodName

from experiments.setups import load_func_dataset
from experiments.setups import setup_models

from sgkigp.interp.sparse.construct import G as SGsize
from sgkigp.kernels.sgkernel import SparseGridKernel
from sgkigp.kernels.sginterpkernel import SparseGridInterpolationKernel
from sgkigp.kernels.gridinterpkernel import ModifiedGridInterpolationKernel
from sgkigp.interp.sginterp import SparseInterpolation
from gpytorch.utils.interpolation import left_interp, left_t_interp

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


def get_memory_info(unit='gb'):
    keys = ['memory_allocated', 'max_memory_allocated', 'memory_reserved', 'max_memory_reserved']
    return {key: _get_memory_info(key, unit, rval=True) for key in keys}


def print_memory_info(unit='gb'):
    print(_get_memory_info('memory_allocated', unit))
    print(_get_memory_info('max_memory_allocated', unit))
    print(_get_memory_info('memory_reserved', unit))
    print(_get_memory_info('max_memory_reserved', unit))
    print('')


class ProfileTask(Enum):
    INTERCOEFF = 0
    KERNELMAT = 1
    INTERPKERNEL = 2
    INFERENCE = 3
    LEARNING = 4


def dump_results(results, log_path):
    if log_path == '':
        print(results)
        return
    json.dump(results, open(log_path + ".result", "w"))


def main(method: int = 2, dataset: str = None, log_int: int = 1, seed: int = 1337,
         device: int = 0, lr: int = 1e-1, p_epochs: int = 200, lanc_iter: int = 100,
         pre_size: int = 100, n_inducing: int = 512, kernel_type: int = 0, min_noise: float = 1e-4,
         grid_level: int = 4, boundary_slack: float = 0.2, cg_tol: float = 1.0, cg_eval_tol: float = 1e-2,
         ndim: int = 2, interp_type: int = 1, basis_type: int = 1, dtype: str = 'float32',
         profile_task: int = 4, log_path: str = ''):

    # Handling config variables
    epochs = 4
    kernel_type = config.KernelsType(kernel_type)
    interp_type = InterpType(interp_type)
    basis_type = SgBasisType(basis_type)
    profile_task = ProfileTask(profile_task)
    method = MethodName(method)
    # gp.settings.max_cholesky_size._set_value(-1)

    # Set-seeds and device
    config.set_seeds(seed=seed)
    device = config.get_device(device=device)
    dtype = config.dtype(dtype, use_torch=True)

    # Preparing data iterations
    train_x, train_y, val_x, val_y, test_x, test_y = load_func_dataset(ndim, device, dtype, func_val=3)

    # Reporting basic data statistics
    print(f'"{dataset}": D = {train_x.size(-1)}, Train N = {train_x.size(0)},'
          f' Val N = {val_x.size(0)} Test N = {test_x.size(0)}')

    if profile_task == ProfileTask.INTERCOEFF:
        results = profile_interp_mat(method, ndim, train_x, boundary_slack, grid_level, device, dtype, interp_type)

    elif profile_task == ProfileTask.KERNELMAT:
        results = profile_kernel_matrix(method, ndim, train_x, boundary_slack, grid_level, device, dtype, interp_type)

    elif profile_task == ProfileTask.INTERPKERNEL:

        results = profile_interp_kernel_matrix(method, ndim, train_x, boundary_slack,
                                               grid_level, device, dtype, interp_type)

    elif profile_task == ProfileTask.INFERENCE:

        model = setup_models(
            method, train_x, train_y, n_inducing,
            kernel_type, min_noise, device, dtype,
            val_x, test_x, boundary_slack, grid_level, interp_type, basis_type
        )
        mll = gp.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        results = profile_inference(
            method, train_x, train_y, n_inducing, kernel_type, min_noise, device, dtype,
            val_x, test_x, boundary_slack, grid_level, interp_type, basis_type,
            test_y, pre_size, lanc_iter, cg_eval_tol, mll)

    elif profile_task == ProfileTask.LEARNING:

        model = setup_models(
            method, train_x, train_y, n_inducing,
            kernel_type, min_noise, device, dtype,
            val_x, test_x, boundary_slack, grid_level, interp_type, basis_type
        )
        mll = gp.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        results = profile_learning(
            method, train_x, train_y, n_inducing, kernel_type, min_noise, device, dtype,
            val_x, test_x, boundary_slack, grid_level, interp_type, basis_type,
            test_y, pre_size, lanc_iter, cg_eval_tol, mll,  lr, epochs=4, cg_tol=1.0)

    else:
        raise NotImplementedError

    dump_results(results, log_path)


def profile_learning(method, train_x, train_y, n_inducing, kernel_type, min_noise, device, dtype,
            val_x, test_x, boundary_slack, grid_level, interp_type, basis_type,
            test_y, pre_size, lanc_iter, cg_eval_tol, mll,  lr, epochs, cg_tol):

    # Setting up models
    model = setup_models(method, train_x, train_y, n_inducing, kernel_type, min_noise, device, dtype,
                         val_x, test_x, boundary_slack, grid_level, interp_type, basis_type)

    # Setting optimizer and early stopper
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Running over epochs
    epochs_results = []
    for i in tqdm(range(epochs)):
        train_dict = train(train_x, train_y, model, mll, optimizer,
                           lanc_iter=lanc_iter, pre_size=pre_size, cg_tol=cg_tol)
        epochs_results += train_dict,

    return epochs_results


def profile_inference(method, train_x, train_y, n_inducing, kernel_type, min_noise, device, dtype,
                      val_x, test_x, boundary_slack, grid_level, interp_type, basis_type,
                      test_y, pre_size, lanc_iter, cg_eval_tol, mll):

    model = setup_models(method, train_x, train_y, n_inducing, kernel_type, min_noise, device, dtype,
                         val_x, test_x, boundary_slack, grid_level, interp_type, basis_type)

    return test(test_x, test_y, model, mll, pre_size=pre_size, lanc_iter=lanc_iter, cg_tol=cg_eval_tol)


def profile_kernel_matrix(method, ndim, train_x,
                       boundary_slack, grid_level,
                       device, dtype, interp_type):

    # Setting about same size as Sparse GRID
    grid_size_dim = max(int(np.floor(SGsize(grid_level, ndim) ** (1 / ndim))), 5)
    grid_size = [grid_size_dim] * ndim
    umin = torch.min(train_x, 0).values.cpu().numpy() - boundary_slack
    umax = torch.max(train_x, 0).values.cpu().numpy() + boundary_slack

    if method == MethodName.SKI:

        t_start = timer()
        grid_bounds = tuple((umin[i], umax[i]) for i in range(ndim))
        grid = gp.utils.grid.create_grid(grid_size, grid_bounds)
        grid_kernel = gp.kernels.GridKernel(
            base_kernel=gp.kernels.RBFKernel(),
            grid=grid,
            interpolation_mode=True,
        ).to(dtype=dtype, device=device)
        grid_kernel_tensor = grid_kernel.forward(grid, grid)
        t_construct = timer() - t_start

        num_grid_points = np.prod(grid_size)

        # record mvms
        rhs_vector = torch.randn(num_grid_points, 1, device=device, dtype=dtype)

        num_repeats = 10
        mvm_time = np.zeros(num_repeats)
        for i in range(num_repeats):
            t_start = timer()
            _ = grid_kernel_tensor._matmul(rhs_vector)
            mvm_time[i] = timer() - t_start
        t_mvm = np.mean(mvm_time)

        print("Time: ", t_mvm)

    elif method == MethodName.SPARSE:

        umin = [0.0]*ndim
        umax = [1.0]*ndim
        base_covar_module = gp.kernels.RBFKernel()
        if config.get_device() != base_covar_module.lengthscale.device:
            base_covar_module.to(device=config.get_device())

        t_start = timer()
        print("Setting up kernel ...")
        kernel_module = SparseGridKernel(
            umin=umin,
            umax=umax,
            basis=config.SgBasisType.NAIVE,
            base_kernel=base_covar_module,
            grid_level=grid_level,
            covar_dtype=dtype,
            ndim=ndim,
            algo_type=config.MatmulAlgo.ITERATIVE,
            device=device,
        )
        kernel_module.to(dtype=dtype, device=device)
        kernel_tensor = kernel_module.forward()
        t_construct = timer() - t_start

        # record mvms
        num_grid_points = kernel_tensor.shape[-1]
        rhs_vector = torch.randn(num_grid_points, 1, device=device, dtype=dtype)

        num_repeats = 10
        mvm_time = np.zeros(num_repeats)
        for i in range(num_repeats):
            t_start = timer()
            _ = kernel_tensor._matmul(rhs_vector)
            mvm_time[i] = timer() - t_start
        t_mvm = np.mean(mvm_time)

    else:
        raise NotImplementedError

    results = {'construct': t_construct, 'mvm': t_mvm}
    results.update(get_memory_info())

    return results


def profile_interp_kernel_matrix(method, ndim, train_x, boundary_slack, grid_level, device, dtype, interp_type):

    # Setting about same size as Sparse GRID
    grid_size_dim = max(int(np.floor(SGsize(grid_level, ndim) ** (1 / ndim))), 5)
    grid_size = [grid_size_dim] * ndim
    umin = torch.min(train_x, 0).values.cpu().numpy() - boundary_slack
    umax = torch.max(train_x, 0).values.cpu().numpy() + boundary_slack

    # construct interp  matrix
    if method == MethodName.SKI:

        grid_bounds = tuple((umin[i], umax[i]) for i in range(ndim))

        t_start = timer()
        grid_interp_kernel = gp.kernels.GridInterpolationKernel(
            base_kernel=gp.kernels.RBFKernel(),
            grid_size=grid_size,
            num_dims=ndim,
            grid_bounds=grid_bounds
        ).to(device=device, dtype=dtype)
        grid_tensor = grid_interp_kernel.forward(train_x, train_x)
        t_construct = timer() - t_start
        print_memory_info()

        num_grid_points = np.prod(grid_size)

        # record mvms
        rhs_vector = torch.randn(grid_tensor.shape[0], 1, device=device, dtype=dtype)

        num_repeats = 10
        mvm_time = np.zeros(num_repeats)
        for i in range(num_repeats):
            t_start = timer()
            _ = grid_tensor.matmul(rhs_vector)
            mvm_time[i] = timer() - t_start
        t_mvm = np.mean(mvm_time)

    elif method == MethodName.SPARSE:

        print("Before creating sparse grid interp kernel ...")
        print_memory_info(unit='gb')

        t_start = timer()
        grid_interp_kernel = SparseGridInterpolationKernel(
            base_kernel=gp.kernels.RBFKernel(),
            grid_level=grid_level,
            ndim=train_x.size(-1),
            comb=True,
            algo_type=config.MatmulAlgo.ITERATIVE,
            use_toeplitz=False,
            interp_type=interp_type,
            basis=config.SgBasisType.NAIVE,
            umin=umin,
            umax=umax,
            device=device,
            dtype=dtype
        )
        t_construct = timer() - t_start

        print("After creating sparse grid interp kernel ...")
        print_memory_info(unit='gb')

        grid_tensor = grid_interp_kernel.forward(train_x, train_x)
        print("After interp tensors ...")
        print_memory_info(unit='gb')

        # record mvms
        rhs_vector = torch.randn(grid_tensor.shape[-1], 1, device=device, dtype=dtype)

        num_repeats = 10
        mvm_time = np.zeros(num_repeats)
        for i in range(num_repeats):
            t_start = timer()
            _ = grid_tensor.matmul(rhs_vector)
            mvm_time[i] = timer() - t_start
        t_mvm = np.mean(mvm_time)

    else:
        raise NotImplementedError

    results = {'construct': t_construct, 'mvm': t_mvm}
    results.update(get_memory_info())

    return results


def profile_interp_mat(method, ndim, train_x,
                       boundary_slack, grid_level,
                       device, dtype, interp_type):

    # Setting about same size as Sparse GRID
    grid_size_dim = max(int(np.floor(SGsize(grid_level, ndim) ** (1 / ndim))), 5)
    grid_size = [grid_size_dim] * ndim
    umin = torch.min(train_x, 0).values.cpu().numpy() - boundary_slack
    umax = torch.max(train_x, 0).values.cpu().numpy() + boundary_slack

    # construct interp  matrix
    if method == MethodName.SKI:

        grid_bounds = tuple((umin[i], umax[i]) for i in range(ndim))

        t_start = timer()
        grid_interp_kernel = gp.kernels.GridInterpolationKernel(
            base_kernel=gp.kernels.RBFKernel(),
            grid_size=grid_size,
            num_dims=ndim,
            grid_bounds=grid_bounds
        ).to(device=device, dtype=dtype)
        print("Before interp tensors ...")
        print_memory_info(unit='gb')
        left_interp_indices, left_interp_values = grid_interp_kernel._compute_grid(train_x, last_dim_is_batch=False)
        print("After interp tensors ...")
        print_memory_info(unit='gb')
        # interp_tensor = InterpolationUtils.unpack_and_sparse_tensor(
        #     left_interp_indices, left_interp_values, train_x.shape[0], np.prod(grid_size))
        print("After sparse tensor ...")
        print_memory_info()
        t_construct = timer() - t_start
        # del left_interp_indices
        # del left_interp_values
        print("After deleting tensor ...")
        print_memory_info()

        num_grid_points = np.prod(grid_size)

        # record mvms
        rhs_vector = torch.randn(num_grid_points, 1, device=device, dtype=dtype)

        num_repeats = 10
        mvm_time = np.zeros(num_repeats)
        for i in range(num_repeats):
            t_start = timer()
            _ = left_interp(left_interp_indices, left_interp_values, rhs_vector)
            mvm_time[i] = timer() - t_start
        t_mvm = np.mean(mvm_time)

    elif method == MethodName.SPARSE:

        print("Before creating sparse grid interp kernel ...")
        print_memory_info(unit='gb')

        t_start = timer()
        grid_interp_kernel = SparseGridInterpolationKernel(
            base_kernel=gp.kernels.RBFKernel(),
            grid_level=grid_level,
            ndim=train_x.size(-1),
            comb=True,
            algo_type=config.MatmulAlgo.ITERATIVE,
            use_toeplitz=False,
            interp_type=interp_type,
            basis=config.SgBasisType.NAIVE,
            umin=umin,
            umax=umax,
            device=device,
            dtype=dtype
        )
        t_construct = timer() - t_start

        print("After creating sparse grid interp kernel ...")
        print_memory_info(unit='gb')

        left_interpolation_coeff = grid_interp_kernel._compute_grid(train_x, is_left=True)
        print("After interp tensors ...")
        print_memory_info(unit='gb')

        # record mvms
        rhs_vector = torch.randn(left_interpolation_coeff.shape[-1], 1, device=device, dtype=dtype)

        num_repeats = 10
        mvm_time = np.zeros(num_repeats)
        for i in range(num_repeats):
            t_start = timer()
            _ = left_interpolation_coeff.matmul(rhs_vector)
            mvm_time[i] = timer() - t_start
        t_mvm = np.mean(mvm_time)

    else:
        raise NotImplementedError

    results = {'construct': t_construct, 'mvm': t_mvm}
    results.update(get_memory_info())

    return results


def train(x, y, model, mll, optim, lanc_iter=100, pre_size=100, cg_tol=1.0):

    model.train()
    optim.zero_grad()
    with gp.settings.cg_tolerance(cg_tol), \
         gp.settings.max_preconditioner_size(pre_size), \
         gp.settings.max_root_decomposition_size(lanc_iter):

        t_start = timer()
        output = model(x)
        loss = -mll(output, y)

        loss_ts = timer() - t_start

        t_start = timer()

        loss.backward()
        optim.step()

        bw_ts = timer() - t_start

    results = {
        'train/mll': -loss.detach().item(),
        'train/loss_ts': loss_ts,
        'train/bw_ts': bw_ts,
        'train/total_ts': loss_ts + bw_ts
    }
    results.update(get_memory_info())
    return results


def test(x, y, model, mll, lanc_iter=100, pre_size=100, label='test', cg_tol=1e-2):

    model.eval()
    with gp.settings.eval_cg_tolerance(cg_tol), \
         gp.settings.max_preconditioner_size(pre_size), \
         gp.settings.max_root_decomposition_size(lanc_iter), \
         gp.settings.fast_pred_var(), torch.no_grad():

        t_start = timer()
        pred_y = model(x)
        pred_ts = timer() - t_start
        rmse = (pred_y - y).pow(2).mean(0).sqrt()
        mae = (pred_y - y).abs().mean(0)
        #nll = -mll(pred_y, y)

    results = {
        f'{label}/rmse': rmse.item(),
        f'{label}/mae': mae.item(),
        f'{label}/pred_ts': pred_ts,
        f'{label}/nll': 0.0,
    }

    results.update(get_memory_info())
    return results


if __name__ == "__main__":
    fire.Fire(main)

