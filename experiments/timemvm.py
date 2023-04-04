import os
import wandb
import fire
import pickle
import tqdm
import torch
from timeit import default_timer as timer
import numpy as np
import gpytorch as gp

import sgkigp.config as config

from sgkigp.config import SgBasisType, MatmulAlgo, set_seeds
from sgkigp.config import DEFAULT_DTYPE, WANDB_PATH

from sgkigp.interp.sparse.sglocations import get_sg_points_nd
from sgkigp.kernels.sgkernel import SparseGridKernel

from sgkigp.utils import print_memory_info, get_memory_info


mean_std_err = lambda X: (np.mean(X), 1.96*(np.sqrt(np.std(X)/len(X))))


def perform_mvm_and_report(kernel_tensor, construction_time, grid_level,
                           ndim, num_repeat, log_dir_path, algo_name='naive'):

    torch.set_num_threads(1)

    if num_repeat > 1:
        mean_c_error, std_c_err = mean_std_err(construction_time[1:])
    else:
        mean_c_error, std_c_err = mean_std_err(construction_time)

    torch.cuda.empty_cache()
    vector = torch.rand(kernel_tensor.shape[-1])

    try:
        vector = vector.to(device=kernel_tensor.device, dtype=kernel_tensor.dtype)
    except IndexError:
        kt = kernel_tensor._kwargs['covars'][0][0]
        vector = vector.to(device=kt.device, dtype=kt.dtype)

    mvm_times_full = np.zeros(num_repeat)
    for i in tqdm.tqdm(range(num_repeat)):
        t_start = timer()
        kernel_tensor.matmul(vector)
        mvm_times_full[i] = timer() - t_start

    if num_repeat > 1:
        mean_mvm_error, std_mvm_err = mean_std_err(mvm_times_full[1:])
    else:
        mean_mvm_error, std_mvm_err = mean_std_err(mvm_times_full)

    memory_required = get_memory_info(unit='gb')
    print("C Times: ", construction_time, mvm_times_full, memory_required)

    times = (mean_c_error, std_c_err, mean_mvm_error, std_mvm_err)
    print(algo_name, ": ", times, "\tgl & ndim: ", grid_level, ndim, kernel_tensor.shape[-1])
    with open(log_dir_path, 'wb') as outfile:
        pickle.dump((times, grid_level, ndim, kernel_tensor.shape[-1], memory_required),
                    outfile, protocol=pickle.HIGHEST_PROTOCOL)

    print_memory_info(unit='gb')


def main(
        log_dir,
        grid_level: int = 2,
        ndim: int = 3,
        num_repeat: int = 9,
        seed=1337,
        basis_type=1,
        algo_type=1,
        device=0,
        use_t=-1,
    ):

    # processing arguments
    set_seeds(seed)
    ndim = ndim
    grid_level = grid_level
    num_repeat = num_repeat
    basis_type = SgBasisType(basis_type)
    algo_type = MatmulAlgo(algo_type)
    dtype = config.dtype(DEFAULT_DTYPE, use_torch=True)
    device = config.get_device(device)

    # if log_dir is None:
    #     sweep_name = os.environ.get(wandb.env.SWEEP_ID, 'solo')
    #     log_dir = WANDB_PATH + '/sweep-' + sweep_name
    #     os.makedirs(log_dir, exist_ok=True)
    #     log_dir_path = log_dir + "/gl_" + str(grid_level) + "_ndim_" + str(ndim) + ".yaml"

    if algo_type == MatmulAlgo.NAIVE:
        torch.set_num_threads(1)
        construction_time = np.zeros(num_repeat)
        torch.cuda.empty_cache()
        for i in tqdm.tqdm(range(num_repeat)):

            t_start = timer()
            points = torch.tensor(get_sg_points_nd(grid_level, ndim), dtype=dtype, device=device)
            print("Points shape: ", points.shape)

            kernel = gp.kernels.RBFKernel()
            kernel.to(device=device, dtype=dtype)
            kernel_tensor = kernel.forward(points, points)
            construction_time[i] = timer() - t_start
            torch.cuda.empty_cache()

        perform_mvm_and_report(kernel_tensor, construction_time, grid_level,
                               ndim, num_repeat, log_dir, algo_name='naive')

    elif algo_type == MatmulAlgo.RECURSIVE:

        umin = [0.0]*ndim
        umax = [1.0]*ndim
        base_covar_module = gp.kernels.RBFKernel()
        if config.get_device() != base_covar_module.lengthscale.device:
            base_covar_module.to(device=config.get_device())

        construction_time = np.zeros(num_repeat)
        torch.cuda.empty_cache()
        for i in tqdm.tqdm(range(num_repeat)):
            t_start = timer()
            print("Setting up kernel ...")
            kernel_module = SparseGridKernel(
                umin=umin,
                umax=umax,
                basis=basis_type,
                base_kernel=base_covar_module,
                grid_level=grid_level,
                covar_dtype=dtype,
                ndim=ndim,
                algo_type=algo_type,
                device=device,
                use_toeplitz=False
            )
            kernel_module.to(dtype=dtype, device=device)
            kernel_tensor = kernel_module.forward()
            construction_time[i] = timer() - t_start

        perform_mvm_and_report(kernel_tensor, construction_time, grid_level, ndim,
                               num_repeat, log_dir, algo_name="recursive")

    elif algo_type == MatmulAlgo.ITERATIVE:

        umin = [0.0]*ndim
        umax = [1.0]*ndim
        base_covar_module = gp.kernels.RBFKernel()
        if config.get_device() != base_covar_module.lengthscale.device:
            base_covar_module.to(device=config.get_device())

        use_toeplitz = True if use_t >= 0 else False
        print("Toeplitz status: ", use_toeplitz)

        construction_time = np.zeros(num_repeat)
        torch.cuda.empty_cache()
        for i in tqdm.tqdm(range(num_repeat)):
            t_start = timer()
            print("Setting up kernel ...")
            kernel_module = SparseGridKernel(
                umin=umin,
                umax=umax,
                basis=basis_type,
                base_kernel=base_covar_module,
                grid_level=grid_level,
                covar_dtype=dtype,
                ndim=ndim,
                algo_type=algo_type,
                device=device,
                use_toeplitz=use_toeplitz
            )
            kernel_module.to(dtype=dtype, device=device)
            kernel_tensor = kernel_module.forward()
            construction_time[i] = timer() - t_start

        perform_mvm_and_report(kernel_tensor, construction_time, grid_level, ndim,
                               num_repeat, log_dir, algo_name="iterative")

    elif algo_type == MatmulAlgo.TOEPLITZ:

        umin = [0.0]*ndim
        umax = [1.0]*ndim
        base_covar_module = gp.kernels.RBFKernel()
        if config.get_device() != base_covar_module.lengthscale.device:
            base_covar_module.to(device=config.get_device())

        construction_time = np.zeros(num_repeat)
        torch.cuda.empty_cache()
        for i in tqdm.tqdm(range(num_repeat)):
            t_start = timer()
            print("Setting up kernel ...")
            kernel_module = SparseGridKernel(
                umin=umin,
                umax=umax,
                basis=basis_type,
                base_kernel=base_covar_module,
                grid_level=grid_level,
                covar_dtype=dtype,
                ndim=ndim,
                algo_type=algo_type,
                device=device,
                use_toeplitz=True
            )
            kernel_module.to(dtype=dtype, device=device)
            kernel_tensor = kernel_module.forward()
            construction_time[i] = timer() - t_start

        perform_mvm_and_report(kernel_tensor, construction_time, grid_level, ndim,
                               num_repeat, log_dir, algo_name="toeplitz")

    else:
        raise NotImplementedError


if __name__ == "__main__":
    # os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')
    fire.Fire(main)
