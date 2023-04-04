import torch
import numpy as np

from copy import deepcopy
from torch.nn import ModuleList
from gpytorch.kernels import Kernel

import sgkigp.config as config

from sgkigp.interp.sparse.sgindices import compute_LI, compute_LI_pairs
from sgkigp.kernels.sgcovars import compute_covars
from sgkigp.lazy.sgkerneltensor import SGKernelLazyTensor

from sgkigp.config import InterpType, SgBasisType, MatmulAlgo


def process_length_scales(ls, ndim, device, dtype):
    return [torch.Tensor([np.sqrt(ls[i])]).view(1, 1).type(dtype).to(device) for i in range(ndim)]


class SparseGridKernel(Kernel):

    def __init__(
            self,
            base_kernel,
            grid_level,
            ndim,
            use_toeplitz=False,
            algo_type=MatmulAlgo.ITERATIVE,
            umin=-0.2,
            umax=1.2,
            comb=False,
            interp_type=InterpType.LINEAR,
            basis=SgBasisType.NAIVE,
            active_dims=None,
            covar_dtype=config.dtype(use_torch=True),
            device=config.get_device(),
    ):

        if not base_kernel.is_stationary:
            raise RuntimeError("The base_kernel for GridKernel must be stationary.")

        super().__init__(active_dims=active_dims)

        self.covar_dtype = covar_dtype
        self.base_kernel = ModuleList([deepcopy(base_kernel) for _ in range(ndim)])
        self.grid_level = grid_level
        self.ndim = ndim
        self.use_toeplitz = use_toeplitz
        self.interp_type = interp_type
        self.basis = basis
        self.comb = comb

        # Setting up lexicographic indices
        if algo_type == MatmulAlgo.RECURSIVE:
            levels_and_indices = compute_LI(grid_level, ndim, comb=comb, basis=basis)
            levels_and_indices = torch.from_numpy(levels_and_indices).to(dtype=torch.long, device=device)

        elif algo_type == MatmulAlgo.ITERATIVE:
            levels_and_indices = compute_LI_pairs(grid_level, ndim, comb=False, basis=basis)
        else:
            raise NotImplementedError

        self.levels_and_indices = levels_and_indices

        # TODO: process these to be in torch
        if type(umin) == list or type(umin) == np.ndarray or torch.is_tensor(umin):
            self.umin, self.umax = umin, umax
        else:
            self.umin, self.umax = [umin]*ndim, [umax]*ndim
        assert len(self.umin) == len(self.umax) == ndim

        self.algo_type = algo_type
        self.device = device

    def _clear_cache(self):
        if hasattr(self, "_cached_kernel_mat"):
            del self._cached_kernel_mat

    def forward(self, x1=None, x2=None, diag=False, last_dim_is_batch=False, **params):

        if last_dim_is_batch:
            raise NotImplementedError("I've not thought through this option yet ~ MY")

        if last_dim_is_batch and not self.interpolation_mode:
            raise ValueError("last_dim_is_batch is only valid with interpolation model")

        if not self.training and hasattr(self, "_cached_kernel_mat"):
            return self._cached_kernel_mat

        covars, sorting_orders = compute_covars(
            grid_level=self.grid_level,
            ndim=self.ndim,
            umin=self.umin,
            umax=self.umax,
            base_kernel=self.base_kernel,
            basis=self.basis,
            device=self.device,
            dtype=self.covar_dtype,
            use_toeplitz=self.use_toeplitz,
        )

        covar = SGKernelLazyTensor(
            covars=covars,
            LI=self.levels_and_indices,
            grid_level=self.grid_level,
            ndim=self.ndim,
            basis_type=self.basis,
            algo_type=self.algo_type,
            use_toeplitz=self.use_toeplitz,
            sorting_orders=sorting_orders,
            comb=self.comb
        )
        if not self.training:
            self._cached_kernel_mat = covar
        return covar

    def num_outputs_per_input(self, x1, x2):

        # TODO: revisit this if there is a better way
        return self.base_kernel[0].num_outputs_per_input(x1, x2)

    def initialize_hypers(self, lengthscales):

        assert len(lengthscales) == self.ndim
        lengthscales = process_length_scales(lengthscales, self.ndim, device=self.device, dtype=self.dtype)
        hypers = {'base_kernel.' + str(i) + '.lengthscale': lengthscales[i] for i in range(self.ndim)}
        self.initialize(**hypers)
