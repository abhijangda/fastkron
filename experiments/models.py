import torch
import gpytorch as gp

import sgkigp.config as config
from sgkigp.config import MatmulAlgo, SgBasisType
from sgkigp.config import KernelsType, SobolevKernType, InterpType, SgShifted

from sgkigp.models.bypasscovargp import BypasscovarExactGP, ExactGP
from sgkigp.kernels.gridinterpkernel import ModifiedGridInterpolationKernel

from sgkigp.kernels.sginterpkernel import SparseGridInterpolationKernel
from sgkigp.kernels.basekernels import SobolevKernel, SpectralMixtureOneDim


class BaseGPModel(ExactGP):
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)


class BypassGPModel(BypasscovarExactGP):
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)


def get_kernel_by_type(kernel_type=KernelsType.RBFKERNEL, ard_num_dims=None):

    if kernel_type == KernelsType.RBFKERNEL:
        return gp.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        # return gp.kernels.RBFKernel(ard_num_dims=ard_num_dims,
        #                             lengthscale_constraint=gp.constraints.GreaterThan(1e-4))

    elif kernel_type == KernelsType.MATTERNHALF:
        return gp.kernels.MaternKernel(nu=0.5, ard_num_dims=ard_num_dims)

    elif kernel_type == KernelsType.MATTERNONEANDHALF:
        return gp.kernels.MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)

    elif kernel_type == KernelsType.MATTERNTWOANDHALF:
        return gp.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)

    elif kernel_type == KernelsType.SOBOLEVONE:
        return SobolevKernel(sobolev_type=SobolevKernType.SOBOLEV1)

    elif kernel_type == KernelsType.SOBOLEVTWO:
        return SobolevKernel(sobolev_type=SobolevKernType.SOBOLEV2)

    elif kernel_type == KernelsType.SOBOLEVTHREE:
        return SobolevKernel(sobolev_type=SobolevKernType.SOBOLEV3)

    elif kernel_type == KernelsType.SpectralMixtureOneDim:
        return SpectralMixtureOneDim()


class SKIPModel(gp.models.ExactGP):
    def __init__(self, train_x, train_y, kernel_type, grid_size=100, min_noise=1e-4):
        likelihood = gp.likelihoods.GaussianLikelihood(
                      noise_constraint=gp.constraints.GreaterThan(min_noise))
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        base_covar_module = get_kernel_by_type(kernel_type=kernel_type)
        self.covar_module = gp.kernels.ProductStructureKernel(
          gp.kernels.ScaleKernel(
            gp.kernels.GridInterpolationKernel(base_covar_module, grid_size=50, num_dims=1)
          ), num_dims=train_x.size(-1)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)


# SGPR model
class SGPRModel(BaseGPModel):
    def __init__(self, train_x, train_y, n_inducing=500, kernel_type=KernelsType.RBFKERNEL, min_noise=1e-4):

        likelihood = gp.likelihoods.GaussianLikelihood(
            noise_constraint=gp.constraints.GreaterThan(min_noise))

        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gp.means.ConstantMean()
        self.base_covar_module = get_kernel_by_type(kernel_type=kernel_type, ard_num_dims=train_x.size(-1))

        print("Base covar module: ", self.base_covar_module, "Nu: ", kernel_type)
        self.covar_module = gp.kernels.InducingPointKernel(
            gp.kernels.ScaleKernel(self.base_covar_module),
            inducing_points=train_x[torch.randperm(train_x.size(0))[:n_inducing]],
            likelihood=likelihood
        )


# Sparse grid model
class SparseGridGpModel(BaseGPModel):
    def __init__(
            self,
            train_x,
            train_y,
            grid_level,
            umin,
            umax,
            algo_type=MatmulAlgo.ITERATIVE,
            use_toeplitz=False,
            interp_type=InterpType.CUBIC,
            basis_type=SgBasisType.MODIFIED,
            kernel_type=KernelsType.RBFKERNEL,
            sg_shifted=SgShifted.ZERO,
            min_noise=1e-4,
            dtype=config.dtype(use_torch=True)
    ):
        likelihood = gp.likelihoods.GaussianLikelihood(noise_constraint=gp.constraints.GreaterThan(min_noise))

        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gp.means.ConstantMean()

        base_covar_module = get_kernel_by_type(kernel_type=kernel_type)

        print("Base covar module: ", base_covar_module, "Nu: ", kernel_type)
        self.covar_module = SparseGridInterpolationKernel(
            base_kernel=base_covar_module,
            grid_level=grid_level,
            ndim=train_x.size(-1),
            comb=True,
            algo_type=algo_type,
            use_toeplitz=use_toeplitz,
            interp_type=interp_type,
            basis=basis_type,
            sg_shifted=sg_shifted,
            umin=umin,
            umax=umax,
            dtype=dtype,
        )


def get_ski_modules(
        kernel_type, ard_num_dims, use_modified, grid_size, grid_bounds, interp_type,
        *args, **kwargs
    ):

    base_covar_module = get_kernel_by_type(kernel_type=kernel_type, ard_num_dims=ard_num_dims)
    # base_covar_module = get_kernel_by_type(kernel_type=kernel_type, ard_num_dims=None)
    mean_module = gp.means.ConstantMean()

    if interp_type in [InterpType.LINEAR, InterpType.SIMPLEX] or use_modified:
        covar_module = ModifiedGridInterpolationKernel(
                gp.kernels.ScaleKernel(base_covar_module),
                grid_size=grid_size,
                grid_bounds=grid_bounds,
                interp_type=interp_type,
                *args, **kwargs
            )

    else:
        assert interp_type == InterpType.CUBIC
        covar_module = gp.kernels.ScaleKernel(
            gp.kernels.GridInterpolationKernel(
                gp.kernels.RBFKernel(), grid_size=grid_size, grid_bounds=grid_bounds, *args, **kwargs
            )
        )

    return mean_module, covar_module


class SKIModel_base(BaseGPModel):
    def __init__(self, train_x, train_y, grid_size=100, grid_bounds=None,
                 kernel_type=KernelsType.RBFKERNEL, min_noise=1e-4,
                 use_modified=False, interp_type=InterpType.CUBIC):
        likelihood = gp.likelihoods.GaussianLikelihood(
            noise_constraint=gp.constraints.GreaterThan(min_noise))

        super(SKIModel_base, self).__init__(train_x, train_y, likelihood)

        self.mean_module, self.covar_module = get_ski_modules(
            kernel_type=kernel_type, ard_num_dims=train_x.size(1), use_modified=use_modified,
            grid_size=grid_size, grid_bounds=grid_bounds, interp_type=interp_type
        )


class SKIModel_bypass(BypassGPModel):
    def __init__(self, train_x, train_y, grid_size=100, grid_bounds=None,
                 kernel_type=KernelsType.RBFKERNEL, min_noise=1e-4,
                 use_modified=False, interp_type = InterpType.CUBIC):
        likelihood = gp.likelihoods.GaussianLikelihood(
            noise_constraint=gp.constraints.GreaterThan(min_noise))

        super(SKIModel_bypass, self).__init__(train_x, train_y, likelihood)

        self.mean_module, self.covar_module = get_ski_modules(
            kernel_type=kernel_type, ard_num_dims=train_x.size(1), use_modified=use_modified,
            grid_size=grid_size, grid_bounds=grid_bounds, interp_type=interp_type
        )
