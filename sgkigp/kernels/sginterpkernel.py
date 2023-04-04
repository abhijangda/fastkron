import math
import torch
import sgkigp.utils as utils

from gpytorch.lazy import lazify
from gpytorch.lazy import MatmulLazyTensor, DiagLazyTensor

from sgkigp.interp.sparse.nbhors import compute_eff_B as compute_B
from sgkigp.interp.sparse.nbhors import compute_comb_B_diag
from gpytorch.utils.memoize import cached

#from sgkigp.interp.sparse.weights import compute_phi
from sgkigp.interp.sginterp import SparseInterpolation
from sgkigp.models.sgpredictionstrategy import SGInterpolatedPredictionStrategy
from sgkigp.lazy.sginterptensor import SymSGInterpolatedLazyTensor, ASymSGInterpolatedLazyTensor

from sgkigp.kernels.sgkernel import SparseGridKernel

import sgkigp.config as config
from sgkigp.config import InterpType, SgBasisType, MatmulAlgo, SgShifted


class SparseGridInterpolationKernel(SparseGridKernel):

    def __init__(
        self,
            base_kernel,
            grid_level,
            ndim,
            comb=True,
            algo_type=MatmulAlgo.ITERATIVE,
            use_toeplitz=False,
            interp_type=InterpType.LINEAR,
            basis=SgBasisType.NAIVE,
            sg_shifted=SgShifted.ZERO,
            umin=-0.2,
            umax=1.2,
            active_dims=None,
            device=config.get_device(),
            dtype=config.dtype(use_torch=True),
    ):

        # use combination technique

        if sg_shifted == SgShifted.ZERO:
            if comb:
                B = compute_comb_B_diag(grid_level, ndim, basis=basis, device=device, dtype=dtype)
            else:
                B = compute_B(grid_level, ndim)  # TODO: this should be cached directly from disk
                B = utils.scipy_coo_to_torch(B, dtype=dtype)
            B = B.to(device)
        else:
            B = None

        self.B = B
        self.sg_shifted = sg_shifted

        if sg_shifted != config.SgShifted.ZERO:
            comb = False

        print("Grid size: ", B.shape[0] if B is not None else -1)
        super().__init__(
            base_kernel=base_kernel,
            grid_level=grid_level,
            ndim=ndim,
            use_toeplitz=use_toeplitz,
            algo_type=algo_type,
            umin=umin,
            umax=umax,
            comb=comb,
            interp_type=interp_type,
            basis=basis,
            active_dims=active_dims,
            covar_dtype=dtype,
            device=device,
        )

        self._phis = []
        self._inputs = []

    # @property
    # @cached(name="inputs_memo")
    # def inputs(self):
    #     return self._inputs

    def check_in_list(self, inputs):

        i = 0
        while i < len(self._inputs):
            if self._inputs[i].size() != inputs.size():
                i += 1
                continue
            if torch.allclose(self._inputs[i], inputs, atol=1e-6):
                break

        if i < len(self._inputs):
            return True, self._phis[i]

        return False, None

    def compute_phi(self, inputs):
        is_computed, phi = self.check_in_list(inputs)

        if is_computed:
            return phi

        phi = SparseInterpolation().sparse_interpolate(
            grid_level=self.grid_level,
            ndim=self.ndim,
            umin=self.umin, umax=self.umax,
            x_target=inputs,
            comb=self.comb,
            interp_type=self.interp_type,
            basis=self.basis,
            dtype=self.dtype,
            device=self.device,
            shifted=self.sg_shifted
        )

        if self.B is None:
            if self.sg_shifted == config.SgShifted.ONE:
                B_diag = 1 / self.ndim
            elif self.sg_shifted == config.SgShifted.TWO:
                B_diag = 1 / (math.comb(self.ndim, 2) + self.ndim)
            else:
                raise NotImplementedError
            B_diag = B_diag * torch.ones(phi.shape[1], device=phi.device, dtype=phi.dtype)
            B = DiagLazyTensor(B_diag)
            self.B = B

        # store for future usage
        self._phis += phi,
        self._inputs += inputs,
        return phi

    def _compute_grid(self, inputs, is_left=True, last_dim_is_batch=False):

        # if self.grid_level < 2:
        #     raise NotImplementedError

        phi = self.compute_phi(inputs)
        # if self.training:
        #     phi = self.compute_phi(inputs)
        # else:
        #     phi = SparseInterpolation().sparse_interpolate(
        #         grid_level=self.grid_level,
        #         ndim=self.ndim,
        #         umin=self.umin, umax=self.umax,
        #         x_target=inputs,
        #         comb=self.comb,
        #         interp_type=self.interp_type,
        #         basis=self.basis,
        #         dtype=self.dtype,
        #         device=self.device,
        #     )

        # Phi = compute_phi(X=inputs, grid_level=self.grid_level,
        #                   ndim=self.ndim, umin=self.umin, umax=self.umax,
        #                   comb=self.comb,  interp_type=self.interp_type, basis=self.basis,
        #                   use_torch=True, dtype=self.dtype,
        #                   device=self.device)

        if is_left:
            return MatmulLazyTensor(phi, self.B)

        # right tensor
        return MatmulLazyTensor(self.B.transpose(1, 0), phi.transpose(1, 0))

    def _inducing_forward(self, last_dim_is_batch, **params):
        return super().forward(None, None, last_dim_is_batch=last_dim_is_batch, **params)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):

        base_lazy_tsr = lazify(self._inducing_forward(last_dim_is_batch=last_dim_is_batch, **params))

        if torch.equal(x1, x2):
            res = SymSGInterpolatedLazyTensor(
                base_lazy_tensor=base_lazy_tsr,
                left_interp_coefficient=self._compute_grid(x1, is_left=True)
            )

        else:
            res = ASymSGInterpolatedLazyTensor(
                base_lazy_tensor=base_lazy_tsr,
                left_interp_coefficient=self._compute_grid(x1, is_left=True),
                right_interp_coefficient=self._compute_grid(x2, is_left=False)
            )

        if diag:
            return res.diag()
        else:
            return res

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return SGInterpolatedPredictionStrategy(train_inputs, train_prior_dist, train_labels, likelihood)

