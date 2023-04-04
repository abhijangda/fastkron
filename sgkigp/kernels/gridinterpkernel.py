#!/usr/bin/env python3

import torch
from typing import List, Optional, Tuple, Union

from gpytorch.lazy import lazify
from gpytorch.utils.grid import create_grid
from gpytorch.kernels.grid_kernel import GridKernel
from gpytorch.kernels.kernel import Kernel


from sgkigp.config import InterpType
from sgkigp.interp.rginterp import Interpolation
import sgkigp.tensorize as tensorize

from sgkigp.lazy.interptensor import ModifiedInterpolatedLazyTensor as InterpolatedLazyTensor
from sgkigp.models.predictionstrategy import ModifiedInterpolatedPredictionStrategy as InterpolatedPredictionStrategy


class ModifiedGridInterpolationKernel(GridKernel):

    def __init__(
        self,
        base_kernel: Kernel,
        grid_size: Union[int, List[int]],
        num_dims: int = None,
        grid_bounds: Optional[Tuple[float, float]] = None,
        active_dims: Tuple[int, ...] = None,
        interp_type: InterpType = InterpType.CUBIC,
        adjust_boundary: bool = True
    ):
        has_initialized_grid = 0
        grid_is_dynamic = True

        # Make some temporary grid bounds, if none exist
        if grid_bounds is None:
            if num_dims is None:
                raise RuntimeError("num_dims must be supplied if grid_bounds is None")
            else:
                # Create some temporary grid bounds - they'll be changed soon
                grid_bounds = tuple((-1.0, 1.0) for _ in range(num_dims))
        else:
            has_initialized_grid = 1
            grid_is_dynamic = False
            if num_dims is None:
                num_dims = len(grid_bounds)
            elif num_dims != len(grid_bounds):
                raise RuntimeError(
                    "num_dims ({}) disagrees with the number of supplied "
                    "grid_bounds ({})".format(num_dims, len(grid_bounds))
                )

        if isinstance(grid_size, int):
            grid_sizes = [grid_size for _ in range(num_dims)]
        else:
            grid_sizes = list(grid_size)

        if len(grid_sizes) != num_dims:
            raise RuntimeError("The number of grid sizes provided through grid_size do not match num_dims.")

        # Initialize values and the grid
        self.grid_is_dynamic = grid_is_dynamic
        self.num_dims = num_dims
        self.grid_sizes = grid_sizes
        self.interp_type = interp_type

        if adjust_boundary:
            self.grid_bounds = self._tight_grid_bounds(grid_bounds, grid_sizes)
        else:
            self.grid_bounds = grid_bounds

        grid = create_grid(self.grid_sizes, self.grid_bounds)

        super(ModifiedGridInterpolationKernel, self).__init__(
            base_kernel=base_kernel, grid=grid, interpolation_mode=True,
            active_dims=active_dims,
        )
        self.register_buffer("has_initialized_grid", torch.tensor(has_initialized_grid, dtype=torch.bool))

    @staticmethod
    def _tight_grid_bounds(grid_bounds, grid_sizes):
        grid_spacings = tuple((bound[1] - bound[0]) / grid_sizes[i] for i, bound in enumerate(grid_bounds))
        return tuple(
            (bound[0] - 1.01 * spacing, bound[1] + 1.01 * spacing)
            for bound, spacing in zip(grid_bounds, grid_spacings)
        )

    def _compute_grid(self, inputs, last_dim_is_batch=False):
        n_data, n_dimensions = inputs.size(-2), inputs.size(-1)
        if last_dim_is_batch:
            inputs = inputs.transpose(-1, -2).unsqueeze(-1)
            n_dimensions = 1
        batch_shape = inputs.shape[:-2]

        inputs = inputs.reshape(-1, n_dimensions)
        interp_indices, interp_values \
            = Interpolation().interpolate(self.grid_bounds, self.grid_sizes,
                                          inputs, interp_type=self.interp_type)
        interp_indices = interp_indices.view(*batch_shape, n_data, -1)
        interp_values = interp_values.view(*batch_shape, n_data, -1)
        return interp_indices, interp_values

    def _inducing_forward(self, last_dim_is_batch, **params):
        return super().forward(self.grid, self.grid, last_dim_is_batch=last_dim_is_batch, **params)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # See if we need to update the grid or not

        base_lazy_tsr = lazify(self._inducing_forward(last_dim_is_batch=last_dim_is_batch, **params))

        if last_dim_is_batch and base_lazy_tsr.size(-3) == 1:
            base_lazy_tsr = base_lazy_tsr.repeat(*x1.shape[:-2], x1.size(-1), 1, 1)

        left_interp_indices, left_interp_values = self._compute_grid(x1, last_dim_is_batch)

        left_interp_tensor = tensorize.unpack_and_sparse_tensor(
            left_interp_indices, left_interp_values, x1.shape[0], base_lazy_tsr.shape[0])

        if torch.equal(x1, x2):
            right_interp_tensor = left_interp_tensor.transpose(0, 1)
        else:
            right_interp_indices, right_interp_values = self._compute_grid(x2, last_dim_is_batch)

            right_interp_tensor = tensorize.unpack_and_sparse_tensor(
                interp_values=right_interp_values, interp_indices=right_interp_indices,
                n_rows=base_lazy_tsr.shape[0], n_cols=x2.shape[0], right_tensor=True
            )

        res = InterpolatedLazyTensor(
            base_lazy_tsr,
            left_interp_tensor=left_interp_tensor,
            right_interp_tensor=right_interp_tensor
        )

        if diag:
            return res.diag()
        else:
            return res

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return InterpolatedPredictionStrategy(train_inputs, train_prior_dist, train_labels, likelihood)
