import torch
import tqdm

from functools import reduce
from operator import mul

import sgkigp.config as config
from sgkigp.config import InterpType, SgBasisType

import sgkigp.tensorize as sparseInterpTensor
from sgkigp.interp.misc import CreateGrids
from sgkigp.interp.rginterp import Interpolation
from sgkigp.config import DEFAULT_INDEX_DTYPE
from sgkigp.interp.kernels import InterpKernels

from sgkigp.interp.sparse.sgindices import compute_LI_order


ZERO_KERNEL_DIST = 10.0


class SparseInterpolation(Interpolation):

    def sparse_interpolate(
            self,
            grid_level,
            ndim, umin, umax,
            x_target: torch.Tensor,
            comb=False,
            interp_type=config.InterpType.LINEAR,
            basis=config.SgBasisType.NAIVE,
            dtype=config.dtype(use_torch=True),
            device=config.get_device(),
            shifted=config.SgShifted.ZERO
    ):

        grid_bounds = CreateGrids().compute_grid_bounds(umin=umin, umax=umax, ndim=ndim)
        sub_grids = CreateGrids().create_sparse_grid(
            grid_level=grid_level,
            ndim=ndim,
            grid_bounds=grid_bounds,
            comb=comb,
            basis=basis,
            shifted=shifted,
            device=device,
            dtype=dtype
        )

        #print("Computing interpolation for ...", x_target.shape, shifted)
        # Note: this is highly sub-optimal wrt to time # TODO: improve this only if time is an issue
        coefficients = []
        for sub_grid in tqdm.tqdm(sub_grids):
            if type(sub_grid) == list:
                coefficients += self.sg_get_basis(x_target=x_target, grid=sub_grid,
                                                  interp_type=interp_type, basis=basis,
                                                  comb=comb, dtype=dtype, shifted=shifted),
            else:
                coefficients += torch.empty(
                    [x_target.shape[0], sub_grid], layout=torch.sparse_coo, dtype=dtype, device=x_target.device
                ),

        coefficients = torch.hstack(coefficients).coalesce()

        # TODO: improve this ... this probably is like np.put_along_axis
        # Re-ordering columns of coefficients #

        if shifted == config.SgShifted.ZERO:
            order_ = config.np2torch(compute_LI_order(grid_level, ndim, comb=comb, basis=basis),
                                     device=device, dtype=torch.long)
        else:
            order_ = config.np2torch(compute_LI_order(grid_level, ndim, comb=False, basis=basis),
                                     device=device, dtype=torch.long)

        reverse_order = torch.zeros_like(order_)
        for i, d in enumerate(order_):
            reverse_order[d] = i

        nrows, ncols = coefficients.shape
        values = coefficients.values()
        c_indices = coefficients.indices()
        new_indices = torch.zeros_like(c_indices).to(device=device)
        new_indices[0, :] = c_indices[0, :]
        new_indices[1, :] = reverse_order[c_indices[1, :]]

        return sparseInterpTensor.make_sparse_tensor(
            index_tensor=new_indices, value_tensor=values, n_rows=nrows, n_cols=ncols
        )

    def sg_get_basis(
            self,
            x_target,
            grid,
            interp_type=InterpType.LINEAR,
            basis=SgBasisType.NAIVE,
            comb=False,
            dtype=config.dtype(use_torch=True),
            shifted=config.SgShifted.ZERO
    ):

        # processing x_target
        x_target = x_target.to(dtype=dtype)
        if x_target.ndim == 1:
            x_target = x_target.reshape(-1, 1)
        if x_target.ndim > 2:
            raise ValueError("x should have at most two-dimensions")
        n, d = x_target.shape

        if len(grid) != d:
            raise ValueError("Second dim of x (shape (%d, %d)) must match len(grid)=%d" % (n, d, len(grid)))

        if not comb and (shifted == config.SgShifted.ZERO):
            if interp_type != InterpType.LINEAR:
                raise NotImplementedError

            return self._hierarchical_interpolate_new(
                x_target=x_target,
                x_grid=grid,
                basis=basis
            )

        return self._interpolation_combination(x_target, x_grid=grid, interp_type=interp_type, basis=basis)

    def _interpolation_combination(self, x_target, x_grid, interp_type=InterpType.LINEAR, basis=SgBasisType.NAIVE):

        grid_sizes = CreateGrids().get_grid_sizes(basis, x_grid, x_target.device, torch.long)

        interp_indices, interp_values = self.interpolate(
            grid_bounds=None, grid_sizes=grid_sizes, basis=basis,
            x_target=x_target, interp_type=interp_type, x_grid=x_grid
        )

        # print("Out: ", grid_sizes, x_grid)
        # print(interp_indices)
        return sparseInterpTensor.unpack_and_sparse_tensor(
            interp_indices, interp_values, n_rows=x_target.shape[0], n_cols=reduce(mul, grid_sizes)
        )

    @staticmethod
    def _compute_hierarchical_zero_level(umax, umin, x_target):
        du = umax - umin
        center = umin + du / 2.0
        dist = torch.abs((x_target - center) / (du / 2.0))
        dim_interp_indices = torch.zeros_like(dist)     # TODO: setup dtype correct, it should be torch long
        return dist, dim_interp_indices

    @staticmethod
    def grid_pos_hierarchical(z, umin, du, dtype=DEFAULT_INDEX_DTYPE):
        return torch.floor(((z - umin) / du)).to(dtype)

    def _hierarchical_interpolate_new(self, x_target, x_grid, basis=SgBasisType.NAIVE):

        assert basis == SgBasisType.NAIVE

        device, dtype = x_target.device, x_target.dtype
        num_target_points, num_dim = x_target.size(0), x_target.size(-1)
        grid_sizes = [2**grid[-1] for grid in x_grid]

        interp_values = torch.ones(num_target_points, 1, dtype=dtype, device=device)
        interp_indices = torch.zeros(num_target_points, 1, dtype=torch.long, device=device)

        for i in range(num_dim):

            umin, umax, __ = x_grid[i]

            if grid_sizes[i] > 1:
                du = (umax - umin) / grid_sizes[i]
                u = torch.linspace(umin + du / 2.0, umax - du / 2.0, grid_sizes[i]).to(device=device)
                dim_interp_indices = self.grid_pos_hierarchical(x_target[:, i], umin, du, dtype=torch.long)
                dist = 2 * torch.abs((x_target[:, i] - u[dim_interp_indices])) / du

            else:
                dist, dim_interp_indices = self._compute_hierarchical_zero_level(umax, umin, x_target[:, i])

            dim_interp_values = InterpKernels().apply_kernel(dist=dist, interp_type=InterpType.LINEAR)
            index_coeff = reduce(mul, grid_sizes[i + 1:], 1)
            dim_interp_indices = dim_interp_indices.unsqueeze(-1).repeat(1, 1, 1)

            # compute the lexicographical position of the indices in the d-dimensional grid points
            interp_indices = interp_indices.add(
                dim_interp_indices.view(num_target_points, -1).mul(index_coeff)).to(torch.long)
            interp_values = interp_values.mul(dim_interp_values.view(num_target_points, -1))

        return sparseInterpTensor.unpack_and_sparse_tensor(interp_indices, interp_values,
                                                           n_rows=num_target_points, n_cols=reduce(mul, grid_sizes))
