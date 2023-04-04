#!/usr/bin/env python3

from functools import reduce
from operator import mul

import torch

from sgkigp.interp.kernels import InterpKernels
from sgkigp.config import DEFAULT_INDEX_DTYPE, InterpType, SgBasisType

from sgkigp.interp.simplicial import get_simplicial_basis

ZERO_KERNEL_DIST = 10.0


class Interpolation(object):

    @staticmethod
    def get_offsets(interp_type, device, dtype=DEFAULT_INDEX_DTYPE):
        """
        :param interp_type: interpolation type
        :param device: device name
        :param dtype: data type
        :return:
        """
        assert interp_type in [InterpType.LINEAR, InterpType.CUBIC]
        return InterpKernels().get_offsets(interp_type=interp_type, device=device, dtype=dtype)

    def compute_dim_interp_values(self, dist, interp_type, i, grid_sizes, dim_interp_indices):
        """
        :param dist: distance from the grid points
        :param interp_type: interpolation type
        :param i: index of dimensions
        :param grid_sizes: size of grid for each dimension
        :param dim_interp_indices: interpolation indices
        :return:
        """

        num_dim = len(grid_sizes)
        num_coefficients = self.get_num_coefficients(interp_type)

        dim_interp_values = InterpKernels().apply_kernel(dist=dist, interp_type=interp_type)
        dim_interp_values = dim_interp_values / torch.sum(dim_interp_values, axis=1).reshape(-1, 1)

        assert torch.isnan(dim_interp_values).sum() == 0

        n_inner_repeat = num_coefficients ** i
        n_outer_repeat = num_coefficients ** (num_dim - i - 1)

        index_coeff = reduce(mul, grid_sizes[i + 1:], 1)
        dim_interp_indices = dim_interp_indices.unsqueeze(-1).repeat(1, n_inner_repeat, n_outer_repeat)
        dim_interp_values = dim_interp_values.unsqueeze(-1).repeat(1, n_inner_repeat, n_outer_repeat)

        return index_coeff, dim_interp_indices, dim_interp_values

    @staticmethod
    def assign_in_lex_order(dim_interp_indices, dim_interp_values, num_target_points, index_coeff,
                            interp_indices, interp_values):

        """
        Puts in lexical order.
        """
        # compute the lexicographical position of the indices in the d-dimensional grid points
        interp_indices = interp_indices.add(dim_interp_indices.view(num_target_points, -1).mul(index_coeff))
        interp_values = interp_values.mul(dim_interp_values.view(num_target_points, -1))
        return interp_indices, interp_values

    @staticmethod
    def get_num_coefficients(interp_type):
        assert interp_type in [InterpType.LINEAR, InterpType.CUBIC]
        return 2**(interp_type.value + 1)  # Works for InterpType.LINEAR and InterpType.CUBIC

    def process_grid_options(self, grid_bounds, grid_sizes, grid, num_dim):
        if grid is not None:
            return grid
        assert grid_bounds is not None
        x_grid = [(*grid_bounds[i], grid_sizes[i]) for i in range(num_dim)]
        return x_grid

    def interpolate(self, grid_bounds, grid_sizes,
                    x_target: torch.Tensor, interp_type=InterpType.LINEAR,
                    x_grid=None, basis=SgBasisType.MODIFIED):

        # Setting up basic info
        device, dtype = x_target.device, x_target.dtype
        num_target_points, num_dim = x_target.size(0), x_target.size(-1)

        # print("Interp type 1: ", interp_type)
        assert num_dim == len(grid_sizes), "Mismatch : " + str(num_dim) + " != " + str(len(grid_sizes))

        x_grid = self.process_grid_options(grid_bounds, grid_sizes, x_grid, num_dim)

        if interp_type == InterpType.SIMPLEX:
            # TODO: remove this bit inconsistency as x_grid contains grid_level not grid size
            x_grid = [(item[0], item[1], grid_sizes[i]) for i, item in enumerate(x_grid)]
            return get_simplicial_basis(x_target, x_grid)

        # tensor values interpolation
        num_coefficients = self.get_num_coefficients(interp_type)
        interp_values = torch.ones(
            num_target_points, num_coefficients ** num_dim, dtype=dtype, device=device
        )
        interp_indices = torch.zeros(
            num_target_points, num_coefficients ** num_dim, dtype=torch.short, device=device
        )

        for i in range(num_dim):

            dim_interp_indices, dist = self._compute_1d_interp_indices(
                x_grid=x_grid[i], x_target=x_target[:, i], grid_size=grid_sizes[i],
                interp_type=interp_type, basis=basis
            )

            index_coeff, dim_interp_indices, dim_interp_values = self.compute_dim_interp_values(
                dist, interp_type, i, grid_sizes, dim_interp_indices
            )

            # print("dim_intper_values: ", dim_interp_values)
            interp_indices, interp_values = self.assign_in_lex_order(
                dim_interp_indices, dim_interp_values, num_target_points, index_coeff,
                interp_indices, interp_values
            )

        return interp_indices, interp_values

    @staticmethod
    def grid_pos(z, umin, du):
        return (z - (umin + du / 2.0)) / du

    @staticmethod
    def grid_pos_boundary(z, umin, du):
        return (z - umin) / du

    def get_index_tensors_data_points(self, num_target_points, interp_type, device, dtype=DEFAULT_INDEX_DTYPE):
        offsets = self.get_offsets(interp_type, device)
        num_points = torch.arange(num_target_points, device=device, dtype=dtype)
        offsets_index = torch.arange(len(offsets), device=device, dtype=dtype)
        I, K = torch.meshgrid(num_points, offsets_index)
        return offsets, I, K

    def _compute_1d_interp_indices(
            self, x_grid, x_target, grid_size,
            interp_type, basis
    ):

        num_target_points = x_target.size(0)
        # print("Num target points: ", num_target_points)

        assert basis in [SgBasisType.MODIFIED]

        umin, umax, __ = x_grid
        device = x_target.device

        if basis == SgBasisType.CONSSTART:
            raise NotImplementedError

        elif basis == SgBasisType.BOUNDSTART:
            raise NotImplementedError
            # return self._compute_1d_basis_boundary(
            #     x_grid=x_grid, x_target=x_target, grid_size=grid_size,
            #     num_coefficients=num_coefficients, num_target_points=num_target_points,
            #     interp_type=interp_type, device=device
            # )

        # Handling zero level grid in a special manner
        # if grid_size == 1:
        #     return self._compute_1d_interp_indices_zero_level(
        #         umax, umin, x_target, num_coefficients,
        #         num_target_points, device, dtype=DEFAULT_INDEX_DTYPE, basis=basis
        #     )

        du = (umax - umin) / grid_size
        u = torch.linspace(umin + du / 2.0, umax - du / 2.0, grid_size).to(device=device, dtype=x_target.dtype)
        # print("u: ", u)

        # Returns index of the nearest grid point to left of z
        offsets, I, K = self.get_index_tensors_data_points(num_target_points, interp_type, device)

        # Index of neighboring grid point for each (input point, offset) pair
        J_tilde = self.grid_pos(x_target[I], umin, du)
        J = torch.floor(J_tilde).to(dtype=torch.long) + offsets[K]

        # Drop (input point, grid point) pairs where grid index is out of bounds
        # -- some points may require support outside grid ....
        I, J, valid_inds = self._drop_out_of_bounds(I, J, grid_size)

        # Compute distance of each (inputs point, grid point) pair and scale by du\
        dist = torch.abs((x_target[I] - u[J])) / du

        # print("dist: ", dist)

        # Fixing boundary cases
        ndist = (torch.ones_like(valid_inds) * ZERO_KERNEL_DIST).to(device=device, dtype=dist.dtype)
        ndist[valid_inds] = dist

        nJ = torch.zeros_like(valid_inds, dtype=torch.long)
        nJ[valid_inds] = J
        J = nJ
        return J, ndist

    def get_basis_functions_1d(self, x_target, grid_size, interp_type, basis, grid_bounds):

        x_grid = None
        x_grid = self.process_grid_options(grid_bounds, [grid_size], x_grid, 1)

        if interp_type == InterpType.SIMPLEX:
            if len(x_target.size()) == 1:
                x_target = x_target.reshape(-1, 1)
            return get_simplicial_basis(x_target, x_grid)

        dim_interp_indices, dist = self._compute_1d_interp_indices(
            x_grid=x_grid[0], x_target=x_target.squeeze(), grid_size=grid_size,
            interp_type=interp_type, basis=basis
        )

        index_coeff, dim_interp_indices, dim_interp_values = self.compute_dim_interp_values(
            dist, interp_type, 0, [grid_size], dim_interp_indices
        )
        return dim_interp_indices.squeeze(), dim_interp_values.squeeze()

    def _drop_out_of_bounds(self, I, J, grid_size):
        valid_inds = (J >= 0) & (J < grid_size)
        I = I[valid_inds]
        J = J[valid_inds]
        return I, J, valid_inds

    # def _compute_1d_basis_boundary(
    #         self, x_grid, x_target, grid_size, num_coefficients, num_target_points,
    #         interp_type, device
    # ):
    #
    #     umin, umax, gl = x_grid
    #     assert gl > -1
    #     if gl >= 1:  # Note: this is twisted logic. Will simplify these calls ...
    #         return self._compute_1d_interp_indices(
    #             x_grid=x_grid, x_target=x_target, grid_size=grid_size,
    #             num_coefficients=num_coefficients, num_target_points=num_target_points,
    #             interp_type=interp_type, basis=SgBasisType.NAIVE,
    #         )
    #
    #     grid_size = 3
    #     umin, umax, __ = x_grid  # grid min, max, # points
    #
    #     du = (umax - umin) / 2
    #     u = torch.linspace(umin, umax, grid_size)  # grid points
    #
    #     # Generate indices for all (input point, offset) pairs
    #     offsets, I, K = self.get_index_tensors_data_points(num_target_points, interp_type, device)
    #
    #     # Index of neighboring grid point for each (input point, offset) pair
    #     J_tilde = self.grid_pos_boundary(x_target[I], umin, du)
    #     J = torch.floor(J_tilde).to(dtype=torch.long) + offsets[K]
    #
    #     # Drop (input point, grid point) pairs where grid index is out of bounds
    #     valid_inds = (J >= 0) & (J < grid_size)
    #     I = I[valid_inds]
    #     J = J[valid_inds]
    #
    #     # Compute distance of each (inputs point, grid point) pair and scale by du\
    #     dist = torch.abs((x_target[I] - u[J])) / du
    #     return self._fix_boundary_cases(valid_inds, dist, J, device)
    #

    #
    # def _fix_boundary_cases(self, valid_inds, dist, J, device):
    #
    #     # Fixing boundary cases
    #     ndist = (torch.ones_like(valid_inds) * ZERO_KERNEL_DIST).to(device=device, dtype=dist.dtype)
    #     ndist[valid_inds] = dist
    #     nJ = torch.zeros_like(valid_inds, dtype=torch.long)
    #     nJ[valid_inds] = J
    #     J = nJ
    #     dist = ndist
    #     dim_interp_indices = J
    #     return dim_interp_indices, dist
    #
    # def _compute_1d_interp_indices_zero_level(
    #         self, umax, umin, x_target, num_coefficients,
    #         num_target_points, device, dtype=DEFAULT_INDEX_DTYPE, basis = SgBasisType.NAIVE
    # ):
    #
    #     du = umax - umin
    #     center = umin + du / 2.0
    #
    #     if basis == SgBasisType.NAIVE:
    #         dist = torch.abs((x_target - center) / (du / 2.0))
    #
    #     elif basis == SgBasisType.MODIFIED:
    #         dist = torch.zeros_like((x_target - center))
    #
    #     else:
    #         raise NotImplementedError
    #
    #     # appending with dist > 1 such that compatible num_coefficients
    #     dist = dist.reshape(-1, 1).repeat(1, num_coefficients)
    #     dist[:, 1:] = ZERO_KERNEL_DIST  # basically zero kernel -- for linear kernel at-least
    #     dim_interp_indices = torch.zeros(num_target_points, num_coefficients)
    #     dim_interp_indices = dim_interp_indices.to(dtype=dtype, device=device)
    #     return dim_interp_indices, dist
