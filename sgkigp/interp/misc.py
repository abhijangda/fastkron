import scipy
import numpy as np

import torch
from typing import List, Tuple

import sgkigp.config as config

from sgkigp.config import SgBasisType
from sgkigp.interp.sparse.sgindices import compute_levels

check_one_shift = lambda mis_, gl: gl in mis_

check_two_shift = lambda mis_, gl: ((sum(mis_) == gl) and (gl - 1 in mis_)) or check_one_shift(mis_, gl)



class CreateGrids(object):

    @staticmethod
    def get_grid_sizes(basis, x_grid, device, dtype):

        if basis in [SgBasisType.NAIVE, SgBasisType.MODIFIED]:
            grid_sizes =[2**grid[-1] for grid in x_grid]

        elif basis == SgBasisType.BOUNDSTART:
            grid_sizes = [2**g[-1] if g[-1] > 0 else 3 for g in x_grid]

        else:
            raise NotImplementedError
        return grid_sizes

        #
        # if basis in [SgBasisType.NAIVE, SgBasisType.MODIFIED]:
        #     grid_sizes = torch.Tensor([2**grid[-1] for grid in x_grid])
        #
        # elif basis == SgBasisType.BOUNDSTART:
        #     grid_sizes = torch.Tensor([2**g[-1] if g[-1] > 0 else 3 for g in x_grid])
        #
        # else:
        #     raise NotImplementedError
        # return grid_sizes.to(device=device, dtype=dtype)

    @staticmethod
    def create_grid(
            grid_levels: List[int], grid_bounds: List[Tuple[float, float]]) -> List[torch.Tensor]:
        grid = []
        for i in range(len(grid_bounds)):
            grid += (grid_bounds[i][0], grid_bounds[i][1], grid_levels[i]),
        return grid

    def create_sparse_grid(
            self,
            grid_level: int,
            ndim: int,
            grid_bounds: List[Tuple[float, float]],
            comb=False,
            basis=config.SgBasisType.NAIVE,
            shifted=config.SgShifted.ZERO,
            device=config.get_device(),
            dtype=config.dtype(use_torch=True)
    ):

        # if ndim > 1:
        assert ndim > 1

        if shifted == config.SgShifted.ZERO:
            mis = compute_levels(grid_level=grid_level, ndim=ndim, comb=comb, basis=basis)
            return [self.create_grid(grid_levels=mis_, grid_bounds=grid_bounds) for mis_ in mis]

        mis = compute_levels(grid_level=grid_level, ndim=ndim, comb=False, basis=basis)
        sub_grids = []

        for mis_ in mis:
            if shifted == config.SgShifted.ONE:
                if check_one_shift(mis_, grid_level):
                    sub_grids += self.create_grid(grid_levels=mis_, grid_bounds=grid_bounds),
                else:
                    sub_grids += 2**np.sum(mis_),
            elif shifted == config.SgShifted.TWO:
                if check_two_shift(mis_, grid_level):
                    sub_grids += self.create_grid(grid_levels=mis_, grid_bounds=grid_bounds),
                else:
                    sub_grids += 2**np.sum(mis_),
            else:
                raise NotImplementedError

        return sub_grids

    @staticmethod
    def compute_grid_bounds(umin, umax, ndim):

        if type(umin) == np.ndarray:
            umin = list(umin)
            umax = list(umax)

        if type(umin) == list:
            assert len(umax) == len(umin) == ndim
            return [(umin[i], umax[i]) for i in range(ndim)]
        elif type(umin) == float and type(umax) == float:
            return [(umin, umax) for _ in range(ndim)]
        else:
            print("umin type", type(umin))
            raise NotImplementedError


def sparse_outer_by_row(A, B):
    n, m1 = A.shape
    n2, m2 = B.shape

    assert (n == n2)

    A_row_size = np.diff(A.indptr)
    B_row_size = np.diff(B.indptr)

    # Size of each row of C is product of A and B sizes
    C_row_size = A_row_size * B_row_size

    # Construct indptr for C (indices of first entry of each row)
    C_indptr = np.zeros(n + 1, dtype='int')
    C_indptr[1:] = np.cumsum(C_row_size)

    # These arrays have one entry for each entry of C
    #
    #   C_row_num    what row entry is in
    #   C_row_start  start index of the row
    #   C_row_pos    position within nonzeros of this row
    #
    C_row_num = np.repeat(np.arange(n), C_row_size)
    C_row_start = np.repeat(C_indptr[:-1], C_row_size)
    C_nnz = np.sum(C_row_size)
    C_row_pos = np.arange(C_nnz) - C_row_start

    # Now compute corresponding row positions for A and B
    second_dim_size = B_row_size[C_row_num]
    A_row_pos = C_row_pos // second_dim_size
    B_row_pos = C_row_pos % second_dim_size

    # Convert row positions to absolute positions in the data/indices arrays for A and B
    A_pos = A.indptr[C_row_num] + A_row_pos
    B_pos = B.indptr[C_row_num] + B_row_pos

    # Construct the indices and data for C
    C_indices = A.indices[A_pos] * m2 + B.indices[B_pos]
    C_data = A.data[A_pos] * B.data[B_pos]

    # Finally, make C
    C = scipy.sparse.csr_matrix((C_data, C_indices, C_indptr), shape=(n, m1 * m2))
    return C
