import numpy as np

from math import comb

from sgkigp.config import SgBasisType
from sgkigp.interp.sparse.sgindices import compute_levels


def G(l, d):
    return np.sum([comb(gl+d-1, d-1)*(2**gl) for gl in range(l+1)])


# Return total number of grid points for a sub-grid
def sparse_subgrid_size(grid, basis=SgBasisType.NAIVE):
    if basis in [SgBasisType.NAIVE, SgBasisType.MODIFIED]:
        return np.prod([2**g[2] for g in grid])

    elif basis == SgBasisType.CONSSTART:
        return np.prod([2**g[2] if g[2] >= 0 else (2 *np.abs(g[2] == -1) + np.abs(g[2] == -2)) for g in grid])

    else:  # SgBasisType.BOUNDSTART:
        return np.prod([2**g[2] if g[2] > 0 else 3 for g in grid])


def sparse_grid_size_table(grid_level, ndim, basis=SgBasisType.NAIVE):
    if basis not in [SgBasisType.NAIVE, SgBasisType.MODIFIED, SgBasisType.BOUNDSTART]:
        raise NotImplementedError

    gs = np.zeros((grid_level + 1, ndim), dtype=np.int)

    # fill zero level
    for dim in range(ndim):
        if basis == SgBasisType.BOUNDSTART:
            gs[0, dim] = 3 ** (dim+1)
        else:
           gs[0, dim] = 1

    # fill rect_grid and sparse_grid for d = 1
    rect_grid = np.zeros(grid_level + 1, dtype=np.int)
    rect_grid[0] = 1 if basis != SgBasisType.BOUNDSTART else 3
    for level in range(1, grid_level + 1):
        rect_grid[level] = 2 ** level
        gs[level, 0] = gs[level - 1, 0] + rect_grid[level]

    # fill table slowly from left to right
    for dim in range(1, ndim):
        for level in range(1, grid_level + 1):
            gs[level, dim] = np.sum([rect_grid[l_dash] * gs[level - l_dash, dim - 1] for l_dash in range(level + 1)])
    return gs, rect_grid


def get_subgrids(grid_level, ndim, umin=0, umax=1,  comb=False, basis=SgBasisType.NAIVE):

    if ndim == 1:
        if basis in [SgBasisType.NAIVE, SgBasisType.MODIFIED]:
            start, end = 0, grid_level + 1

        elif basis == SgBasisType.CONSSTART:
            start, end = -2, grid_level + 1

        elif basis == SgBasisType.BOUNDSTART:
            start, end = -1, grid_level + 1

        else:
            raise NotImplementedError

        return [[(umin, umax, 2 ** mis)] for mis in range(start, end)]

    mis = compute_levels(grid_level=grid_level, ndim=ndim, comb=comb, basis=basis)

    if type(umin) == list or type(umin) == np.ndarray:
        umin, umax = list(umin), list(umax)
        return [[(umin_, umax_, np.sum(mis__))
                 for mis__, umin_, umax_ in zip(mis_, umin, umax)] for mis_ in mis]
    return [[(umin, umax, np.sum(mis__)) for mis__ in mis_] for mis_ in mis]
