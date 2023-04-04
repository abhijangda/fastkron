import torch
import numpy as np
import itertools as it

from sgkigp.config import SgBasisType


def get_multi_index_sequence(grid_level, ndimensions):

    """
    :param grid_level: grid level
    :param ndimensions: dimension of the grid
    :return: multi index sequences
    """

    # trivial cases
    if ndimensions < 0:
        raise ValueError('Invalid input')
    elif ndimensions == 1:
        return list(np.arange(grid_level))

    grid_level = grid_level
    levels_it = it.combinations(range(grid_level + ndimensions - 1), ndimensions - 1)
    nlevels = len(['' for _ in levels_it])
    seq = np.zeros(shape=(nlevels, ndimensions), dtype=int)

    seq[0, 0] = grid_level
    maxi = grid_level

    for level in range(1, nlevels):
        if seq[level - 1, 0] > int(0):
            seq[level, 0] = seq[level - 1, 0] - 1
            for dim in range(1, ndimensions):
                if seq[level - 1, dim] < maxi:
                    seq[level, dim] = seq[level - 1, dim] + 1
                    for next_dim in range(dim + 1, ndimensions):
                        seq[level, next_dim] = seq[level - 1, next_dim]
                    break
        else:
            sum1 = int(0)
            for dim in range(1, ndimensions):
                if seq[level - 1, dim] < maxi:
                    seq[level, dim] = seq[level - 1, dim] + 1
                    sum1 += seq[level, dim]
                    for next_dim in range(dim + 1, ndimensions):
                        seq[level, next_dim] = seq[level - 1, next_dim]
                        sum1 += seq[level, next_dim]
                    break
                else:
                    temp = int(0)
                    for next_dim in range(dim + 2, ndimensions):
                        temp += seq[level - 1, next_dim]
                    maxi = grid_level - temp
                    seq[level, dim] = 0
            seq[level, 0] = grid_level - sum1
            maxi = grid_level - sum1
    return seq


def lexorder_matrix(mat, rmat=False):
    order = np.lexsort(np.rot90(mat))
    if rmat:
        return mat[order, :]
    else:
        return order


def compute_levels(grid_level, ndim, comb=False, basis=SgBasisType.NAIVE):
    if ndim == 1:
        assert (not comb, "Combination technique is not applicable to 1-D grids")
        return list(range(grid_level + 1))

    if basis in [SgBasisType.CONSSTART, SgBasisType.ANOVA]:
        grid_level += 2

    assert grid_level >= 0
    if comb:
        start_gl = max(grid_level - ndim + 1, 0)
    else:
        start_gl = 0

    mis = np.vstack(
        [get_multi_index_sequence(grid_level=_gl, ndimensions=ndim) for _gl in range(start_gl, grid_level + 1)])
    mis = lexorder_matrix(mis, rmat=True)

    # Assuming lexical order does not matter if we shift levels by integer
    if basis in [SgBasisType.CONSSTART, SgBasisType.ANOVA]:
        mis = mis - 2

    return list(mis)


def help_level_indices(level, basis=SgBasisType.NAIVE):

    if basis in [SgBasisType.NAIVE, SgBasisType.MODIFIED]:
        return np.array(range(1, 2 ** (level + 1), 2))
    elif basis == SgBasisType.BOUNDSTART:
        assert level >= 0
        if level > 0:
            return np.array(range(1, 2 ** (level + 1), 2))
        else:
            return np.array([-1, 0, 1])
    else:
        raise NotImplementedError


def get_level_indices(level, basis=SgBasisType.NAIVE):
    """
    :param level: sparse grid for subgrid associated with level
    :return: value of I vectors for n-dimensional grid
    """

    point_array = np.meshgrid(*[help_level_indices(level_, basis=basis) for level_ in level], indexing='ij')
    point_array = [arr.reshape(-1, 1) for arr in point_array]
    point_array = np.concatenate(point_array, axis=1)
    return point_array


def compute_indices(mis, basis=SgBasisType.NAIVE):
    level_arrays = []
    reverse_level_arrays = dict()
    index_arrays = []

    start = 0
    for level in mis:
        index_arrays += get_level_indices(level, basis=basis),
        level_arrays += np.array(list(level) * index_arrays[-1].shape[0]).reshape(-1, len(level)),
        reverse_level_arrays[tuple(level)] = np.arange(start, start + index_arrays[-1].shape[0])
        start = start + index_arrays[-1].shape[0]
    return np.concatenate(level_arrays), np.concatenate(index_arrays), reverse_level_arrays


def compute_LI_1d(gl, basis=SgBasisType.NAIVE):
    L = []
    I = []
    for i in range(gl+1):
        I += help_level_indices(i, basis=basis),
        L += [i]*len(I[-1]),
    L, I = np.hstack(L), np.hstack(I)
    return np.vstack([L, I]).T


def compute_LI_order(grid_level, ndim, comb=False, basis=SgBasisType.NAIVE):
    return compute_LI(grid_level, ndim, comb=comb, basis=basis, rmat=False)


def compute_LI(grid_level, ndim, comb=False, basis=SgBasisType.NAIVE, rmat=True):
    """
    Computes LI pairs.

    :param grid_level:
    :param ndim:
    :param comb:
    :param basis:
    :param rmat:
    :return:
    """
    if ndim == 1:
        assert comb is False,  "Combination technique isn't applicable to 1-dimensional setting ..."
        LI = compute_LI_1d(grid_level, basis=basis)
        return LI if rmat else np.lexsort(np.rot90(LI))

    mis = compute_levels(grid_level, ndim, comb=comb, basis=basis)
    L, I, R = compute_indices(mis, basis=basis)
    LI = np.zeros((L.shape[0], 2*L.shape[1]))
    LI[:, ::2] = L
    LI[:, 1::2] = I
    return lexorder_matrix(LI, rmat=rmat).astype(int)


def compute_LI_pairs(grid_level, ndim, comb=False, basis=SgBasisType.NAIVE):
    """
    Computes LI pairs, i.e., computes all Li for any (gl, d).

    :param grid_level:
    :param ndim:
    :param comb:
    :param basis:
    :return:
    """
    LI = compute_LI(grid_level, ndim, comb=comb, basis=basis)
    LI = torch.from_numpy(LI).to(torch.long)
    LIs = {}
    d = ndim
    while d > 0:
        point_levels = torch.sum(LI[:, ::2], axis=1)
        for gl in range(grid_level+1):
            LIs[(gl, d)] = point_levels <= gl
        i = 0

        if basis == SgBasisType.BOUNDSTART:
            X_i_indices = (LI[:, 0] == i) & (LI[:, 1] == i)
        else:
            X_i_indices = LI[:, 0] == i
        LI = LI[X_i_indices, 2:]
        d = d - 1
    return LIs
