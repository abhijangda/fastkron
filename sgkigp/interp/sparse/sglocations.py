import numpy as np

from sgkigp.config import SgBasisType
from sgkigp.interp.sparse.sgindices import compute_levels


# Computing sparse grid locations for 1-dimension
def get_sg_level_locations(level, basis=SgBasisType.NAIVE, umin=0, umax=1):

    assert umin < umax

    # Naive and
    if basis in [SgBasisType.NAIVE, SgBasisType.MODIFIED]:
        assert level >= 0
        return umin + np.array(range(1, 2 ** (level + 1), 2)) * (2.0 ** (-level - 1)) * (umax - umin)

    elif basis == SgBasisType.CONSSTART:

        assert level >= -2
        if level == -2:
            return np.array([(umin + umax) / 2])

        elif level == -1:
            return np.array([umin, umax])

        # otherwise
        return get_sg_level_locations(level, basis=SgBasisType.NAIVE, umin=umin, umax=umax)

    elif basis == SgBasisType.BOUNDSTART:
        assert level >= -1
        if level == 0:
            return np.array([umin, (umin + umax) / 2, umax])
        return get_sg_level_locations(level, basis=SgBasisType.NAIVE, umin=umin, umax=umax)

    elif basis == SgBasisType.ANOVA:
        assert level >= -2

        if level == -2:
            return np.array([umin])
        elif level == -1:
            return np.array([umax])
        return get_sg_level_locations(level, basis=SgBasisType.NAIVE, umin=umin, umax=umax)

    else:
        raise NotImplementedError


def get_sg_points_1d(grid_level, basis=SgBasisType.NAIVE, umin=0, umax=1):

    if basis in [SgBasisType.NAIVE, SgBasisType.MODIFIED]:
        start, end = 0, grid_level + 1

    elif basis == SgBasisType.CONSSTART:
        start, end = -2, grid_level + 1

    elif basis == SgBasisType.BOUNDSTART:
        start, end = 0, grid_level + 1

    elif basis == SgBasisType.ANOVA:
        start, end = -2, grid_level + 1
    else:
        raise NotImplementedError

    return np.hstack([get_sg_level_locations(gl, basis=basis, umin=umin, umax=umax) for gl in range(start, end)])


def help_get_sg_points_nd(levels, basis=SgBasisType.NAIVE, umin=0, umax=1):
    point_array = np.meshgrid(*[get_sg_level_locations(level_, basis=basis, umin=umin, umax=umax) for level_ in levels],
                              indexing='ij')
    point_array = [arr.reshape(-1, 1) for arr in point_array]
    return np.concatenate(point_array, axis=1)


# Computing sparse grid locations for n-dimensions
def get_sg_points_nd(grid_level, ndim, basis=SgBasisType.NAIVE, comb=False, umin=0, umax=1, ordered=True):

    """
    :param grid_level:
    :param ndim:
    :param basis:
    :param umin:
    :param umax:
    :param ordered: order by level norms not lexiographic order
    :return:
    """

    if ndim == 1:
        return get_sg_points_1d(grid_level, basis=basis, umin=umin, umax=umax)

    levels = compute_levels(grid_level=grid_level, comb=comb, ndim=ndim, basis=basis)

    if ordered:
        subgrids_level_norms = np.array(levels).sum(axis=1)
        order = np.argsort(subgrids_level_norms)
        levels = [levels[level_i] for level_i in order]

    return np.vstack([help_get_sg_points_nd(level_i, basis=basis, umin=umin, umax=umax) for level_i in levels])

