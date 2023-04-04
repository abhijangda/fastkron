import torch
import scipy.sparse
import numpy as np
from math import comb

from sgkigp.interp.sparse.construct import get_subgrids
from sgkigp.interp.sparse.sgindices import compute_levels
from sgkigp.interp.sparse.construct import sparse_subgrid_size

from sgkigp.interp.sparse.basis import get_sg_subgrid_basis as get_sg_basis

from sgkigp.config import SgBasisType, InterpType
from sgkigp.interp.sparse.sglocations import get_sg_points_nd
from sgkigp.interp.sparse.sgindices import compute_LI
from sgkigp.interp.sginterp import SparseInterpolation
from sgkigp.interp.sparse.nbhors import compute_comb_B_diag
from gpytorch.lazy import MatmulLazyTensor


def compute_B_diag(ndim, qvals, subgrids, sufficient_levels, basis=SgBasisType.NAIVE):
    B_diag = np.hstack([[sl*comb(ndim-1, q)*((-1)**q)] *sparse_subgrid_size(subgrid, basis=basis)
                        for q, subgrid, sl in zip(qvals, subgrids, sufficient_levels)])
    return B_diag


def spinterpolate(func, X, gl, ndim, umin, umax,
                  kind=InterpType.LINEAR, basis=SgBasisType.NAIVE):
    """
    Performs interpolation on sparse gris using combination technique.
    This function is not efficient as this is meant only for the proof of concept of combination technique.

    :param func:
    :param X:
    :param gl:
    :param ndim:
    :param umin:
    :param umax:
    :param kind:
    :param basis:
    :return:
    """
    # evaluating function on grid
    sg_locs = get_sg_points_nd(gl, ndim, umin=umin, umax=umax, basis=basis)
    f_sg = func(sg_locs)

    # sorting sub-grids by their level norm
    subgrids_level = np.array(compute_levels(grid_level=gl, ndim=ndim, basis=basis)).sum(axis=1)
    right_subgrid_order = np.argsort(subgrids_level)
    subgrids = get_subgrids(gl, ndim, umin=umin, umax=umax, basis=basis)

    subgrids = [subgrids[i] for i in right_subgrid_order]
    subgrids_levels = subgrids_level[right_subgrid_order]

    # computing phi
    phi = scipy.sparse.hstack([get_sg_basis(X, subgrid, kind=kind, comb=True, basis=basis) for subgrid in subgrids])

    # computing B using combination technique
    if basis == SgBasisType.CONSSTART:
        adjust = 2
    else:
        adjust = 0
    qvals = (gl - subgrids_level[right_subgrid_order] - adjust).astype(int)
    sufficient_levels = np.array(subgrids_levels > gl - ndim - adjust)
    B_diag = compute_B_diag(ndim, qvals, subgrids, sufficient_levels, basis=basis)
    f_h = phi @ (B_diag * f_sg)
    return f_h, phi.shape[1]


def spinterpolate_tch_interpolation(func, X, gl, ndim, umin, umax,
                  kind=InterpType.LINEAR, basis=SgBasisType.NAIVE, comb=True):
    # evaluating function on grid
    sg_locs = get_sg_points_nd(gl, ndim, umin=umin, umax=umax, basis=basis, comb=True, ordered=False)
    order = compute_LI(gl, ndim, basis=basis, comb=comb, rmat=False)
    sg_locs = sg_locs[order, :]

    f_sg = torch.from_numpy(func(sg_locs))

    x_target = torch.from_numpy(X) if not torch.is_tensor(X) else X
    phi = SparseInterpolation().sparse_interpolate(
        grid_level=gl,
        ndim=ndim, umin=umin, umax=umax,
        x_target=x_target,
        comb=comb,
        interp_type=kind,
        basis=basis,
        device=x_target.device,
    )

    B = compute_comb_B_diag(gl, ndim, basis=basis, device=x_target.device, dtype=x_target.dtype)
    W = MatmulLazyTensor(phi, B)

    f_h = W.matmul(f_sg).detach().numpy()  # .reshape(-1)
    return f_h, phi.shape[1]



