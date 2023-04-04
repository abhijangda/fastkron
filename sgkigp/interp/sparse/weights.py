import numpy as np
import scipy.sparse
from gpytorch.lazy import MatmulLazyTensor

import sgkigp.utils as utils
import sgkigp.config as config
from sgkigp.config import SgBasisType, InterpType

from sgkigp.interp.sparse.nbhors import compute_B, compute_comb_B_diag

from sgkigp.interp.sparse.sgindices import compute_LI
from sgkigp.interp.sginterp import SparseInterpolation
from sgkigp.interp.sparse.construct import get_subgrids
from sgkigp.interp.sparse.basis import get_sparse_basis_1d, get_sg_subgrid_basis


def compute_phi_1d(X, grid_level, umin=-0.2, umax=1.2,
                   nparray=True, comb=False, basis=SgBasisType.NAIVE):
    if comb:
        raise NotImplementedError('Combination technique is not applicable to 1-D cases ...')

    if basis != SgBasisType.NAIVE:
        raise NotImplementedError('this function is not extended with non-linear basis function ...')

    if nparray:
        coeffs = [get_sparse_basis_1d(X, (umin, umax, 2 ** gl)).toarray().reshape(X.shape[0], -1)
                  for gl in range(grid_level + 1)]
        return np.hstack(coeffs)

    # Non-comb case keeping for compatibility
    order = compute_LI(grid_level, 1, rmat=False)
    return utils.swaps_scipy_coo_columns(
        scipy.sparse.hstack([get_sparse_basis_1d(X, (umin, umax, 2 ** gl)) for gl in range(grid_level + 1)]),
        desired_order=order
    )


def compute_phi_numpy(X, grid_level, ndim, umin=-0.2, umax=1.2, nparray=True,
                      interp_type=InterpType.LINEAR,
                      comb=False, basis=SgBasisType.NAIVE):
    if ndim == 1:
        assert not comb
        return compute_phi_1d(X, grid_level, umin=umin, umax=umax, comb=comb, basis=basis)

    # Below function have a duplicate call to compute_levels
    subgrids = get_subgrids(grid_level, ndim, umin=umin, umax=umax, comb=comb, basis=basis)
    order = compute_LI(grid_level, ndim, comb=comb, rmat=False)

    if nparray:
        coeffs = [get_sg_subgrid_basis(X, subgrid, kind=interp_type,
                                       basis=basis, comb=comb).toarray().reshape(X.shape[0], -1)
                  for subgrid in subgrids]
        return np.hstack(coeffs)[:, order]

    basis_evaluated = scipy.sparse.hstack([get_sg_subgrid_basis(X, subgrid, grid_in_level=True,
                                                                kind=interp_type, basis=basis, comb=comb)
                                           for subgrid in subgrids])

    return utils.swaps_scipy_coo_columns(basis_evaluated, order)


def compute_phi(X, grid_level, ndim, umin, umax,
                comb=False,
                interp_type=InterpType.LINEAR,
                basis=SgBasisType.NAIVE,
                use_torch=True,
                device=config.get_device(),
                dtype=config.dtype(use_torch=True)):

    if use_torch:
        return SparseInterpolation().sparse_interpolate(
            grid_level=grid_level,
            ndim=ndim,
            umin=umin,
            umax=umax,
            interp_type=interp_type,
            basis=basis,
            comb=comb,
            x_target=X,
            device=device, dtype=dtype
        )

    phi = compute_phi_numpy(X, grid_level, ndim, umin=umin, umax=umax,
                            comb=comb, nparray=False, interp_type=interp_type, basis=basis)
    return utils.scipy_coo_to_torch(phi)


def compute_W(X, grid_level, ndim, umin=-0.2, umax=1.2,
              is_left=True,
              comb=False,
              interp_type=InterpType.LINEAR,
              basis=SgBasisType.MODIFIED,
              use_torch=True,
              device=config.get_device(),
              dtype=config.dtype(use_torch=True)):

    if grid_level < 2:  # Why this?
        raise NotImplementedError

    Phi = SparseInterpolation().sparse_interpolate(
        grid_level=grid_level,
        ndim=ndim,
        umin=umin, umax=umax,
        x_target=X,
        comb=comb,
        interp_type=interp_type,
        basis=basis,
    )

    # use combination technique
    if comb:
        B = compute_comb_B_diag(grid_level, ndim, basis=basis, device=device, dtype=dtype)
    else:
        B = compute_B(grid_level, ndim)  # TODO: this should be cached directly from disk
        B = utils.scipy_coo_to_torch(B, dtype=dtype)

    # print(Phi.shape, B.shape)
    if is_left:
        return MatmulLazyTensor(Phi, B)

    return MatmulLazyTensor(B.transpose(1, 0), Phi.transpose(1, 0))
