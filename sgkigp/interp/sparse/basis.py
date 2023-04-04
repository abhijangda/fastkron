import scipy
import operator
import functools
import numpy as np
import scipy.sparse

from sgkigp.config import InterpType, SgBasisType
from sgkigp.interp.misc import sparse_outer_by_row
from sgkigp.interp.kernels import linear_kernel, eval_interp_kernel


# Constructing basis functions for sparse grid
# all definitions for valid for grid with end points on [0, 1]^d
phi = lambda x : np.maximum(1 - np.abs(x), np.zeros_like(x))  # 1-D had function
phi_li = lambda x, l, i: phi((x - i * (2.0**(-l-1)))/2.0**(-l-1)) # scaled and shifted hat function
phi_li_nd = lambda X, L, I: \
    functools.reduce(operator.mul,
                     [phi_li(X[:, i], L[i], I[i]) for i in range(X.shape[1])]) # multi-dimensional  hat function


# Get sparse_grid-based basis function expansion for points in x in 1d
def get_sparse_basis_1d(x, grid, grid_in_level=False, **kwargs):

    # Note: this is implemented assuming hierarchical surplus setting with linear interpolation.
    n = len(x)
    umin, umax, m = grid  # grid min, max, # points

    if grid_in_level:
        m = 2**m

    assert np.all(x < umax) and np.all(x > umin)

    if m == 1:
        du = umax - umin
        center = umin + du / 2.0
        dist = np.abs((x - center) / (du / 2.0))
        vals = linear_kernel(dist, **kwargs)
        # Return sparse matrix in sparse row format
        return scipy.sparse.coo_matrix((vals, (np.arange(n), [0] * n)), shape=(n, 1)).tocsr()

    assert (m > 1)

    du = (umax - umin) / m
    u = np.linspace(umin + du / 2.0, umax - du / 2.0, m)  # grid points

    # Returns index of closest grid point to left of z
    def grid_pos(z):
        return np.floor(((z - umin) / du)).astype(int)

    # Index of neighboring grid point for each (input point, offset) pair
    J = grid_pos(x)  # + offsets[K]

    # Compute distance of each (inputs point, grid point) pair and scale by du
    dist = 2 * np.abs((x - u[J])) / du

    vals = linear_kernel(dist, **kwargs)

    # Return sparse matrix in sparse row format
    W = scipy.sparse.coo_matrix((vals, (np.arange(n), J)), shape=(n, m)).tocsr()
    return W


def get_sg_basis_constant(x, grid, kind=InterpType.LINEAR, **kwargs):
    umin, umax, gl = grid
    assert gl >= -2
    if gl >= 0:
        return get_sg_comb_basis_1d(x, grid, kind=kind, boundary=SgBasisType.NAIVE, **kwargs)
    elif gl == -1:
            grid = (umin - (umax - umin), umax + (umax - umin), 1)
            return get_sg_comb_basis_1d(x, grid, kind=kind, basis=SgBasisType.NAIVE, **kwargs)

    n = len(x)
    dist = np.zeros_like(x)
    I, J = np.arange(n), [0] * n

    # Now evaluate the kernel
    vals = eval_interp_kernel(kind, dist)
    # Return sparse matrix in sparse row format
    W = scipy.sparse.coo_matrix((vals, (I, J)), shape=(n, 1)).tocsr()
    return W


def get_sg_basis_boundary(x, grid, kind=InterpType.LINEAR, **kwargs):

    # TODO: Remove redundant code
    umin, umax, gl = grid
    assert gl >= -1
    if gl >= 1:
        return get_sg_comb_basis_1d(x, grid, kind=kind, boundary=SgBasisType.NAIVE, **kwargs)

    m = 3
    n = len(x)
    umin, umax, gl = grid  # grid min, max, # points

    du = (umax - umin) / 2
    u = np.linspace(umin, umax, m)  # grid points

    # Returns index of closest grid point to left of z
    def grid_pos(z):
        return (z - umin) / du

    # Specifies which grid points will have nonzero basis function
    # values for a point relative to it's grid position
    if kind == InterpType.CUBIC:
        offsets = np.array([-1, 0, 1, 2])
    elif kind == InterpType.LINEAR:
        offsets = np.array([0, 1])
    else:
        raise ValueError('unrecognized kind')

    # Generate indices for all (input point, offset) pairs
    I, K = np.mgrid[0:n, 0:len(offsets)]

    # Index of neighboring grid point for each (input point, offset) pair
    J_tilde = grid_pos(x[I])

    J = np.floor(J_tilde).astype(int) + offsets[K]

    # Drop (input point, grid point) pairs where grid index is out of bounds
    valid_inds = (J >= 0) & (J < m)
    I = I[valid_inds]
    J = J[valid_inds]

    # Compute distance of each (inputs point, grid point) pair and scale by du\
    dist = np.abs((x[I] - u[J])) / du

    # correcting for boundary points
    # Now evaluate the kernel
    vals = eval_interp_kernel(kind, dist)

    # Return sparse matrix in sparse row format
    W = scipy.sparse.coo_matrix((vals, (I, J)), shape=(n, m)).tocsr()
    return W


def get_sg_comb_basis_1d(x, grid, kind=InterpType.LINEAR, basis=SgBasisType.NAIVE, **kwargs):

    if basis == SgBasisType.CONSSTART:
        return get_sg_basis_constant(x, grid, kind=kind, **kwargs)

    elif basis == SgBasisType.BOUNDSTART:
        return get_sg_basis_boundary(x, grid, kind=kind, **kwargs)

    n = len(x)
    umin, umax, gl = grid  # grid min, max, # points

    m = 2 ** gl

    if m == 1:

        if basis == SgBasisType.NAIVE:
            du = umax - umin
            center = umin + du / 2.0
            dist = np.abs((x - center) / (du / 2.0))

        elif basis == SgBasisType.MODIFIED:
            dist = np.zeros_like(x)

        else:
            raise NotImplementedError

        I, J = np.arange(n), [0] * n

    else:
        du = (umax - umin) / m
        u = np.linspace(umin + du / 2.0, umax - du / 2.0, m)  # grid points

        # Returns index of closest grid point to left of z
        def grid_pos(z):
            return (z - (umin + du / 2.0)) / du

        # Specifies which grid points will have nonzero basis function
        # values for a point relative to it's grid position
        if kind == InterpType.CUBIC:
            offsets = np.array([-1, 0, 1, 2])

        elif kind == InterpType.LINEAR:
            offsets = np.array([0, 1])

        else:
            raise ValueError('unrecognized kind')

        # Generate indices for all (input point, offset) pairs
        I, K = np.mgrid[0:n, 0:len(offsets)]

        # Index of neighboring grid point for each (input point, offset) pair
        J_tilde = grid_pos(x[I])
        if basis == SgBasisType.NAIVE:
            du_diff = (x < umin + du / 2.0) | (x > umax - du / 2.0)
            du_diff = du_diff.reshape(-1, 1).repeat(len(offsets), axis=1)

        J = np.floor(J_tilde).astype(int) + offsets[K]

        # Drop (input point, grid point) pairs where grid index is out of bounds
        valid_inds = (J >= 0) & (J < m)
        I = I[valid_inds]
        J = J[valid_inds]
        if basis == SgBasisType.NAIVE:
            du_diff = du_diff[valid_inds]

        # Compute distance of each (inputs point, grid point) pair and scale by du\
        dist = np.abs((x[I] - u[J])) / du

        # correcting for boundary points
        if basis == SgBasisType.NAIVE:
            dist[du_diff] = 2.0 * (dist[du_diff])

    # Now evaluate the kernel
    vals = eval_interp_kernel(kind, dist)

    # Return sparse matrix in sparse row format
    W = scipy.sparse.coo_matrix((vals, (I, J)), shape=(n, m)).tocsr()

    return W


def get_sg_subgrid_basis(x, grid, kind=InterpType.LINEAR, basis=SgBasisType.NAIVE, comb=False,
                         grid_in_level=False,
                         **kwargs):
    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim > 2:
        raise ValueError("x should have at most two-dimensions")
    n, d = x.shape

    if len(grid) != d:
        raise ValueError("Second dim of x (shape (%d, %d)) must match len(grid)=%d" % (n, d, len(grid)))

    # Get basis expansions in each dimension
    if not comb: # hierarchical surplus interpolation
        W_dims = [get_sparse_basis_1d(x[:, j], grid[j], grid_in_level=grid_in_level, **kwargs) for j in range(d)]
    else:
        W_dims = [get_sg_comb_basis_1d(x[:, j], grid[j], kind=kind, basis=basis, **kwargs) for j in range(d)]

    # Compute kron product of rows
    W = W_dims[0]
    for j in range(1, d):
        W = sparse_outer_by_row(W, W_dims[j])
    return W

