import scipy as scipy
import numpy as np

from sgkigp.interp.kernels import linear_kernel, cubic_kernel
from sgkigp.interp.misc import sparse_outer_by_row


# Return the grid coordinates in each dimension
def grid_coords(grid):
    assert all([g[2] > 1] for g in grid) # check n_points > 1 in each dimension
    return [np.linspace(*g) for g in grid]


# Return total number of grid points
def grid_size(grid):
    return np.prod([g[2] for g in grid])


# Return the actual grid points in an m x d array
def grid_points(grid):
    d = len(grid)
    coords = np.meshgrid(*grid_coords(grid), indexing='ij')    # grid points with coordinates in separate array
    points = np.stack(coords).reshape(d, -1).T   # stack arrays and reshape to m x d
    return points


# Get grid-based basis function expansion for points in x in 1d
def get_basis_1d(x, grid, kind='cubic', **kwargs):
    n = len(x)
    umin, umax, m = grid  # grid min, max, # points

    assert (m > 1)
    u = np.linspace(umin, umax, m)  # grid points
    du = u[1] - u[0]  # grid spacing

    # Returns index of closest grid point to left of z
    def grid_pos(z):
        return np.floor(((z - umin) / du)).astype(int)

    # Specifies which grid points will have nonzero basis function
    # values for a point relative to it's grid position
    if kind == 'cubic':
        offsets = np.array([-1, 0, 1, 2])
    elif kind == 'linear':
        offsets = np.array([0, 1])
    else:
        raise ValueError('unrecognized kind')

    # Generate indices for all (input point, offset) pairs
    I, K = np.mgrid[0:n, 0:len(offsets)]

    # Index of neighboring grid point for each (input point, offset) pair
    J = grid_pos(x[I]) + offsets[K]

    # Drop (input point, grid point) pairs where grid index is out of bounds
    valid_inds = (J >= 0) & (J < m)
    I = I[valid_inds]
    J = J[valid_inds]

    # Compute distance of each (inputs point, grid point) pair and scale by du
    dist = np.abs((x[I] - u[J]) / du)

    # Now evaluate the kernel
    if kind == 'cubic':
        vals = cubic_kernel(dist, **kwargs)
    elif kind == 'linear':
        vals = linear_kernel(dist, **kwargs)
    else:
        raise NotImplementedError

    # Return sparse matrix in sparse row format
    W = scipy.sparse.coo_matrix((vals, (I, J)), shape=(n, m)).tocsr()

    return W


def get_basis(x, grid, **kwargs):
    '''
    Get the KISS-GP "W" matrix, which is a basis expansion
    In multiple dimensions, the basis a product of 1d basis expansion, so this
    functions calls get_basis_1d separately for each dimension separately and
    the combines the results.
    '''

    x = np.array(x)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim > 2:
        raise ValueError("x should have at most two-dimensions")

    n, d = x.shape

    if len(grid) != d:
        raise ValueError("Second dim of x (shape (%d, %d)) must match len(grid)=%d" % (n, d, len(grid)))

    m = grid_size(grid)

    # Get basis expansions in each dimension
    W_dims = [get_basis_1d(x[:, j], grid[j], **kwargs) for j in range(d)]

    # Compute kron product of rows
    W = W_dims[0]
    for j in range(1, d):
        W = sparse_outer_by_row(W, W_dims[j])
    return W


def get_basis_test(X, Y, grid):
    XY = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    W = get_basis(XY, grid).toarray()
    return W

