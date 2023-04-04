import torch
import scipy
import scipy.sparse
import numpy as np

from functools import reduce
from sgkigp.interp.rectgrid import grid_coords


def inv_permutation(I, device=None):

    # TODO: fix torch long d-type issue
    n, d = I.shape
    if torch.is_tensor(I):
        J = torch.zeros(n, d).to(device=device, dtype=torch.long)
        J = J.scatter_(1, I, torch.arange(d).to(device=device, dtype=torch.long) * torch.ones(n, d).to(device=device, dtype=torch.long)) #
    else:
        J = np.zeros_like(I)
        np.put_along_axis(J, I, np.arange(d), 1)
    return J


def get_simplicial_basis(x, grid):

    if torch.is_tensor(x):
        return get_simplicial_basis_torch(x, grid)

    n, d = x.shape

    # Get info about grid
    coords = grid_coords(grid)
    spacing = np.array([c[1] - c[0] for c in coords])
    origin = np.array([c[0] for c in coords])
    grid_sz = tuple(g[2] for g in grid)

    # Get grid cell multi_index and local coordinates for each data point
    grid_multi_index = np.floor(((x - origin) / spacing)).astype(int)
    grid_point = origin + grid_multi_index * spacing
    local_coords = (x - grid_point) / spacing

    assert np.allclose(x, origin + grid_multi_index * spacing + local_coords * spacing)
    assert np.allclose(x, grid_point + local_coords * spacing)

    # Each data point will have d+1 neighboring grid points
    # (the corners of the simplex containing the point)

    # Get the d+1 basis values for each data point
    sorted_coords = np.sort(local_coords, axis=1)
    basis_values_rev = np.hstack((sorted_coords[:, 0, None],
                                  np.diff(sorted_coords, axis=1),
                                  1 - sorted_coords[:, -1, None]))  # n x d+1
    basis_values = np.flip(basis_values_rev, axis=1)  # n x d+1

    # Construct the reference simplex. It looks like this. Note
    # that row's coordinates are monotonically non-decreasing
    #     [[0 0 0]
    #      [0 0 1]
    #      [0 1 1]
    #      [1 1 1]]
    reference_simplex_corners = np.vstack((np.zeros(d), np.tril(np.ones(d))[:, ::-1])).astype('int')  # d+1 x d

    # The neighbors are obtained by sorting the coordinates (columns) of
    # the reference simplex so they follow the same sorting order as the
    # local coordinates. That is, we sort the reference coordinates by
    # the inverse argsort of the local coordinates
    I = np.argsort(local_coords, axis=1)  # n x d
    J = inv_permutation(I)  # n x d
    neighbor_offsets = reference_simplex_corners[:, J]  # d+1 x n x d

    # Compute indices of neighbors
    neighbor_multi_indices = np.transpose(grid_multi_index + neighbor_offsets, (2, 1, 0))  # d x n x d+1
    neighbor_indices = np.ravel_multi_index(tuple(neighbor_multi_indices), grid_sz, mode='clip')  # n x d+1

    data_point_indices = np.repeat(np.arange(n), d + 1).ravel()

    m = np.prod(grid_sz)
    W = scipy.sparse.coo_matrix((basis_values.ravel(),
                                 (data_point_indices, neighbor_indices.ravel())),
                                shape=(n, m))
    return W.tocsr()


def get_simplicial_basis_torch(x, grid):

    n, d = x.shape

    # Get info about grid
    coords = [torch.from_numpy(i).to(dtype=x.dtype, device=x.device) for i in grid_coords(grid)]

    check_for_singletons = [len(coord) == 1 for coord in coords]

    if any(check_for_singletons):

        if all(check_for_singletons):
            neighbor_indices = torch.zeros(n).to(dtype=torch.long, device=x.device)
            basis_values = torch.ones(n).to(dtype=x.dtype, device=x.device)
            return neighbor_indices, basis_values

        new_grid = [g for i, g in enumerate(grid) if not check_for_singletons[i]]

        #print("grids: ", grid, new_grid)
        check_for_singletons = [not i for i in check_for_singletons]
        neighbor_indices, basis_values = get_simplicial_basis(x[:, check_for_singletons], new_grid)
        return neighbor_indices, basis_values

    device = x.device
    dtype = x.dtype
    put_on = lambda trch_tensor: trch_tensor.to(device=device, dtype=dtype)

    spacing = put_on(torch.Tensor([c[1]-c[0] for c in coords]))
    origin = put_on(torch.Tensor([c[0] for c in coords]))
    grid_sz = tuple(g[2] for g in grid)

    # Get grid cell multi_index and local coordinates for each data point
    grid_multi_index = torch.floor(((x-origin)/spacing)).to(dtype=torch.long)
    grid_point = origin + grid_multi_index*spacing
    local_coords = (x - grid_point)/spacing

    #assert torch.allclose(x, origin + grid_multi_index*spacing + local_coords*spacing,  atol=1e-03)
    #assert torch.allclose(x, grid_point + local_coords*spacing, atol=1e-03)

    # Each data point will have d+1 neighboring grid points
    # (the corners of the simplex containing the point)

    # Get the d+1 basis values for each data point
    sorted_coords = torch.sort(local_coords, axis=1)[0]
    basis_values_rev = torch.hstack((sorted_coords[:, 0, None],
                                  torch.diff(sorted_coords, axis=1),
                                  1-sorted_coords[:, -1, None]))  # n x d+1
    basis_values = torch.flip(basis_values_rev, [1])  # n x d+1

    # Construct the reference simplex. It looks like this. Note
    # that row's coordinates are monotonically non-decreasing
    #     [[0 0 0]
    #      [0 0 1]
    #      [0 1 1]
    #      [1 1 1]]
    reference_simplex_corners = torch.vstack((torch.zeros(d),
                                              torch.tril(torch.ones(d, d)).flip([1]))).to(dtype=torch.long) # d+1 x d

    reference_simplex_corners = reference_simplex_corners.to(device=device)
    # The neighbors are obtained by sorting the coordinates (columns) of
    # the reference simplex so they follow the same sorting order as the
    # local coordinates. That is, we sort the reference coordinates by
    # the inverse argsort of the local coordinates
    I = torch.argsort(local_coords, axis=1) # n x d
    J = inv_permutation(I, x.device)               # n x d
    neighbor_offsets = reference_simplex_corners[:, J] # d+1 x n x d

    # Compute indices of neighbors
    neighbor_multi_indices = grid_multi_index + neighbor_offsets
    neighbor_multi_indices \
        = neighbor_multi_indices.clamp(
        max=(torch.Tensor(grid_sz).to(dtype=torch.long, device=device) - 1)).transpose(0, 2)  # d x n x d+1

    # ravel_multi_index
    grid_ravel_vals = [np.prod(grid_sz[i + 1:]).astype('int') for i in range(d)]

    # n x d+1
    neighbor_indices = reduce(torch.add, [gs*ii for gs, ii in zip(grid_ravel_vals, tuple(neighbor_multi_indices))])

    return neighbor_indices, basis_values

