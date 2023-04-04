import torch

from sgkigp.config import DEFAULT_INDEX_DTYPE


def swap_dimension_index_tensor(indices):
    new_indices = torch.zeros_like(indices)
    new_indices[0, :] = indices[1, :]
    new_indices[1, :] = indices[0, :]
    return new_indices


def unpack_and_sparse_tensor(interp_indices, interp_values, n_rows, n_cols, right_tensor=False):
    index_tensor, value_tensor = unpack_interp_indices_values(
        interp_values=interp_values, interp_indices=interp_indices)
    if right_tensor:
        index_tensor = swap_dimension_index_tensor(index_tensor)
    return make_sparse_tensor(index_tensor, value_tensor.squeeze(), n_rows, n_cols)


def unpack_interp_indices_values(interp_indices, interp_values):

    # TODO: revisit this step and see if this could be done in place
    if len(interp_values.shape) == 1:
        n_target_points = len(interp_values)
        row_tensor = torch.arange(0, n_target_points, dtype=DEFAULT_INDEX_DTYPE, device=interp_values.device)
        row_tensor = row_tensor.unsqueeze_(1).repeat(1, 1).view(-1)
        index_tensor = torch.stack([row_tensor, interp_indices.reshape(-1)], 0)
        value_tensor = interp_values.reshape(-1)
        nonzero_indices = value_tensor.nonzero(as_tuple=False)
        nonzero_indices.squeeze_()
        index_tensor = index_tensor.index_select(1, nonzero_indices)
        value_tensor = value_tensor.index_select(0, nonzero_indices)

    else:
        n_target_points, n_coefficients = interp_values.shape[-2:]

        row_tensor = torch.arange(0, n_target_points, dtype=DEFAULT_INDEX_DTYPE, device=interp_values.device)
        row_tensor = row_tensor.unsqueeze_(1).repeat(1, n_coefficients).view(-1)
        index_tensor = torch.stack([row_tensor, interp_indices.reshape(-1)], 0)
        value_tensor = interp_values.reshape(-1)
        nonzero_indices = value_tensor.nonzero(as_tuple=False)
        nonzero_indices.squeeze_()
        index_tensor = index_tensor.index_select(1, nonzero_indices)
        value_tensor = value_tensor.index_select(0, nonzero_indices)
    return index_tensor, value_tensor


def make_sparse_tensor(index_tensor, value_tensor, n_rows, n_cols):
    # Make the sparse tensor
    type_name = value_tensor.type().split(".")[-1]  # e.g. FloatTensor
    interp_size = torch.Size((n_rows, n_cols))
    if index_tensor.is_cuda:
        cls = getattr(torch.cuda.sparse, type_name)
    else:
        cls = getattr(torch.sparse, type_name)
    try:
        return cls(index_tensor, value_tensor, interp_size)
    except RuntimeError:
        index_tensor = torch.clamp(index_tensor, min=0)
        return cls(index_tensor, value_tensor, interp_size)

