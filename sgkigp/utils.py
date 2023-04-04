import torch
import logging
import numpy as np

from datetime import datetime

from itertools import chain
from collections import OrderedDict
from gpytorch.utils.toeplitz import toeplitz_getitem


_cache_ = {
  'memory_allocated': 0,
  'max_memory_allocated': 0,
  'memory_reserved': 0,
  'max_memory_reserved': 0,
}


def _get_memory_info(info_name, unit, rval=False):

    tab = '\t'
    if info_name == 'memory_allocated':
        current_value = torch.cuda.memory.memory_allocated()
    elif info_name == 'max_memory_allocated':
        current_value = torch.cuda.memory.max_memory_allocated()
    elif info_name == 'memory_reserved':
        tab = '\t\t'
        current_value = torch.cuda.memory.memory_reserved()
    elif info_name == 'max_memory_reserved':
        current_value = torch.cuda.memory.max_memory_reserved()
    else:
        raise ValueError()

    divisor = 1
    if unit.lower() == 'kb':
        divisor = 1024
    elif unit.lower() == 'mb':
        divisor = 1024*1024
    elif unit.lower() == 'gb':
        divisor = 1024*1024*1024
    else:
        raise ValueError()

    diff_value = current_value - _cache_[info_name]
    _cache_[info_name] = current_value

    if rval:
        return current_value/divisor
    else:
        return f"{info_name}: \t {current_value} ({current_value/divisor:.3f} {unit.upper()})" \
           f"\t diff_{info_name}: {diff_value} ({diff_value/divisor:.3f} {unit.upper()})"


def get_memory_info(unit='kb'):
    mem_keys = ['memory_allocated', 'max_memory_allocated', 'memory_reserved', 'max_memory_reserved']
    return [_get_memory_info(mem_key, unit, True) for mem_key in mem_keys]


def print_memory_info(unit='kb'):

    print(_get_memory_info('memory_allocated', unit))
    print(_get_memory_info('max_memory_allocated', unit))
    print(_get_memory_info('memory_reserved', unit))
    print(_get_memory_info('max_memory_reserved', unit))
    print('')


def get_in_out_orders(points):
    in_order = np.argsort(points)
    out_order = np.zeros_like(in_order)
    for i, d in enumerate(in_order):
        out_order[d] = i
    return in_order, out_order


def covar_matmul(covar, right_vecs, use_toeplitz=False, orders=None):
    if use_toeplitz:
        assert orders is not None
        output = covar._matmul(right_vecs[orders[0]])[orders[1]]
        return output
    return torch.matmul(covar, right_vecs)


def torch_einsum_2d_times_3d(t1, t2, use_toeplitz=False, orders=None):

    i, j = t1.size()
    m, k, l = t2.size()
    if use_toeplitz:
        assert orders is not None
        output = t1._matmul(t2.view(j, k*l)[orders[0]]).view(i, k, l)[orders[1]]
        return output
    return torch.einsum('ij, jkl->ikl', t1, t2)


def get_sg_toeplitz_column(level1, level2=None, ls=1, device='cpu', dtype=torch.float64):

    if level2 is None:
        level2 = level1

    max_level = max(level1, level2)
    num_points = 2 ** (max_level + 1) - 1
    dist = np.array(range(0, num_points)) * (2.0 ** (-max_level-1))

    # TODO: remove this hard coding for RBF kernel
    toeplitz_column = np.exp(- dist**2/ls)
    return torch.from_numpy(toeplitz_column).to(dtype=dtype, device=device)


def swaps_scipy_coo_columns(mat, desired_order, inplace=False):
    reverse_order = np.zeros_like(desired_order)
    for i, d in enumerate(desired_order):
        reverse_order[d] = i
    try:
        mat.col = reverse_order[mat.col]
    except AttributeError:
        mat = mat.tocoo()
        mat.col = reverse_order[mat.col]
    if inplace:
        return
    return mat


def scipy_coo_to_torch(matsp, dtype='float32'):

    # TODO: set dtype with some consistent rule
    values = matsp.data
    indices = np.vstack((matsp.row, matsp.col))
    i = torch.LongTensor(indices)
    shape = matsp.shape

    if dtype == 'float32' or dtype == torch.FloatTensor or dtype == torch.float32:
        v = torch.FloatTensor(values)
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))
    else:
        v = torch.DoubleTensor(values)
        return torch.sparse.DoubleTensor(i, v, torch.Size(shape))


# Basic logging level
def set_loglevel(level=logging.INFO, simple=True):

    if simple:
        logging.basicConfig(level=level,
                            format=f'%(levelname)s → %(message)s')
    else:
        logging.basicConfig(level=level,
                            format=f'%(levelname)s → {datetime.now()} → %(name)s:%(message)s')

    logging.getLogger().setLevel(level=level)


def log_np(x):
    try:
        return np.array_repr(x)
    except AttributeError:
        return np.array_repr(np.array(x))


def append_ordered_dicts(od1, od2):
    if len(od1) > 0:
        return OrderedDict(chain(od1.items(), od2.items()))
    else:
        return od2


def compute_rows_to_assign(tensor_rows, new_rows):
    assert is_power_of_two(new_rows + 1) and is_power_of_two(tensor_rows)

    rows_to_assign = torch.arange(0, new_rows)
    previous_ = None
    count_times = 0
    while len(rows_to_assign) >= tensor_rows:
        previous_ = rows_to_assign
        rows_to_assign = rows_to_assign[1:][::2]
        count_times += 1
        if count_times > 30:
            print("Something wrong!")
            break
    rows_to_assign = previous_[::2]
    assert len(rows_to_assign) == tensor_rows
    return rows_to_assign


def extract_matrix(k_x, i, j, all_previous=False):
    max_i_j = max(i, j)
    num_points = 2 ** (max_i_j + 1) - 1
    matrix = torch.zeros(num_points, num_points)
    for _i in range(num_points):
        for _j in range(num_points):
            matrix[_i, _j] = toeplitz_getitem(toeplitz_column=k_x, toeplitz_row=k_x, i=_i, j=_j)
    if i > j:
        if not all_previous:
            matrix = matrix[::2, :]
        columns = compute_rows_to_assign(2**j, num_points)
        matrix = matrix[:, columns]
    elif i == j:
        if not all_previous:

            matrix = matrix[::2, ::2]
    else:  # j > i
        if not all_previous:
            matrix = matrix[:, ::2]
        rows = compute_rows_to_assign(2**i, num_points)
        matrix = matrix[rows, :]
    return matrix


def is_power_of_two(n):
    return (n != 0) and (n & (n - 1) == 0)


def kronecker(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0) * B.size(0), A.size(1) * B.size(1))
