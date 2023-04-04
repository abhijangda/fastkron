import os
import scipy
import pickle

import numpy as np
import gpytorch as gp
import sgkigp.config as config
from math import comb

from sgkigp.interp.sparse.sgindices import compute_LI_1d, compute_LI
from sgkigp.config import SgBasisType


def compute_comb_B_diag(
        grid_level, ndim,
        basis=SgBasisType.NAIVE,
        device=config.get_device(),
        dtype=config.dtype(use_torch=True)
    ):

    assert basis in [SgBasisType.NAIVE, SgBasisType.MODIFIED, SgBasisType.BOUNDSTART]
    LI = compute_LI(grid_level=grid_level, ndim=ndim, basis=basis, comb=True, rmat=True)

    subgrids_levels = np.sum(LI[:, ::2], axis=1)
    qvals = (grid_level - subgrids_levels).astype(int)

    B_diag = config.np2torch(np.array([comb(ndim - 1, qval) * ((-1) ** qval) for qval in qvals]),
                             device=device, dtype=dtype)
    return gp.lazy.DiagLazyTensor(B_diag)


def _generatedelta(prefix, d):
    if d == 0:
        return prefix
    allcases = []
    for i in [-1, 0, 1]:
        newPrefix = prefix + [i]
        allcases += _generatedelta(newPrefix, d - 1),
    if d ==1:
        return allcases
    else:
        return allcases[0] + allcases[1] + allcases[2]


def generatedelta(d):
    return _generatedelta([], d)


def compute_B(grid_level, ndim, sparse=True, file_name=None):
    """
    This function compute neighbourhood matrix in hierarchical plus setting of interpolation.
    Also, this is probably only write for SgBasisType.NAIVE.

    :param grid_level:
    :param ndim:
    :param sparse:
    :param file_name:
    :return:
    """
    if file_name is None:
        BPATH = os.path.expanduser("~/sgkigp/data/gridB/")
        file_name = BPATH + "gl_" + str(grid_level) + "_ndim_" + str(ndim) + ".pkl"
        if os.path.exists(file_name):
            cols, rows, values, shape = pickle.load(open(file_name, "rb"))
            return scipy.sparse.coo_matrix((values, (rows, cols)), shape=shape)
        store_true = True

    if ndim == 1:
        LI = compute_LI_1d(grid_level)
    else:
        LI = compute_LI(grid_level, ndim)

    L, I = LI[:, ::2], LI[:, 1::2]
    if sparse:
        rows, cols, values = [], [], []
    else:
        B = np.zeros((len(L), len(L)))

    B_dict = {tuple(l) + tuple(i): ind for ind, (l, i) in enumerate(zip(L, I))}
    B_li = lambda l, i: B_dict[tuple(l) + tuple(i)]
    deltas = generatedelta(ndim)  # 3**ndim X 3

    # Note: this isn't optimized yet as it may not be required.
    # Below two for loops can be converted which will improve time worsen memory.
    # Since, we don't care about time and care about memory for only high level grids. It might be best to have 2 loops.
    for l, i in zip(L, I):
        for delta in deltas:
            i_dash = i + delta
            hash2 = prime_exp(i_dash)
            l_dash = l - hash2
            i_dash = [int(_i_dash / (2 ** _hash2)) for _i_dash, _hash2 in zip(i_dash, hash2)]
            try:
                if sparse:
                    cols += B_li(l_dash, i_dash),
                    rows += B_li(l, i),
                    values += (-2.0) ** (-np.sum(np.abs(delta))),
                else:
                    B[B_li(l, i), B_li(l_dash, i_dash)] = (-2.0) ** (-np.sum(np.abs(delta)))
            except KeyError:
                pass  # this is on purpose as it allows to handle boundary cases
    if False and store_true:
        # dump stuff
        pickle.dump((cols, rows, values, (len(L), len(L))), open(file_name, "wb"))

    if sparse:
        B = scipy.sparse.coo_matrix((values, (rows, cols)), shape=(len(L), len(L)))
    return B


# vectorized code
def prime_exp(x):
    x = x.astype(int)
    shape = x.shape
    x = x.flatten()
    rval = np.zeros_like(x).astype(int)

    power = 1
    plsb = np.zeros_like(rval).astype(int)
    while sum(x) != 0:
        lsb = x & 1
        plsb = plsb | lsb
        rval[plsb == 0] = power
        power += 1
        x = x >> 1
    return rval.reshape(shape)


def compute_eff_B(grid_level, ndim, sparse=True, store_true=False, file_name=None):
    """
    Efficient implementation of compute_B wrt time and worse in memory.

    :param grid_level:
    :param ndim:
    :param sparse:
    :param store_true:
    :param file_name:
    :return:
    """
    if not sparse:
        raise NotImplementedError

    if file_name is None:
        BPATH = os.path.expanduser("~/sgkigp/data/gridB/")
        file_name = BPATH + "gl_" + str(grid_level) + "_ndim_" + str(ndim) + ".pkl"
        if os.path.exists(file_name):
            cols, rows, values, shape = pickle.load(open(file_name, "rb"))
            return scipy.sparse.coo_matrix((values, (rows, cols)), shape=shape)
        store_true = True

    LI = compute_LI(grid_level, ndim).astype(int)

    L, I = LI[:, ::2], LI[:, 1::2]

    B_dict = {tuple(l) + tuple(i): ind for ind, (l, i) in enumerate(zip(L, I))}
    B_li = lambda l, i: B_dict.get(tuple(l) + tuple(i), None)
    deltas = generatedelta(ndim)  # 3^ndim X 3

    idash = np.expand_dims(np.array(deltas), axis=0) + np.expand_dims(I, axis=1)  # G X 3^ndim X ndim
    hash2 = prime_exp(idash)  # G X 3^ndim X ndim
    ldash = np.expand_dims(L, axis=1) - hash2
    idash = (idash / (2 ** hash2)).astype(int)

    cols = [B_li(l_dash, i_dash) for l_dash, i_dash in zip(ldash.reshape(-1, ndim), idash.reshape(-1, ndim))]
    rows = list(np.expand_dims(np.array(range(L.shape[0])), 1).repeat(3 ** ndim, axis=1).flatten())
    values = list(np.expand_dims((-2.0) ** (-np.sum(np.abs(np.array(deltas)), axis=1)), axis=1).repeat(L.shape[0],
                                                                                                       axis=1).T.flatten())
    filter_ = np.array(cols) != None
    apply_f = lambda X: np.array(X)[filter_]

    cols = apply_f(cols)
    rows = apply_f(rows)
    values = apply_f(values)

    if store_true:
        # dump stuff
        pickle.dump((cols, rows, values, (len(L), len(L))), open(file_name, "wb"))

    B = scipy.sparse.coo_matrix((values, (rows, cols)), shape=(len(L), len(L)))
    return B
