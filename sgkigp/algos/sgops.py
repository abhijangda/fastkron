import torch
import numpy as np


def mat_X_i(X, i, LI):
    """
    :param X: G(l, d) X nvecs
    :param i: scalar
    :param L: G(l, d) X d - level vectors
    :param I: G(l, d) X d - index vectors
    :return:
        X_i: 2^i X G(l-i, d-1) X nvecs
        L: X G(l-i, d-1) X  d-1
        I: X G(l-i, d-1) X  d-1
    """

    assert len(LI.shape) > 1

    X_i_indices = LI[:, 0] == i
    X_i = X[X_i_indices, :]  # shape: (2^i * G(l-1, d-1)) X nvecs
    X_i = X_i.reshape(2 ** i, -1, X_i.shape[-1])  # reshape: 2^i X G(l-1, d-1) X nvecs
    return X_i, LI[X_i_indices, 2:][:X_i.shape[1], :]


def S_l_G_transpose(X, l, gl):
    """
    :param X: input shape --> 2^l X G(gl-l, d-1) X nvecs
    :param l: first dimensional level index
    :param gl: sparse grid level
    :return: output shape --> G(l, 1) X G(gl-l, d-1) X nvecs
    """
    d1, d2, d3 = X.shape
    assert d1 == 2 ** l

    if torch.is_tensor(X):
        S_mat_X_i = torch.zeros((2 ** (gl + 1) - 1, d2, d3)).to(dtype=X.dtype, device=X.device)
    else:
        S_mat_X_i = np.zeros((2 ** (gl + 1) - 1, d2, d3))

    S_mat_X_i[2**l-1:2 ** (l + 1) - 1, :, :] = X
    return S_mat_X_i


def S_l_G(X, l, gl):
    return X[2**l - 1:2 ** (l + 1) - 1, :, :]


def S_G_G(X, LI, new):
    d1, d2, d3 = X.shape

    if torch.is_tensor(X):
        S_G_G_X = torch.zeros((d1, len(LI), d3)).to(dtype=X.dtype, device=X.device)
        S_G_G_X[:, torch.sum(LI[:, ::2], axis=1) <= new, :] = X
    else:
        S_G_G_X = np.zeros((d1, len(LI), d3))
        S_G_G_X[:, np.sum(LI[:, ::2], axis=1) <= new, :] = X
    return S_G_G_X


def S_G_G_transpose(X, LI, new):

    if torch.is_tensor(X):
        return X[:, torch.sum(LI[:, ::2], axis=1) <= new, :]
    else:
        return X[:, np.sum(LI[:, ::2], axis=1) <= new, :]
