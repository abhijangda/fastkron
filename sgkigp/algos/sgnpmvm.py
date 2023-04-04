import numpy as np
from collections import OrderedDict

from sgkigp.interp.sparse.construct import G
from sgkigp.interp.sparse.sglocations import get_sg_points_nd
from sgkigp.interp.sparse.sgindices import compute_LI_1d, compute_LI_order

from sgkigp.algos.sgops import S_l_G
from sgkigp.algos.sgops import S_G_G
from sgkigp.algos.sgops import mat_X_i
from sgkigp.algos.sgops import S_l_G_transpose
from sgkigp.algos.sgops import S_G_G_transpose

from sgkigp.config import SgBasisType


# Computing kernel matrix
def get_kernel_matrix(LI, ls, basis=SgBasisType.NAIVE, comb=False):
    # dimension check
    L = LI[:, ::2]
    I = LI[:, 1::2]
    assert L.shape[0] == I.shape[0]
    assert L.shape[1] == I.shape[1] == len(ls)

    order = compute_LI_order(max(LI[:, 0]), int(LI.shape[1] / 2), basis=basis, comb=comb)
    LI = LI[order]

    if basis == SgBasisType.BOUNDSTART:
        def compute_dist(a, b, ls):
            b = a.reshape(a.shape[0], 1, a.shape[1])
            return np.sum(np.einsum('ijk, ijk->ijk', a - b, a - b) / np.array(ls), axis=-1)

        order = compute_LI_order(max(LI[:, 0]), int(LI.shape[1]/2), basis=basis, comb=comb)
        data = get_sg_points_nd(max(LI[:, 0]), int(LI.shape[1]/2), basis=basis, comb=comb, ordered=False)[order, :]
        return np.exp(-compute_dist(data, data, ls)/2)

    dist = np.zeros((L.shape[0], L.shape[0]))
    for d in range(len(ls)):
        xli = (I[:, d] * (2.0 ** (-L[:, d] - 1))).reshape(-1, 1)
        if basis == SgBasisType.BOUNDSTART:
            zero_level = L[:, d] == 0
            for ii, val in zip([-1, 0, 1], [0.0, 0.5, 1.0]):
                xli[zero_level][I[:, d][zero_level] == ii] = val
        dist += (np.abs(xli - xli.T)) ** 2 / ls[d]

    return np.exp(-dist/2)


def kernel_matrix_1d(gl, ls=1):
    LI = compute_LI_1d(gl)
    L, I = LI[:, 0], LI[:, 1]
    xli = (I*(2.0**(-L-1))).reshape(-1, 1)
    dist = np.abs(xli - xli.T)
    return np.exp(-dist**2/(2*ls))


def fast_mvm_algorithm(K, X, LI, gl, dim):
    """
    :param K: list of dictionaries of size dim, in which each item represents kernel in that dimension.
    :param X: sg_size X nvecs, input vectors.
    :param LI: sz_size X 2*dim, indicates the index within the grid level of the points indexed for X vector.
    :param gl: a scalar representing grid_level to which K is computed and X is indexed.
    :param dim: a scalar representing dimension of the input space.
    :return: K times X
    """

    assert len(K) == dim, "Size of kernel matrix doesn't match."
    # assert (len(L.shape) == 1 and dim == 1) or L.shape[1] == dim, "L isn't in acceptable shape."

    # Base case
    if dim == 1:
        return np.matmul(kernel_matrix_1d(gl, K[0]), X)

    # Pre-computation
    u, v = OrderedDict(), OrderedDict()
    LI_minus = OrderedDict()

    for _gl in range(gl + 1):
        # computing u_gl
        matXi, LInew = mat_X_i(X, i=_gl, LI=LI)
        LI_minus[_gl] = LInew
        S_mat_X_i = S_l_G_transpose(X=matXi, l=_gl, gl=gl)

        # u[_gl] shape: G(gl, 1) X G(gl-_gl, dim-1) X nvecs
        u[_gl] = np.einsum('ij,jkl->ikl', kernel_matrix_1d(gl, K[0]), S_mat_X_i)
        assert (u[_gl].shape[0] == G(gl, 1)) and (u[_gl].shape[1] == G(gl - _gl, dim - 1))

        # computing v_gl
        d1, d2, d3 = matXi.shape  # 2**_gl X G(gl-_gl, dim-1) X nvecs
        matXi = matXi.swapaxes(0, 1)  # G(gl-_gl, dim-1) X 2**_gl X nvecs
        assert (matXi.shape[0] == G(gl - _gl, dim - 1)) and (matXi.shape[1] == 2 ** _gl)

        # batching all dimensions for MVM
        matXi = matXi.reshape(d2, -1)  # G(gl-_gl, dim-1) X (2**_gl * nvecs)

        # input X.shape: G(gl-_gl, dim-1) X (2**_gl * nvecs)
        # output omatXi.shape: G(gl-_gl, dim-1) X (2**_gl * nvecs)
        omatXi = fast_mvm_algorithm(K=K[1:], X=matXi, LI=LInew, gl=gl - _gl, dim=dim - 1)

        # reformulating output
        v[_gl] = omatXi.reshape(d2, d1, d3).swapaxes(0, 1)  # 2**_gl X G(gl-_gl, dim-1) X nvecs
        assert (v[_gl].shape[0] == 2 ** _gl) and (v[_gl].shape[1] == G(gl - _gl, dim - 1))

    # Main-loop computation
    w = np.zeros_like(X)
    for _gl in range(gl + 1):

        # computing inner sum in a_gl via re-arrangement and recursive call
        a_gl_prevec = None
        for _gl_dash in range(_gl, gl + 1):
            SmatXi = S_l_G(X=u[_gl_dash], l=_gl, gl=gl)
            if a_gl_prevec is None:
                a_gl_prevec = S_G_G(SmatXi, LI=LI_minus[_gl], new=gl - _gl_dash)
            else:
                a_gl_prevec += S_G_G(SmatXi, LI=LI_minus[_gl], new=gl - _gl_dash)

        # computing a_gl
        d1, d2, d3 = a_gl_prevec.shape  # shape: 2**_gl X G(gl-_gl, dim-1) X nvecs
        assert a_gl_prevec.shape[0] == 2 ** _gl and a_gl_prevec.shape[1] == G(gl - _gl, dim - 1)

        a_gl_prevec = a_gl_prevec.swapaxes(0, 1)  # shape: G(gl-_gl, dim-1) X 2**_gl X nvecs
        a_gl_prevec = a_gl_prevec.reshape(d2, -1)  # shape: G(gl-_gl, dim-1) X (2**_gl X nvecs)
        a_gl = fast_mvm_algorithm(K=K[1:], X=a_gl_prevec, LI=LI_minus[_gl], gl=gl - _gl, dim=dim - 1)
        a_gl = a_gl.reshape(d2, d1, d3).swapaxes(0, 1)  # shape: 2**_gl X G(gl-_gl, dim-1) X nvecs
        assert a_gl.shape[0] == 2 ** _gl and a_gl.shape[1] == G(gl - _gl, dim - 1)

        # computing inner sum in b_gl via re-arrangement
        b_gl_prevec = None  # Expected shape: G(_gl, 1) X G(gl-_gl, dim-1) X nvecs

        #         print("Ref .....", G(_gl, 1), "X", G(gl-_gl, dim-1))
        for _gl_dash in range(_gl):

            # v[_gl_dash] shape: 2**_gl_dash X G(gl-_gl_dash, dim-1) X nvecs
            # S_mat_X_i shape: G(_gl, 1) X G(gl-_gl_dash, dim-1) X nvecs
            S_mat_X_i = S_l_G_transpose(X=v[_gl_dash], l=_gl_dash, gl=_gl)
            assert S_mat_X_i.shape[0] == G(_gl, 1) and S_mat_X_i.shape[1] == G(gl - _gl_dash, dim - 1)

            # Transforming:
            # from --> G(_gl, 1) X G(gl-_gl_dash, dim-1) X nvecs
            # to --> G(_gl, 1) X G(gl-_gl, dim-1) X nvecs
            if b_gl_prevec is None:
                # print(gl-_gl_dash, gl-_gl, LI)
                b_gl_prevec = S_G_G_transpose(S_mat_X_i, LI=LI_minus[_gl_dash], new=gl - _gl)
                assert b_gl_prevec.shape[0] == G(_gl, 1) and b_gl_prevec.shape[1] == G(gl - _gl, dim - 1)
            else:
                assert b_gl_prevec.shape[0] == G(_gl, 1) and b_gl_prevec.shape[1] == G(gl - _gl, dim - 1)
                b_gl_prevec += S_G_G_transpose(S_mat_X_i, LI=LI_minus[_gl_dash], new=gl - _gl)

        # computing b_gl
        if b_gl_prevec is not None:
            # Expected shape: G(_gl, 1) X G(gl-_gl, dim-1) X nvecs
            #             print("shapes: ", kernel_matrix_1d(_gl, K[0]).shape, b_gl_prevec.shape)
            #             print("expectd: ", G(_gl, 1), G(gl-_gl, dim-1))
            #             print("gl: ", gl, "  _gl:", _gl, "dim-1: ", dim-1)
            assert b_gl_prevec.shape[0] == G(_gl, 1) and b_gl_prevec.shape[1] == G(gl - _gl, dim - 1)

            # b_gl shape: G(_gl, 1) X G(gl-_gl, dim-1) X nvecs
            b_gl = np.einsum('ij,jkl->ikl', kernel_matrix_1d(_gl, K[0]), b_gl_prevec)

            # b_gl shape: 2**_gl X G(gl-_gl, dim-1) X nvecs
            b_gl = S_l_G(X=b_gl, l=_gl, gl=gl)
            assert b_gl.shape[0] == 2 ** _gl and b_gl.shape[1] == G(gl - _gl, dim - 1)

            # collecting all results in a_gl
            a_gl += b_gl

        w[LI[:, 0] == _gl, :] = a_gl.reshape(-1, w.shape[1])

    return w
