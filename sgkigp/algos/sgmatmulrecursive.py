from collections import OrderedDict

import torch
from sgkigp.algos.sgops import S_l_G
from sgkigp.algos.sgops import S_G_G
from sgkigp.algos.sgops import mat_X_i
from sgkigp.algos.sgops import S_l_G_transpose
from sgkigp.algos.sgops import S_G_G_transpose

from sgkigp.config import SgBasisType
from sgkigp.utils import torch_einsum_2d_times_3d
from sgkigp.interp.sparse.construct import G


def _sg_kernel_matmul(covars, LI, grid_level, ndim, X, basis=SgBasisType.NAIVE):

    # Base case
    if ndim == 1:
        assert covars.shape[0] == 1, "Unexpected input encountered ..."
        return torch.matmul(covars[0, :G(grid_level, 1), :G(grid_level, 1)], X)

    # Pre-computation
    u, v = OrderedDict(), OrderedDict()
    LI_minus = OrderedDict()

    for _gl in range(grid_level + 1):

        # computing u_gl
        matXi, LInew = mat_X_i(X, i=_gl, LI=LI)
        LI_minus[_gl] = LInew
        S_mat_X_i = S_l_G_transpose(X=matXi, l=_gl, gl=grid_level)

        # u[_gl] shape: G(gl, 1) X G(gl-_gl, dim-1) X nvecs
        u[_gl] = torch_einsum_2d_times_3d(covars[0][:G(grid_level, 1), :G(grid_level, 1)], S_mat_X_i)
        assert (u[_gl].shape[0] == G(grid_level, 1)) and (u[_gl].shape[1] == G(grid_level - _gl, ndim - 1))

        # computing v_gl
        d1, d2, d3 = matXi.shape  # 2**_gl X G(gl-_gl, dim-1) X nvecs
        matXi = matXi.swapaxes(0, 1)  # G(gl-_gl, dim-1) X 2**_gl X nvecs
        assert (matXi.shape[0] == G(grid_level - _gl, ndim - 1)) and (matXi.shape[1] == 2 ** _gl)

        # batching all dimensions for MVM
        matXi = matXi.reshape(d2, -1)  # G(gl-_gl, dim-1) X (2**_gl * nvecs)

        # input X.shape: G(gl-_gl, dim-1) X (2**_gl * nvecs)
        # output omatXi.shape: G(gl-_gl, dim-1) X (2**_gl * nvecs)
        omatXi = _sg_kernel_matmul(
            covars=covars[1:, :, :], X=matXi, LI=LInew, grid_level=grid_level - _gl, ndim=ndim - 1
        )

        # reformulating output
        v[_gl] = omatXi.reshape(d2, d1, d3).swapaxes(0, 1)  # 2**_gl X G(gl-_gl, dim-1) X nvecs
        assert (v[_gl].shape[0] == 2 ** _gl) and (v[_gl].shape[1] == G(grid_level - _gl, ndim - 1))

    # Main-loop computation
    w = torch.zeros_like(X)
    for _gl in range(grid_level + 1):

        # computing inner sum in a_gl via re-arrangement and recursive call
        a_gl_prevec = None

        for _gl_dash in range(_gl, grid_level + 1):
            SmatXi = S_l_G(X=u[_gl_dash], l=_gl, gl=grid_level)
            if a_gl_prevec is None:
                a_gl_prevec = S_G_G(SmatXi, LI=LI_minus[_gl], new=grid_level - _gl_dash)
            else:
                a_gl_prevec += S_G_G(SmatXi, LI=LI_minus[_gl], new=grid_level - _gl_dash)

        # computing a_gl
        d1, d2, d3 = a_gl_prevec.shape  # shape: 2**_gl X G(gl-_gl, dim-1) X nvecs
        assert a_gl_prevec.shape[0] == 2 ** _gl and a_gl_prevec.shape[1] == G(grid_level - _gl, ndim - 1)

        a_gl_prevec = a_gl_prevec.swapaxes(0, 1)  # shape: G(gl-_gl, dim-1) X 2**_gl X nvecs
        a_gl_prevec = a_gl_prevec.reshape(d2, -1)  # shape: G(gl-_gl, dim-1) X (2**_gl X nvecs)

        a_gl = _sg_kernel_matmul(
            covars=covars[1:, :, :], X=a_gl_prevec, LI=LI_minus[_gl], grid_level=grid_level - _gl, ndim=ndim - 1
        )

        a_gl = a_gl.reshape(d2, d1, d3).swapaxes(0, 1)  # shape: 2**_gl X G(gl-_gl, dim-1) X nvecs
        assert a_gl.shape[0] == 2 ** _gl and a_gl.shape[1] == G(grid_level - _gl, ndim - 1)

        # computing inner sum in b_gl via re-arrangement
        b_gl_prevec = None  # Expected shape: G(_gl, 1) X G(gl-_gl, dim-1) X nvecs

        #         print("Ref .....", G(_gl, 1), "X", G(gl-_gl, dim-1))
        for _gl_dash in range(_gl):

            # v[_gl_dash] shape: 2**_gl_dash X G(gl-_gl_dash, dim-1) X nvecs
            # S_mat_X_i shape: G(_gl, 1) X G(gl-_gl_dash, dim-1) X nvecs
            S_mat_X_i = S_l_G_transpose(X=v[_gl_dash], l=_gl_dash, gl=_gl)
            assert S_mat_X_i.shape[0] == G(_gl, 1) and S_mat_X_i.shape[1] == G(grid_level - _gl_dash, ndim - 1)

            # Transforming:
            # from --> G(_gl, 1) X G(gl-_gl_dash, dim-1) X nvecs
            # to --> G(_gl, 1) X G(gl-_gl, dim-1) X nvecs
            if b_gl_prevec is None:
                # print(gl-_gl_dash, gl-_gl, LI)
                b_gl_prevec = S_G_G_transpose(S_mat_X_i, LI=LI_minus[_gl_dash], new=grid_level - _gl)
                assert b_gl_prevec.shape[0] == G(_gl, 1) and b_gl_prevec.shape[1] == G(grid_level - _gl, ndim - 1)
            else:
                assert b_gl_prevec.shape[0] == G(_gl, 1) and b_gl_prevec.shape[1] == G(grid_level - _gl, ndim - 1)
                b_gl_prevec += S_G_G_transpose(S_mat_X_i, LI=LI_minus[_gl_dash], new=grid_level - _gl)

        # computing b_gl
        if b_gl_prevec is not None:
            assert b_gl_prevec.shape[0] == G(_gl, 1) and b_gl_prevec.shape[1] == G(grid_level - _gl, ndim - 1)

            # b_gl shape: G(_gl, 1) X G(gl-_gl, dim-1) X nvecs
            b_gl = torch_einsum_2d_times_3d(covars[0][:G(_gl, 1), :G(_gl, 1)], b_gl_prevec)

            # b_gl shape: 2**_gl X G(gl-_gl, dim-1) X nvecs
            b_gl = S_l_G(X=b_gl, l=_gl, gl=grid_level)
            assert b_gl.shape[0] == 2 ** _gl and b_gl.shape[1] == G(grid_level - _gl, ndim - 1)

            # collecting all results in a_gl
            a_gl += b_gl

        w[LI[:, 0] == _gl, :] = a_gl.reshape(-1, w.shape[1])
    return w
