import torch
import numpy as np

from sgkigp.utils import covar_matmul
from sgkigp.utils import torch_einsum_2d_times_3d

# # import line_profiler
# # profile = line_profiler.LineProfiler()
# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)


def multiply_3d_splits(left_tensors, right_tensors, orders, use_toeplitz=False,):
    return [torch_einsum_2d_times_3d(left_, right_, use_toeplitz=use_toeplitz, orders=order_)
            for left_, right_, order_ in zip(left_tensors, right_tensors, orders)]


def club_3d_tensors(*tensors):
    return torch.vstack(list(*tensors)).swapaxes(0, 1).reshape(tensors[0][0].shape[1], -1)


def split_and_map(V, sp_ids, sg_table, rg_sizes, gl, ndim, sg_1d_sizes, only_rect=False):
    """
    This function splits vector onto rectangular grids in the first dimension and
     then map vector splits to corresponding sparse grids of same level in the first dimension.    :param V:
    :param sp_ids:
    :param sg_table:
    :param rg_sizes:
    :param gl:
    :param ndim:
    :param sg_1d_sizes:
    :param only_rect:
    :return:
    """

    splits = [None]*(gl+1)
    for g_dash, v_l_dash in zip(range(gl + 1), torch.split(V, sp_ids)):

        if only_rect:
            splits[g_dash] = v_l_dash.reshape(rg_sizes[g_dash], sg_table[gl - g_dash, ndim], -1)

        else:
            splits[g_dash] = torch.zeros(sg_1d_sizes[g_dash], sg_table[gl - g_dash, ndim],
                                         V.shape[-1], dtype=V.dtype, device=V.device)

            splits[g_dash][sg_1d_sizes[g_dash] - rg_sizes[g_dash]:sg_1d_sizes[g_dash], :, :] \
                = v_l_dash.reshape(rg_sizes[g_dash], sg_table[gl-g_dash, ndim], -1)

    return splits

# @profile
def _sg_kernel_matmul_efficient(covars, LI, grid_level, ndim, X,
                                sg_table, rg_sizes, use_toeplitz=False,
                                sorting_orders=None):
    gl = grid_level

    # Basic size tests
    assert sg_table[gl, ndim-1] == X.shape[0]
    assert sg_table.shape == (gl+1, ndim)
    assert len(rg_sizes) == gl+1
    sg_1d_sizes = np.cumsum(rg_sizes)
    device, dtype = X.device, X.dtype

    input_is_vec = False
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
        input_is_vec = True

    V = []  # Holds vectors for next iterations
    a_pre_shapes = {}

    all_gl_i = np.array([[_gl, _i] for _gl in range(gl + 1) for _i in range(_gl+1)])
    gl_i_s = {i: all_gl_i[(all_gl_i[:, 0] - all_gl_i[:, 1]) == i].tolist() for i in range(gl+1)}

    covar_d_gl = lambda _d_, _gl_: covars[_d_][_gl_] if use_toeplitz \
        else covars[_d_, :sg_1d_sizes[_gl_], :sg_1d_sizes[_gl_]]
    so_gl = lambda _gl_: None if sorting_orders is None else sorting_orders[_gl_]

    # Step-1: Rolling iterations forward: computing mat_v_d and a_pre_d
    for d in range(ndim):

        # Step-1: split all input vectors
        mat_v = {}  # Keeps splits and merge of current iterations
        if d == 0:  # just split as no merging required for first step

            # Handling 1s-D case (i.e, last dimension)
            sp_ids = tuple(rg_sizes[:gl+1]) if d == ndim-1 else tuple(sg_table[:gl+1, ndim-d-2][::-1]*rg_sizes[:gl+1])

            mat_v[gl] = split_and_map(
                V=X, sp_ids=sp_ids, sg_table=sg_table, rg_sizes=rg_sizes,
                gl=gl, ndim=ndim-d-2, sg_1d_sizes=sg_1d_sizes
            )

        elif d == ndim-1:

            # Handling 1-D case (i.e, last dimension)
            V += [covar_matmul(covar_d_gl(d, _gl), v_l, use_toeplitz, orders=so_gl(_gl))
                  for _gl, v_l in enumerate(V[d-1])],
            break

        else:
            # assuming that first entry requires MVM with kernel at grid level 0 and vector at grid level l
            for _gl, v_l in enumerate(V[d-1]):
                sp_ids = tuple(rg_sizes[:_gl+1]) if d == ndim - 1 \
                    else tuple(sg_table[:_gl+1, ndim-d-2][::-1]*rg_sizes[:_gl+1])
                mat_v[_gl] = split_and_map(
                    V=v_l, sp_ids=sp_ids, sg_table=sg_table, rg_sizes=rg_sizes,
                    gl=_gl, ndim=ndim-d-2, sg_1d_sizes=sg_1d_sizes
                )

        # Step 2: Computing a_hat
        a_pre = {}
        a_pre_shapes[d] = {}
        for _gl in mat_v.keys():

            a_hat = multiply_3d_splits(
                [covar_d_gl(d, i) for i in range(_gl+1)],
                [mat_v[_gl][i] for i in range(_gl+1)],
                [so_gl(iii) for iii in range(_gl + 1)],
                use_toeplitz=use_toeplitz,
            )
            # collect results from all A_hat computations
            a_pre[_gl] = [torch.zeros(rg_sizes[l], a_hat[l].shape[1], a_hat[l].shape[2],
                                      device=device, dtype=dtype) for l in range(_gl+1)]

            for i, j in [(i, j) for i in range(0, _gl + 1) for j in range(i, _gl + 1)]:

                # S_{i, G_j^1} \times \hat{a}_j
                rhs_to_sum = a_hat[j][sg_1d_sizes[i] - rg_sizes[i]:sg_1d_sizes[i], :, :]

                # applying S_{G_{l-j}^{ndim-d-1}, G_{l-j}^{ndim-d-1}}^T
                cols_to_sum_to = LI[(int(_gl-j), ndim-d-1)][LI[(int(_gl-i), ndim-d-1)]]

                # Collecting all results into \tilde{a} for MVM with remaining dimensions
                a_pre[_gl][i][:, cols_to_sum_to, :] += rhs_to_sum

            a_pre_shapes[d][_gl] = [(2*rg_sizes[l], a_hat[l].shape[1], a_hat[l].shape[2]) for l in range(_gl+1)]

        # Setting vectors for next round of computation -- compress all vector multiplication into gl+1 cases
        if d == 0:
            # Note that this order is very important
            V += [club_3d_tensors([mat_v[gl][i][sg_1d_sizes[i] - rg_sizes[i]:sg_1d_sizes[i], :, :], a_pre[gl][i]])
                  for i in range(gl, -1, -1)],
            continue

        # Creating unique gl MVM tasks for remaining dimensions
        # For d > 0 & d < ndim -1
        V += [],
        for i in range(gl+1):  # Important: Iterating over gl-i

            clubbed_tensors = [club_3d_tensors([mat_v[_gl][_i][sg_1d_sizes[_i] - rg_sizes[_i]:sg_1d_sizes[_i], :, :],
                                                a_pre[_gl][_i]]) for _gl, _i in gl_i_s[i]]
            V[-1] += torch.cat(clubbed_tensors, dim=1),

    # Rolling iteration backwards: compute_b_d and final solutions
    for d in range(ndim-2, -1, -1):

        if d == 0:
            a_i = {gl: [None] * (gl + 1)}
            b_hats = {gl: [None] * (gl + 1)}
        else:
            a_i = {_gl: [None] * (_gl + 1) for _gl in range(gl+1)}
            b_hats = {_gl: [None] * (_gl + 1) for _gl in range(gl+1)}

        # De-club processed tensor into b_hats and a_i
        for i in range(len(V[d + 1])):

            if d == 0:
                a_pre_ = a_pre_shapes[d][gl][i]
                b_hats[gl][i], a_i[gl][i] = torch.tensor_split(
                    V[d+1][gl-i].reshape(a_pre_[1], a_pre_[0], a_pre_[2]).swapaxes(0, 1), 2, dim=0)
                continue

            split_ids = tuple([np.prod(list(a_pre_shapes[d][_gl][_i]))//V[d+1][i].shape[0] for _gl, _i in gl_i_s[i]])
            vector_sp = torch.split(V[d+1][i], split_ids, dim=1)

            for ii, (_gl, _i) in enumerate(gl_i_s[i]):  # reverse levels
                b_hats[_gl][_i], a_i[_gl][_i] = torch.tensor_split(
                    vector_sp[ii].reshape(a_pre_shapes[d][_gl][_i][1], a_pre_shapes[d][_gl][_i][0],
                                          a_pre_shapes[d][_gl][_i][2]).swapaxes(0, 1), 2, dim=0)

        # Shape assertions
        if d == 0:
            for i in range(gl+1):
                expected_shape = (a_pre_shapes[d][gl][i][0] // 2, a_pre_shapes[d][gl][i][1],
                                  a_pre_shapes[d][gl][i][2])
                assert expected_shape == a_i[gl][i].shape == b_hats[gl][i].shape

        else:
            for _gl in a_pre_shapes[d].keys():
                for l in range(_gl+1):
                    expected_shape = (a_pre_shapes[d][_gl][l][0]//2, a_pre_shapes[d][_gl][l][1],
                                      a_pre_shapes[d][_gl][l][2])
                    assert expected_shape == a_i[_gl][l].shape == b_hats[_gl][l].shape

        # Computing b_pre and multiply with kernel matrix
        b_pre = {}
        for _gl in a_i.keys():

            # d > 0
            b_pre[_gl] = [torch.zeros(sg_1d_sizes[l], a_i[_gl][l].shape[1], a_i[_gl][l].shape[2],
                                      device=device, dtype=dtype) for l in range(_gl+1)]

            for i, j in [(i, j) for i in range(0, _gl + 1) for j in range(0, i)]:
                cols_to_sum_to = LI[(int(_gl - i), ndim - d - 1)][LI[(int(_gl - j), ndim - d - 1)]]
                b_pre[_gl][i][sg_1d_sizes[j]-rg_sizes[j]:sg_1d_sizes[j], :, :] += b_hats[_gl][j][:, cols_to_sum_to, :]

            # Multiply with the kernel
            b_pre[_gl] = [torch_einsum_2d_times_3d(covar_d_gl(d, i), b_pre[_gl][i], use_toeplitz=use_toeplitz,
                                                   orders=so_gl(i))
                          [sg_1d_sizes[i] - rg_sizes[i]:sg_1d_sizes[i], :, :] for i in range(_gl+1)]

        # Collecting results for this iteration and over-writing vector array
        if d > 0:
            V[d] = [torch.cat([(a_i[_gl][l] + b_pre[_gl][l]).reshape(-1, b_pre[_gl][l].shape[2])
                               for l in range(_gl+1)], dim=0) for _gl in range(gl+1)]
        else:
            V[d] = torch.cat([(a_i[gl][l] + b_pre[gl][l]).reshape(-1, b_pre[gl][l].shape[2])
                              for l in range(gl+1)], dim=0)

    if input_is_vec:
        return V[0].squeeze()
    return V[0]

