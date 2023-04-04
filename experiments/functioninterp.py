import os
import fire
import torch
import wandb
import pickle
import numpy as np
from collections import defaultdict

from sgkigp.config import SgBasisType, InterpType, set_seeds

from sgkigp.interp.sparse.construct import G
from sgkigp.interp.rectgrid import get_basis, grid_points
from sgkigp.interp.sparse.combination import spinterpolate_tch_interpolation

from sgkigp.interp.rginterp import Interpolation
import sgkigp.tensorize as sparseInterpTensor


#np.random.seed(1337)

bj = np.array([9.0, 7.25, 1.85, 7.03, 20.4, 4.3, 1.0, 20.4, 4.3, 1.0])
f_gaussian = lambda X, w, c: np.exp(-np.sum((c**2)*(X - w)**2, axis=1))
f_sine = lambda X, w, c: np.sin(X[:, 0] + X[:, 1])
f_cos = lambda X, w, c: np.cos(X[:, 0] + X[:, 1])
f_sine2 = lambda X, w, c: np.sin((1/X.shape[1])*np.sum(X, axis=1))
f_cos2 = lambda X, w, c: np.cos((1/X.shape[1])*np.sum(X, axis=1))
f_sine3 = lambda X, w, c: np.sin(np.sum(X, axis=1))
f_cos3 = lambda X, w, c: np.cos(np.sum(X, axis=1))
bj = np.array([9.0, 7.25, 1.85, 7.03, 20.4, 4.3, 1.0, 1.0, 3.7, 2.3])

# ocillatory
f_osc = lambda X, w, c : np.cos(np.pi*2*w[0] + np.matmul(X, c))

# product peak
f_product = lambda X, w, c : np.prod((1/(c**(-2) + (X - w)**2)), axis=1)

# corner peak
f_corner = lambda X, w, c: (1 + np.matmul(X, c))**(- X.shape[1] - 1)

# guassian peak
f_gaussian_2 = lambda X, w, c: np.exp(-np.sum((c**2)*(X)**2, axis=1))

# continuous peak
f_continuous = lambda X, w, c: np.exp(- np.sum(c*(np.abs(X- w)), axis=1))


def get_cj(i, ndim):
    i = i - 1
    tt = np.random.rand(ndim)
    return bj[i]*tt/np.sum(tt)


def err(f1, f2):
    return np.mean(np.abs(f1-f2))


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj][0]


def get_test(ndim):

    assert ndim in [2, 4, 6, 8, 10]
    dirpath = os.path.expanduser("~/sgkigp/data/")

    if ndim == 4:
        return pickle.load(open(dirpath + "dim4.yaml", "rb"))

    elif ndim == 6:
        return pickle.load(open(dirpath + "dim6.yaml", "rb"))

    elif ndim == 10:
        return pickle.load(open(dirpath + "dim10.yaml", "rb"))

    raise NotImplementedError


def get_test_w_c(task_id, ndim):
    datapath = os.path.expanduser("~/sgkigp/data/")
    datapath = datapath + 'f' + str(task_id) + "_ndim" + str(ndim) + ".pt"
    _, _, _, _, test_x, test_y, c, w = torch.load(open(datapath, 'rb'))
    test_x, test_y, c, w = test_x.numpy(), test_y.numpy(), c, w

    return test_x, test_y, c, w


def get_rg_basis(umin, umax, grid_sizes, x_target, interp_type, basis):
    grid_bounds = tuple((umin, umax) for i in range(x_target.shape[1]))

    interp_indices, interp_values = Interpolation().interpolate(grid_bounds, grid_sizes, x_target, interp_type,
                                                                basis=basis)

    return sparseInterpTensor.unpack_and_sparse_tensor(interp_indices.squeeze(),
                                                       interp_values.squeeze(), x_target.shape[0],
                                                       np.prod(grid_sizes)).to_dense()


def main(task_id: int = 1, ndim: int = 4, grid_level: int = 2,
         basis: int = 1, seed: int = 1337, log_dir=None):

    set_seeds(seed)
    if log_dir is None:
        sweep_name = os.environ.get(wandb.env.SWEEP_ID, 'solo')
        log_dir = os.path.expanduser('~/sgkigp/') + 'wandb/sweep-' + sweep_name
    os.makedirs(log_dir, exist_ok=True)
    file_prefix = log_dir + "/gl_" + str(grid_level) + "_ndim_" + str(ndim) + "_task_" \
                  + str(task_id) + "_basis_" + str(basis) + "_seed_" + str(seed) + ".yaml"

    ntest = 1000
    umin = -0.1
    umax = 1.1

    funcs = {1: f_gaussian, 2: f_sine, 3: f_cos, 4: f_sine2, 5: f_cos2, 6:f_sine3, 7: f_cos3,
             11:f_osc, 12:f_product, 13: f_corner, 14: f_gaussian_2}
    func = funcs[task_id]
    print("Function: ", namestr(funcs[task_id], globals()))

    if task_id <= 10:
        Xtest = get_test(ndim=ndim)
        c = get_cj(task_id, ndim)
        w = np.random.rand(ndim)
        f_t = func(Xtest, w, c)
    else:
        #test_x, test_y, c, w
        Xtest, f_t, c, w = get_test_w_c(task_id, ndim)

    sg_func = lambda x: func(x, w, c)

    rg_methods = defaultdict(list)
    sg_methods = defaultdict(list)
    # regular-grid
    grid_size_dim = int(np.floor(G(grid_level, ndim)**(1/ndim)))
    grid = [(umin, umax, grid_size_dim)]*ndim
    X = grid_points(grid)
    f_rg = func(X, w, c)

    Xtest = torch.from_numpy(Xtest)
    f_rg = f_rg #.to(torch.float64)

    # W0 = get_rg_basis(umin, umax, [grid_size_dim] * ndim, Xtest, InterpType(0), SgBasisType(basis))
    # f_h = (W0 @ f_rg.reshape(-1, 1)).reshape(-1).detach().numpy()
    # rg_methods['linear'] += (err(f_h, f_t), X.shape[0]),

    W = get_basis(Xtest, grid, kind='cubic')
    # W = W.to(torch.float64)
    f_h = (W @ f_rg.reshape(-1, 1)).reshape(-1)
    rg_methods['cubic'] += (err(f_h, f_t), X.shape[0]),

    W = get_rg_basis(umin, umax, [grid_size_dim] * ndim, Xtest, InterpType(2), SgBasisType(basis))
    # W = W.to(torch.float64)
    f_h = (W @ f_rg.reshape(-1, 1)).reshape(-1).detach().numpy()
    rg_methods['simplex'] += (err(f_h, f_t), X.shape[0]),

    # sparse-grid
    for kind in [1, 2]:
        f_h, numpoints = spinterpolate_tch_interpolation(sg_func, Xtest, grid_level, ndim,
                                                         umin, umax, kind=InterpType(kind),
                                                         basis=SgBasisType(basis))
        sg_error = err(f_h, f_t),
        sg_methods[(kind, basis)] += (sg_error, numpoints),

    print(rg_methods, sg_methods)
    pickle.dump((rg_methods, sg_methods, task_id, grid_level, ndim, (w, c)),
                open(file_prefix, "wb"))


if __name__ == "__main__":
    fire.Fire(main)
