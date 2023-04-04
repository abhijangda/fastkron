import torch
import scipy
import gpytorch as gp
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt


def grid2extent(grid):
    assert (len(grid) == 2)

    ymin, ymax, ny = grid[0]
    xmin, xmax, nx = grid[1]
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    return [xmin - 0.5 * dx, xmax + 0.5 * dx, ymin - 0.5 * dy, ymax + 0.5 * dy]


def eval_kernel(func, X, X_dash=None):
    X = torch.from_numpy(X).type(torch.float32)
    X_dash = torch.from_numpy(X_dash).type(torch.float32) if X_dash is not None else X
    return func(X, X_dash).evaluate().detach().numpy()


# rbf
def rbf_kernel(X, X_dash=None):
    return eval_kernel(gp.kernels.RBFKernel(), X, X_dash)


# mattern
def matern_kernel(X, X_dash=None, nu=2.5):
    return eval_kernel(gp.kernels.MaternKernel(nu=nu), X, X_dash)


def compute_abs_diff(xa, xb):
    return np.abs(xa.reshape(-1, 1) - xb.reshape(1, -1))


def sobolev1(xa, xb=None):
    xb = xa if xb is None else xb
    if len(xa.shape) > 1 and xa.shape[1] > 1:
        return np.multiply(*[sobolev1(xa[:, i], xb[:, i]) for i in range(xa.shape[1])])
    cdist = compute_abs_diff(xa, xb)
    return np.exp(-0.5*cdist)


def sobolev2(xa, xb=None):
    xb = xa if xb is None else xb
    if len(xa.shape) > 1 and xa.shape[1] > 1:
        return np.multiply(*[sobolev2(xa[:, i], xb[:, i]) for i in range(xa.shape[1])])
    l1 = compute_abs_diff(xa, xb)
    return (1 / np.sqrt(3)) * np.exp(-(l1 ** (np.sqrt(3) / 2))) * np.sin(l1 / 2 + np.pi / 6)


def sobolevinf(xa, xb=None):
    xb = xa if xb is None else xb
    if len(xa.shape) > 1 and xa.shape[1] > 1:
        return np.multiply(*[sobolevinf(xa[:, i], xb[:, i]) for i in range(xa.shape[1])])
    l1 = compute_abs_diff(xa, xb)
    fvals = (2 / (np.pi * 3)) * np.ones_like(l1)
    l1_nnz = l1 > 0
    l1 = l1[l1_nnz]
    fvals_nnz = (2 / (np.pi * (l1 ** 3))) * (np.sin(l1) - l1 * np.cos(l1))
    fvals[l1_nnz] = fvals_nnz
    return fvals


def plot_kernel_function(kern_f, name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    xlim = (-3, 3)
    X = np.expand_dims(np.linspace(*xlim, 25), 1)
    Σ = kern_f(X, X)

    # Plot covariance matrix
    im = ax1.imshow(Σ, cmap=cm.YlGnBu)
    cbar = plt.colorbar(
        im, ax=ax1, fraction=0.045, pad=0.05)
    cbar.ax.set_ylabel('$k(x,x)$', fontsize=10)
    ax1.set_title((name + '\n'
                          'example of covariance matrix'))
    ax1.set_xlabel('x', fontsize=13)
    ax1.set_ylabel('x', fontsize=13)
    ticks = list(range(xlim[0], xlim[1] + 1))
    ax1.set_xticks(np.linspace(0, len(X) - 1, len(ticks)))
    ax1.set_yticks(np.linspace(0, len(X) - 1, len(ticks)))
    ax1.set_xticklabels(ticks)
    ax1.set_yticklabels(ticks)
    ax1.grid(False)

    # Show covariance with X=0
    xlim = (-4, 4)
    X = np.expand_dims(np.linspace(*xlim, num=50), 1)
    zero = np.array([[0]])
    Σ0 = kern_f(X, zero)
    # Make the plots
    ax2.plot(X[:, 0], Σ0[:, 0], label='$k(x,0)$')
    ax2.set_xlabel('x', fontsize=13)
    ax2.set_ylabel('covariance', fontsize=13)
    ax2.set_title((name +
                   ' covariance\n'
                   'between $x$ and $0$'))
    # ax2.set_ylim([0, 1.1])
    ax2.set_xlim(*xlim)
    ax2.legend(loc=1)

    fig.tight_layout()
    plt.show()
