import copy
import scipy
import scipy.sparse
from abc import abstractmethod

from sgkigp.interp.misc import sparse_outer_by_row
from sgkigp.interp.sparse.sgindices import get_multi_index_sequence

import numpy as np
from matplotlib import pyplot as plt


def make_1D_data(grid=100, Ntx=601):
    grid = [0.0, 1.0, grid]
    x = np.linspace(0, 1, grid[2])
    x_test = np.linspace(0, 1, Ntx)
    return grid, x, x_test


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def fun_gaussian(x, mu = 0.5, sigma = 0.12):
    mu = np.array(mu)
    sigma = np.array(sigma)
    return gaussian(x, mu, sigma)


def make_data(dim, grid=[10, 10], Ntx=61, Nty=61):
    grid = [[0.0, 1.0, grid[0]], [0.0, 1.0, grid[1]]]
    x = np.linspace(0, 1, grid[0][2])
    y = np.linspace(0, 1, grid[1][2])
    X,  Y = np.meshgrid(x, y)
    x_test = np.linspace(0, 1, Ntx)
    y_test = np.linspace(0, 1, Nty)
    X_test,  Y_test = np.meshgrid(x_test, y_test)
    return dim, grid, X, Y, X_test, Y_test


def fun_2D_gaussian(x, mu=[0.5, 0.5], sigma=[0.2, 0.2]):
    mu = np.array(mu)
    sigma = np.array(sigma)
    vals = gaussian(x[:, 0], mu[0], sigma[0])*gaussian(x[:, 1], mu[1], sigma[1])
    fs = [lambda x: gaussian(x, mu[0], sigma[0]), lambda x: gaussian(x, mu[1], sigma[1])]
    return vals, fs


def get_gt_values(X, Y, func):
    f_values = func[0](X.reshape(-1)) *  func[1](Y.reshape(-1))
    f_values = f_values.reshape(X.shape[0], X.shape[1])
    return f_values


def plot_results(X, Y, f_org, f_pred):
    plt.subplot(121)
    plt.contourf(X, Y, f_org, levels=14, cmap="RdBu_r")
    plt.figaspect(1.0)
    plt.colorbar()

    plt.subplot(122)
    plt.contourf(X, Y, f_pred, levels=14, cmap="RdBu_r")
    plt.colorbar()
    plt.figaspect(1.0)
    plt.show()


class BasicSGInterpolationModule(object):
    def __init__(self):
        self.points = None
        self.f_values = None
        self.num_points = None

    @abstractmethod
    def get_points(self):
        raise NotImplementedError

    def store_points(self):
        self.points = self.get_points()


class GridLevel1d(BasicSGInterpolationModule):

    def __init__(self, level, umin=0, umax=1):
        super().__init__()

        assert level >= 0
        assert umax == 1 and umin == 0

        self.umin = umin
        self.umax = umax
        self.level = level
        self.quanta = 2.0 ** (-level-1)
        self.num_points = 2 ** level

    def get_points(self, ):
        return np.array(range(1, 2**(self.level+1), 2))*self.quanta

    def interpolate(self, f):
        if self.points is None:
            self.store_points()

        # TODO: duplicate function calls that can be optimized
        self.f_values = f(self.points) - (1/2) * f(self.points + self.quanta) - (1/2) * f(self.points - self.quanta)

    def predict(self, X):

        if self.f_values is None:
            raise NotImplementedError("Function values are not assigned yet!")

        Wx = self.compute_coefficients(X)
        return Wx @ self.f_values

    def compute_coefficients(self, X):
        # Returns index of closest grid point to left of z
        n = len(X)
        I = np.linspace(0, n-1, n).astype(int)

        if self.num_points == 1:
            J = np.zeros_like(X).astype(int)
            M = np.logical_or(X == self.umin, X == self.umax)
        else:
            J, M = self.grid_pos(X)

        J[M] = 0  # over-riding integer points for sometime
        dist = np.abs((X - self.points[J]) / self.quanta)
        vals = 1 - dist
        vals[M] = 0  # assigning ZERO to integer points

        return scipy.sparse.coo_matrix((vals, (I, J)), shape=(n, self.num_points)).tocsr()

    def grid_pos(self, z):
        scales_dist = ((z - self.umin) / (2 * self.quanta))
        mask_dist = np.mod(scales_dist, 1) == 0
        floor_dist = np.floor(scales_dist)
        floor_dist[mask_dist] = 0
        return floor_dist.astype(int), mask_dist


class GridLevelNd(BasicSGInterpolationModule):
    def __init__(self, level, ndim=None):
        """
        If required stores grid points in the meshgrid style.

        :param level: a list of levels across dimensions.
        :param ndim:
        """
        assert type(level) in [list, np.ndarray, np.array]
        if ndim is not None:
            assert len(level) == ndim
        super().__init__()

        self.level = level
        self.ndim = ndim
        self.levels = [GridLevel1d(level_) for level_ in level]
        self.quanta = [level_.quanta for level_ in self.levels]
        self.num_points = np.prod(
            [self.levels[i].num_points for i in range(len(self.levels))])

    def get_points(self, ):
        point_array = np.meshgrid(*[level_.get_points() for level_ in self.levels],  indexing='ij')
        point_array = [arr.reshape(-1, 1) for arr in point_array]
        return np.concatenate(point_array, axis=1)

    def plot_sg_level(self, ax=None, desc='o'):

        if self.points is None:
            self.store_points()

        if ax is None:
            fig, axs = plt.subplots(1, 1, figsize=(10, 10))
            axs.plot(self.points[:, 0], self.points[:, 1], desc)
            plt.grid()
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            return plt
        else:
            ax.plot(self.points[:, 0], self.points[:, 1], desc)

    def interpolate(self, fs):

        # Interpolating at function level
        for level, f, in zip(self.levels, fs):
             level.interpolate(f)

        # Accumulating function values at all points
        vals = [copy.deepcopy(level.f_values) for level in self.levels]
        if self.num_points == 1:
            self.f_values = np.prod(vals)
        else:
            fvals = copy.deepcopy(vals[0])
            for i in range(1, self.ndim):
                fvals = self.outer_multiply(fvals, copy.deepcopy(vals[i]))
            self.f_values = fvals.reshape(-1, 1)

        assert self.f_values.size == self.num_points

    @staticmethod
    def outer_multiply(A, B):
        A = A.reshape(-1, 1)
        B = B.reshape(-1, 1)
        return A@B.T

    def compute_coefficients(self, X):
        W_dims = [self.levels[j].compute_coefficients(X[:, j]) for j in range(self.ndim)]

        # Compute kron product of rows
        W = W_dims[0]
        for j in range(1, self.ndim):
            W = sparse_outer_by_row(W, W_dims[j])
        return W

    def predict(self, X):

        if self.f_values is None:
            raise NotImplementedError("Function values are not assigned yet!")

        W = self.compute_coefficients(X)
        try:
            return W @ self.f_values
        except ValueError:
            return W * self.f_values


class SparseGridInterpolator(BasicSGInterpolationModule):
    def __init__(self, level, ndim):
        """
        sparse grid levels are coming with L1 norm <= (level + ndim - 1)
        :param level:
        :param ndim:
        """
        super().__init__()
        self.level = level
        self.ndim = ndim
        self.mi_seqs = []

        for level_ in range(0, level+1):
            if ndim > 1:
                for mis in get_multi_index_sequence(level_, ndim):
                    self.mi_seqs += GridLevelNd(level=mis, ndim=ndim),
            else:
                self.mi_seqs += GridLevel1d(level=level_),

        self.num_points = np.sum([mis.num_points for mis in self.mi_seqs])

    def compute_coefficients(self, X):
        return [grid_level.compute_coefficients(X) for grid_level in self.mi_seqs]

    def get_points(self,):
        if self.ndim > 1:
            return np.vstack([grid_level.get_points() for grid_level in self.mi_seqs])
        else:
            return np.hstack([grid_level.get_points() for grid_level in self.mi_seqs])

    def store_points_at_level(self):
        for mis in self.mi_seqs:
            for level in mis.levels:
                level.store_points()

    def store_points_at_mis(self):
        for mis in self.mi_seqs:
            mis.store_points()

    def plot_grid(self, figsize=(10, 10)):

        self.store_points_at_mis()
        fig, axs = plt.subplots(1, 1, figsize=figsize)
        clrs = ['k']
        for i in range(len(self.mi_seqs)):
            shift = i * 0.000
            points = self.mi_seqs[i].points
            axs.plot(points[:, 0] + shift, points[:, 1] + shift, 'o' + clrs[i%len(clrs)])
        plt.grid()
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.show()

    def fit(self, f):
        """
        :param f: function required for interpolation
        :return:
        """
        for mis in self.mi_seqs:
            mis.interpolate(f)

    def predict(self, X):
        if len(X.shape) <= 1:
            predict_X = np.zeros(len(X))
        else:
            predict_X = np.zeros_like(len(X))

        for mis in self.mi_seqs:
            predict_X += mis.predict(X)

        return np.array(predict_X).flatten()


if __name__ == '__main__':

    VERBOSE = True

    # testing num points
    for l in range(5, -1, -1):
        gl1D = GridLevel1d(level=l)
        gl1D.store_points()
        assert gl1D.num_points == len(gl1D.points), str(gl1D.num_points) + "!= " + str(len(gl1D.points))
        if VERBOSE:
            print(gl1D.points,  gl1D.quanta, gl1D.level, gl1D.num_points)

    # testing gridlevel ND
    for level in [[2, 3], [4, 5], [2, 3, 4]]:
        ndim = len(level)
        glNd = GridLevelNd(level=level, ndim=ndim)
        glNd.store_points()
        assert glNd.points.shape[0] == glNd.num_points
        assert glNd.points.shape[1] == ndim

    # testing sparse grid for 1D
    for level in range(3, 9, 2):
        sg1D = SparseGridInterpolator(level=level, ndim=1)
        sg1D.store_points()
        assert sg1D.num_points == len(sg1D.points)

    # # testing sparse grid for ND
    for level in range(3, 9, 2):
        sgND = SparseGridInterpolator(level=level, ndim=2 + np.random.choice([2, 3, 1]))
        sgND.store_points()
        assert sgND.points.shape[0] == sgND.num_points, str(sgND.points.shape[0]) + "!=" + str(sgND.num_points)
        assert sgND.points.shape[1] == sgND.ndim
