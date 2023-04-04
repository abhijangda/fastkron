import os
import torch
import numpy as np
from enum import Enum
from gpytorch.lazy import LazyTensor

from sgkigp.config import SgBasisType, MatmulAlgo

from sgkigp.interp.sparse.construct import sparse_grid_size_table
from sgkigp.algos.sgmatmulrecursive import _sg_kernel_matmul
from sgkigp.algos.sgmatmuliterative import _sg_kernel_matmul_efficient


class SGKernelLazyTensor(LazyTensor):

    def __init__(self,
                 covars,
                 LI=None,
                 grid_level=None,
                 ndim=None,
                 algo_type=MatmulAlgo.ITERATIVE,
                 basis_type=SgBasisType.NAIVE,
                 sorting_orders=None,
                 use_toeplitz=False,
                 comb=False,
                 ):
        """
        represents sparse kernel grid matrix K_sg

        :param covars: covariances for each dimension for each grid_level
        :param LI: level and indices
        :param grid_level: grid_level
        :param ndim: dimension of the sparse grid
        :param algo_type: recursive versus iterative algorithm
        """
        super().__init__(covars, LI=LI, grid_level=grid_level, ndim=ndim,
                         algo_type=algo_type, basis_type=basis_type,
                         sorting_orders=sorting_orders,
                         use_toeplitz=use_toeplitz, comb=comb)

        self.covars = covars
        self.grid_level = grid_level
        self.LI = LI
        self.ndim = ndim

        try:
            grid_level = int(grid_level)
            ndim = int(ndim)
        except TypeError:
            grid_level = grid_level.numpy()[0]
            ndim = ndim.numpy()[0]
        assert type(grid_level) == int and type(ndim) == int

        self.sg_table, self.rg_sizes = sparse_grid_size_table(grid_level, ndim, basis=basis_type)
        self.algo_type = algo_type
        self.sorting_orders = sorting_orders
        self.use_toeplitz = use_toeplitz
        self.comb = comb

    @property
    def dtype(self):
        try:
            return self.covars.dtype
        except AttributeError:
            return self.covars[0][0].dtype # toeplitz mode

    @property
    def device(self):
        try:
            return self.covars.device
        except AttributeError:
            return self.covars[0][0].device # toeplitz mode

    def diag(self):

        # TODO: check if this shall always be 1.0 for any kernel as distance will always be zero.
        # #this is most likely incorrect ... with reformulated conversions
        #
        # levels = compute_levels(grid_level=self.grid_level, ndim=self.ndim)
        # subgrid_diags = []
        # for level in levels:
        #     subgrid_diags += KroneckerProductLazyTensor(*[self.covars[i, :G(level[i], 1), :G(level[i], 1)]
        #                                                   for i in range(self.ndim)]).diag(),
        # # TODO: this shall be saved and used here
        # # Also, same computation is utilized during interpolation -- try reusing
        # diag = torch.hstack(subgrid_diags)
        # order = compute_LI_order(self.grid_level, self.ndim)
        # return diag[order]

        return torch.ones(self._size()[0], device=self.device, dtype=self.dtype)

    def _matmul(self, rhs):
        """
        performs a matrix multiplication with the sparse grid kernel, i.e., K_sg * rhs
        :param rhs:
        :return:
        """

        # processing vector
        is_vec = rhs.ndimension() == 1
        if is_vec:
            rhs = rhs.unsqueeze(-1)
        X = rhs.contiguous()

        # performing MVM
        if self.algo_type == MatmulAlgo.RECURSIVE:

            grid_level = self.grid_level.detach().numpy()[0] if torch.is_tensor(self.grid_level) else self.grid_level

            res = _sg_kernel_matmul(
                covars=self.covars, LI=self.LI,
                grid_level=grid_level, ndim=self.ndim, X=X,
            )

        elif self.algo_type == MatmulAlgo.ITERATIVE:

            processed = False
            if self.comb and (self.grid_level - self.ndim) >= 0:
                processed = True
                rhs = torch.zeros(self.sg_table[-1, -1], X.shape[1], device=X.device, dtype=X.dtype)
                rhs[~ self.LI[(int(self.grid_level-self.ndim), int(self.ndim))], :] = X
                X = rhs

            res = _sg_kernel_matmul_efficient(
                covars=self.covars,
                LI=self.LI,
                ndim=int(self.ndim), X=X, grid_level=int(self.grid_level),
                sg_table=self.sg_table,
                rg_sizes=self.rg_sizes,
                use_toeplitz=self.use_toeplitz,
                sorting_orders=self.sorting_orders,
            )
            if processed:
                res = res[~self.LI[(int(self.grid_level-self.ndim), int(self.ndim))], :]
        else:
            raise NotImplementedError

        # processing result
        if is_vec:
            res = res.squeeze(-1)
        return res

    def _compute_size(self):
        if self.comb and (self.grid_level - self.ndim) >= 0:
            return self.sg_table[-1, -1] - sum(self.LI[(self.grid_level-self.ndim, int(self.ndim))])
        return self.sg_table[-1, -1]

    def _size(self):
        """
        returns a :class:`torch.Size` containing the dimensions of the sparse grid kernel matrix K_sg
        :return:
        """
        sg_size = self._compute_size()
        return torch.Size([sg_size, sg_size])

    def _t_matmul(self, rhs):
        return self._matmul(rhs)  # Matrix is symmetric

    def _transpose_nonbatch(self):
        """
        returns a transposed version of the LazyTensor
        :return:
        """
        return self.__class__(self.covars, self.LI, self.grid_level, self.ndim)

    def _cholesky_solve(self, rhs, upper: bool = False):
        raise NotImplementedError
