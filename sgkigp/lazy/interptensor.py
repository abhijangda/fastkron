#!/usr/bin/env python3

import torch

from gpytorch.lazy import LazyTensor
from gpytorch.lazy import NonLazyTensor, lazify
from gpytorch.lazy import RootLazyTensor

""" Source: https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/lazy/interpolated_lazy_tensor.py
"""


class ModifiedInterpolatedLazyTensor(LazyTensor):

    def _cholesky_solve(self, rhs, upper: bool = False):
        raise NotImplementedError

    def __init__(
        self,
        base_lazy_tensor,
        left_interp_tensor,
        right_interp_tensor,
        symmetric=True,
    ):
        base_lazy_tensor = lazify(base_lazy_tensor)

        super(ModifiedInterpolatedLazyTensor, self).__init__(
            base_lazy_tensor, left_interp_tensor=left_interp_tensor, right_interp_tensor=right_interp_tensor,
            symmetric=symmetric
        )

        self.base_lazy_tensor = base_lazy_tensor
        self.left_interp_tensor = left_interp_tensor
        self.right_interp_tensor = right_interp_tensor
        self.symmetric = symmetric

    def _approx_diag(self):

        # TODO:
        #  1. We assume that tensor is symmetric.
        #  2. Check if we really need sqrt() as for most kernel diag is one vector.

        if not self.symmetric:
            raise NotImplementedError
        base_diag_root = self.base_lazy_tensor.diag().sqrt()
        left_res = self.left_interp_tensor.matmul(base_diag_root)
        # right_res = self.right_interp_tensor.tranpose(0, 1).matmul(base_diag_root)
        res = left_res * left_res
        return res.squeeze(-1)

    def _expand_batch(self, batch_shape):
        raise NotImplementedError

    def _matmul(self, rhs):

        if rhs.ndimension() == 1:
            is_vector = True
            rhs = rhs.unsqueeze(-1)
        else:
            is_vector = False

        # right_interp^T * rhs
        right_interp_res = self.right_interp_tensor.matmul(rhs)

        # base_lazy_tensor * right_interp^T * rhs
        base_res = self.base_lazy_tensor.matmul(right_interp_res)

        # left_interp * base_lazy_tensor * right_interp^T * rhs
        res = self.left_interp_tensor.matmul(base_res)

        # Squeeze if necessary
        if is_vector:
            res = res.squeeze(-1)

        if not torch.equal(res, res):
            raise RuntimeError("NaNs encountered when trying to perform matrix-vector multiplication")

        # assert res.isnan().sum()
        return res

    def _mul_constant(self, other):
        # We're using a custom method here - the constant mul is applied to the base_lazy tensor
        # This preserves the interpolated structure
        return self.__class__(
            self.base_lazy_tensor._mul_constant(other),
            self.left_interp_tensor,
            self.right_interp_tensor
        )

    def _t_matmul(self, rhs):
        # Get sparse tensor representations of left/right interp matrices

        if rhs.ndimension() == 1:
            is_vector = True
            rhs = rhs.unsqueeze(-1)
        else:
            is_vector = False

        # right_interp^T * rhs
        left_interp_res = self.right_interp_tensor.matmul(rhs)

        # base_lazy_tensor * right_interp^T * rhs
        base_res = self.base_lazy_tensor._t_matmul(left_interp_res)

        # left_interp * base_lazy_tensor * right_interp^T * rhs
        res = self.left_interp_tensor.matmul(base_res)

        # Squeeze if necessary
        if is_vector:
            res = res.squeeze(-1)
        return res

    def _quad_form_derivative(self, left_vecs, right_vecs):

        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(1)
            right_vecs = right_vecs.unsqueeze(1)

        # base_lazy_tensor grad
        left_res = self.left_interp_tensor.transpose(0, 1).matmul(left_vecs)
        right_res = self.right_interp_tensor.matmul(right_vecs)
        base_lv_grad = list(self.base_lazy_tensor._quad_form_derivative(left_res, right_res))

        # Return zero grad for interp indices
        res = tuple(
            base_lv_grad
        )
        return res

    def _size(self):
        return torch.Size([self.left_interp_tensor.shape[0], self.right_interp_tensor.shape[-1]])

    def _transpose_nonbatch(self):

        # Assuming interpolated tensor is symmetric
        res = self.__class__(
            self.base_lazy_tensor.transpose(-1, -2),
            self.left_interp_tensor,
            self.right_interp_tensor,
            self.symmetric
        )
        return res

    def _sum_batch(self, dim):
        raise NotImplementedError

    def diag(self):
        #return torch.ones(self.shape[0]).to(device=self.device, dtype=self.dtype)
        if isinstance(self.base_lazy_tensor, RootLazyTensor) and isinstance(self.base_lazy_tensor.root, NonLazyTensor):
            base_lazy_tensor = self.base_lazy_tensor.root.evaluate()
            left_interp_vals = self.left_interp_tensor.matmul(base_lazy_tensor)
            right_interp_vals = self.right_interp_tensor.matmul(base_lazy_tensor)
            return (left_interp_vals * right_interp_vals).sum(-1)
        else:
            return super(ModifiedInterpolatedLazyTensor, self).diag()
            #return

    def matmul(self, tensor):
        return self._matmul(rhs=tensor)

    def zero_mean_mvn_samples(self, num_samples):
        base_samples = self.base_lazy_tensor.zero_mean_mvn_samples(num_samples)
        batch_iter = tuple(range(1, base_samples.dim()))
        base_samples = base_samples.permute(*batch_iter, 0)
        res = self.left_interp_tensor(base_samples).contiguous()
        batch_iter = tuple(range(res.dim() - 1))
        return res.permute(-1, *batch_iter).contiguous()


