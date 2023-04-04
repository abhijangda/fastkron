import torch

from gpytorch.lazy import LazyTensor, lazify


class SymSGInterpolatedLazyTensor(LazyTensor):

    def __init__(
            self,
            base_lazy_tensor,
            left_interp_coefficient,
    ):

        base_lazy_tensor = lazify(base_lazy_tensor)
        left_interp_coefficient = lazify(left_interp_coefficient)

        super().__init__(
            base_lazy_tensor, left_interp_coefficient,
        )
        self.base_lazy_tensor = base_lazy_tensor
        self.left_interp_coefficient = left_interp_coefficient

    @property
    def right_interp_coefficient(self):
        return self.left_interp_coefficient.transpose(0, 1)

    def _matmul(self, rhs):

        if rhs.ndimension() == 1:
            is_vector = True
            rhs = rhs.unsqueeze(-1)
        else:
            is_vector = False

        # right_interp^T * rhs
        right_interp_res = self.left_interp_coefficient._t_matmul(rhs)

        # base_lazy_tensor * right_interp^T * rhs
        base_res = self.base_lazy_tensor._matmul(right_interp_res)


        # left_interp * base_lazy_tensor * right_interp^T * rhs
        res = self.left_interp_coefficient.matmul(base_res)

        # Squeeze if necessary
        if is_vector:
            res = res.squeeze(-1)
        return res

    def _mul_constant(self, other):
        # We're using a custom method here - the constant mul is applied to the base_lazy tensor
        # This preserves the interpolated structure
        return self.__class__(
            self.base_lazy_tensor._mul_constant(other),
            self.left_interp_coefficient,
        )

    def _t_matmul(self, rhs):
        return self._matmul(rhs=rhs)  # Because this is a symmetric tensor

    def _quad_form_derivative(self, left_vecs, right_vecs):

        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(1)
            right_vecs = right_vecs.unsqueeze(1)

        # base_lazy_tensor grad
        left_res = self.left_interp_coefficient._t_matmul(left_vecs)
        right_res = self.left_interp_coefficient._t_matmul(right_vecs)
        base_lv_grad = list(self.base_lazy_tensor._quad_form_derivative(left_res, right_res))

        # Return zero grad for interp indices
        res = tuple(
            base_lv_grad
            + [
                None,
                None
            ]
        )
        return res

    def _size(self):
        left_hand_size = self.left_interp_coefficient.size(0)
        return torch.Size((left_hand_size, left_hand_size))

    def _transpose_nonbatch(self):
        return self

    def matmul(self, tensor):
        return self._matmul(rhs=tensor)

    def _approx_diag(self):
        base_diag_root = self.base_lazy_tensor.diag().sqrt()

        left_res = self.left_interp_coefficient.matmul(base_diag_root)
        right_res = left_res
        res = left_res * right_res
        return res.squeeze(-1)


class ASymSGInterpolatedLazyTensor(LazyTensor):

    def __init__(
            self,
            base_lazy_tensor,
            left_interp_coefficient,
            right_interp_coefficient
    ):

        base_lazy_tensor = lazify(base_lazy_tensor)
        left_interp_coefficient = lazify(left_interp_coefficient)
        right_interp_coefficient = lazify(right_interp_coefficient)

        super().__init__(
            base_lazy_tensor, left_interp_coefficient, right_interp_coefficient
        )
        self.base_lazy_tensor = base_lazy_tensor
        self.left_interp_coefficient = left_interp_coefficient
        self.right_interp_coefficient = right_interp_coefficient

    def _matmul(self, rhs):

        if rhs.ndimension() == 1:
            is_vector = True
            rhs = rhs.unsqueeze(-1)
        else:
            is_vector = False

        # right_interp^T * rhs
        right_interp_res = self.right_interp_coefficient._matmul(rhs)

        # base_lazy_tensor * right_interp^T * rhs
        base_res = self.base_lazy_tensor._matmul(right_interp_res)

        # left_interp * base_lazy_tensor * right_interp^T * rhs
        res = self.left_interp_coefficient.matmul(base_res)

        # Squeeze if necessary
        if is_vector:
            res = res.squeeze(-1)
        return res

    def _mul_constant(self, other):
        # We're using a custom method here - the constant mul is applied to the base_lazy tensor
        # This preserves the interpolated structure
        return self.__class__(
            self.base_lazy_tensor._mul_constant(other),
            self.left_interp_coefficient,
            self.right_interp_coefficient
        )

    def _t_matmul(self, rhs):

        if rhs.ndimension() == 1:
            is_vector = True
            rhs = rhs.unsqueeze(-1)
        else:
            is_vector = False

        # left_coefficient^T * rhs
        left_interp_res = self.left_interp_coefficient._t_matmul(rhs)

        # base_lazy_tensor * left_coefficient^T * rhs
        base_res = self.base_lazy_tensor._t_matmul(left_interp_res)

        # base_lazy_tensor * left_coefficient^T * rhs
        res = self.right_interp_coefficient._t_matmul(base_res)

        # Squeeze if necessary
        if is_vector:
            res = res.squeeze(-1)
        return res

    def _quad_form_derivative(self, left_vecs, right_vecs):

        if left_vecs.ndimension() == 1:
            left_vecs = left_vecs.unsqueeze(1)
            right_vecs = right_vecs.unsqueeze(1)

        # base_lazy_tensor grad
        left_res = self.left_interp_coefficient._t_matmul(left_vecs)
        right_res = self.right_interp_coefficient._matmul(right_vecs)

        base_lv_grad = list(self.base_lazy_tensor._quad_form_derivative(left_res, right_res))

        # Return zero grad for interp indices
        # TODO: this will require revision but not required unless we are shooting for Bayesian optimization
        res = tuple(
            base_lv_grad
            + [
                None,
                None,
                None,
                None
            ]
        )
        return res

    def _size(self):
        left_hand_size = self.left_interp_coefficient.size(0)
        right_hand_size = self.right_interp_coefficient.size(1) if self.right_interp_coefficient else left_hand_size
        return torch.Size((left_hand_size, right_hand_size))

    def _transpose_nonbatch(self):
        return self

    def matmul(self, tensor):
        return self._matmul(rhs=tensor)

    def _approx_diag(self):
        base_diag_root = self.base_lazy_tensor.diag().sqrt()

        left_res = self.left_interp_coefficient.matmul(base_diag_root)
        right_res = self.right_interp_coefficient.matmul(base_diag_root)
        res = left_res * right_res
        return res.squeeze(-1)
