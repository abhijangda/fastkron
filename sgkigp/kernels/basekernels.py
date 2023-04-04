import torch
from gpytorch.kernels.kernel import Kernel
from gpytorch.constraints import Interval

from sgkigp.config import SobolevKernType

sqrt_3_by_2 = 0.86602
pi_by_6 = 0.523598
sqrt_2 = 1.414213
one_by_sqrt = 0.707106
DEFAULT_LOWER_BOUND = -torch.ones(1).squeeze()*1e5
DEFAULT_UPPER_BOUND = torch.ones(1).squeeze()*1e5
two_pi = 6.283185
two_pi_square = 19.7392088


def default_postprocess(dist_mat):
    return dist_mat


def postprocess_sobolev1(dist_mat):
    return dist_mat.mul_(-1.0).exp_()


def postprocess_sobolev2(dist_mat):
    return dist_mat.mul_(-sqrt_3_by_2).exp_().mul_(torch.sin(dist_mat.div_(2.0) + pi_by_6))


def postprocess_sobolev3(dist_mat):
    return dist_mat.mul_(-1.0).exp_() \
           + dist_mat.mul_(-one_by_sqrt).mul_(sqrt_2).mul_(torch.sin(dist_mat.mul_(one_by_sqrt)))


def smkernelpostprocess(dist_mat1, dist_mat2):
    return dist_mat1.mul_(two_pi).cos().mul_(dist_mat2.mul_(-two_pi_square))


class SobolevKernel(Kernel):
    has_lengthscale = True

    def __init__(self, sobolev_type=SobolevKernType.SOBOLEV1, *args, **kwargs):
        super().__init__(args=args, kwargs=kwargs)

        if sobolev_type == SobolevKernType.SOBOLEV1:
            self.postprocess = postprocess_sobolev1

        elif sobolev_type == SobolevKernType.SOBOLEV2:
            self.postprocess = postprocess_sobolev2

        elif sobolev_type == SobolevKernType.SOBOLEV3:
            self.postprocess = postprocess_sobolev3

        else:
            raise NotImplementedError

    def forward(self, x1, x2, diag=False, **params):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)

        dist_mat = torch.cdist(x1_, x2_, 1.0)
        return self.postprocess(dist_mat)


class SpectralMixtureOneDim(Kernel):
    has_lengthscale = True

    def __init__(self, lower_bound_mean=DEFAULT_LOWER_BOUND, upper_bound_mean=DEFAULT_UPPER_BOUND, *args, **kwargs):
        super().__init__(args=args, kwargs=kwargs)

        lengthscale_num_dims = 1 if self.ard_num_dims is None else self.ard_num_dims
        self.register_parameter(
            name="raw_meancenter",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, lengthscale_num_dims)),
        )

        meancenter_constraint = Interval(lower_bound=lower_bound_mean, upper_bound=upper_bound_mean)
        self.register_constraint("raw_meancenter", meancenter_constraint)

    @property
    def meancenter(self):
        return self.raw_meancenter_constraint.transform(self.raw_meancenter)

    @meancenter.setter
    def meancenter(self, value):
        self._set_meancenter(value)

    def _set_meancenter(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_meancenter)

        self.initialize(raw_meancenter=self.raw_meancenter_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):

        x1_ = x1.mul(self.meancenter)
        x2_ = x2.mul(self.meancenter)
        dist_mat1 = torch.cdist(x1_, x2_, 1.0)

        x1_ = x1.mul(self.lengthscale)
        x2_ = x2.mul(self.lengthscale)
        dist_mat2 = torch.cdist(x1_, x2_, 2.0)

        return smkernelpostprocess(dist_mat1, dist_mat2)


if __name__ == '__main__':
    pass

    # ndim = 2
    # npoints = 160 + 1
    # on_sparse_grid = False
    #
    # basis = SgBasisType.MODIFIED
    # interp_type = InterpType.CUBIC
    # comb_case = True
    #
    # ndimpoints = 5
    # epsilon = 10 ** (-7)
    # x1s = np.linspace(0 + epsilon, 1 - epsilon, num=ndimpoints)
    # x2s = np.linspace(0 + epsilon, 1 - epsilon, num=ndimpoints)
    # x1, x2 = np.meshgrid(x1s, x2s)  # Generate grid
    # X = np.vstack([x1.ravel(), x2.ravel()]).T
    # npoints = X.shape[0]
    # func = lambda x: np.sin(4 * np.pi * (x[:, 0] + x[:, 1]))
    #
    # for gl in [6]:
    #
    #     base_covar_module = SobolevKernel()
    #     covar_module = SparseGridInterpolationKernel(
    #         base_kernel=base_covar_module,
    #         grid_level=gl,
    #         ndim=2,
    #         umin=-0.2,
    #         umax=1.2,
    #         interp_type=interp_type,
    #         basis=basis,
    #         comb=True,
    #     )
    #     # X = torch.from_numpy(X)
    #     order = compute_LI(gl, ndim, basis=basis, comb=comb_case, rmat=False)
    #     sg_locs = get_sg_points_nd(gl, ndim, basis, comb=comb_case, umin=0, umax=1, ordered=False)
    #     sg_locs = sg_locs[order, :]
    #
    #     if on_sparse_grid:
    #         X = torch.from_numpy(sg_locs)
    #     else:
    #         X = torch.from_numpy(X)
    #
    #     interp_kernel = covar_module.forward(x1=X, x2=X)
    #     eye_matrix = torch.eye(X.shape[0]).to(dtype=torch.float64)
    #     interp_matrix = interp_kernel.matmul(eye_matrix)
    #     actual_interp = interp_matrix.detach().cpu().numpy()
    #     actual_on_grid = interp_kernel.base_lazy_tensor.evaluate().detach().cpu().numpy()
    #
    #     base_covariance_kernel = base_covar_module(X).evaluate()
    #     expected = base_covariance_kernel.detach().cpu().numpy()

        # print("True kernel ...", expected.shape)
        # plt.imshow(expected)
        # plt.colorbar()
        # plt.show()
        #
        # print("Interpolated kernel ...", actual_interp.shape)
        # plt.imshow(2 * actual_interp)
        # plt.colorbar()
        # plt.show()
        #
        # print("Diff 1...")
        # plt.imshow(np.abs(expected - actual_interp))
        # plt.colorbar()
        # plt.show()
        #
        # if on_sparse_grid:
        #     print("Diff 2...")
        #     plt.imshow(np.abs(expected - actual_on_grid))
        #     plt.colorbar()
        #     plt.show()


