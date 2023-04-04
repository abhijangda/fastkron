import torch
import gpytorch.settings as settings

from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy
from gpytorch.utils.memoize import cached, pop_from_cache

from gpytorch.lazy import RootLazyTensor


class ModifiedInterpolatedPredictionStrategy(DefaultPredictionStrategy):

    def __init__(self, train_inputs, train_prior_dist, train_labels, likelihood):

        train_prior_dist = train_prior_dist.__class__(
            train_prior_dist.mean, train_prior_dist.lazy_covariance_matrix.evaluate_kernel()
        )
        super().__init__(train_inputs, train_prior_dist, train_labels, likelihood)

    def _exact_predictive_covar_inv_quad_form_cache(self, train_train_covar_inv_root, test_train_covar):
        # train_interp_indices = test_train_covar.right_interp_indices
        # train_interp_values = test_train_covar.right_interp_values

        base_lazy_tensor = test_train_covar.base_lazy_tensor
        #base_size = base_lazy_tensor.size(-1)
        res = base_lazy_tensor.matmul(
            test_train_covar.right_interp_tensor.matmul(train_train_covar_inv_root)
        )
        return res

    def _exact_predictive_covar_inv_quad_form_root(self, precomputed_cache, test_train_covar):
        # Here the precomputed cache represents K_UU W S,
        # where S S^T = (K_XX + sigma^2 I)^-1
        # test_interp_indices = test_train_covar.left_interp_indices
        # test_interp_values = test_train_covar.left_interp_values
        res = test_train_covar.left_interp_tensor.matmul(precomputed_cache)
        return res

    @property
    @cached(name="mean_cache")
    def mean_cache(self):
        train_train_covar = self.train_prior_dist.lazy_covariance_matrix

        mvn = self.likelihood(self.train_prior_dist, self.train_inputs)
        train_mean, train_train_covar_with_noise = mvn.mean, mvn.lazy_covariance_matrix
        mean_diff = (self.train_labels - train_mean).unsqueeze(-1)
        train_train_covar_inv_labels = train_train_covar_with_noise.inv_matmul(mean_diff)

        # train_interp_indices = train_train_covar.left_interp_indices
        # train_interp_values = train_train_covar.left_interp_values

        mean_cache = train_train_covar.base_lazy_tensor.matmul(
            train_train_covar.right_interp_tensor.matmul(train_train_covar_inv_labels))

        # Prevent backprop through this variable
        if settings.detach_test_caches.on():
            return mean_cache.detach()
        else:
            return mean_cache

    @property
    @cached(name="covar_cache")
    def covar_cache(self):
        # Get inverse root
        train_train_covar = self.train_prior_dist.lazy_covariance_matrix
        # train_interp_indices = train_train_covar.left_interp_tensor._indices()
        # train_interp_values = train_train_covar.left_interp_values

        # Get probe vectors for inverse root
        dtype = train_train_covar.dtype
        device = train_train_covar.device
        num_probe_vectors = settings.fast_pred_var.num_probe_vectors()
        num_inducing = train_train_covar.base_lazy_tensor.size(-1)
        vector_indices = torch.randperm(num_inducing)

        probe_test_vector_indices = vector_indices[:2*num_probe_vectors]
        rhs_vectors = torch.zeros(num_inducing, 2*num_probe_vectors, dtype=dtype, device=device)
        rhs_vectors[probe_test_vector_indices, range(2*num_probe_vectors)] = 1.0

        probe_test_vectors = train_train_covar.left_interp_tensor.matmul(
            train_train_covar.base_lazy_tensor.matmul(rhs_vectors)
        )

        if probe_test_vectors.sum().detach().cpu().numpy() < 1e-6:
            print("In covar_cache: ", "probe_test_vectors got very small values ..")
            probe_test_vectors = torch.randn(probe_test_vectors.shape, dtype=dtype, device=device)

        probe_vectors = probe_test_vectors[:, :num_probe_vectors]
        test_vectors = probe_test_vectors[:, num_probe_vectors:]

        # Put data through the likelihood
        dist = self.train_prior_dist.__class__(
            torch.zeros_like(self.train_prior_dist.mean), self.train_prior_dist.lazy_covariance_matrix
        )
        train_train_covar_plus_noise = self.likelihood(dist, self.train_inputs).lazy_covariance_matrix

        # Get inverse root
        train_train_covar_inv_root = train_train_covar_plus_noise.root_inv_decomposition(
            initial_vectors=probe_vectors, test_vectors=test_vectors
        ).root
        train_train_covar_inv_root = train_train_covar_inv_root.evaluate()

        # New root factor
        root = self._exact_predictive_covar_inv_quad_form_cache(train_train_covar_inv_root, self._last_test_train_covar)

        # Precomputed factor
        if settings.fast_pred_samples.on():
            inside = train_train_covar.base_lazy_tensor + RootLazyTensor(root).mul(-1)
            inside_root = inside.root_decomposition().root.evaluate()
            # Prevent backprop through this variable
            if settings.detach_test_caches.on():
                inside_root = inside_root.detach()
            covar_cache = inside_root, None
        else:
            # Prevent backprop through this variable
            if settings.detach_test_caches.on():
                root = root.detach()
            covar_cache = None, root

        return covar_cache

    def exact_prediction(self, joint_mean, joint_covar):
        # Find the components of the distribution that contain test data
        test_mean = joint_mean[..., self.num_train:]
        test_test_covar = joint_covar[..., self.num_train:, self.num_train:].evaluate_kernel()
        test_train_covar = joint_covar[..., self.num_train:, : self.num_train].evaluate_kernel()

        return (
            self.exact_predictive_mean(test_mean, test_train_covar),
            self.exact_predictive_covar(test_test_covar, test_train_covar),
        )

    def exact_predictive_mean(self, test_mean, test_train_covar):
        precomputed_cache = self.mean_cache
        res = test_train_covar.left_interp_tensor.matmul(precomputed_cache).squeeze(-1) + test_mean
        return res

    def exact_predictive_covar(self, test_test_covar, test_train_covar):

        if settings.fast_pred_var.off() and settings.fast_pred_samples.off():
            return super(
                ModifiedInterpolatedPredictionStrategy,
                self
            ).exact_predictive_covar(test_test_covar, test_train_covar)

        self._last_test_train_covar = test_train_covar
        # test_interp_indices = test_train_covar.left_interp_indices
        # test_interp_values = test_train_covar.left_interp_values

        precomputed_cache = self.covar_cache
        fps = settings.fast_pred_samples.on()
        if (fps and precomputed_cache[0] is None) or (not fps and precomputed_cache[1] is None):
            pop_from_cache(self, "covar_cache")
            precomputed_cache = self.covar_cache

        # Compute the exact predictive posterior
        if settings.fast_pred_samples.on():
            res = self._exact_predictive_covar_inv_quad_form_root(precomputed_cache[0], test_train_covar)
            res = RootLazyTensor(res)
        else:
            root = test_train_covar.left_interp_tensor.matmul(precomputed_cache[1])
            res = test_test_covar + RootLazyTensor(root).mul(-1)
        return res
