import torch
import gpytorch.settings as settings

from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy
from gpytorch.utils.memoize import cached, pop_from_cache

from gpytorch.lazy import RootLazyTensor


class SGInterpolatedPredictionStrategy(DefaultPredictionStrategy):

    def __init__(self, train_inputs, train_prior_dist, train_labels, likelihood):

        # this converts the lazy covariance matrix to exact kernel, N \times N. -- really???
        # TODO: understand this
        train_prior_dist = train_prior_dist.__class__(
            train_prior_dist.mean, train_prior_dist.lazy_covariance_matrix.evaluate_kernel()
        )
        super().__init__(train_inputs, train_prior_dist, train_labels, likelihood)

    def exact_prediction(self, joint_mean, joint_covar):

        test_mean = joint_mean[..., self.num_train:]
        test_test_covar = joint_covar[..., self.num_train:, self.num_train:].evaluate_kernel()
        test_train_covar = joint_covar[..., self.num_train:, : self.num_train].evaluate_kernel()

        predictive_mean = self.exact_predictive_mean(test_mean, test_train_covar)
        predictive_covar = self.exact_predictive_covar(test_test_covar, test_train_covar)

        return predictive_mean, predictive_covar

    def exact_predictive_mean(self, test_mean, test_train_covar):
        return test_train_covar.left_interp_coefficient.matmul(self.mean_cache) + test_mean

    def exact_predictive_covar(self, test_test_covar, test_train_covar):
        if settings.fast_pred_var.off() and settings.fast_pred_samples.off():
            return super(
                SGInterpolatedPredictionStrategy,
                self
            ).exact_predictive_covar(test_test_covar, test_train_covar)

        self._last_test_train_covar = test_train_covar

        precomputed_cache = self.covar_cache
        fps = settings.fast_pred_samples.on()
        if (fps and precomputed_cache[0] is None) or (not fps and precomputed_cache[1] is None):
            pop_from_cache(self, "covar_cache")
            precomputed_cache = self.covar_cache

        # Compute the exact predictive posterior
        if settings.fast_pred_samples.on():
            res = self._exact_predictive_covar_inv_quad_form_root(precomputed_cache[0], test_train_covar)
            res = RootLazyTensor(res)
        else: # -- no lanczos
            root = test_train_covar.left_interp_coefficient.matmul(precomputed_cache[1])
            res = test_test_covar + RootLazyTensor(root).mul(-1)
        return res

    def _exact_predictive_covar_inv_quad_form_cache(self, train_train_covar_inv_root, test_train_covar):
        base_lazy_tensor = test_train_covar.base_lazy_tensor
        return base_lazy_tensor.matmul(test_train_covar.right_interp_coefficient.matmul(train_train_covar_inv_root))

    @property
    @cached(name="mean_cache")
    def mean_cache(self):
        train_train_covar = self.train_prior_dist.lazy_covariance_matrix  # K_{X, X}

        # Computing [K_{X, X} + \sigma^2 I]^{-1}
        mvn = self.likelihood(self.train_prior_dist, self.train_inputs)
        train_mean, train_train_covar_with_noise = mvn.mean, mvn.lazy_covariance_matrix

        # Computing [K_{X, X} + \sigma^2 I]^{-1} (Y - \mu{X})
        mean_diff = (self.train_labels - train_mean).unsqueeze(-1)  # Y - \mu{X}
        train_train_covar_inv_labels = train_train_covar_with_noise.inv_matmul(mean_diff)

        # New root factor
        # base_size = train_train_covar.base_lazy_tensor.size(-1)
        # mean_cache = K W^T [K_{X, X} + \sigma^2 I]^{-1} (Y - \mu{X}) -- in terms of SKI matrices
        mean_cache = train_train_covar.base_lazy_tensor.matmul(
            train_train_covar.left_interp_coefficient._t_matmul(train_train_covar_inv_labels)
        ).squeeze()

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

        # Get probe vectors for inverse root
        num_probe_vectors = settings.fast_pred_var.num_probe_vectors()
        num_inducing = train_train_covar.base_lazy_tensor.size(-1)

        device = train_train_covar.device
        vector_indices = torch.randperm(num_inducing).to(dtype=torch.long, device=train_train_covar.device)
        vector_indices = vector_indices[:2*num_probe_vectors]
        vector_indices = torch.stack([vector_indices, torch.linspace(0, len(vector_indices) - 1, len(vector_indices),
                                                                     dtype=torch.long, device=device)])

        vector_values = torch.ones(2*num_probe_vectors, 1, dtype=train_train_covar.dtype,
                                   device=train_train_covar.device).squeeze(-1)

        prob_test_select = torch.sparse_coo_tensor(vector_indices, vector_values, (num_inducing, 2*num_probe_vectors))
        prob_test_select = prob_test_select.to_dense()

        # K_sg{}^T x selection matrix
        prob_test_select_ = train_train_covar.base_lazy_tensor.matmul(prob_test_select)

        # multiply with left x selection matrix
        prob_test_vectors = train_train_covar.left_interp_coefficient._matmul(prob_test_select_)

        probe_vectors = prob_test_vectors[:, :num_probe_vectors]
        test_vectors = prob_test_vectors[:, num_probe_vectors:-1]

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
        if settings.fast_pred_samples.on():  # using lanczos
            inside = train_train_covar.base_lazy_tensor + RootLazyTensor(root).mul(-1)
            inside_root = inside.root_decomposition().root.evaluate()

            # Prevent back-prop through this variable
            if settings.detach_test_caches.on():
                inside_root = inside_root.detach()
            covar_cache = inside_root, None
        else:

            # Prevent back-prop through this variable -- no lanczos
            if settings.detach_test_caches.on():
                root = root.detach()
            covar_cache = None, root

        return covar_cache
