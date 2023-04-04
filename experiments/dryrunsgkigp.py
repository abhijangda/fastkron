import torch
import gpytorch as gp
from timeit import default_timer as timer

from experiments.data import prepare_dataset
from sgkigp.kernels.sginterpkernel import SparseGridInterpolationKernel


# Sparse grid model
class SparseGridGpModel(gp.models.ExactGP):
    def __init__(
        self, 
        train_x, 
        train_y, 
        grid_level, 
        umin,
        umax,
        min_noise=1e-4
    ):            
            
        likelihood = gp.likelihoods.GaussianLikelihood(noise_constraint=gp.constraints.GreaterThan(min_noise))
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gp.means.ConstantMean()
        base_covar_module = gp.kernels.RBFKernel()
        self.covar_module = SparseGridInterpolationKernel(
            base_kernel=base_covar_module,
            grid_level=grid_level,
            ndim=train_x.size(-1),
            umin=umin,
            umax=umax,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)


# Run test on sparse grid model
def test_model(x, y, model, pre_size=100, label='test'):
    model.eval()

    with gp.settings.eval_cg_tolerance(1e-2), \
         gp.settings.max_preconditioner_size(pre_size),  \
         gp.settings.max_cholesky_size(-1), \
         gp.settings.fast_pred_var(), torch.no_grad():
        
        t_start = timer()
        pred_y = model(x)
        pred_ts = timer() - t_start
        rmse = (pred_y.mean - y).pow(2).mean(0).sqrt()
        mae = (pred_y.mean - y).abs().mean(0)
        print("MAE: ", mae)
        print("RMSE: ", rmse)
        print("Pres_ts: ", pred_ts)


def main():

    # loading dataset
    dataset = "servo"
    data_iter = prepare_dataset(dataset, uci_data_dir='/Users/ymohit/sgkigp/data/uci')
    _, train_x, train_y = next(data_iter)
    _, val_x, val_y = next(data_iter)
    _, test_x, test_y = next(data_iter)
    print(f'"{dataset}": D = {train_x.size(-1)}, Train N = {train_x.size(0)}, Val N = {val_x.size(0)} Test N = {test_x.size(0)}')

    # setting up the model and likelihood
    pre_size = 100
    epochs = 100
    p_epochs = 3
    lr = 1e-2
    lanc_iter = 100
    grid_level = 4

    umin = torch.min(train_x, 0).values.numpy() - 0.2
    umax = torch.max(train_x, 0).values.numpy() + 0.2
    gp.settings.max_cholesky_size._set_value(-1) # To exclude the possibility of Cholesky
    model = SparseGridGpModel(
        train_x=train_x,
        train_y=train_y,
        umin=umin,
        umax=umax,
        grid_level=grid_level
    )
    mll = gp.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    # Setting up the optimizer and stopper
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # Training the model
    mll_loss = []
    for iter_ in range(epochs):
        model.train()
        optim.zero_grad()

        with gp.settings.cg_tolerance(1.0), \
             gp.settings.max_preconditioner_size(pre_size), \
             gp.settings.max_root_decomposition_size(lanc_iter):

            output = model(train_x)
            loss = -mll(output, train_y)

            loss.backward()
            optim.step()
            mll_loss += -loss.detach().item(),
            print("Iter: ", iter_, " , loss: ", mll_loss[-1])

    # Perform prediction
    test_model(test_x, test_y, model, pre_size=pre_size)


if __name__ == '__main__':
    main()
