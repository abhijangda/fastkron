import os
import math
import torch
import gpytorch
import gpytorch as gp
from matplotlib import pyplot as plt
import numpy as np

from tqdm import tqdm


def train(model, mll, training_iterations=30):

    model.train()
    model.likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    print('Tuning hyper-parameters ...')
    for i in tqdm(range(training_iterations)):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        print(i, loss.item())
        # print(i, list(model.named_hyperparameters()))
    return


def eval(model, test_x, test_y):
    # Set model and likelihood into evaluation mode
    model.eval()
    model.likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_cholesky_size(1):
        observed_pred = model.likelihood(model(test_x))
        pred_labels = observed_pred.mean

    # Calc absolute error
    errors = torch.abs(pred_labels - test_y).detach().numpy()
    print("MSE:", np.mean(errors), ', Zero pred error: ', np.mean( torch.abs(test_y).detach().numpy()))
    return


# Define a plotting function
def ax_plot(f, ax, y_labels, title):
    im = ax.imshow(y_labels)
    ax.set_title(title)
    f.colorbar(im)
    plt.show()


class SKIModel(gp.models.ExactGP):
    def __init__(self, train_x, train_y, grid_size=100, grid_bounds=None, min_noise=1e-4):
        likelihood = gp.likelihoods.GaussianLikelihood(
                      noise_constraint=gp.constraints.GreaterThan(min_noise))
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        self.base_covar_module = gp.kernels.RBFKernel(ard_num_dims=train_x.size(-1))

        self.covar_module = gp.kernels.ScaleKernel(
            gp.kernels.GridInterpolationKernel(
                self.base_covar_module, grid_size=grid_size, grid_bounds=grid_bounds,
                num_dims=train_x.size(-1)
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)


if __name__ == '__main__':

    # Source: https://docs.gpytorch.ai/en/latest/examples/02_Scalable_Exact_GPs/KISSGP_Regression.html

    func_val = 0
    ndim = 2
    datapath = os.path.expanduser("~/sgkigp/data/")
    datapath = datapath + 'f' + str(func_val) + "_ndim" + str(ndim) + ".pt"
    train_x, train_y, val_x, val_y, test_x, test_y = torch.load(datapath)


    model = SKIModel(train_x, train_y, grid_size=143)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    train(model=model, mll=mll, training_iterations=30)

    eval(model, test_x, test_y)

    print("Done!")
