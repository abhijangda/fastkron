import os, sys
import gc
import copy
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np

from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
#from sgkigp.gpyexample.gridkernel import CustomGridInterpolationKernel

from gpytorch.kernels import GridInterpolationKernel

# Source: https://docs.gpytorch.ai/en/latest/examples/02_Scalable_Exact_GPs/KISSGP_Regression.html

n = 1024
dims = 7
train_x = torch.zeros(n, dims)
# for i in range(n):
#     for j in range(n):
#         train_x[i * n + j][0] = float(i) / (n-1)
#         train_x[i * n + j][1] = float(j) / (n-1)
# True function is sin( 2*pi*(x0+x1))
train_y = torch.ones(train_x.shape[0])#torch.sin((train_x[:, 0] + train_x[:, 1]) * (2 * math.pi)) + torch.randn_like(train_x[:, 0]).mul(0.01)
train_x = train_x.cuda()
train_y = train_y.cuda()
grid_size = 8

print(train_x.shape)
print(train_y.shape)
with gpytorch.settings.use_toeplitz(False):
    with gpytorch.settings.num_trace_samples(1023): # gpytorch.settings.fast_computations.log_prob(False):
        class GPRegressionModel(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood):
                super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

                # SKI requires a grid size hyperparameter. This util can help with that

                self.mean_module = gpytorch.means.ConstantMean()
                # self.covar_module = gpytorch.kernels.ScaleKernel(
                #     GridInterpolationKernel(
                #         gpytorch.kernels.RBFKernel(), grid_size=grid_size, num_dims=2
                #     )
                # )

                self.covar_module = GridInterpolationKernel(gpytorch.kernels.ScaleKernel(
                        gpytorch.kernels.CosineKernel()), grid_size=grid_size, num_dims=dims
                    )

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(train_x, train_y, likelihood)
        
        # model = train_to_covergence(model, train_x, train_y)


        def train(training_iterations=10):

            # Find optimal model hyperparameters
            model.cuda()
            likelihood.cuda()

            model.train()
            likelihood.train()

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            # Use the adam optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
            torch.cuda.synchronize()
            import time
            print('Tuning hyper-parameters ...')
            start = time.time()
            for i in tqdm(range(training_iterations)):
                # optimizer.zero_grad()
                output = model(train_x)
                # output@torch.ones(dims**grid_size, 11)
                loss = -mll(output, train_y)
                print("91", loss.shape)
                # loss.backward()
                # optimizer.step()
            torch.cuda.synchronize()
            end = time.time()
            print("Total time ", (end - start)*1e3)

        train()