#!/usr/bin/env python
# coding: utf-8

# # Exact GP Regression with Multiple GPUs and Kernel Partitioning
# ## Introduction
# In this notebook, we'll demonstrate training exact GPs on large datasets using two key features from the paper https://arxiv.org/abs/1903.08114: 
# 
# 1. The ability to distribute the kernel matrix across multiple GPUs, for additional parallelism.
# 2. Partitioning the kernel into chunks computed on-the-fly when performing each MVM to reduce memory usage.
# 
# We'll be using the `protein` dataset, which has about 37000 training examples. The techniques in this notebook can be applied to much larger datasets, but the training time required will depend on the computational resources you have available: both the number of GPUs available and the amount of memory they have (which determines the partition size) have a significant effect on training time.

# In[1]:


import math
import torch
import gpytorch
import sys
from matplotlib import pyplot as plt
sys.path.append('../')
# from LBFGS import FullBatchLBFGS

# We will be using the Protein UCI dataset which contains a total of 40000+ data points. The next cell will download this dataset from a Google drive and load it.

# In[2]:


import os
import urllib.request
from scipy.io import loadmat
dataset = 'protein'
if not os.path.isfile(f'../{dataset}.mat'):
    print(f'Downloading \'{dataset}\' UCI dataset...')
    urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1nRb8e7qooozXkNghC5eQS0JeywSXGX2S',
                               f'../{dataset}.mat')
    
# data = torch.Tensor(loadmat(f'../{dataset}.mat')['data'])


# ### Normalization and train/test Splits
# 
# In the next cell, we split the data 80/20 as train and test, and do some basic z-score feature normalization.

# In[3]:


import numpy as np


output_device = torch.device('cuda:0')
train_x = torch.ones(1000, 4).cuda()
train_y = torch.ones(1000).cuda()

# ## How many GPUs do you want to use?
# 
# In the next cell, specify the `n_devices` variable to be the number of GPUs you'd like to use. By default, we will use all devices available to us.

# In[4]:


n_devices = torch.cuda.device_count()
print('Planning to run on {} GPUs.'.format(n_devices))


# In the next cell we define our GP model and training code. For this notebook, the only thing different from the Simple GP tutorials is the use of the `MultiDeviceKernel` to wrap the base covariance module. This allows for the use of multiple GPUs behind the scenes.

# In[5]:


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_devices):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        self.covar_module = gpytorch.kernels.MultiDeviceKernel(
            base_covar_module, device_ids=range(n_devices),
            output_device=output_device
        )
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train(train_x,
          train_y,
          n_devices,
          output_device,
          checkpoint_size,
          preconditioner_size,
          n_training_iter,
):
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(output_device)
    model = ExactGPModel(train_x, train_y, likelihood, n_devices).to(output_device)
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    
    with gpytorch.beta_features.checkpoint_kernel(checkpoint_size), \
         gpytorch.settings.max_preconditioner_size(preconditioner_size):

        def closure():
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            return loss

        loss = closure()
        loss.backward()

        for i in range(n_training_iter):
            options = {'closure': closure, 'current_loss': loss, 'max_ls': 10}
            loss, _, _, _, _, _, _, fail = optimizer.step(options)
            
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, n_training_iter, loss.item(),
                model.covar_module.module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
            
            if fail:
                print('Convergence reached!')
                break
    
    print(f"Finished training on {train_x.size(0)} data points using {n_devices} GPUs.")
    return model, likelihood


# ## Automatically determining GPU Settings
# 
# In the next cell, we automatically determine a roughly reasonable partition or *checkpoint* size that will allow us to train without using more memory than the GPUs available have. Not that this is a coarse estimate of the largest possible checkpoint size, and may be off by as much as a factor of 2. A smarter search here could make up to a 2x performance improvement.

# In[6]:


import gc

def find_best_gpu_setting(train_x,
                          train_y,
                          n_devices,
                          output_device,
                          preconditioner_size
):
    N = train_x.size(0)
    
    # Find the optimum partition/checkpoint size by decreasing in powers of 2
    # Start with no partitioning (size = 0)
    settings = [0] + [int(n) for n in np.ceil(N / 2**np.arange(1, np.floor(np.log2(N))))]

    for checkpoint_size in settings:
        print('Number of devices: {} -- Kernel partition size: {}'.format(n_devices, checkpoint_size))
        try:
            # Try a full forward and backward pass with this setting to check memory usage
            _, _ = train(train_x, train_y,
                         n_devices=n_devices, output_device=output_device,
                         checkpoint_size=checkpoint_size,
                         preconditioner_size=preconditioner_size, n_training_iter=1)
            
            # when successful, break out of for-loop and jump to finally block
            break
        except RuntimeError as e:
            print('RuntimeError: {}'.format(e))
        except AttributeError as e:
            print('AttributeError: {}'.format(e))
        finally:
            # handle CUDA OOM error
            gc.collect()
            torch.cuda.empty_cache()
    return checkpoint_size

# Set a large enough preconditioner size to reduce the number of CG iterations run
preconditioner_size = 100
# checkpoint_size = find_best_gpu_setting(train_x, train_y,
#                                         n_devices=n_devices, 
#                                         output_device=output_device,
#                                         preconditioner_size=preconditioner_size)


# ### Training

# In[7]:


model, likelihood = train(train_x, train_y,
                          n_devices=n_devices, output_device=output_device,
                          checkpoint_size=10000,
                          preconditioner_size=100,
                          n_training_iter=20)


# ## Computing test time caches

# In[9]:


# Get into evaluation (predictive posterior) mode
# model.eval()
# likelihood.eval()

# with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.beta_features.checkpoint_kernel(1000):
#     # Make predictions on a small number of test points to get the test time caches computed
#     latent_pred = model(test_x[:2, :])
#     del latent_pred  # We don't care about these predictions, we really just want the caches.


# ### Testing: Computing predictions

# In[11]:


# with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.beta_features.checkpoint_kernel(1000):
#     get_ipython().run_line_magic('time', 'latent_pred = model(test_x)')
    
# test_rmse = torch.sqrt(torch.mean(torch.pow(latent_pred.mean - test_y, 2)))
# print(f"Test RMSE: {test_rmse.item()}")


# In[10]:





# In[ ]:



