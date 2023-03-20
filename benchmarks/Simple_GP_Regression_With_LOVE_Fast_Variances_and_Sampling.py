#!/usr/bin/env python
# coding: utf-8

# # GP Regression with LOVE for Fast Predictive Variances and Sampling
# 
# ## Overview
# 
# In this notebook, we demonstrate that LOVE (the method for fast variances and sampling introduced in this paper https://arxiv.org/abs/1803.06058) can significantly reduce the cost of computing predictive distributions. This can be especially useful in settings like small-scale Bayesian optimization, where predictions need to be made at enormous numbers of candidate points.
# 
# In this notebook, we will train a KISS-GP model on the `skillcraft `UCI dataset, and then compare the time required to make predictions with each model.
# 
# **NOTE**: The timing results reported in the paper compare the time required to compute (co)variances __only__. Because excluding the mean computations from the timing results requires hacking the internals of GPyTorch, the timing results presented in this notebook include the time required to compute predictive means, which are not accelerated by LOVE. Nevertheless, as we will see, LOVE achieves impressive speed-ups.

# In[1]:


import math
import torch
import gpytorch
from tqdm import tqdm

# from matplotlib import pyplot as plt

# Make plots inline
# get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading Data
# 
# For this example notebook, we'll be using the `elevators` UCI dataset used in the paper. Running the next cell downloads a copy of the dataset that has already been scaled and normalized appropriately. For this notebook, we'll simply be splitting the data using the first 40% of the data as training and the last 60% as testing.
# 
# **Note**: Running the next cell will attempt to download a small dataset file to the current directory.

# In[14]:


import urllib.request
import os
from scipy.io import loadmat
from math import floor

import sys

# this is for running the notebook in our testing framework
smoke_test = True #('CI' in os.environ)


if not smoke_test and not os.path.isfile('../elevators.mat'):
    print('Downloading \'elevators\' UCI dataset...')
    urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk', '../elevators.mat')

n = int(sys.argv[1])
dims = int(sys.argv[2])
grid_size = int(sys.argv[3])
num_trace_samples = int(sys.argv[4])

if smoke_test:  # this is for running the notebook in our testing framework
    X = torch.randn(n, dims)
    y = torch.sin((X[:, 0] + X[:, 1]) * (2 * math.pi)) + torch.randn_like(X[:, 0]).mul(0.01)
else:
    data = torch.Tensor(loadmat('../elevators.mat')['data'])
    X = data[:, :-1]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = data[:, -1]


train_n = int(floor(0.8 * len(X)))
train_x = X #X[:train_n, :].contiguous()
train_y = y # y[:train_n].contiguous()
print(train_x.shape, train_y.shape)
test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()


# LOVE can be used with any type of GP model, including exact GPs, multitask models and scalable approximations. Here we demonstrate LOVE in conjunction with KISS-GP, which has the amazing property of producing **constant time variances.**
# 
# ## The KISS-GP + LOVE GP Model
# 
# We now define the GP model. For more details on the use of GP models, see our simpler examples. This model uses a `GridInterpolationKernel` (SKI) with an Deep RBF base kernel. The forward method passes the input data `x` through the neural network feature extractor defined above, scales the resulting features to be between 0 and 1, and then calls the kernel.
# 
# The Deep RBF kernel (DKL) uses a neural network as an initial feature extractor. In this case, we use a fully connected network with the architecture `d -> 1000 -> 500 -> 50 -> 2`, as described in the original DKL paper. All of the code below uses standard PyTorch implementations of neural network layers.

# In[3]:


class LargeFeatureExtractor(torch.nn.Sequential):           
    def __init__(self, input_dim):                                      
        super(LargeFeatureExtractor, self).__init__()        
        self.add_module('linear1', torch.nn.Linear(input_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())                  
        self.add_module('linear2', torch.nn.Linear(1000, 500))     
        self.add_module('relu2', torch.nn.ReLU())                  
        self.add_module('linear3', torch.nn.Linear(500, 50))       
        self.add_module('relu3', torch.nn.ReLU())                  
        self.add_module('linear4', torch.nn.Linear(50, dims))         


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
            grid_size=grid_size, num_dims=dims,
        )
        
        # Also add the deep net
        self.feature_extractor = LargeFeatureExtractor(input_dim=train_x.size(-1))

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        # We're also scaling the features so that they're nice values
        projected_x = self.feature_extractor(x)
        projected_x = projected_x - projected_x.min(0)[0]
        projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1
        
        # The rest of this looks like what we've seen
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()


# ### Training the model
# 
# The cell below trains the GP model, finding optimal hyperparameters using Type-II MLE. We run 20 iterations of training using the `Adam` optimizer built in to PyTorch. With a decent GPU, this should only take a few seconds.

# In[5]:


training_iterations = 1 if smoke_test else 20


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

import time


def train():
    iterator = tqdm(range(training_iterations))
    for i in iterator:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()
        
start = time.time()
with gpytorch.settings.use_toeplitz(False):
    with gpytorch.settings.num_trace_samples(num_trace_samples): # gpytorch.settings.fast_computations.log_prob(False):
        # with gpytorch.settings.max_preconditioner_size(0):
        #     with gpytorch.settings.min_preconditioning_size(0):
                with gpytorch.settings.debug(False):
                    train()
end = time.time()

print("Total time ", (end - start)*1e3)
import sys
sys.exit(0)
# ## Computing predictive variances (KISS-GP or Exact GPs)
# 
# ### Using standard computaitons (without LOVE)
# 
# The next cell gets the predictive covariance for the test set (and also technically gets the predictive mean, stored in `preds.mean`) using the standard SKI testing code, with no acceleration or precomputation. 
# 
# **Note:** Full predictive covariance matrices (and the computations needed to get them) can be quite memory intensive. Depending on the memory available on your GPU, you may need to reduce the size of the test set for the code below to run. If you run out of memory, try replacing `test_x` below with something like `test_x[:1000]` to use the first 1000 test points only, and then restart the notebook.

# In[6]:


import time

# Set into eval mode
model.eval()
likelihood.eval()

with torch.no_grad():
    start_time = time.time()
    preds = likelihood(model(test_x))
    exact_covar = preds.covariance_matrix
    exact_covar_time = time.time() - start_time
    
print(f"Time to compute exact mean + covariances: {exact_covar_time:.2f}s")


# ### Using LOVE
# 
# Next we compute predictive covariances (and the predictive means) for LOVE, but starting from scratch. That is, we don't yet have access to the precomputed cache discussed in the paper. This should still be faster than the full covariance computation code above.
# 
# To use LOVE, use the context manager `with gpytorch.settings.fast_pred_var():`
# 
# You can also set some of the LOVE settings with context managers as well. For example, `gpytorch.settings.max_root_decomposition_size(100)` affects the accuracy of the LOVE solves (larger is more accurate, but slower).
# 
# In this simple example, we allow a rank 100 root decomposition, although increasing this to rank 20-40 should not affect the timing results substantially.

# In[7]:


# Clear the cache from the previous computations
model.train()
likelihood.train()

# Set into eval mode
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(100):
    start_time = time.time()
    preds = model(test_x)
    fast_time_no_cache = time.time() - start_time


# The above cell additionally computed the caches required to get fast predictions. From this point onwards, unless we put the model back in training mode, predictions should be extremely fast. The cell below re-runs the above code, but takes full advantage of both the mean cache and the LOVE cache for variances.

# In[8]:


with torch.no_grad(), gpytorch.settings.fast_pred_var():
    start_time = time.time()
    preds = likelihood(model(test_x))
    fast_covar = preds.covariance_matrix
    fast_time_with_cache = time.time() - start_time


# In[9]:


print('Time to compute mean + covariances (no cache) {:.2f}s'.format(fast_time_no_cache))
print('Time to compute mean + variances (cache): {:.2f}s'.format(fast_time_with_cache))


# ### Compute Error between Exact and Fast Variances
# 
# Finally, we compute the mean absolute error between the fast variances computed by LOVE (stored in fast_covar), and the exact variances computed previously. 
# 
# Note that these tests were run with a root decomposition of rank 10, which is about the minimum you would realistically ever run with. Despite this, the fast variance estimates are quite good. If more accuracy was needed, increasing `max_root_decomposition_size` would provide even better estimates.

# In[10]:


mae = ((exact_covar - fast_covar).abs() / exact_covar.abs()).mean()
print(f"MAE between exact covar matrix and fast covar matrix: {mae:.6f}")


# ## Computing posterior samples (KISS-GP only)
# 
# With KISS-GP models, LOVE can also be used to draw fast posterior samples. (The same does not apply to exact GP models.)
# 
# ### Drawing samples the standard way (without LOVE)
# 
# We now draw samples from the posterior distribution. Without LOVE, we accomlish this by performing Cholesky on the posterior covariance matrix. This can be slow for large covariance matrices.

# In[11]:


import time
num_samples = 20 if smoke_test else 20000


# Set into eval mode
model.eval()
likelihood.eval()

with torch.no_grad():
    start_time = time.time()
    exact_samples = model(test_x).rsample(torch.Size([num_samples]))
    exact_sample_time = time.time() - start_time
    
print(f"Time to compute exact samples: {exact_sample_time:.2f}s")


# ### Using LOVE
# 
# Next we compute posterior samples (and the predictive means) using LOVE.
# This requires the additional context manager `with gpytorch.settings.fast_pred_samples():`.
# 
# Note that we also need the `with gpytorch.settings.fast_pred_var():` flag turned on. Both context managers respond to the `gpytorch.settings.max_root_decomposition_size(100)` setting.

# In[12]:


# Clear the cache from the previous computations
model.train()
likelihood.train()

# Set into eval mode
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(200):
    # NEW FLAG FOR SAMPLING
    with gpytorch.settings.fast_pred_samples():
        start_time = time.time()
        _ = model(test_x).rsample(torch.Size([num_samples]))
        fast_sample_time_no_cache = time.time() - start_time
    
# Repeat the timing now that the cache is computed
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    with gpytorch.settings.fast_pred_samples():
        start_time = time.time()
        love_samples = model(test_x).rsample(torch.Size([num_samples]))
        fast_sample_time_cache = time.time() - start_time
    
print('Time to compute LOVE samples (no cache) {:.2f}s'.format(fast_sample_time_no_cache))
print('Time to compute LOVE samples (cache) {:.2f}s'.format(fast_sample_time_cache))


# ### Compute the empirical covariance matrices
# 
# Let's see how well LOVE samples and exact samples recover the true covariance matrix.

# In[13]:


# Compute exact posterior covar
with torch.no_grad():
    start_time = time.time()
    posterior = model(test_x)
    mean, covar = posterior.mean, posterior.covariance_matrix

exact_empirical_covar = ((exact_samples - mean).t() @ (exact_samples - mean)) / num_samples
love_empirical_covar = ((love_samples - mean).t() @ (love_samples - mean)) / num_samples

exact_empirical_error = ((exact_empirical_covar - covar).abs()).mean()
love_empirical_error = ((love_empirical_covar - covar).abs()).mean()

print(f"Empirical covariance MAE (Exact samples): {exact_empirical_error}")
print(f"Empirical covariance MAE (LOVE samples): {love_empirical_error}")


# In[ ]:




