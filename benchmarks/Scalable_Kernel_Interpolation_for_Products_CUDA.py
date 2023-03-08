#!/usr/bin/env python
# coding: utf-8

# # Scalable Kernel Interpolation for Product Kernels (SKIP)
# 
# ## Overview
# 
# In this notebook, we'll overview of how to use SKIP, a method that exploits product structure in some kernels to reduce the dependency of SKI on the data dimensionality from exponential to linear. 
# 
# The most important practical consideration to note in this notebook is the use of `gpytorch.settings.max_root_decomposition_size`, which we explain the use of right before the training loop cell.

# In[1]:


import math
import torch
import gpytorch
from matplotlib import pyplot as plt

# Make plots inline
# get_ipython().run_line_magic('matplotlib', 'inline')


# For this example notebook, we'll be using the `elevators` UCI dataset used in the paper. Running the next cell downloads a copy of the dataset that has already been scaled and normalized appropriately. For this notebook, we'll simply be splitting the data using the first 80% of the data as training and the last 20% as testing.
# 
# **Note**: Running the next cell will attempt to download a ~400 KB dataset file to the current directory.

# In[3]:


import urllib.request
import os
from scipy.io import loadmat
from math import floor


# this is for running the notebook in our testing framework
smoke_test = True #('CI' in os.environ)


if not smoke_test and not os.path.isfile('../elevators.mat'):
    print('Downloading \'elevators\' UCI dataset...')
    urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk', '../elevators.mat')


if smoke_test:  # this is for running the notebook in our testing framework
    X, y = torch.randn(1000, 3), torch.randn(1000)
else:
    data = torch.Tensor(loadmat('../elevators.mat')['data'])
    X = data[:, :-1]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = data[:, -1]


train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()

test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()


# In[4]:


X.size()


# ## Defining the SKIP GP Model
# 
# We now define the GP model. For more details on the use of GP models, see our simpler examples. This model uses a `GridInterpolationKernel` (SKI) with an RBF base kernel. To use SKIP, we make two changes:
# 
# - First, we use only a 1 dimensional `GridInterpolationKernel` (e.g., by passing `num_dims=1`). The idea of SKIP is to use a product of 1 dimensional `GridInterpolationKernel`s instead of a single `d` dimensional one.
# - Next, we create a `ProductStructureKernel` that wraps our 1D `GridInterpolationKernel` with `num_dims=18`. This specifies that we want to use product structure over 18 dimensions, using the 1D `GridInterpolationKernel` in each dimension.
# 
# **Note:** If you've explored the rest of the package, you may be wondering what the differences between `AdditiveKernel`, `AdditiveStructureKernel`, `ProductKernel`, and `ProductStructureKernel` are. The `Structure` kernels (1) assume that we want to apply a single base kernel over a fully decomposed dataset (e.g., every dimension is additive or has product structure), and (2) are significantly more efficient as a result, because they can exploit batch parallel operations instead of using for loops.

# In[5]:


from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, ProductStructureKernel, GridInterpolationKernel
from gpytorch.distributions import MultivariateNormal

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = RBFKernel()
        self.covar_module = ProductStructureKernel(
            ScaleKernel(
                GridInterpolationKernel(self.base_covar_module, grid_size=100, num_dims=1)
            ), num_dims=18
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# In[6]:


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()


# ### Training the model
# 
# The training loop for SKIP has one main new feature we haven't seen before: we specify the `max_root_decomposition_size`. This controls how many iterations of Lanczos we want to use for SKIP, and trades off with time and--more importantly--space. Realistically, the goal should be to set this as high as possible without running out of memory.
# 
# In some sense, this parameter is the main trade-off of SKIP. Whereas many inducing point methods care more about the number of inducing points, because SKIP approximates one dimensional kernels, it is able to do so very well with relatively few inducing points. The main source of approximation really comes from these Lanczos decompositions we perform.

# In[7]:


training_iterations = 2 if smoke_test else 50

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

def train():
    for i in range(training_iterations):
        # Zero backprop gradients
        optimizer.zero_grad()
        with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(30):
            # Get output from model
            output = model(train_x)
            # Calc loss and backprop derivatives
            loss = -mll(output, train_y)
            print(loss.shape)
            loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()
        torch.cuda.empty_cache()
        
# See dkl_mnist.ipynb for explanation of this flag
# with gpytorch.settings.use_toeplitz(True):
#     get_ipython().run_line_magic('time', 'train()')


# ### Making Predictions
# 
# The next cell makes predictions with SKIP. We use the same max_root_decomposition size, and we also demonstrate increasing the max preconditioner size. Increasing the preconditioner size on this dataset is **not** necessary, but can make a big difference in final test performance, and is often preferable to increasing the number of CG iterations if you can afford the space.

# In[8]:

train()

# model.eval()
# likelihood.eval()
# with gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
#     with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_root_decomposition_size(30), gpytorch.settings.fast_pred_var():
#         preds = model(test_x)


# # In[9]:


# print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - test_y))))


# # In[ ]:


