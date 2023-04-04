import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np

#from experiments.setups import setup_ski
from experiments.models import SKIModel_base as SKIModel
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
        # print(i, loss.item())
        # print(i, list(model.named_hyperparameters()))
    return


def eval(model):
    # Set model and likelihood into evaluation mode
    model.eval()
    model.likelihood.eval()

    # Generate nxn grid of test points spaced on a grid of size 1/(n-1) in [0,1]x[0,1]
    n = 10
    test_x = torch.zeros(int(pow(n, 2)), 2)
    for i in range(n):
        for j in range(n):
            test_x[i * n + j][0] = float(i) / (n-1)
            test_x[i * n + j][1] = float(j) / (n-1)

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_cholesky_size(1):
        observed_pred = model.likelihood(model(test_x))
        pred_labels = observed_pred.mean.view(n, n)

    # Calc absolute error
    test_y_actual = torch.sin(((test_x[:, 0] + test_x[:, 1]) * (2 * math.pi))).view(n, n)
    delta_y = torch.abs(pred_labels - test_y_actual).detach().numpy()
    errors = delta_y.flatten()
    print("Error:", np.mean(errors), ' +/- ', np.std(errors))
    return pred_labels, test_y_actual, delta_y


# Define a plotting function
def ax_plot(f, ax, y_labels, title):
    im = ax.imshow(y_labels)
    ax.set_title(title)
    f.colorbar(im)
    plt.show()


if __name__ == '__main__':

    # Source: https://docs.gpytorch.ai/en/latest/examples/02_Scalable_Exact_GPs/KISSGP_Regression.html

    n = 40
    train_x = torch.zeros(pow(n, 2), 2)
    for i in range(n):
        for j in range(n):
            train_x[i * n + j][0] = float(i) / (n - 1)
            train_x[i * n + j][1] = float(j) / (n - 1)

    # True function is sin( 2*pi*(x0+x1))
    train_y = torch.sin((train_x[:, 0] + train_x[:, 1]) * (2 * math.pi)) + torch.randn_like(train_x[:, 0]).mul(0.01)

    model = SKIModel(train_x, train_y, use_modified=False, grid_size=27, num_dims=2)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    train(model=model, mll=mll, training_iterations=30)

    pred_labels, test_y_actual, delta_y = eval(model)

    # Plot our predictive means
    # f, observed_ax = plt.subplots(1, 1, figsize=(4, 3))
    # ax_plot(f, observed_ax, pred_labels, 'Predicted Values (Likelihood)')
    #
    # # Plot the true values
    # f, observed_ax2 = plt.subplots(1, 1, figsize=(4, 3))
    # ax_plot(f, observed_ax2, test_y_actual, 'Actual Values (Likelihood)')

    # Plot the absolute errors
    f, observed_ax3 = plt.subplots(1, 1, figsize=(4, 3))
    ax_plot(f, observed_ax3, delta_y, 'Absolute Error Surface')

    print("Done!")
