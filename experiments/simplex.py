# Credits: part of the code in this file is adapted from: https://github.com/activatedgeek/simplex-gp/
import os
import pickle
import fire
import wandb
import torch
from tqdm.auto import tqdm
import numpy as np
import gpytorch as gp
from pathlib import Path

from timeit import default_timer as timer
from experiments.misc import EarlyStopper
from experiments.data import prepare_dataset

import shutil

import sgkigp.config as config

from gpytorch_lattice_kernel import RBFLattice, MaternLattice


class SimplexGPModel(gp.models.ExactGP):

    def __init__(self, train_x, train_y, kernel_type=config.KernelsType.RBFKERNEL, order=1, min_noise=1e-4):
        likelihood = gp.likelihoods.GaussianLikelihood(
                      noise_constraint=gp.constraints.GreaterThan(min_noise))
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gp.means.ConstantMean()

        if kernel_type == config.KernelsType.RBFKERNEL:
            base_covar_module = RBFLattice(ard_num_dims=train_x.size(-1), order=order)

        elif kernel_type == config.KernelsType.MATTERNHALF:
            base_covar_module = MaternLattice(nu=0.5, ard_num_dims=train_x.size(-1), order=order)

        elif kernel_type == config.KernelsType.MATTERNONEANDHALF:
            base_covar_module = MaternLattice(nu=1.5, ard_num_dims=train_x.size(-1), order=order)

        elif kernel_type == config.KernelsType.MATTERNTWOANDHALF:
            base_covar_module = MaternLattice(nu=2.5, ard_num_dims=train_x.size(-1), order=order)

        else:
            raise NotImplementedError

        self.base_covar_module = base_covar_module
        self.covar_module = gp.kernels.ScaleKernel(self.base_covar_module)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)


def main(method: str = 'SIMPLEX', dataset: str = None, data_dir: str = None, log_int: int = 1, seed: int = 1337,
         device: int = 0, epochs: int = 100, lr: int = 1e-1, p_epochs: int = 200, lanc_iter: int = 100,
         pre_size: int = 100, n_inducing: int = 512, kernel_type: int = 0, min_noise: float = 1e-4,
         grid_level: int = 3, boundary_slack: float = 0.2, cg_tol: float = 1.0, cg_eval_tol: float = 1e-2,
         func_val: int = 0, ndim: int = 4, order=1,
         log_dir: str = os.path.expanduser('~/sgkigp/logs/tmp')):

    # Initiate wandb config for the experimentation
    wandb_run = wandb.init(entity="swjindal", project="sgkigp", dir=log_dir,
               config={
                'method': method,
                'dataset': dataset,
                'lr': lr,
                'lanc_iter': lanc_iter,
                'pre_size': pre_size,
                'n_inducing': n_inducing,
                'kernel_type': kernel_type
    })

    # Set-seeds and device
    config.set_seeds(seed=seed)
    device = config.get_device(device=device)
    dtype = config.dtype(default='float32', use_torch=True)
    kernel_type = config.KernelsType(kernel_type)

    # Preparing data iterations
    if dataset is None and func_val >= 0:  # synthetic dataset
        datapath = os.path.expanduser("~/sgkigp/data/")
        datapath = datapath + 'f' + str(func_val) + "_ndim" + str(ndim) + ".pt"
        train_x, train_y, val_x, val_y, test_x, test_y = torch.load(datapath)

    elif dataset == 'airline':
        airline_path = os.path.expanduser("~/sgkigp/data/airline/ss_5929413_idx_0.pkl")
        X, Y, X_test, Y_test, X_val, Y_val = pickle.load(open(airline_path, 'rb'))

        n_test = X_test.shape[0]
        train_x = torch.from_numpy(X)
        train_y = torch.from_numpy(Y).squeeze()
        val_x = torch.from_numpy(X_test[:n_test//2 +1])
        val_y = torch.from_numpy(Y_test[:n_test//2 +1]).squeeze()
        test_x = torch.from_numpy(X_test[n_test//2 +1:-1])
        test_y = torch.from_numpy(Y_test[n_test//2 +1:-1]).squeeze()

    else: # real dataset
        data_iter = prepare_dataset(dataset, uci_data_dir=data_dir, device=device)
        _, train_x, train_y = next(data_iter)
        _, val_x, val_y = next(data_iter)
        _, test_x, test_y = next(data_iter)
        print("Train shape:", train_x.shape)

    # Move to device and dtype
    train_x = train_x.to(device=device, dtype=dtype)
    train_y = train_y.to(device=device, dtype=dtype)
    val_x = val_x.to(device=device, dtype=dtype)
    val_y = val_y.to(device=device, dtype=dtype)
    test_x = test_x.to(device=device, dtype=dtype)
    test_y = test_y.to(device=device, dtype=dtype)

    # Reporting basic data statistics
    print(
        f'"{dataset}": D = {train_x.size(-1)}, Train N = {train_x.size(0)},'
        f' Val N = {val_x.size(0)} Test N = {test_x.size(0)}')

    wandb.config.update({
        'D': train_x.size(-1),
        'N_train': train_x.size(0),
        'N_test': test_x.size(0),
        'N_val': val_x.size(0)
    })

    # Setting up model and likelihood
    if method == 'SIMPLEX':

        model = SimplexGPModel(train_x, train_y,
                               kernel_type=kernel_type, order=order, min_noise=min_noise
                               ).to(device=device, dtype=dtype)

    else:
        raise NotImplementedError

    mll = gp.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    # Setting optimizer and early stopper
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopper(patience=p_epochs)   # p_epochs is patience of epochs

    # Running over epochs
    for i in tqdm(range(epochs)):

        # Running and reporting progress of an epoch
        train_dict = train(train_x, train_y, model, mll, optimizer,
                           lanc_iter=lanc_iter, pre_size=pre_size, cg_tol=cg_tol)

        wandb.log(train_dict, step=i + 1)

        # Doing validation
        if (i % log_int) == 0:

            # Running validation and testing
            val_dict = test(val_x, val_y, model, mll, pre_size=pre_size, lanc_iter=lanc_iter, label='val',
                            cg_tol=cg_eval_tol)

            test_dict = test(test_x, test_y, model, mll, pre_size=pre_size, lanc_iter=lanc_iter,
                             cg_tol=cg_eval_tol)

            # Recording results
            stopper(-val_dict['val/rmse'], dict(
                state_dict=model.state_dict(),
                summary={
                    'test/best_rmse': test_dict['test/rmse'],
                    'test/best_nll': test_dict['test/nll'],
                    'val/best_step': i + 1
                }
            ))
            wandb.log(val_dict, step=i + 1)
            wandb.log(test_dict, step=i + 1)
            for k, v in stopper.info().get('summary').items():
                wandb.run.summary[k] = v
            torch.save(stopper.info().get('state_dict'), Path(wandb.run.dir) / 'model.pt')
            wandb.save('*.pt')
            wandb_run.save()

            if stopper.is_done():
                break

    # Recording final summary
    for k, v in stopper.info().get('summary').items():
        wandb.run.summary[k] = v

    torch.save(stopper.info().get('state_dict'), Path(wandb.run.dir) / 'model.pt')
    wandb.save('*.pt')
    wandb_run.save()


def train(x, y, model, mll, optim, lanc_iter=100, pre_size=100, cg_tol=1.0):

    model.train()
    optim.zero_grad()

    with gp.settings.cg_tolerance(cg_tol), \
         gp.settings.max_preconditioner_size(pre_size), \
         gp.settings.max_root_decomposition_size(lanc_iter):
        t_start = timer()

        output = model(x)
        loss = -mll(output, y)

        loss_ts = timer() - t_start

        t_start = timer()

        loss.backward()
        optim.step()

        bw_ts = timer() - t_start

    return {
        'train/mll': -loss.detach().item(),
        'train/loss_ts': loss_ts,
        'train/bw_ts': bw_ts,
        'train/total_ts': loss_ts + bw_ts
    }


def test(x, y, model, mll, lanc_iter=100, pre_size=100, label='test', cg_tol=1e-2):

    model.eval()
    with gp.settings.eval_cg_tolerance(cg_tol), \
         gp.settings.max_preconditioner_size(pre_size), \
         gp.settings.max_root_decomposition_size(lanc_iter), \
         gp.settings.fast_pred_var(), torch.no_grad():
        t_start = timer()

        pred_y = model(x)
        pred_ts = timer() - t_start

        rmse = (pred_y.mean - y).pow(2).mean(0).sqrt()
        mae = (pred_y.mean - y).abs().mean(0)

        nll = -mll(pred_y, y)

    return {
        f'{label}/rmse': rmse.item(),
        f'{label}/mae': mae.item(),
        f'{label}/pred_ts': pred_ts,
        f'{label}/nll': nll.item()
    }


if __name__ == "__main__":
    #os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')
    fire.Fire(main)

