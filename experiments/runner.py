# Credits: part of the code in this file is adapted from: https://github.com/activatedgeek/simplex-gp/
import os
import fire
import wandb
import pickle
import torch
from tqdm.auto import tqdm
import numpy as np
import gpytorch as gp
from pathlib import Path

from timeit import default_timer as timer
from experiments.misc import EarlyStopper
from experiments.data import prepare_dataset

from experiments.setups import setup_models

import sgkigp.config as config
from sgkigp.config import SgBasisType, InterpType, MethodName

DEBUG_MODE = True


def main(
        # experiments
        dataset: str = None,
        data_dir: str = None,
        log_int: int = 1,
        seed: int = 1337,
        device: int = 0,
        func_val: int = -1,
        ndim: int = 2,
        dtype: str = 'float32',
        skip_wandb: int = 0,
        verbose: int = 0,
        log_dir: str = os.path.expanduser('~/sgkigp/logs/tmp'),

        # optimization
        epochs: int = 10,
        lr: float = 1e-1,
        p_epochs: int = 200,
        lanc_iter: int = 100,
        pre_size: int = 0,  # changed during the rebuttal
        cg_tol: float = 1.0,
        cg_eval_tol: float = 1e-2,

        # model related
        method: int = 0,
        n_inducing: int = 512,
        kernel_type: int = 0,
        min_noise: float = 1e-4,
        grid_level: int = 4,
        boundary_slack: float = 0.1,
        interp_type: int = 1,
        basis_type: int = 1,
        use_modified_ski: int = 1,
        bypass_covar: int = 0,
        grid_size_dim: int = 2,
        sg_shifted: int = 0
):

    # Handling config variables
    kernel_type = config.KernelsType(kernel_type)
    interp_type = InterpType(interp_type)
    basis_type = SgBasisType(basis_type)
    bypass_covar = True if bypass_covar > 0 else False
    skip_wandb = True if skip_wandb > 0 else False
    verbose = True if verbose > 0 else False
    method = MethodName(method) # SKI = 0, SPARSE = 1, SGPR = 2
    use_modified_ski = True if use_modified_ski > 0 else False
    sg_shifted = config.SgShifted(sg_shifted)

    print("Basic config vars:")
    print("kernel type: ", kernel_type)
    print("interp_type: ", interp_type)
    print("basis: ", basis_type)
    print("bypass_covar: ", bypass_covar)
    print("skip wandb: ", skip_wandb)
    print("verbose: ", verbose)
    print("method: ", method)
    print("use modified: ", use_modified_ski)
    print("grid_level: ", grid_level)
    print("lr: ", lr)
    print("grid_size_dim: ", grid_size_dim)
    print("sg_shifted: ", sg_shifted)

    # if sg_shifted != config.SgShifted.ZERO:
    #     comb = False

    #gp.settings.max_cholesky_size._set_value(-1)

    # Initiate wandb config for the experimentation
    if not skip_wandb:
        wandb_run = wandb.init(entity="swjindal", project="sgkigp", dir = log_dir, config={
            'method': method,
            'dataset': dataset,
            'lr': lr,
            'lanc_iter': lanc_iter,
            'pre_size': pre_size,
            'n_inducing': n_inducing,
            'kernel_type': kernel_type.value
        })

        print("Wandb run dir: ", wandb.run.dir)

    # Set-seeds and device
    config.set_seeds(seed=seed)
    device = config.get_device(device=device)
    dtype = config.dtype(dtype, use_torch=True)

    # Preparing data iterations
    if func_val >= 11:
        datapath = os.path.expanduser("~/sgkigp/data/")
        datapath = datapath + 'f' + str(func_val) + "_ndim" + str(ndim) + ".pt"
        train_x, train_y, val_x, val_y, test_x, test_y, __, __ = torch.load(datapath)

    elif dataset is None and func_val >= 0:  # synthetic dataset
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

    # Move to device and dtype
    train_x = train_x.to(device=device, dtype=dtype)
    train_y = train_y.to(device=device, dtype=dtype)
    val_x = val_x.to(device=device, dtype=dtype)
    val_y = val_y.to(device=device, dtype=dtype)
    test_x = test_x.to(device=device, dtype=dtype)
    test_y = test_y.to(device=device, dtype=dtype)

    n_inducing = min(train_x.size(0), n_inducing)
    print("num inducing points: ", n_inducing)

    # Reporting basic data statistics
    print(
        f'"{dataset}": D = {train_x.size(-1)}, Train N = {train_x.size(0)},'
        f' Val N = {val_x.size(0)} Test N = {test_x.size(0)}')

    if not skip_wandb:
        wandb.config.update({
            'D': train_x.size(-1),
            'N_train': train_x.size(0),
            'N_test': test_x.size(0),
            'N_val': val_x.size(0)
        })

    grid_size = 0
    model = setup_models(
        method, train_x, train_y, n_inducing,
        kernel_type, min_noise, device, dtype,
        val_x, test_x, boundary_slack, grid_level, interp_type, basis_type,
        bypass_covar, use_modified=use_modified_ski, grid_size_dim=grid_size_dim,
        sg_shifted=sg_shifted,
    )

    mll = gp.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    # Setting optimizer and early stopper
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopper(patience=p_epochs)   # p_epochs is patience of epochs

    # Running over epochs
    for i in tqdm(range(epochs)):

        # Running and reporting progress of an epoch
        train_dict = train(train_x, train_y, model, mll, optimizer,
                           lanc_iter=lanc_iter, pre_size=pre_size, cg_tol=cg_tol)

        if not skip_wandb:
            wandb.log(train_dict, step=i + 1)

        # Doing validation
        if (i % log_int) == 0:

            # Running validation and testing
            val_dict = test(val_x, val_y, model, mll, pre_size=pre_size, lanc_iter=lanc_iter, label='val',
                            cg_tol=cg_eval_tol, bypass_covar=bypass_covar)

            test_dict = test(test_x, test_y, model, mll, pre_size=pre_size, lanc_iter=lanc_iter,
                             cg_tol=cg_eval_tol,  bypass_covar=bypass_covar)

            # Recording results
            test_results = {
                'test/best_rmse': test_dict['test/rmse'],
                'test/best_nll': test_dict['test/nll'],
                'val/best_step': i + 1
            }
            stopper(-val_dict['val/rmse'], dict(
                state_dict=model.state_dict(),
                summary=test_results
            ))

            if verbose:
                print("I: ", i, "\n", test_results)

            if not skip_wandb:
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
    if not skip_wandb:
        for k, v in stopper.info().get('summary').items():
            wandb.run.summary[k] = v
        wandb.run.summary['grid_size'] = grid_size
        torch.save(stopper.info().get('state_dict'), Path(wandb.run.dir) / 'model.pt')
        wandb.save('*.pt')
        wandb_run.save()


def train(x, y, model, mll, optim, lanc_iter=100, pre_size=100, cg_tol=1.0):

    model.train()
    optim.zero_grad()

    # with gp.settings.cg_tolerance(cg_tol), \
    #      gp.settings.max_preconditioner_size(pre_size), \
    #      gp.settings.max_cholesky_size(-1), \
    #      gp.settings.max_root_decomposition_size(lanc_iter):

    with gp.settings.cg_tolerance(cg_tol), \
            gp.settings.max_cholesky_size(-1), \
        gp.settings.max_preconditioner_size(pre_size), \
        gp.settings.max_root_decomposition_size(lanc_iter):

        t_start = timer()
        output = model(x)
        loss = -mll(output, y)
        loss_ts = timer() - t_start

        if DEBUG_MODE:
            print("Before update: ")
            print("loss: ", loss.detach().cpu().numpy())
            for param_name, param in model.named_parameters():
                print('Parameter name: ', param_name,  'value = ', param.detach().cpu().numpy())

        t_start = timer()
        loss.backward()
        optim.step()

        if DEBUG_MODE:
            print("After update: ")
            for param_name, param in model.named_parameters():
                print('Parameter name: ', param_name,  'value = ', param.detach().cpu().numpy())

        bw_ts = timer() - t_start

    return {
        'train/mll': -loss.detach().item(),
        'train/loss_ts': loss_ts,
        'train/bw_ts': bw_ts,
        'train/total_ts': loss_ts + bw_ts
    }


def test(x, y, model, mll, lanc_iter=100,
         pre_size=100, label='test', cg_tol=1e-2,
         bypass_covar=False):

    model.eval()
    model.likelihood.eval()

    # with gp.settings.eval_cg_tolerance(cg_tol), \
    #      gp.settings.max_cholesky_size(-1), \
    #      gp.settings.max_preconditioner_size(pre_size), \
    #      gp.settings.max_root_decomposition_size(lanc_iter), \
    #      torch.no_grad():

    with gp.settings.eval_cg_tolerance(cg_tol), \
            gp.settings.max_cholesky_size(-1), \
         gp.settings.max_preconditioner_size(pre_size), \
         gp.settings.max_root_decomposition_size(lanc_iter), \
         gp.settings.fast_pred_var(), torch.no_grad():

        t_start = timer()
        pred_y = model(x)
        pred_ts = timer() - t_start

        if bypass_covar:
            rmse = (pred_y - y).pow(2).mean(0).sqrt()
            mae = (pred_y - y).abs().mean(0)
            nll = 0.0
        else:
            #nll = -mll(pred_y, y)
            rmse = (pred_y.mean - y).pow(2).mean(0).sqrt()
            mae = (pred_y.mean - y).abs().mean(0)
            #nll = nll.item()
            nll = 0.0
    return {
        f'{label}/rmse': rmse.item(),
        f'{label}/mae': mae.item(),
        f'{label}/pred_ts': pred_ts,
        f'{label}/nll': nll,
    }


if __name__ == "__main__":
    os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')
    fire.Fire(main)

