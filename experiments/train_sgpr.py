import os
import torch
import pickle
import gpytorch as gp
from tqdm.auto import tqdm
import wandb
from pathlib import Path
from timeit import default_timer as timer
from experiments.data import prepare_dataset
from experiments.misc import EarlyStopper
import random
import  numpy as np
import sgkigp.config as config
#from utils import  prepare_dataset, EarlyStopper


## Default to double.
# torch.set_default_dtype(torch.float64)

class SGPRModel(gp.models.ExactGP):
    def __init__(self, train_x, train_y, n_inducing=500, nu=None, min_noise=1e-4):
        likelihood = gp.likelihoods.GaussianLikelihood(
            noise_constraint=gp.constraints.GreaterThan(min_noise))
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        self.base_covar_module = gp.kernels.MaternKernel(nu=nu, ard_num_dims=train_x.size(-1)) \
            if nu is not None else gp.kernels.RBFKernel(ard_num_dims=train_x.size(-1))
        self.covar_module = gp.kernels.InducingPointKernel(gp.kernels.ScaleKernel(self.base_covar_module),
                                                           inducing_points=train_x[
                                                               torch.randperm(train_x.size(0))[:n_inducing]],
                                                           likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)


def train(x, y, model, mll, optim, lanc_iter=100, pre_size=100):
    model.train()

    optim.zero_grad()

    with gp.settings.cg_tolerance(1.0), \
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


def test(x, y, model, mll, lanc_iter=100, pre_size=100, label='test'):
    model.eval()

    with gp.settings.eval_cg_tolerance(1e-2), \
            gp.settings.max_preconditioner_size(pre_size), \
            gp.settings.max_root_decomposition_size(lanc_iter), \
            gp.settings.fast_pred_var(), torch.no_grad():
        t_start = timer()

        # pred_y = model.likelihood(model(x))
        pred_y = model(x)
        pred_ts = timer() - t_start

        rmse = (pred_y.mean - y).pow(2).mean(0).sqrt()
        mae = (pred_y.mean - y).abs().mean(0)
        # nll = - torch.distributions.Normal(pred_y.mean,
        #   pred_y.variance.add(model.likelihood.noise).sqrt()).log_prob(y).mean()

    return {
        f'{label}/rmse': rmse.item(),
        f'{label}/mae': mae.item(),
        f'{label}/pred_ts': pred_ts,
        f'{label}/nll': 0.0  # nll.item()
    }

def set_seeds(seed=None):
  if seed is not None and seed >= 0:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(dataset: str = None, data_dir: str = None, log_int: int = 1, seed: int = None, device: int = 0,
         epochs: int = 100, lr: int = 1e-3, p_epochs: int = 200, lanc_iter: int = 100, pre_size: int = 100,
         n_inducing: int = 512, nu: float = None, min_noise: float = 1e-4, func_val: int = 0, ndim: int = 4,
         log_dir: str = os.path.expanduser('~/sgkigp/logs/tmp')):

    wandb_run = wandb.init(entity="swjindal", project="sgkigp", dir=log_dir, config={
        'method': 'SGPR',
        'dataset': dataset,
        'lr': lr,
        'lanc_iter': lanc_iter,
        'pre_size': pre_size,
        'n_inducing': n_inducing,
        'nu': nu
    })

    set_seeds(seed)
    device = f"cuda:{device}" if (device >= 0 and torch.cuda.is_available()) else "cpu"
    dtype = config.dtype(default='float32', use_torch=True)

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


    print(
        f'"{dataset}": D = {train_x.size(-1)}, Train N = {train_x.size(0)}, Val N = {val_x.size(0)} Test N = {test_x.size(0)}')

    wandb.config.update({
        'D': train_x.size(-1),
        'N_train': train_x.size(0),
        'N_test': test_x.size(0),
        'N_val': val_x.size(0)
    })

    model = SGPRModel(train_x, train_y, n_inducing=n_inducing, nu=nu, min_noise=min_noise).to(device)
    mll = gp.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopper(patience=p_epochs)

    for i in tqdm(range(epochs)):
        train_dict = train(train_x, train_y, model, mll, optimizer,
                           lanc_iter=lanc_iter, pre_size=pre_size)
        wandb.log(train_dict, step=i + 1)

        if (i % log_int) == 0:
            val_dict = test(val_x, val_y, model, mll,
                            pre_size=pre_size, lanc_iter=lanc_iter,
                            label='val')

            test_dict = test(test_x, test_y, model, mll,
                             pre_size=pre_size, lanc_iter=lanc_iter)

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

    for k, v in stopper.info().get('summary').items():
        wandb.run.summary[k] = v
    torch.save(stopper.info().get('state_dict'), Path(wandb.run.dir) / 'model.pt')
    wandb.save('*.pt')
    wandb_run.save()


if __name__ == "__main__":
    os.environ['WANDB_MODE'] = os.environ.get('WANDB_MODE', default='dryrun')
    import fire
    fire.Fire(main)