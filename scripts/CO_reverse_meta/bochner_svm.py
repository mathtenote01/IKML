import argparse
import pickle as pkl
import warnings
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.optimize as sco
from tqdm import tqdm 
import optuna
from implicit_kernel_meta_learning.algorithms import SupportVectorMachine
# import cvxopt
# from cvxopt import matrix
from implicit_kernel_meta_learning.data_utils import AirQualityDataLoader
from implicit_kernel_meta_learning.data_utils import ethyleneCOLoader
from implicit_kernel_meta_learning.experiment_utils import set_seed
from implicit_kernel_meta_learning.kernels import BochnerKernel
from concurrent import futures

warnings.filterwarnings("ignore")
class SupportVectorMachine(nn.Module):
    def __init__(self, lam, kernel, device=None):
        super(SupportVectorMachine, self).__init__()
        self.lam = torch.tensor(lam)
        self.kernel = kernel
        self.alphas = None
        self.X_tr = None
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def fit(self, X, Y):
        eps = 0.001
        C = 1
        if len(X.size()) == 3:
            b, n, d = X.size()
            b, m, l = Y.size()
            # self.K = self.kernel(X, X)
            # self.K = self.K.to('cpu').detach().numpy().copy()
            # X = X.to('cpu').detach().numpy().copy()
            # Y = Y.to('cpu').detach().numpy().copy()
            # def min_func_var(alphas):
            #     S = 0
            #     for i in range(n):
            #         for j in range(n):
            #             S = S + 0.5 * (alphas[i] - alphas[i + n]) * (alphas[j] - alphas[j + n]) * self.K[0, i, j]
            #         S = S + self.lam * (alphas[i] + alphas[i + n])
            #         S = S - (alphas[i] - alphas[i + n]) * Y[i]
            #     return S
            # cur_alphas = [1 for _ in range(2 * n)]
            # bnds = [(0, C) for _ in range(2 * n)]
            # opts = sco.minimize(fun=min_func_var, x0=cur_alphas, method='SLSQP', bounds=bnds)
            # self.alphas = torch.from_numpy(opts['x']).clone()
        elif len(X.size()) == 2:
            n, d = X.size()
            m, l = Y.size()
        assert (
            n == m
        ), "Tensors need to have same dimension, dimensions are {} and {}".format(n, m)
        self.K = self.kernel(X, X)
        self.K = self.K.to('cpu').detach().numpy().copy()
        X = X.to('cpu').detach().numpy().copy()
        Y = Y.to('cpu').detach().numpy().copy()
        
        def min_func_var(alphas):
            S = 0
            for i in range(n):
                for j in range(n):
                    S = S + 0.5 * (alphas[i] - alphas[i + n]) * (alphas[j] - alphas[j + n]) * self.K[0, i, j]
                    if i == j:
                        S += torch.exp(self.lam)
                S = S + torch.exp(self.lam) * (alphas[i] + alphas[i + n])
                S = S - (alphas[i] - alphas[i + n]) * Y[i]
            return S
        cur_alphas = [1 for _ in range(2 * n)]
        bnds = [(0, C) for _ in range(2 * n)]
        opts = sco.minimize(fun=min_func_var, x0=cur_alphas, method='SLSQP', bounds=bnds)
        self.alphas = torch.from_numpy(opts['x']).clone()
        # NOTE kernel(X, X) = cos(X^{T}omega) x cos(X^{T}omega)
        # NOTE omegaは, ラテントを関数でpush forwardしたもの
        # TODO Kは行列ではないので, 二次計画問題にすることはできない
        # self.K = self.kernel(X, X)
        self.K = torch.from_numpy(self.K).clone()
        self.X_tr = torch.from_numpy(X).clone()
        self.Y_tr = torch.from_numpy(Y).clone()

    def predict(self, X):
        # self.last_alpha = torch.tensor(self.alphas["alpha"]) - torch.tensor(self.alphas["alpha_tilde"])
        # return torch.matmul(self.kernel(X, self.X_tr), self.last_alpha)
        result = 0
        b_array = [self.Y_tr[i] - self.lam for i in range(torch.tensor(self.X_tr).size()[0])]
        b_mean = 0
        n = torch.tensor(self.X_tr).size()[0]
        for i in range(n):
            for j in range(n):
                # b_array[i] -= (self.alphas[j] - self.alphas[j + n]) * self.kernel(self.X_tr[j], self.X_tr[i])
                b_array[i] -= (self.alphas[j] - self.alphas[j + n]) * self.K[0, i, j]
            b_mean += b_array[i] / torch.tensor(self.X_tr).size()[0]
        
        for i in range(torch.tensor(self.X_tr).size()[0]):
            result += (self.alphas[i] - self.alphas[i + n]) * self.kernel(X, self.X_tr[i])
            
        return result + b_mean

def visualise_run(result):
    t_val = result["meta_val_every"] * np.arange(len(result["meta_valid_error"]))
    t = np.arange(len(result["meta_train_error"]))
    fig, ax = plt.subplots()
    ax.plot(t, result["meta_train_error"], label="Meta train MSE")
    ax.plot(t_val, result["meta_valid_error"], label="Meta val MSE")
    ax.legend()
    ax.set_title(
        "meta-(val, test) holdout MSE: ({:.4f}, {:.4f})".format(
            result["holdout_meta_valid_error"][0], result["holdout_meta_test_error"][0]
        )
    )
    return fig, ax

def fast_adapt_boch(batch, model, loss, D, device):
    # Unpack data
    X_tr, y_tr = batch["train"]
    X_tr = X_tr.to(device).float()
    y_tr = y_tr.to(device).float()
    X_val, y_val = batch["valid"]
    X_val = X_val.to(device).float()
    y_val = y_val.to(device).float()

    # adapt algorithm
    model.kernel.sample_features(D)
    # print("type of X_tr is {}".format(type(X_tr)))
    model.fit(X_tr, y_tr)

    # Predict
    y_hat = model.predict(X_val)
    return loss(y_val, y_hat)


def get_nonlinearity(nonlinearity):
    nonlinearity = nonlinearity.lower()
    if nonlinearity == "relu":
        return nn.ReLU
    elif nonlinearity == "sigmoid":
        return nn.Sigmoid
    elif nonlinearity == "tanh":
        return nn.Tanh


def mlp_layer(in_dim, out_dim, nonlinearity, batch_norm=True):
    if batch_norm:
        layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(num_features=out_dim),
            nonlinearity(),
        )
    else:
        layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nonlinearity(),
        )

    return layer


def create_mlp(num_layers, hidden_dim, in_dim, out_dim, nonlinearity, batch_norm=True):
    if num_layers == 0:
        mlp = nn.Linear(in_dim, out_dim)
    else:
        if batch_norm:
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nonlinearity(),
                nn.BatchNorm1d(num_features=hidden_dim),
                *(
                    mlp_layer(
                        hidden_dim, hidden_dim, nonlinearity, batch_norm=batch_norm
                    )
                    for _ in range(num_layers)
                ),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nonlinearity(),
                *(
                    mlp_layer(
                        hidden_dim, hidden_dim, nonlinearity, batch_norm=batch_norm
                    )
                    for _ in range(num_layers)
                ),
                nn.Linear(hidden_dim, out_dim),
            )
    return mlp


        
def main(
    seed,
    k_support,
    k_query,
    num_iterations,
    meta_batch_size,
    meta_val_batch_size,
    meta_val_every,
    holdout_size,
    num_layers,
    latent_d,
    D,
    hidden_dim,
    nonlinearity,
    meta_lr,
    lam,
):
    nonlinearity = get_nonlinearity(nonlinearity)
    result = OrderedDict(
        meta_train_error=[],
        meta_valid_error=[],
        holdout_meta_test_error=[],
        holdout_meta_valid_error=[],
        meta_val_every=meta_val_every,
        num_iterations=num_iterations,
        name="Bochner IKML",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed, False)

    # Load train/validation/test data
    traindata_meta = AirQualityDataLoader(k_support, k_query, split="train")
    valdata_meta = AirQualityDataLoader(k_support, k_query, split="valid")
    testdata_meta = AirQualityDataLoader(k_support, k_query, split="test")
    traindata = ethyleneCOLoader(k_support, k_query, split="train")
    valdata = ethyleneCOLoader(k_support, k_query, split="valid")
    testdata = ethyleneCOLoader(k_support, k_query, split="test")

    # Holdout errors
    valid_batches = [valdata_meta.sample() for _ in range(holdout_size)]
    test_batches = [testdata_meta.sample() for _ in range(holdout_size)]

    # Define model
    in_dim = latent_d
    out_dim = 14
    pf_map = create_mlp(
        num_layers, hidden_dim, in_dim, out_dim, nonlinearity, batch_norm=False
    )
    latent_dist = torch.distributions.Normal(
        torch.tensor(0.0).to(device), torch.tensor(1.0).to(device)
    )
    kernel = BochnerKernel(latent_d, latent_dist, pf_map, device=device)
    model = SupportVectorMachine(np.log(lam), kernel).to(device)
    opt = optim.Adam(model.parameters(), meta_lr)
    parameters = model.parameters()
    for val in parameters:
        print(val)
    print("parameters of model are {}".format(model.parameters()))
    loss = nn.MSELoss("mean")

    # Keep best model around
    best_val_iteration = 0
    best_val_mse = np.inf

    for iteration in tqdm(range(num_iterations), desc="training"):
        validate = True if iteration % meta_val_every == 0 else False

        train_batches = [traindata.sample() for _ in range(meta_batch_size)]
        opt.zero_grad()
        meta_train_error = 0.0
        meta_valid_error = 0.0
        def _fast_adapt_boch_train(batch, model, loss, D, device):
            nonlocal meta_train_error
            # Unpack data
            X_tr, y_tr = batch["train"]
            X_tr = X_tr.to(device).float()
            y_tr = y_tr.to(device).float()
            X_val, y_val = batch["valid"]
            X_val = X_val.to(device).float()
            y_val = y_val.to(device).float()
            # adapt algorithm
            model.kernel.sample_features(D)
            model.fit(X_tr, y_tr)
            # Predict
            y_hat = model.predict(X_val)
            cur = loss(y_val, y_hat)
            cur.backward()
            meta_train_error += cur.item()
        
        def _fast_adapt_boch_valid(batch, model, loss, D, device):
            nonlocal meta_valid_error
            # Unpack data
            X_tr, y_tr = batch["train"]
            X_tr = X_tr.to(device).float()
            y_tr = y_tr.to(device).float()
            X_val, y_val = batch["valid"]
            X_val = X_val.to(device).float()
            y_val = y_val.to(device).float()
            # adapt algorithm
            model.kernel.sample_features(D)
            model.fit(X_tr, y_tr)
            # Predict
            y_hat = model.predict(X_val)
            meta_valid_error += torch.norm(y_hat - y_val).to('cpu').detach().numpy().copy() ** 2
        with futures.ThreadPoolExecutor() as executor:
            for train_batch in train_batches:
                # タスクを追加する。
                executor.submit(_fast_adapt_boch_train, train_batch, model, loss, D, device)
        if validate:
            val_batches = [valdata_meta.sample() for _ in range(meta_val_batch_size)]
            torch.save(model.state_dict(), "model.pth")
            # 保存したモデルを読み込む。
            tmp = SupportVectorMachine(np.log(lam), kernel).to(device)
            tmp.load_state_dict(torch.load("model.pth"))
            with futures.ThreadPoolExecutor() as executor:
                for val_batch in val_batches:
                    # タスクを追加する。
                    executor.submit(_fast_adapt_boch_valid, val_batch, tmp, loss, D, device)
            meta_valid_error /= meta_val_batch_size
            result["meta_valid_error"].append(meta_valid_error)
            print("Iteration {}".format(iteration))
            print("meta_valid_error: {}".format(meta_valid_error))
            if meta_valid_error < best_val_mse:
                best_val_iteration = iteration
                best_val_mse = meta_valid_error
                best_state_dict = model.state_dict()

        meta_train_error /= meta_batch_size
        result["meta_train_error"].append(meta_train_error)
        # Average the accumulated gradients and optimize
        for p in model.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        opt.step()

    # Load best model
    print("best_valid_iteration: {}".format(best_val_iteration))
    print("best_valid_mse: {}".format(best_val_mse))
    model.load_state_dict(best_state_dict)

    meta_valid_error = 0.0
    meta_test_error = 0.0
    for (valid_batch, test_batch) in tqdm(zip(valid_batches, test_batches), desc="validation, test"):
        evaluation_error = fast_adapt_boch(
            batch=valid_batch,
            model=model,
            loss=loss,
            D=D,
            device=device,
        )
        meta_valid_error += evaluation_error.item()
        evaluation_error = fast_adapt_boch(
            batch=test_batch,
            model=model,
            loss=loss,
            D=D,
            device=device,
        )
        meta_test_error += evaluation_error.item()

    meta_valid_error /= holdout_size
    meta_test_error /= holdout_size
    print("holdout_meta_valid_error: {}".format(meta_valid_error))
    print("holdout_meta_test_error: {}".format(meta_test_error))
    result["holdout_meta_valid_error"].append(meta_valid_error)
    result["holdout_meta_test_error"].append(meta_test_error)

    with open("result.pkl", "wb") as f:
        pkl.dump(result, f)

    # Visualise
    fig, ax = visualise_run(result)
    plt.tight_layout()
    fig.savefig("learning_curves.pdf", bbox_inches="tight")
    fig.savefig("learning_curves.png", bbox_inches="tight")
    
    ####################################### objective と元々のmainの境界線 #################################################################
    
    # nonlinearity = get_nonlinearity(nonlinearity)
    
    # def objective(trial):
    #     # param = {'seed': 42,
    #     #         'k_support': 20,
    #     #         'k_query': 20,
    #     #         'num_iterations': 30000,
    #     #         'meta_batch_size': 4,
    #     #         'meta_val_batch_size': 1000,
    #     #         'meta_val_every': 250,
    #     #         'holdout_size': 3000,
    #     #         'num_layers': 2,
    #     #         'latent_d': 64,
    #     #         'D': 20000,
    #     #         'hidden_dim': 64,
    #     #         'nonlinearity': 'relu',
    #     #         'meta_lr': 0.00001,
    #     #         'lam': 0.001}
    #     nonlocal seed
    #     nonlocal k_support
    #     nonlocal k_query
    #     num_iterations = 300
    #     nonlocal meta_batch_size
    #     nonlocal meta_val_batch_size
    #     meta_val_every = 25
    #     nonlocal holdout_size
    #     nonlocal num_layers
    #     nonlocal latent_d
    #     nonlocal D
    #     nonlocal hidden_dim
    #     nonlocal nonlinearity
    #     meta_lr = trial.suggest_float("meta_lr", 0.00001, 1)
    #     lam = trial.suggest_float("lam", 0.001, 1)

        
    #     result = OrderedDict(
    #         meta_train_error=[],
    #         meta_valid_error=[],
    #         holdout_meta_test_error=[],
    #         holdout_meta_valid_error=[],
    #         meta_val_every=meta_val_every,
    #         num_iterations=num_iterations,
    #         name="Bochner IKML",
    #     )
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     set_seed(seed, False)

    # traindata_meta = AirQualityDataLoader(k_support, k_query, split="train")
    # valdata_meta = AirQualityDataLoader(k_support, k_query, split="valid")
    # testdata_meta = AirQualityDataLoader(k_support, k_query, split="test")
    # traindata = ethyleneCOLoader(k_support, k_query, split="train")
    # valdata = ethyleneCOLoader(k_support, k_query, split="test")
    # testdata = ethyleneCOLoader(k_support, k_query, split="test")
    #     # Holdout errors
    #     valid_batches = [valdata_meta.sample() for _ in range(holdout_size)]
    #     test_batches = [testdata_meta.sample() for _ in range(holdout_size)]

    #     # Define model
    #     in_dim = latent_d
    #     out_dim = 14
    #     pf_map = create_mlp(
    #         num_layers, hidden_dim, in_dim, out_dim, nonlinearity, batch_norm=False
    #     )
    #     latent_dist = torch.distributions.Normal(
    #         torch.tensor(0.0).to(device), torch.tensor(1.0).to(device)
    #     )
    #     kernel = BochnerKernel(latent_d, latent_dist, pf_map, device=device)
    #     model = SupportVectorMachine(np.log(lam), kernel).to(device)
    #     opt = optim.Adam(model.parameters(), meta_lr)
    #     # parameters = model.parameters()
    #     # for val in parameters:
    #     #     print(val)
    #     # print("parameters of model are {}".format(model.parameters()))
    #     loss = nn.MSELoss("mean")

    #     # Keep best model around
    #     best_val_iteration = 0
    #     best_val_mse = np.inf

    #     for iteration in tqdm(range(num_iterations), desc="training"):
    #         validate = True if iteration % meta_val_every == 0 else False

    #         train_batches = [traindata.sample() for _ in range(meta_batch_size)]
    #         opt.zero_grad()
    #         meta_train_error = 0.0
    #         meta_valid_error = 0.0
    #         def _fast_adapt_boch_train(batch, model, loss, D, device):
    #             nonlocal meta_train_error
    #             # Unpack data
    #             X_tr, y_tr = batch["train"]
    #             X_tr = X_tr.to(device).float()
    #             y_tr = y_tr.to(device).float()
    #             X_val, y_val = batch["valid"]
    #             X_val = X_val.to(device).float()
    #             y_val = y_val.to(device).float()
    #             # adapt algorithm
    #             model.kernel.sample_features(D)
    #             model.fit(X_tr, y_tr)
    #             # Predict
    #             y_hat = model.predict(X_val)
    #             cur = loss(y_val, y_hat)
    #             cur.backward()
    #             meta_train_error += cur.item()

    #         def _fast_adapt_boch_valid(batch, model, loss, D, device):
    #             nonlocal meta_valid_error
    #             # Unpack data
    #             X_tr, y_tr = batch["train"]
    #             X_tr = X_tr.to(device).float()
    #             y_tr = y_tr.to(device).float()
    #             X_val, y_val = batch["valid"]
    #             X_val = X_val.to(device).float()
    #             y_val = y_val.to(device).float()
    #             # adapt algorithm
    #             model.kernel.sample_features(D)
    #             model.fit(X_tr, y_tr)
    #             # Predict
    #             y_hat = model.predict(X_val)
    #             meta_valid_error += torch.norm(y_hat - y_val).to('cpu').detach().numpy().copy() ** 2
    #         with futures.ThreadPoolExecutor() as executor:
    #             for train_batch in train_batches:
    #                 # タスクを追加する。
    #                 executor.submit(_fast_adapt_boch_train, train_batch, model, loss, D, device)
    #         if validate:
    #             val_batches = [valdata_meta.sample() for _ in range(meta_val_batch_size)]
    #             torch.save(model.state_dict(), "model.pth")
    #             # 保存したモデルを読み込む。
    #             tmp = SupportVectorMachine(np.log(lam), kernel).to(device)
    #             tmp.load_state_dict(torch.load("model.pth"))
    #             with futures.ThreadPoolExecutor() as executor:
    #                 for val_batch in val_batches:
    #                     # タスクを追加する。
    #                     executor.submit(_fast_adapt_boch_valid, val_batch, tmp, loss, D, device)
    #             meta_valid_error /= meta_val_batch_size
    #             result["meta_valid_error"].append(meta_valid_error)
    #             print("Iteration {}".format(iteration))
    #             print("meta_valid_error: {}".format(meta_valid_error))
    #             if iteration == num_iterations - meta_val_every:
    #                 return meta_valid_error
    #             if meta_valid_error < best_val_mse:
    #                 best_val_iteration = iteration
    #                 best_val_mse = meta_valid_error
    #                 best_state_dict = model.state_dict()

    #         meta_train_error /= meta_batch_size
    #         result["meta_train_error"].append(meta_train_error)
    #         # Average the accumulated gradients and optimize
    #         for p in model.parameters():
    #             p.grad.data.mul_(1.0 / meta_batch_size)
    #         opt.step()
    #     # Load best model
    #     print("best_valid_iteration: {}".format(best_val_iteration))
    #     print("best_valid_mse: {}".format(best_val_mse))
    #     model.load_state_dict(best_state_dict)

    #     meta_valid_error = 0.0
    #     meta_test_error = 0.0
    #     for (valid_batch, test_batch) in tqdm(zip(valid_batches, test_batches), desc="validation, test"):
    #         evaluation_error = fast_adapt_boch(
    #             batch=valid_batch,
    #             model=model,
    #             loss=loss,
    #             D=D,
    #             device=device,
    #         )
    #         meta_valid_error += evaluation_error.item()
    #         evaluation_error = fast_adapt_boch(
    #             batch=test_batch,
    #             model=model,
    #             loss=loss,
    #             D=D,
    #             device=device,
    #         )
    #         meta_test_error += evaluation_error.item()

    #     meta_valid_error /= holdout_size
    #     meta_test_error /= holdout_size

    # study = optuna.create_study()
    # study.optimize(objective, n_trials=100)
    # trial = study.best_trial
    # print('  Value: {}'.format(trial.value))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_support", type=int, default=10)
    parser.add_argument("--k_query", type=int, default=10)
    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument("--meta_batch_size", type=int, default=4)
    parser.add_argument("--meta_val_batch_size", type=int, default=100)
    parser.add_argument("--meta_val_every", type=int, default=100)
    parser.add_argument("--holdout_size", type=int, default=3000)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--nonlinearity", type=str, default="relu")
    parser.add_argument("--latent_d", type=int, default=64)
    parser.add_argument("--D", type=int, default=5000)
    parser.add_argument("--lam", type=float, default=0.001)
    parser.add_argument("--meta_lr", type=float, default=0.001)
    args = parser.parse_args()
    main(**vars(args))
