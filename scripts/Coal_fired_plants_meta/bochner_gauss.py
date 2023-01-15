import argparse
import pickle as pkl
import warnings
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from implicit_kernel_meta_learning.algorithms import GaussProcess
from implicit_kernel_meta_learning.data_utils import GasSensorDataLoader
from implicit_kernel_meta_learning.data_utils import AirQualityDataLoader
from implicit_kernel_meta_learning.experiment_utils import set_seed
from implicit_kernel_meta_learning.kernels import BochnerKernel
from concurrent import futures
import copy
import optuna
warnings.filterwarnings("ignore")


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


class RidgeRegression(nn.Module):
    def __init__(self, log_lam, kernel, device=None):
        super(RidgeRegression, self).__init__()
        self.log_lam = nn.Parameter(torch.tensor(log_lam))
        self.kernel = kernel
        self.alphas = None
        self.X_tr = None
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def fit(self, X, Y):
        if len(X.size()) == 3:
            b, n, d = X.size()
            b, m, l = Y.size()
        elif len(X.size()) == 2:
            n, d = X.size()
            m, l = Y.size()
        assert (
            n == m
        ), "Tensors need to have same dimension, dimensions are {} and {}".format(n, m)

        self.K = self.kernel(X, X)
        K_nl = self.K + torch.exp(self.log_lam) * n * torch.eye(n).to(self.device)
        # To use solve we need to make sure Y is a float
        # and not an int
        self.alphas, _ = torch.solve(Y.float(), K_nl)
        self.X_tr = X

    def predict(self, X):
        return torch.matmul(self.kernel(X, self.X_tr), self.alphas)

class GaussProcess(nn.Module):
    def __init__(self, log_lam, kernel, device=None):
        super(GaussProcess, self).__init__()
        self.log_lam = nn.Parameter(torch.tensor(log_lam))
        self.kernel = kernel
        self.alphas = None
        self.X_tr = None
        self.Y_tr = None
        print("right class was called")
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def fit(self, X, Y):
        if len(X.size()) == 3:
            b, n, d = X.size()
            b, m, l = Y.size()
        elif len(X.size()) == 2:
            n, d = X.size()
            m, l = Y.size()
        assert (
            n == m
        ), "Tensors need to have same dimension, dimensions are {} and {}".format(n, m)
        # print("Now training... the size of training datasets is {}, {}".format(X.size(), Y.size()))
        self.K = self.kernel(X, X)
        self.K_nl = self.K + self.log_lam * n * torch.eye(n).to(self.device)
        # To use solve we need to make sure Y is a float
        # and not an int
        # Yの標本平均が入る.
        self.mu_of_Y = torch.mean(input=Y)
        # self.alphas, _ = torch.solve(Y.float(), self.K_nl)
        self.X_tr = X
        self.Y_tr = Y

    def predict(self, X):
        if len(X.size()) == 3:
            b, n, d = X.size()
            b, m, l = self.Y_tr.size()
            full_mu_of_Y = self.mu_of_Y * torch.ones(n, dtype=torch.float32).reshape(n, 1)
        elif len(X.size()) == 2:
            n, d = X.size()
            m, l = self.Y_tr.size()
            full_mu_of_Y = self.mu_of_Y * torch.ones(n, dtype=torch.float32).reshape(n, 1)
        
        if len(self.X_tr.size()) == 3:
            b, tr_n, d = self.X_tr.size()
            b, tr_m, l = self.Y_tr.size()
            full_mu_of_Y_tr = self.mu_of_Y * torch.ones(tr_n, dtype=torch.float32).reshape(tr_n, 1)
        elif len(self.X_tr.size()) == 2:
            tr_n, d = self.X_tr.size()
            tr_m, l = self.Y_tr.size()
            full_mu_of_Y_tr = self.mu_of_Y * torch.ones(tr_n, dtype=torch.float32).reshape(tr_n, 1)
        # 気合いで書き換えていけ
        # 事後分布の平均を計算して入れる
        self.K_nl = self.K_nl.reshape(tr_n, tr_n)
        # cur = torch.matmul(self.kernel(X, self.X_tr).reshape(n, tr_n), torch.inverse(self.K_nl))
        # print("the size of cur and Y are {}, {}".format(cur.size(), full_mu_of_Y.size()))
        expect = full_mu_of_Y + torch.matmul(torch.matmul(self.kernel(X, self.X_tr).reshape(n, tr_n), torch.inverse(self.K_nl)), self.Y_tr - full_mu_of_Y_tr)
        # expect = full_mu_of_Y
        # 事後分布の分散を計算して入れる
        # print("the size is {}, {}, {}, {}, {}, {}".format(self.kernel(X, self.X_tr).reshape(n, n).size(), self.kernel(X, X).reshape(n, n).size(), X.size(), self.X_tr.size(), self.K_nl.size(), self.Y_tr.size()))
        variation = self.kernel(X, X).reshape(n, n) - torch.matmul(torch.matmul(self.kernel(X, self.X_tr).reshape(n, tr_n), torch.inverse(self.K_nl)), self.kernel(self.X_tr, X).reshape(tr_n, n))
        # print("variation is {}".format(variation))
        # print("the size of expectation and variation are {} ,{}".format(expect.size(), variation.size()))
        # variation = self.kernel(X, X)
        # print("the size of expect and variation are {}, {}".format(expect.size(), variation.size()))
        expect = expect.reshape(-1,).to('cpu').detach().numpy().copy()
        variation = variation.reshape(n, n).to('cpu').detach().numpy().copy()
        cur_list = np.random.multivariate_normal(expect, variation, 1).tolist()
        res = torch.tensor(cur_list, dtype=torch.double, requires_grad=True)
        # print("res.dtype is {}".format(res.dtype))
        return res


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
    model.fit(X_tr, y_tr)

    # Predict
    y_hat = model.predict(X_val)
    # return loss(y_val, y_hat)
    return torch.norm(y_hat - y_val).to('cpu').detach().numpy().copy() ** 2


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

def log_marginal_likelihood(kernel, X, Y, sigma=1e-3):
    mean = torch.mean(input=Y) * torch.ones(Y.size()[0], dtype=torch.float32).reshape(Y.size()[0], 1)
    K = kernel(X.float(), X.float()) + torch.eye(len(X.float()), dtype=float) * sigma**2
    K_inv = torch.linalg.inv(K).float()
    L = -torch.logdet(K.to(torch.float)) - (Y - mean).T@K_inv@(Y - mean)
    return L


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
#     gas_sensor_meta:bochner_ikml_gauss
# 'meta_lr': 0.7607326294063792, 'lam': 0.27271004693357376
    meta_lr = 0.76073
    lam = 0.2727
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
    traindata = GasSensorDataLoader(k_support, k_query, split="train", t=True)
    valdata = GasSensorDataLoader(k_support, k_query, split="valid", t=True)
    testdata = GasSensorDataLoader(k_support, k_query, split="test", t=True)
    traindata_meta = AirQualityDataLoader(k_support, k_query, split="train")
    valdata_meta = AirQualityDataLoader(k_support, k_query, split="valid")
    testdata_meta = AirQualityDataLoader(k_support, k_query, split="test")

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
    model = GaussProcess(np.log(lam), kernel).to(device)
    opt = optim.Adam(model.parameters(), meta_lr)
    print("parameters are {}".format(model.parameters()))

    loss = log_marginal_likelihood

    # Keep best model around
    best_val_iteration = 0
    best_val_mse = np.inf

    # Dump frequencies
    kernel.sample_features(10000)
    sample = kernel.omegas.cpu().detach().numpy()
    with open("omegas-init.npy", "wb") as f:
        np.save(f, sample)

    for iteration in tqdm(range(num_iterations), desc="training"):
        validate = True if iteration % meta_val_every == 0 else False
        # NOTE traindata.sample()
        # NOTE {"train": train_data, "valid": valid_data, "full": full_data}
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
            cur = -loss(model.kernel, X_tr.float(), y_tr.float())
            cur.backward()
            meta_train_error += cur.item()
        
        def _fast_adapt_boch_valid(batch, model, D, device):
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
            # print(type(meta_valid_error))
        
        with futures.ThreadPoolExecutor() as executor:
            for train_batch in train_batches:
                executor.submit(_fast_adapt_boch_train, train_batch, model, loss, D, device)
        # print(meta_train_error)
        # for p in model.parameters():
        #     print(p.grad.data)
        if validate:
            val_batches = [valdata_meta.sample() for _ in range(meta_val_batch_size)]
            with futures.ThreadPoolExecutor() as executor:
                torch.save(model.state_dict(), "model.pth")
                # 保存したモデルを読み込む。
                tmp = GaussProcess(np.log(lam), kernel).to(device)
                tmp.load_state_dict(torch.load("model.pth"))
                for val_batch in val_batches:
                    # タスクを追加する。
                    executor.submit(_fast_adapt_boch_valid, val_batch, tmp, D, device)
            meta_valid_error /= meta_val_batch_size
            result["meta_valid_error"].append(meta_valid_error)
            print("Iteration {}".format(iteration))
            print("meta_valid_error: {}".format(meta_valid_error))
            if meta_valid_error < best_val_mse:
                best_val_iteration = iteration
                best_val_mse = meta_valid_error
                best_state_dict = model.state_dict()

            kernel.sample_features(10000)
            sample = kernel.omegas.cpu().detach().numpy()
            with open("omegas-step{}.npy".format(iteration), "wb") as f:
                np.save(f, sample)

        meta_train_error /= meta_batch_size
        result["meta_train_error"].append(meta_train_error)
        # Average the accumulated gradients and optimize
        
        # for val in model.parameters():
        #     print(val)
        for p in model.parameters():
           if p.grad is not None:
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

    kernel.sample_features(10000)
    sample = kernel.omegas.cpu().detach().numpy()
    with open("omegas-final.npy".format(iteration), "wb") as f:
        np.save(f, sample)
    
    
    # nonlinearity = get_nonlinearity(nonlinearity)
    # def objective(trial):
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

    #     # Load train/validation/test data
    #     traindata = GasSensorDataLoader(k_support, k_query, split="train", t=True)
    #     valdata = GasSensorDataLoader(k_support, k_query, split="valid", t=True)
    #     testdata = GasSensorDataLoader(k_support, k_query, split="test", t=True)
    #     traindata_meta = AirQualityDataLoader(k_support, k_query, split="train")
    #     valdata_meta = AirQualityDataLoader(k_support, k_query, split="test")
    #     testdata_meta = AirQualityDataLoader(k_support, k_query, split="test")

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
    #     model = GaussProcess(np.log(lam), kernel).to(device)
    #     opt = optim.Adam(model.parameters(), meta_lr)
    #     print("parameters are {}".format(model.parameters()))

    #     loss = log_marginal_likelihood

    #     # Keep best model around
    #     best_val_iteration = 0
    #     best_val_mse = np.inf

    #     # Dump frequencies
    #     kernel.sample_features(10000)
    #     sample = kernel.omegas.cpu().detach().numpy()
    #     with open("omegas-init.npy", "wb") as f:
    #         np.save(f, sample)

    #     for iteration in tqdm(range(num_iterations), desc="training"):
    #         validate = True if iteration % meta_val_every == 0 else False
    #         # NOTE traindata.sample()
    #         # NOTE {"train": train_data, "valid": valid_data, "full": full_data}
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
            
    #         def _fast_adapt_boch_valid(batch, model, D, device):
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
    #             # print(type(meta_valid_error))
            
    #         with futures.ThreadPoolExecutor() as executor:
    #             for train_batch in train_batches:
    #                 # executor.submit(_fast_adapt_boch_train, train_batch, model, loss, D, device)
    #                 X_tr, y_tr = train_batch["train"]
    #                 X_tr = X_tr.to(device).float()
    #                 y_tr = y_tr.to(device).float()
    #                 X_val, y_val = train_batch["valid"]
    #                 X_val = X_val.to(device).float()
    #                 y_val = y_val.to(device).float()
    #                 # adapt algorithm
    #                 model.kernel.sample_features(D)
    #                 model.fit(X_tr, y_tr)
    #                 # Predict
    #                 # y_hat = model.predict(X_val).to(torch.float32)
    #                 cur = -loss(model.kernel, X_tr.float(), y_tr.float())
    #                 # print("dtype of datas are {}, {}, {}".format(y_hat.dtype, y_val.dtype, cur.dtype))
    #                 # print(cur.required_grad)
    #                 # opt.zero_grad()
    #                 cur.backward()
    #                 meta_train_error += cur.detach().item()
    #         # print(meta_train_error)
    #         # for p in model.parameters():
    #         #     print(p.grad.data)
    #         if validate:
    #             val_batches = [valdata_meta.sample() for _ in range(meta_val_batch_size)]
    #             with futures.ThreadPoolExecutor() as executor:
    #                 torch.save(model.state_dict(), "model.pth")
    #                 # 保存したモデルを読み込む。
    #                 tmp = GaussProcess(np.log(lam), kernel).to(device)
    #                 tmp.load_state_dict(torch.load("model.pth"))
    #                 for val_batch in val_batches:
    #                     # タスクを追加する。
    #                     # executor.submit(_fast_adapt_boch_valid, val_batch, tmp, D, device)
    #                     X_tr, y_tr = val_batch["train"]
    #                     X_tr = X_tr.to(device).float()
    #                     y_tr = y_tr.to(device).float()
    #                     X_val, y_val = val_batch["valid"]
    #                     X_val = X_val.to(device).float()
    #                     y_val = y_val.to(device).float()
    #                     # print(X_tr.size(), y_tr.size())
    #                     # adapt algorithm
    #                     tmp.kernel.sample_features(D)
    #                     tmp.fit(X_tr, y_tr)
    #                     # Predict
    #                     # print("the size of variation dataset is {}, {}".format(X_val.size(), y_val.size()))
    #                     y_hat = tmp.predict(X_val).to(torch.float32)
    #                     meta_valid_error += torch.norm(y_hat - y_val).to('cpu').detach().numpy().copy()
    #                     # meta_valid_error -= loss(tmp.kernel, X_val, y_val).to('cpu').detach().numpy().copy().item()
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

    #             kernel.sample_features(10000)
    #             sample = kernel.omegas.cpu().detach().numpy()
    #             with open("omegas-step{}.npy".format(iteration), "wb") as f:
    #                 np.save(f, sample)

    #         meta_train_error /= meta_batch_size
    #         result["meta_train_error"].append(meta_train_error)
    #         # Average the accumulated gradients and optimize
            
    #         # for val in model.parameters():
    #         #     print(val)
    #         for p in model.parameters():
    #             if p.grad is not None:
    #                 p.grad.data.mul_(1.0 / meta_batch_size)
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
    #     print("holdout_meta_valid_error: {}".format(meta_valid_error))
    #     print("holdout_meta_test_error: {}".format(meta_test_error))
    #     result["holdout_meta_valid_error"].append(meta_valid_error)
    #     result["holdout_meta_test_error"].append(meta_test_error)

    #     with open("result.pkl", "wb") as f:
    #         pkl.dump(result, f)

    #     # Visualise
    #     fig, ax = visualise_run(result)
    #     plt.tight_layout()
    #     fig.savefig("learning_curves.pdf", bbox_inches="tight")
    #     fig.savefig("learning_curves.png", bbox_inches="tight")

    #     kernel.sample_features(10000)
    #     sample = kernel.omegas.cpu().detach().numpy()
    #     with open("omegas-final.npy".format(iteration), "wb") as f:
    #         np.save(f, sample)
    # study = optuna.create_study()
    # study.optimize(objective, n_trials=100)
    # trial = study.best_trial
    # print('  Value: {}'.format(trial.value))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k_support", type=int, default=20)
    parser.add_argument("--k_query", type=int, default=20)
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
