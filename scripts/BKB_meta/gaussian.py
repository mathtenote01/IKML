import argparse
import pickle as pkl
import warnings
from collections import OrderedDict
from concurrent import futures
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from implicit_kernel_meta_learning.algorithms import RidgeRegression
from implicit_kernel_meta_learning.data_utils import GasSensorDataLoader
from implicit_kernel_meta_learning.data_utils import AirQualityDataLoader
from implicit_kernel_meta_learning.experiment_utils import set_seed
from implicit_kernel_meta_learning.kernels import GaussianKernel

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


def assert_batch_no_nan(batch):
    X_tr, y_tr = batch["train"]
    X_val, y_val = batch["valid"]
    assert not torch.isnan(X_tr).any(), "X_tr has NaN's \n{}".format(X_tr)
    assert not torch.isnan(X_val).any(), "X_val has NaN's \n{}".format(X_val)
    assert not torch.isnan(y_tr).any(), "y_tr has NaN's \n{}".format(y_tr)
    assert not torch.isnan(y_val).any(), "y_val has NaN's \n{}".format(y_val)


def fast_adapt_ker(batch, model, loss, device):
    # Unpack data
    X_tr, y_tr = batch["train"]
    X_tr = X_tr.to(device).float()
    y_tr = y_tr.to(device).float()
    X_val, y_val = batch["valid"]
    X_val = X_val.to(device).float()
    y_val = y_val.to(device).float()

    assert_batch_no_nan(batch)

    # adapt algorithm
    model.fit(X_tr, y_tr)

    # Predict
    y_hat = model.predict(X_val)
    assert not torch.isnan(y_hat).any(), "y_hat has NaN's \n{}".format(y_hat)
    l = loss(y_val, y_hat)
    assert not torch.isnan(l).any(), "loss has NaN's \n{}".format(batch["full"])
    return l


def main(
    seed,
    k_support,
    k_query,
    num_iterations,
    meta_batch_size,
    meta_val_batch_size,
    meta_val_every,
    holdout_size,
    s2,
    lam,
    meta_lr,
):
    # Bochner kernel output dim
    result = OrderedDict(
        meta_train_error=[],
        meta_valid_error=[],
        holdout_meta_test_error=[],
        holdout_meta_valid_error=[],
        meta_val_every=meta_val_every,
        num_iterations=num_iterations,
        name="Gaussian MKRR",
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

    # Gaussian Kernel
    kernel = GaussianKernel(torch.log(torch.tensor(s2)))
    model = RidgeRegression(np.log(lam), kernel).to(device)
    opt = optim.Adam(model.parameters(), meta_lr)

    loss = nn.MSELoss("mean")

    # Keep best model around
    best_val_iteration = 0
    best_val_mse = np.inf

    for iteration in range(num_iterations):
        validate = True if iteration % meta_val_every == 0 else False

        train_batches = [traindata.sample() for _ in range(meta_batch_size)]
        opt.zero_grad()
        meta_train_error = 0.0
        meta_valid_error = 0.0
        for train_batch in train_batches:
            evaluation_error = fast_adapt_ker(
                batch=train_batch,
                model=model,
                loss=loss,
                device=device,
            )
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
        if validate:
            val_batches = [valdata_meta.sample() for _ in range(meta_val_batch_size)]
            with futures.ThreadPoolExecutor() as executor:
                torch.save(model.state_dict(), "model.pth")
                # ???????????????????????????????????????
                tmp = RidgeRegression(np.log(lam), kernel).to(device)
                tmp.load_state_dict(torch.load("model.pth"))
                for val_batch in val_batches:
                    X_tr, y_tr = val_batch["train"]
                    X_tr = X_tr.to(device).float()
                    y_tr = y_tr.to(device).float()
                    X_val, y_val = val_batch["valid"]
                    X_val = X_val.to(device).float()
                    y_val = y_val.to(device).float()

                    assert_batch_no_nan(val_batch)

                    # adapt algorithm
                    model.fit(X_tr, y_tr)

                    # Predict
                    y_hat = model.predict(X_val)
                    meta_valid_error += torch.norm(y_hat - y_tr).to('cpu').detach().numpy().copy()
            # for val_batch in val_batches:
            #     evaluation_error = fast_adapt_ker(
            #         batch=val_batch,
            #         model=model,
            #         loss=loss,
            #         device=device,
            #     )
            #     meta_valid_error += evaluation_error.item()
            meta_valid_error /= meta_val_batch_size
            result["meta_valid_error"].append(meta_valid_error)

            print("Iteration {}".format(iteration))
            print("meta_valid_error: {}".format(meta_valid_error))
            if meta_valid_error < best_val_mse:
                best_val_iteration = iteration
                best_val_mse = meta_valid_error
                best_state_dict = model.state_dict()

            print("kernel.s2: {}".format(np.exp(kernel.logs2.item())))
            print("lambda: {}".format(np.exp(model.log_lam.item())))

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
    for (valid_batch, test_batch) in zip(valid_batches, test_batches):
        evaluation_error = fast_adapt_ker(
            batch=valid_batch,
            model=model,
            loss=loss,
            device=device,
        )
        meta_valid_error += evaluation_error.item()
        evaluation_error = fast_adapt_ker(
            batch=test_batch,
            model=model,
            loss=loss,
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
    parser.add_argument("--s2", type=float, default=1e4)
    parser.add_argument("--lam", type=float, default=0.001)
    parser.add_argument("--meta_lr", type=float, default=0.001)
    args = parser.parse_args()
    main(**vars(args))
