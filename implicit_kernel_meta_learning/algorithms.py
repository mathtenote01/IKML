import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.optimize as sco
import numpy as np
from tabnanny import verbose

class RidgeRegression(nn.Module):
    def __init__(self, lam, kernel, device=None):
        super(RidgeRegression, self).__init__()
        self.lam = lam
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
        K_nl = self.K + self.lam * n * torch.eye(n).to(self.device)
        # To use solve we need to make sure Y is a float
        # and not an int
        self.alphas, _ = torch.solve(Y.float(), K_nl)
        self.X_tr = X

    def predict(self, X):
        return torch.matmul(self.kernel(X, self.X_tr), self.alphas)

class GaussProcess(nn.Module):
    def __init__(self, lam, kernel, device=None):
        super(GaussProcess, self).__init__()
        self.lam = lam
        self.kernel = kernel
        self.alphas = None
        self.X_tr = None
        self.Y_tr = None
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
        self.K_nl = self.K + self.lam * n * torch.eye(n).to(self.device)
        # To use solve we need to make sure Y is a float
        # and not an int
        # Yの標本平均が入る.
        self.mu_of_Y = torch.mean(input=Y)
        self.alphas, _ = torch.solve(Y.float(), self.K_nl)
        self.X_tr = X
        self.Y_tr = Y

    def predict(self, X):
        self.kernel(X, X)
        if len(X.size()) == 3:
            b, n, d = self.X_tr.size()
            b, m, l = self.Y_tr.size()
            full_mu_of_Y = self.mu_of_Y * torch.ones(n, dtype=torch.float32)
        elif len(X.size()) == 2:
            n, d = self.X_tr.size()
            m, l = self.Y_tr.size()
            full_mu_of_Y = self.mu_of_Y * torch.ones(n, dtype=torch.float32)
        # 事後分布の平均を計算して入れる
        expect = self.mu_of_Y + torch.matmul(torch.matmul(self.kernel(X, self.X_tr), torch.inverse(self.K_nl)), self.Y_tr - full_mu_of_Y)
        # 事後分布の分散を計算して入れる
        variation = self.kernel(X, X) - torch.matmul(torch.matmul(self.kernel(X, self.X_tr), torch.inverse(self.K_nl)), self.kernel(self.X_tr, X))
        return torch.normal(mean=expect, std=torch.sqrt(variation))


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
        C = 10
        if len(X.size()) == 3:
            b, n, d = X.size()
            b, m, l = Y.size()
        elif len(X.size()) == 2:
            n, d = X.size()
            m, l = Y.size()
            
            
        assert (
            n == m
        ), "Tensors need to have same dimension, dimensions are {} and {}".format(n, m)
        
        # NOTE kernel(X, X) = cos(X^{T}omega) x cos(X^{T}omega)
        # NOTE omegaは, ラテントを関数でpush forwardしたもの
        # TODO Kは行列ではないので, 二次計画問題にすることはできない
        # self.K = self.kernel(X, X)
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
        self.K = torch.from_numpy(self.K).clone()
        self.X_tr = torch.from_numpy(X).clone()
        self.Y_tr = torch.from_numpy(Y).clone()
        self.alphas = self.alphas.to(self.device)
        self.K = self.K.to(self.device)
        self.X_tr= self.X_tr.to(self.device)
        self.Y_tr = self.Y_tr.to(self.device)

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
class LearnedBiasRidgeRegression(nn.Module):
    def __init__(self, d, log_lam, device=None):
        super(LearnedBiasRidgeRegression, self).__init__()
        self.log_lam = nn.Parameter(torch.tensor(log_lam))
        self.bias = nn.Parameter(torch.tensor(torch.zeros(d)).reshape(-1, 1))
        self.w = None
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

        C = X.transpose(-2, -1).matmul(X) + n * torch.exp(self.log_lam) * torch.eye(
            d
        ).to(self.device)
        a = X.transpose(-2, -1).matmul(Y) + self.bias
        self.w, _ = torch.solve(a, C)

    def predict(self, X):
        return torch.matmul(X, self.w)


class FeatureMapRidgeRegression(nn.Module):
    """Like RidgeRegression but with an additional feature map phi: X \to Phi

    feature_map is a torch module which is learned together with the rest of the parameters

    TODO: Log is wrong since it can go negative, fix"""

    def __init__(self, lam, kernel, feature_map, normalize_features=False, device=None):
        super(FeatureMapRidgeRegression, self).__init__()
        self.lam = nn.Parameter(torch.tensor(lam))
        self.kernel = kernel
        self.feature_map = feature_map
        self.normalize_features = normalize_features
        self.alphas = None
        self.Phi_tr = None
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def fit(self, X, Y):
        Y = F.one_hot(Y)
        n = X.size()[0]

        # Normalize features
        Phi = self.feature_map(X)  # B x N x D
        if self.normalize_features:
            Phi = F.normalize(Phi, dim=-1)

        self.K = self.kernel(Phi, Phi)
        K_nl = self.K + self.lam * n * torch.eye(n).to(self.device)
        # To use solve we need to make sure nY is a float
        # and not an int
        self.alphas, _ = torch.solve(Y.float(), K_nl)
        self.Phi_tr = Phi

    def predict(self, X):
        return torch.matmul(self.kernel(self.feature_map(X), self.Phi_tr), self.alphas)
