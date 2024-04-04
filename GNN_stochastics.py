import matplotlib.pyplot as plt
from tqdm import trange
import os
import scipy.io
from sklearn import metrics
from matplotlib import rc
import matplotlib
import numpy as np
import torch
from torch import nn
from sklearn import datasets
import matplotlib.pyplot as plt
from scipy import stats

# https://www.ritchievink.com/blog/2019/09/16/variational-inference-from-scratch/


def load_dataset(n=150, n_tst=150):

    w0 = 0.125
    b0 = 5.
    x_range = [-20, 60]

    np.random.seed(43)

    def s(x):
        g = (x - x_range[0]) / (x_range[1] - x_range[0])
        return 3 * (0.25 + g**2.)

    x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
    eps = np.random.randn(n) * s(x)
    y = (w0 * x * (1. + np.sin(x)) + b0) + eps
    y = (y - y.mean()) / y.std()
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    return y[:, None], x[:, None]

class MaximumLikelihood(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
    def forward(self, x):
        return self.out(x)

class VI(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_mu = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def reparameterize(self, mu, log_var):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        return self.reparameterize(mu, log_var), mu, log_var

def ll_gaussian(y, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * np.pi * sigma ** 2) - (1 / (2 * sigma ** 2)) * (y - mu) ** 2

def elbo(y_pred, y, mu, log_var):

    # Log(P(D|Z) + Log(P(Z)) - Log(Q(Z)), where Z are the latent variables

    # likelihood of observing y given Z Variational mu and sigma
    likelihood = ll_gaussian(y, mu, log_var)

    # prior probability of y_pred  epsilon N(0,1)
    log_prior = ll_gaussian(y_pred, 0, torch.log(torch.tensor(1.)))

    # variational probability of y_pred
    log_p_q = ll_gaussian(y_pred, mu, log_var)

    # by taking the mean we approximate the expectation
    return (likelihood + log_prior - log_p_q).mean()

def det_loss(y_pred, y, mu, log_var):
    return -elbo(y_pred, y, mu, log_var)

def det_loss_KL(y_pred, y, mu, log_var):
    reconstruction_error = (0.5 * (y - y_pred)**2).sum()
    kl_divergence = (-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp()))

    return (reconstruction_error + kl_divergence).sum()

class EM:
    def __init__(self, k):
        self.k = k
        self.mu = None
        self.std = np.ones(k)
        self.w_ij = None
        self.phi = np.ones(k) / k

    def expectation_step(self, x):
        for z_i in range(self.k):
            self.w_ij[z_i] = stats.norm(self.mu[z_i], self.std[z_i]).pdf(x) * self.phi[z_i]
        self.w_ij /= self.w_ij.sum(0) # normalize zo that marginalizing z would lead to p = 1


    def maximization_step(self, x):
        self.phi = self.w_ij.mean(1)
        self.std = ((self.w_ij * (x - self.mu[:, None])**2).sum(1) / self.w_ij.sum(1))**0.5
        self.mu = (self.w_ij * x).sum(1) / self.w_ij.sum(1)

    def fit(self, x):
        self.mu = np.random.uniform(x.min(), x.max(), size=self.k)
        self.w_ij = np.zeros((self.k, x.shape[0]))

        last_mu = np.ones(self.k) * np.inf
        while ~np.all(np.isclose(self.mu, last_mu)):
            last_mu = self.mu
            self.expectation_step(x)
            self.maximization_step(x)








if __name__ == '__main__':


    w0 = 0.125
    b0 = 5.
    x_range = [-20, 60]


    y, x = load_dataset()

    X = torch.tensor(x, dtype=torch.float)
    Y = torch.tensor(y, dtype=torch.float)

    epochs = 200
    m = MaximumLikelihood()
    optim = torch.optim.Adam(m.parameters(), lr=0.01)

    for epoch in range(epochs):
        optim.zero_grad()
        y_pred = m(X)
        loss = (0.5 * (y_pred - Y) ** 2).mean()
        loss.backward()
        optim.step()


    epochs = 1500

    m = VI()
    optim = torch.optim.Adam(m.parameters(), lr=0.005)

    for epoch in range(epochs):
        optim.zero_grad()
        y_pred, mu, log_var = m(X)
        loss = det_loss(y_pred, Y, mu, log_var)
        loss.backward()
        optim.step()

    with torch.no_grad():
        y_pred = torch.cat([m(X)[0] for _ in range(1000)], dim=1)

    # Get some quantiles
    q1, mu, q2 = np.quantile(y_pred, [0.05, 0.5, 0.95], axis=1)

    matplotlib.use("Qt5Agg")
    plt.figure(figsize=(16, 6))
    plt.scatter(X, Y)
    plt.plot(X, mu)
    plt.fill_between(X.flatten(), q1, q2, alpha=0.2)





    np.random.seed(654)
    # Draw samples from two Gaussian w.p. z_i ~ Bernoulli(phi)
    generative_m = np.array([stats.norm(-5, 2), stats.norm(5, 2)])
    z_i = stats.bernoulli(0.85).rvs(100)
    x_i = np.array([g.rvs() for g in generative_m[z_i]])

    t_i = np.linspace(0, 100, 100)
    y_i = 2*t_i + x_i

    X = torch.tensor(t_i, dtype=torch.float)
    X = X[:,None]
    Y = torch.tensor(y_i, dtype=torch.float)
    Y = Y[:,None]

    epochs = 200
    m = MaximumLikelihood()
    optim = torch.optim.Adam(m.parameters(), lr=0.01)

    for epoch in range(epochs):
        optim.zero_grad()
        y_pred = m(X)
        loss = (0.5 * (y_pred - Y) ** 2).mean()
        loss.backward()
        optim.step()

    plt.figure(figsize=(6, 6))
    plt.scatter(t_i, y_i, c=np.array(['C0', 'C1'])[z_i])
    plt.plot(t_i, m(X).detach().numpy(), color='black', lw=1, ls='-.')

    # plot generated data and the latent distributions
    x = np.linspace(-12, 12, 150)
    plt.figure(figsize=(16, 6))
    plt.plot(x, generative_m[0].pdf(x))
    plt.plot(x, generative_m[1].pdf(x))
    plt.plot(x, generative_m[0].pdf(x) + generative_m[1].pdf(x), lw=1, ls='-.', color='black')
    plt.fill_betweenx(generative_m[0].pdf(x), x, alpha=0.1)
    plt.fill_betweenx(generative_m[1].pdf(x), x, alpha=0.1)
    plt.vlines(x_i, 0, 0.01, color=np.array(['C0', 'C1'])[z_i])

    m = EM(2)
    m.fit(x_i)

    fitted_m = [stats.norm(mu, std) for mu, std in zip(m.mu, m.std)]

    plt.figure(figsize=(16, 6))
    plt.vlines(x_i, 0, 0.01, color=np.array(['C0', 'C1'])[z_i])
    plt.plot(x, fitted_m[0].pdf(x))
    plt.plot(x, fitted_m[1].pdf(x))
    plt.plot(x, generative_m[0].pdf(x), color='black', lw=1, ls='-.')
    plt.plot(x, generative_m[1].pdf(x), color='black', lw=1, ls='-.')

    m = EM(2)
    residual = y_i - y_pred.squeeze().detach().cpu().numpy()
    m.fit(residual)

    fitted_m = [stats.norm(mu, std) for mu, std in zip(m.mu, m.std)]

    plt.figure(figsize=(16, 6))
    plt.vlines(x_i, 0, 0.01, color=np.array(['C0', 'C1'])[z_i])
    plt.plot(x, fitted_m[0].pdf(x))
    plt.plot(x, fitted_m[1].pdf(x))
    plt.plot(x, generative_m[0].pdf(x), color='black', lw=1, ls='-.')
    plt.plot(x, generative_m[1].pdf(x), color='black', lw=1, ls='-.')
