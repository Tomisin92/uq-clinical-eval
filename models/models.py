"""
models.py
---------
Base neural network, MC Dropout, and Bayesian Neural Network (mean-field VI).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Base / MC-Dropout Network
# ─────────────────────────────────────────────────────────────────────────────

class BaseNet(nn.Module):
    """
    Three-hidden-layer feedforward network with optional dropout.
    Shared by deterministic baseline and MC Dropout.
    """
    def __init__(self, input_dim: int, hidden: tuple = (128, 64, 32),
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def mc_dropout_predict(model: BaseNet, x: torch.Tensor,
                       T: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    """
    T stochastic forward passes with dropout kept active.
    Returns (mean, variance) both shape (N,).
    """
    model.train()           # keeps dropout active
    with torch.no_grad():
        preds = torch.stack([model(x) for _ in range(T)], dim=0)  # (T, N)
    model.eval()
    return preds.mean(0), preds.var(0)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Bayesian Neural Network — mean-field variational inference
# ─────────────────────────────────────────────────────────────────────────────

class BayesLinear(nn.Module):
    """
    Linear layer with weight distributions q(w) = N(mu, softplus(rho)^2).
    Uses local reparameterisation for reduced gradient variance.
    Prior: N(0, prior_std^2).
    """
    def __init__(self, in_features: int, out_features: int,
                 prior_std: float = 1.0):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.prior_var    = prior_std ** 2

        # Weight parameters
        self.w_mu  = nn.Parameter(torch.empty(out_features, in_features))
        self.w_rho = nn.Parameter(torch.empty(out_features, in_features))
        # Bias parameters
        self.b_mu  = nn.Parameter(torch.zeros(out_features))
        self.b_rho = nn.Parameter(torch.full((out_features,), -3.0))

        nn.init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        nn.init.constant_(self.w_rho, -3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_sigma = F.softplus(self.w_rho)           # (out, in)
        b_sigma = F.softplus(self.b_rho)           # (out,)

        # Local reparameterisation: sample output activations directly
        # mean = x @ mu^T + b_mu
        # var  = x^2 @ sigma^2^T + b_sigma^2
        out_mu  = F.linear(x, self.w_mu,  self.b_mu)
        out_var = F.linear(x ** 2, w_sigma ** 2, b_sigma ** 2)
        eps     = torch.randn_like(out_mu)
        return out_mu + eps * torch.sqrt(out_var + 1e-8)

    def kl_divergence(self) -> torch.Tensor:
        """KL(q(w) || p(w)) summed over all weight/bias elements."""
        w_sigma = F.softplus(self.w_rho)
        b_sigma = F.softplus(self.b_rho)

        def _kl(mu, sigma):
            return 0.5 * torch.sum(
                (mu ** 2 + sigma ** 2) / self.prior_var
                - torch.log(sigma ** 2 / self.prior_var) - 1
            )

        return _kl(self.w_mu, w_sigma) + _kl(self.b_mu, b_sigma)


class BayesNet(nn.Module):
    """BNN with three Bayesian hidden layers."""
    def __init__(self, input_dim: int, hidden: tuple = (128, 64, 32),
                 prior_std: float = 1.0):
        super().__init__()
        dims = [input_dim] + list(hidden) + [1]
        self.layers = nn.ModuleList([
            BayesLinear(dims[i], dims[i+1], prior_std)
            for i in range(len(dims)-1)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(h) for h in hidden
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (layer, norm) in enumerate(zip(self.layers[:-1], self.norms)):
            x = F.relu(norm(layer(x)))
        return torch.sigmoid(self.layers[-1](x)).squeeze(-1)

    def kl(self) -> torch.Tensor:
        return sum(l.kl_divergence() for l in self.layers)

    def elbo_loss(self, x: torch.Tensor, y: torch.Tensor,
                  n_data: int, beta: float = 1.0) -> torch.Tensor:
        """
        Monte Carlo ELBO:  -E[log p(y|x,w)] + beta * KL(q||p) / N
        beta = 1/M  (M = number of mini-batches) for scale-invariance.
        """
        y_hat     = self.forward(x)
        nll       = F.binary_cross_entropy(y_hat, y, reduction="mean")
        kl_scaled = beta * self.kl() / n_data
        return nll + kl_scaled

    @torch.no_grad()
    def predict(self, x: torch.Tensor, S: int = 30):
        """Posterior predictive mean and variance over S weight samples."""
        self.train()   # sample from q(w) during forward
        preds = torch.stack([self.forward(x) for _ in range(S)], dim=0)
        self.eval()
        return preds.mean(0), preds.var(0)
