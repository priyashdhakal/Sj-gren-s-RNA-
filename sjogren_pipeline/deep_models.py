"""Deep-learning classifier with a scikit-learn interface (works inside Pipeline / GridSearchCV).

``SupervisedAutoencoder``: encoder -> latent code -> {classifier head, decoder}. The loss is
BCE(classification) + ``recon_weight`` * MSE(reconstruction), trained full-batch with Adam and
weight decay; the reconstruction term regularises the tiny-n / high-p problem.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

try:  # torch is only needed for the deep model
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


class _Net(nn.Module if nn is not None else object):
    def __init__(self, n_features: int, hidden: int, latent: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, latent), nn.ReLU())
        self.classifier = nn.Linear(latent, 1)
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, n_features))

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z).squeeze(-1), self.decoder(z)

    def logit(self, x):
        return self.classifier(self.encoder(x)).squeeze(-1)


class SupervisedAutoencoder(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden: int = 64, latent: int = 8, dropout: float = 0.2, weight_decay: float = 1e-3,
                 recon_weight: float = 0.1, epochs: int = 150, lr: float = 1e-3, seed: int = 0):
        self.hidden = hidden
        self.latent = latent
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.recon_weight = recon_weight
        self.epochs = epochs
        self.lr = lr
        self.seed = seed

    # ---- training ----------------------------------------------------------
    def fit(self, X, y):
        if torch is None:
            raise ImportError("torch is required for SupervisedAutoencoder")
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        yb = (y == self.classes_[1]).astype(np.float32)
        net = _Net(X.shape[1], self.hidden, self.latent, self.dropout)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pos_w = torch.tensor(max(1.0, (yb == 0).sum() / max((yb == 1).sum(), 1)), dtype=torch.float32)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        mse = nn.MSELoss()
        xt, yt = torch.from_numpy(X), torch.from_numpy(yb)
        net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            logit, rec = net(xt)
            loss = bce(logit, yt) + self.recon_weight * mse(rec, xt)
            loss.backward()
            opt.step()
        net.eval()
        self.net_ = net
        self.n_features_in_ = X.shape[1]
        return self

    # ---- inference -----------------------------------------------------------
    def decision_function(self, X):
        with torch.no_grad():
            return self.net_.logit(torch.from_numpy(np.asarray(X, dtype=np.float32))).numpy()

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
