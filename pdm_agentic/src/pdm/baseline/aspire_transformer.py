"""ASPIRE-style baseline: the Prediction Agent's Transformer, re-implemented for multi-alarm early warning.

Settings follow our Phase I ASPIRE reproduction: d_model 64, 4 heads, 2 encoder layers,
feed-forward 128, dropout 0.2. Inputs are z-scored 1-minute sequences (WINDOW x channels)
using training-set statistics, as in the baseline. Output: one logit per alarm (multi-label).
This is our re-implementation of the prediction model only, not the full ASPIRE system.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except Exception:  # torch optional so the rest of the framework runs without it
    TORCH_OK = False

from ..agents.early_warning import tune_threshold


if TORCH_OK:
    class _Net(nn.Module):
        def __init__(self, n_in: int, n_out: int, window: int, d_model=64, heads=4, layers=2, ff=128, dropout=0.2):
            super().__init__()
            self.proj = nn.Linear(n_in, d_model)
            self.pos = nn.Parameter(torch.zeros(1, window, d_model))
            enc = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc, layers)
            self.head = nn.Linear(d_model, n_out)

        def forward(self, x):
            z = self.encoder(self.proj(x) + self.pos)
            return self.head(z.mean(dim=1))


class AspireTransformer:
    def __init__(self, codes: list[str], epochs: int = 12, batch: int = 256, lr: float = 1e-3, seed: int = 0):
        if not TORCH_OK:
            raise ImportError("PyTorch is required for the ASPIRE-style baseline: pip install torch")
        self.codes, self.epochs, self.batch, self.lr, self.seed = codes, epochs, batch, lr, seed
        self.thresholds: dict[str, float] = {}

    def _norm(self, X):
        return np.nan_to_num((X - self.mu) / self.sd).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, m: np.ndarray, val_frac: float = 0.2) -> "AspireTransformer":
        torch.manual_seed(self.seed); np.random.seed(self.seed)
        n = len(X); cut = int(n * (1 - val_frac))
        self.mu = X[:cut].reshape(-1, X.shape[2]).mean(0)
        self.sd = X[:cut].reshape(-1, X.shape[2]).std(0) + 1e-6
        Xn = self._norm(X)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = _Net(X.shape[2], y.shape[1], X.shape[1]).to(self.device)
        pos = (y[:cut] * m[:cut]).sum(0); neg = ((1 - y[:cut]) * m[:cut]).sum(0)
        pw = torch.tensor(np.clip(neg / np.maximum(pos, 1), 1, 200), dtype=torch.float32, device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        Xt, yt, mt = (torch.tensor(a, device=self.device) for a in (Xn[:cut], y[:cut], m[:cut]))
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(cut, device=self.device)
            for i in range(0, cut, self.batch):
                b = perm[i:i + self.batch]
                opt.zero_grad()
                l = (loss_fn(self.net(Xt[b]), yt[b]) * mt[b]).sum() / mt[b].sum().clamp(min=1)
                l.backward(); opt.step()
        pv = self._proba(Xn[cut:])
        for j, c in enumerate(self.codes):
            ok = m[cut:, j] > 0
            self.thresholds[c] = tune_threshold(y[cut:][ok, j], pv[ok, j]) if ok.any() else 0.5
        return self

    def _proba(self, Xn):
        self.net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xn), 2048):
                out.append(torch.sigmoid(self.net(torch.tensor(Xn[i:i + 2048], device=self.device))).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.codes)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._proba(self._norm(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        th = np.array([self.thresholds[c] for c in self.codes])
        return (p >= th[None, :]).astype(int)
