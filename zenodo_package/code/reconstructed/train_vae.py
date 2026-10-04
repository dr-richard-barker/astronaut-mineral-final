"""beta-VAE on the same 22 x 779 z-scored matrix, LOAO cross-validation.

Methods: 779 -> 128 -> 64 -> (mu, logvar; 16) -> 64 -> 128 -> 779, reparameterisation
trick, loss = MSE + beta * KL with beta = 0.5, dropout 0.3, L2 weight decay 1e-4,
Adam lr 1e-3, up to 300 epochs, early stopping patience 30. The latent feature is mu.
The Methods do not give the KL reduction; here both terms are per-element means.

RECONSTRUCTION (Oct 2026) of a lost original; see FIDELITY.md. The deposited
data/vae_latent_features.csv remains the input behind the published results.
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from _common import SEED, dirs, io_args, load_meta, zscore_matrix
from train_autoencoder import save_latent


class VAE(nn.Module):
    def __init__(self, n_genes=779, latent=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_genes, 128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3))
        self.mu, self.logvar = nn.Linear(64, latent), nn.Linear(64, latent)
        self.dec = nn.Sequential(nn.Linear(latent, 64), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, n_genes))

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        return self.dec(z), mu, logvar


def vae_loss(m, x, beta):
    recon, mu, logvar = m(x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return nn.functional.mse_loss(recon, x) + beta * kl


def train(X, beta=0.5, epochs=300, patience=30, batch=8, seed=SEED):
    torch.manual_seed(seed)
    m = VAE(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    x = torch.tensor(X, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    best, best_ep, best_state, wait = np.inf, 0, None, 0
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(len(x), generator=g)
        tot = 0.0
        for i in range(0, len(x), batch):
            xb = x[perm[i:i + batch]]
            loss = vae_loss(m, xb, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        ep_loss = tot / len(x)
        if ep_loss < best:
            best, best_ep, wait = ep_loss, ep, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m, best, best_ep


def encode(m, X):
    with torch.no_grad():
        return m(torch.tensor(X, dtype=torch.float32))[1].numpy()


def main():
    ap = io_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--beta", type=float, default=0.5)
    a = ap.parse_args()
    data, out = dirs(a)
    meta, _, astr = load_meta(data)
    X = zscore_matrix(data, meta).astype(np.float32)

    rows, oof = [], np.zeros((len(X), 16))
    for a_id in sorted(set(astr)):
        tr, te = astr != a_id, astr == a_id
        m, loss, ep = train(X[tr], a.beta)
        oof[te] = encode(m, X[te])
        rows.append({"fold": a_id, "test_astronaut": a_id, "vae_loss": loss, "n_train": int(tr.sum())})
        print(f"Fold test={a_id}: best VAE loss {loss:.6f} (epoch {ep})")
    m, loss, ep = train(X, a.beta)
    rows.append({"fold": "ALL", "test_astronaut": "ALL", "vae_loss": loss, "n_train": len(X)})
    print(f"Final VAE: loss {loss:.6f} (epoch {ep})")

    pd.DataFrame(rows).to_csv(out / "vae_recon_losses.csv", index=False)
    save_latent(oof, meta, out / "vae_latent_features.csv")
    save_latent(encode(m, X), meta, out / "vae_latent_features_final.csv")


if __name__ == "__main__":
    main()
