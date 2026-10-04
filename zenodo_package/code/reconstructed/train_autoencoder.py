"""Denoising autoencoder on the 22 x 779 per-study z-scored mineral-pathway matrix
(I4 + AX-1), with leave-one-astronaut-out (LOAO) cross-validation.

Methods (main_integrated.tex): 779 -> 128 -> 64 -> 16 -> 64 -> 128 -> 779, BatchNorm,
ReLU, dropout 0.3, L2 weight decay 1e-4, Gaussian input noise sigma = 0.1, Adam
lr 1e-3, up to 300 epochs with early stopping (patience 30). Latent features of
each held-out astronaut come from the model that did not see them.

RECONSTRUCTION (Oct 2026) of a lost original; see FIDELITY.md. Retraining gives a
different latent space from the deposited one (data/autoencoder_latent_features.csv),
which remains the input behind the published results.
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from _common import SEED, LF, dirs, io_args, load_meta, zscore_matrix


def block(i, o):
    return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(0.3)]


class DenoisingAE(nn.Module):
    def __init__(self, n_genes=779, latent=16):
        super().__init__()
        self.encoder = nn.Sequential(*block(n_genes, 128), *block(128, 64), nn.Linear(64, latent))
        self.decoder = nn.Sequential(*block(latent, 64), *block(64, 128), nn.Linear(128, n_genes))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train(X, epochs=300, patience=30, batch=8, noise=0.1, seed=SEED):
    """Early stopping monitors the training reconstruction loss (the only data a
    fold may see); the best-epoch weights are restored."""
    torch.manual_seed(seed)
    m = DenoisingAE(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    mse = nn.MSELoss()
    x = torch.tensor(X, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    best, best_ep, best_state, wait = np.inf, 0, None, 0
    for ep in range(1, epochs + 1):
        m.train()
        perm = torch.randperm(len(x), generator=g)
        tot = 0.0
        for i in range(0, len(x), batch):
            xb = x[perm[i:i + batch]]
            if len(xb) < 2:          # BatchNorm needs >= 2 samples
                continue
            recon, _ = m(xb + noise * torch.randn_like(xb))
            loss = mse(recon, xb)
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


def save_latent(Z, meta, path):
    df = pd.DataFrame(Z, columns=LF)
    df.insert(0, "sample_id", meta.sample_id.values)
    for c in ["astronaut_id", "study", "flight_status"]:
        df[c] = meta[c].values
    df.to_csv(path, index=False)


def main():
    a = io_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    data, out = dirs(a)
    meta, _, astr = load_meta(data)
    X = zscore_matrix(data, meta).astype(np.float32)
    print(f"Data: {X.shape[0]} samples x {X.shape[1]} genes; astronauts {sorted(set(astr))}")

    rows, oof = [], np.zeros((len(X), 16))
    for a_id in sorted(set(astr)):
        tr, te = astr != a_id, astr == a_id
        m, loss, ep = train(X[tr])
        oof[te] = encode(m, X[te])
        rows.append({"fold": a_id, "test_astronaut": a_id, "recon_loss": loss, "n_train": int(tr.sum())})
        print(f"Fold test={a_id} (train={tr.sum()}): best recon loss {loss:.6f} (epoch {ep})")
    m, loss, ep = train(X)
    rows.append({"fold": "ALL", "test_astronaut": "ALL", "recon_loss": loss, "n_train": len(X)})
    print(f"Final model: recon loss {loss:.6f} (epoch {ep})")

    pd.DataFrame(rows).to_csv(out / "autoencoder_recon_losses.csv", index=False)
    save_latent(oof, meta, out / "autoencoder_latent_features.csv")
    save_latent(encode(m, X), meta, out / "autoencoder_latent_features_final.csv")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
