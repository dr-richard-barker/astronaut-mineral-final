"""Lightweight FT-Transformer on the same 22 x 779 z-scored matrix, LOAO cross-validation.

Methods: each gene is a token = value embedding (Linear 1 -> 16) + learned gene-identity
embedding (779 x 16); a [CLS] token; one pre-LN transformer encoder layer (4 heads,
FFN 32, GELU, dropout 0.5); the CLS output gives the 16-dim latent; decoder
16 -> 128 -> 779 trained on reconstruction (MSE). Adam lr 1e-3, weight decay 1e-4,
early stopping patience 30; the original run log shows best epochs <= 150, so
training is capped at 150 epochs.

This layout has exactly 117,707 parameters, the count printed by the original run.
(The documented parts sum to 117,403; adding a final LayerNorm (32) and a 16 -> 16
latent projection (272) reaches 117,707. Those two layers are inferred from the
parameter count, not stated in the Methods.)

RECONSTRUCTION (Oct 2026) of a lost original; see FIDELITY.md. The deposited
data/transformer_latent_features.csv remains the input behind the published results.
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from _common import SEED, dirs, io_args, load_meta, zscore_matrix
from train_autoencoder import save_latent

N_PARAMS_ORIGINAL = 117_707


class FTTransformer(nn.Module):
    def __init__(self, n_genes=779, d=16, latent=16):
        super().__init__()
        self.value = nn.Linear(1, d)
        self.gene = nn.Embedding(n_genes, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.encoder = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=32, dropout=0.5,
                                                  activation="gelu", norm_first=True, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.to_latent = nn.Linear(d, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, n_genes))
        self.register_buffer("ids", torch.arange(n_genes))

    def forward(self, x):                                    # x: (B, G)
        tok = self.value(x.unsqueeze(-1)) + self.gene(self.ids)
        h = self.encoder(torch.cat([self.cls.expand(len(x), -1, -1), tok], 1))
        z = self.to_latent(self.norm(h[:, 0]))
        return self.decoder(z), z


def train(X, epochs=150, patience=30, batch=8, seed=SEED):
    torch.manual_seed(seed)
    m = FTTransformer(X.shape[1])
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
            loss = nn.functional.mse_loss(m(xb)[0], xb)
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
    a = io_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    data, out = dirs(a)
    meta, _, astr = load_meta(data)
    X = zscore_matrix(data, meta).astype(np.float32)
    n = sum(p.numel() for p in FTTransformer(X.shape[1]).parameters())
    print(f"FT-Transformer parameters: {n:,} (original run: {N_PARAMS_ORIGINAL:,})")
    assert n == N_PARAMS_ORIGINAL

    rows, oof = [], np.zeros((len(X), 16))
    for a_id in sorted(set(astr)):
        tr, te = astr != a_id, astr == a_id
        m, loss, ep = train(X[tr])
        oof[te] = encode(m, X[te])
        rows.append({"fold": a_id, "test_astronaut": a_id, "recon_loss": loss, "n_train": int(tr.sum())})
        print(f"Fold test={a_id}: best recon loss {loss:.6f} (epoch {ep})")
    m, loss, ep = train(X)
    rows.append({"fold": "ALL", "test_astronaut": "ALL", "recon_loss": loss, "n_train": len(X)})
    print(f"Final model: loss {loss:.6f} (epoch {ep})")

    pd.DataFrame(rows).to_csv(out / "transformer_recon_losses.csv", index=False)
    save_latent(oof, meta, out / "transformer_latent_features.csv")
    save_latent(encode(m, X), meta, out / "transformer_latent_features_final.csv")


if __name__ == "__main__":
    main()
