"""Latent-correlation heatmap (DAE vs FT-Transformer latent dimensions) and
per-astronaut AUC figure.
Outputs: dl_latent_correlation_heatmap, dl_per_astronaut_auc (.png and .svg).

RECONSTRUCTION (Oct 2026) of a lost original; see FIDELITY.md.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

plt.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"

from _common import DATA, OUT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA), help="deposited results (fallback)")
    ap.add_argument("--out-dir", default=str(OUT), help="reproduced results (preferred)")
    ap.add_argument("--fig-dir", default=str(OUT / "figures"))
    a = ap.parse_args()
    figs = Path(a.fig_dir)

    def pick(name):
        rep = Path(a.out_dir) / name
        return rep if rep.exists() else Path(a.data_dir) / name
    figs.mkdir(parents=True, exist_ok=True)
    lf = [f"LF{i}" for i in range(1, 17)]

    dae = pd.read_csv(pick("autoencoder_latent_features.csv")).set_index("sample_id")
    tft = pd.read_csv(pick("transformer_latent_features.csv")).set_index("sample_id")
    shared = dae.index.intersection(tft.index)

    # (a) 16x16 Pearson correlation between DAE and Transformer latent dims
    corr = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            corr[i, j] = np.corrcoef(dae.loc[shared, lf[i]],
                                     tft.loc[shared, lf[j]])[0, 1]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                xticklabels=lf, yticklabels=lf,
                cbar_kws={"label": "Pearson r"})
    ax.set_xlabel("FT-Transformer latent dim")
    ax.set_ylabel("DAE latent dim")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(figs / f"dl_latent_correlation_heatmap.{ext}", dpi=200)
    plt.close(fig)

    # (b) Per-astronaut AUC for the best feature space
    roc = pd.read_csv(pick("dl_extended_roc_data.csv"))
    auc_by = roc.groupby(["feature_space", "classifier"]).apply(
        lambda g: roc_auc_score(g.true_label, g.predicted_prob),
        include_groups=False)
    fs, clf = auc_by.idxmax()
    g = roc[(roc.feature_space == fs) & (roc.classifier == clf)]
    per = g.groupby("astronaut").apply(
        lambda x: roc_auc_score(x.true_label, x.predicted_prob)
        if x.true_label.nunique() > 1 else np.nan,
        include_groups=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    per.plot.bar(ax=ax, color="#0279EE", alpha=0.9)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("AUC")
    ax.set_xlabel("Astronaut (LOAO held-out fold)")
    ax.set_title(f"{fs}/{clf}", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(figs / f"dl_per_astronaut_auc.{ext}", dpi=200)
    plt.close(fig)

    print("Saved dl_latent_correlation_heatmap and dl_per_astronaut_auc "
          "(.png/.svg)")


if __name__ == "__main__":
    main()
