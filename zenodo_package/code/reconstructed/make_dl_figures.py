"""UMAP, ROC, JAXA6 concordance and regression figures for the DAE analysis.
Outputs: dl_umap_latent, dl_roc_curves, dl_jaxa6_concordance,
dl_regression_scatter (.png and .svg).

RECONSTRUCTION (Oct 2026) of a lost original; see FIDELITY.md.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from umap import UMAP

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

    # (a) UMAP of the 16-dim latent space
    lat = pd.read_csv(pick("autoencoder_latent_features.csv"))
    lf = [f"LF{i}" for i in range(1, 17)]
    emb = UMAP(n_components=2, random_state=42).fit_transform(lat[lf])
    lat["UMAP1"], lat["UMAP2"] = emb[:, 0], emb[:, 1]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (col, title) in zip(axes, [("flight_status", "Flight status"),
                                       ("study", "Study"),
                                       ("astronaut_id", "Astronaut")]):
        for k, g in lat.groupby(col):
            ax.scatter(g.UMAP1, g.UMAP2, label=str(k), s=45, alpha=0.85)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(figs / f"dl_umap_latent.{ext}", dpi=200)
    plt.close(fig)

    # (b) ROC curves: latent vs raw
    roc = pd.read_csv(pick("dl_classification_roc_data.csv"))
    fig, ax = plt.subplots(figsize=(5, 5))
    for (fs, clf), g in roc.groupby(["feature_space", "classifier"]):
        fpr, tpr, _ = roc_curve(g.true_label, g.predicted_prob)
        auc = roc_auc_score(g.true_label, g.predicted_prob)
        ax.plot(fpr, tpr, label=f"{fs}/{clf} (AUC={auc:.2f})", lw=1.2)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(figs / f"dl_roc_curves.{ext}", dpi=200)
    plt.close(fig)

    # (c) JAXA6 cross-mission concordance
    con = pd.read_csv(pick("jaxa6_concordance_table.csv"))
    fig, ax = plt.subplots(figsize=(5, 5))
    colors = ["#27AE60" if c else "#C0392B" for c in con.concordant]
    ax.scatter(con.i4_logFC, con.jaxa_logFC, c=colors, s=45, alpha=0.85)
    for _, r in con.iterrows():
        if r.Symbol in ("ALAS2", "HFE", "CNNM4"):
            ax.annotate(r.Symbol, (r.i4_logFC, r.jaxa_logFC), fontsize=8)
    lim = con[["i4_logFC", "jaxa_logFC"]].abs().max().max() * 1.15
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("I4 log2FC (R+1 vs pre-flight)")
    ax.set_ylabel("JAXA6 log2FC (flight vs pre)")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(figs / f"dl_jaxa6_concordance.{ext}", dpi=200)
    plt.close(fig)

    # (d) Regression scatter: true vs predicted per target (best model)
    preds = pd.read_csv(pick("dl_regression_predictions.csv"))
    res = pd.read_csv(pick("dl_regression_results.csv"))
    best = res.loc[res.groupby("target").r_pearson.idxmax()]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (_, b) in zip(axes, best.iterrows()):
        g = preds[(preds.target == b.target) & (preds.model == b.model)]
        ax.scatter(g.true_value, g.predicted, s=40, alpha=0.8, color="#3B6E8F")
        lim = g[["true_value", "predicted"]].abs().max().max()
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
        ax.set_title(f"{b.target} ({b.model}, r={b.r_pearson:.2f})", fontsize=10)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(figs / f"dl_regression_scatter.{ext}", dpi=200)
    plt.close(fig)

    print("Saved dl_umap_latent, dl_roc_curves, dl_jaxa6_concordance, "
          "dl_regression_scatter (.png/.svg)")


if __name__ == "__main__":
    main()
