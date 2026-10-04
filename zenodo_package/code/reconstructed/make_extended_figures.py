"""AUC comparison, UMAP comparison and ROC overlay figures for the extended
architecture analysis (DAE / VAE / Transformer / Raw_779 / Multiomics_15 /
ensembles).
Outputs: dl_extended_auc_comparison, dl_extended_umap_comparison,
dl_extended_roc_overlay (.png and .svg).

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

    res = pd.read_csv(pick("dl_extended_classification_results.csv"))
    res["label"] = res.feature_space + "/" + res.classifier

    # (a) AUC comparison bar chart with significance stars
    def star(p):
        return "***" if p < 0.01 else "**" if p < 0.025 else "*" if p < 0.05 else ""
    res["sig"] = res.p_value.apply(star)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = sns.color_palette("colorblind", res.label.nunique())
    ax.bar(res.label, res.pooled_auc, color=colors, alpha=0.9)
    for x, (_, r) in enumerate(res.iterrows()):
        ax.text(x, r.pooled_auc + 0.01, f"{r.pooled_auc:.2f}{r.sig}",
                ha="center", fontsize=8)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("Pooled AUC (LOAO)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(figs / f"dl_extended_auc_comparison.{ext}", dpi=200)
    plt.close(fig)

    # (b) UMAP comparison across feature spaces
    from eval_all_architectures import spaces as load_spaces
    from _common import load_meta
    meta, _, _ = load_meta(Path(a.data_dir))
    a.use_deposited_latents = False
    spaces = load_spaces(Path(a.data_dir), meta, a)
    fig, axes = plt.subplots(1, len(spaces), figsize=(4 * len(spaces), 4))
    for ax, (name, X) in zip(axes, spaces.items()):
        emb = UMAP(n_components=2, random_state=42).fit_transform(X)
        for k, c in [("pre", "#3B6E8F"), ("post", "#FF9400")]:
            m = (meta.flight_status == k).values
            ax.scatter(emb[m, 0], emb[m, 1], label=k, s=25, alpha=0.8, c=c)
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(figs / f"dl_extended_umap_comparison.{ext}", dpi=200)
    plt.close(fig)

    # (c) ROC overlay: best classifier per feature space
    roc = pd.read_csv(pick("dl_extended_roc_data.csv"))
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    aucs = {k: roc_auc_score(g.true_label, g.predicted_prob)
            for k, g in roc.groupby(["feature_space", "classifier"])}
    for fs in roc.feature_space.unique():
        clf = max((c for f, c in aucs if f == fs), key=lambda c: aucs[(fs, c)])
        g = roc[(roc.feature_space == fs) & (roc.classifier == clf)]
        fpr, tpr, _ = roc_curve(g.true_label, g.predicted_prob)
        ax.plot(fpr, tpr, label=f"{fs}/{clf} ({aucs[(fs, clf)]:.2f})", lw=1.1)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(figs / f"dl_extended_roc_overlay.{ext}", dpi=200)
    plt.close(fig)

    # (d) Best |r| per feature space and target (mineral regression, I4 only)
    reg = pd.read_csv(pick("dl_extended_regression_results.csv"))
    best = reg.assign(abs_r=reg.r_pearson.abs()).groupby(["target", "feature_space"]).abs_r.max().unstack()
    order = [s for s in ["DAE", "VAE", "Transformer", "Raw_779", "Multiomics_15"] if s in best.columns]
    best = best.loc[[t for t in ["CALCIUM", "POTASSIUM", "SODIUM", "HEMOGLOBIN"] if t in best.index], order]
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.8 / len(order)
    for i, s in enumerate(order):
        ax.bar(np.arange(len(best)) + (i - (len(order) - 1) / 2) * w, best[s], w, label=s)
    ax.set_xticks(range(len(best)))
    ax.set_xticklabels([t.title() for t in best.index])
    ax.set_ylabel("Best |Pearson r| (LOAO)")
    ax.set_title("Regression: mineral prediction across architectures (I4 only)")
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(figs / f"dl_extended_regression_comparison.{ext}", dpi=200)
    plt.close(fig)

    print("Saved dl_extended_auc_comparison, dl_extended_umap_comparison, "
          "dl_extended_roc_overlay, dl_extended_regression_comparison (.png/.svg)")


if __name__ == "__main__":
    main()
