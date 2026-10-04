"""NHANES iron-marker x CBC correlations.
Merges CBC (LBXHGB, LBXHCT, LBXRBCSI, LBXMCHSI, LBXMC, LBXMCVSI) with iron
status data (BIOPRO serum iron, FERTIN ferritin, FETIB transferrin saturation
and TIBC) plus demographics; restricts to age 40-60 (all sexes); computes 24
Pearson correlations (4 iron markers x 6 CBC parameters).
Writes nhanes_iron_cbc_correlations.csv and three figures
(dl_iron_reference_panel, dl_iron_cbc_correlation_heatmap,
dl_iron_reference_combined; .png and .svg).

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
from scipy.stats import pearsonr

plt.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"

CYCLES = ["H", "I", "J"]
IRON = [("SERUM IRON", "LBXSIR", "Serum Iron (ug/dL)"),
        ("FERRITIN", "LBXFER", "Ferritin (ng/mL)"),
        ("TRANSFERRIN SATURATION", "LBDPCT", "Transferrin Saturation (%)"),
        ("TIBC", "LBDTIB", "TIBC (ug/dL)")]
CBC = [("HEMOGLOBIN", "LBXHGB"), ("HEMATOCRIT", "LBXHCT"),
       ("RED BLOOD CELL COUNT", "LBXRBCSI"), ("MCH", "LBXMCHSI"),
       ("MCHC", "LBXMC"), ("MCV", "LBXMCVSI")]


def main():
    ap = argparse.ArgumentParser()
    from _common import OUT
    ap.add_argument("--raw-dir", default=str(OUT / "nhanes_raw"))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--fig-dir", default=str(OUT / "figures"))
    a = ap.parse_args()
    raw, data, figs = Path(a.raw_dir), Path(a.out_dir), Path(a.fig_dir)
    data.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    # Load and merge per cycle: CBC + BIOPRO(iron) + FETIB + FERTIN + DEMO(age)
    merged = []
    for c in CYCLES:
        demo = pd.read_sas(raw / f"DEMO_{c}.XPT", format="xport")[["SEQN", "RIDAGEYR"]]
        df = demo
        for stem, cols in [("CBC", [x[1] for x in CBC]),
                           ("BIOPRO", ["LBXSIR"]),
                           ("FETIB", ["LBDPCT", "LBDTIB"]),
                           ("FERTIN", ["LBXFER"])]:
            p = raw / f"{stem}_{c}.XPT"
            if p.exists():
                df = df.merge(pd.read_sas(p, format="xport")[["SEQN"] + cols],
                              on="SEQN", how="left")
        merged.append(df)
    df = pd.concat(merged, ignore_index=True)
    df = df[(df.RIDAGEYR >= 40) & (df.RIDAGEYR <= 60)]
    df.to_csv(data / "nhanes_iron_ref_data.csv", index=False)
    print(f"Reference data (age 40-60): {len(df)} participants")

    # 24 Pearson correlations
    rows = []
    for iron_name, iron_code, _ in IRON:
        for cbc_name, cbc_code in CBC:
            sub = df[[iron_code, cbc_code]].dropna()
            r, p = pearsonr(sub[iron_code], sub[cbc_code])
            rows.append({"iron_marker": iron_name, "cbc_marker": cbc_name,
                         "nhanes_iron_code": iron_code, "nhanes_cbc_code": cbc_code,
                         "n_pairs": len(sub), "pearson_r": r, "p_value": p})
            print(f"{iron_name} vs {cbc_name}: r={r:.3f}, p={p:.2e}, n={len(sub)}")
    pd.DataFrame(rows).to_csv(data / "nhanes_iron_cbc_correlations.csv", index=False)

    # Figure 1: 3-panel reference distributions with P2.5/P97.5/mean lines
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, code, label) in zip(axes, IRON[:3]):
        v = df[code].dropna()
        ax.hist(v, bins=50, color="#3B6E8F", alpha=0.75)
        for x, ls in [(v.mean(), "-"), (np.percentile(v, 2.5), "--"),
                      (np.percentile(v, 97.5), "--")]:
            ax.axvline(x, color="#C0392B", ls=ls, lw=1.2)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(label.split("(")[1].rstrip(")"))
        ax.set_ylabel("Participants")
    fig.suptitle("NHANES iron status reference (age 40-60, all sexes): "
                 "mean (solid), P2.5-P97.5 (dashed)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(figs / f"dl_iron_reference_panel.{ext}", dpi=200)
    plt.close(fig)

    # Figure 2: 4x6 correlation heatmap with significance stars
    piv = pd.DataFrame(rows).pivot(index="iron_marker", columns="cbc_marker",
                                   values="pearson_r").loc[
        [x[0] for x in IRON], [x[0] for x in CBC]]
    stars = pd.DataFrame(rows).pivot(index="iron_marker", columns="cbc_marker",
                                     values="p_value").loc[piv.index, piv.columns]
    annot = piv.apply(lambda col: col.map(lambda v: f"{v:.2f}")).values
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            p = stars.iloc[i, j]
            annot[i, j] += "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.heatmap(piv, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=annot, fmt="", cbar_kws={"label": "Pearson r"})
    ax.set_title("NHANES iron status x CBC correlations (age 40-60)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(figs / f"dl_iron_cbc_correlation_heatmap.{ext}", dpi=200)
    plt.close(fig)

    # Figure 3: combined (KDE densities + heatmap)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    for name, code, label in IRON[:3]:
        v = df[code].dropna()
        sns.kdeplot(v, ax=axes[0], label=label, fill=True, alpha=0.25)
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)
    sns.heatmap(piv, ax=axes[1], cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                cbar_kws={"label": "Pearson r"})
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(figs / f"dl_iron_reference_combined.{ext}", dpi=200)
    plt.close(fig)

    print("Saved nhanes_iron_cbc_correlations.csv and 3 iron figures "
          "(.png/.svg)")


if __name__ == "__main__":
    main()
