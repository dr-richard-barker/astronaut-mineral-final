"""I4 vs JAXA6 (OSD-530) cross-mission validation of the mineral-pathway signature.

Inputs
  data/mineral_pathway_de_combined.csv   I4 DE (RNA-seq, I4-FP1, p < 0.05; 20 genes)
  data/ml_selected_features_v2.csv       ML stability-selected genes
  --jaxa-dge TABLE                       JAXA6 flight-vs-pre DGE with columns
                                         Symbol, logFC, adj-P-Val (CSV/TSV/XLSX)

Statistics, as logged by the original run (execution_trace cell 47)
  sign concordance with a one-sided binomial test; Pearson and Spearman on logFC;
  Fisher's exact test for I4-DE vs JAXA6-DE (adj-P < 0.05) over the union of symbols;
  Stouffer meta-analysis weighted by sqrt(n) (I4 n = 4, JAXA6 n = 6), z = sign(logFC) *
  Phi^-1(1 - p), restricted to genes whose JAXA6 adj-P is below 1, then BH-FDR.

SOURCE OF THE JAXA6 TABLE: the original read a 26,844-gene DGE table (columns Row-names,
Symbol, logFC, adj-P-Val, data) that the manuscript calls "the associated data
repository". It is not among the OSD-530 files on OSDR (checked Oct 2026: the EDGE
pairwise tables there do not reproduce its values), so this script cannot fetch it.
The deposited data/jaxa6_*.csv files are the original outputs.

RECONSTRUCTION (Oct 2026) of a lost original; see FIDELITY.md.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, norm, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

N_I4, N_JAXA = 4, 6


def read_table(path):
    p = Path(path)
    if p.suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    return pd.read_csv(p, sep="\t" if p.suffix in (".tsv", ".txt") else ",")


def analyse(i4, jaxa, ml_genes):
    """i4: Symbol, i4_logFC, i4_p.  jaxa: Symbol, jaxa_logFC, jaxa_adjP.  Returns 3 tables."""
    jaxa = jaxa.dropna(subset=["Symbol"]).drop_duplicates("Symbol")
    jaxa_de = set(jaxa.Symbol[jaxa.jaxa_adjP < 0.05])
    i4_genes = set(i4.Symbol)

    con = i4.merge(jaxa, on="Symbol").dropna(subset=["i4_logFC", "jaxa_logFC"])
    con["i4_sign"], con["jaxa_sign"] = np.sign(con.i4_logFC), np.sign(con.jaxa_logFC)
    con["concordant"] = con.i4_sign == con.jaxa_sign
    n, k = len(con), int(con.concordant.sum())
    pr, sr = pearsonr(con.i4_logFC, con.jaxa_logFC), spearmanr(con.i4_logFC, con.jaxa_logFC)

    universe = set(jaxa.Symbol) | i4_genes
    both = len(i4_genes & jaxa_de)
    table = [[both, len(i4_genes) - both], [len(jaxa_de) - both, len(universe) - len(i4_genes | jaxa_de)]]
    odds, fisher_p = fisher_exact(table)

    st = con[(con.jaxa_adjP < 1) & con.i4_p.notna()].copy()
    st["z_i4"] = np.sign(st.i4_logFC) * norm.isf(st.i4_p)
    st["z_jaxa"] = np.sign(st.jaxa_logFC) * norm.isf(st.jaxa_adjP)
    st["z_combined"] = (np.sqrt(N_I4) * st.z_i4 + np.sqrt(N_JAXA) * st.z_jaxa) / np.sqrt(N_I4 + N_JAXA)
    st["p_combined"] = 2 * norm.sf(np.abs(st.z_combined))
    st["fdr_combined"] = multipletests(st.p_combined, method="fdr_bh")[1]
    st = st.rename(columns={"jaxa_adjP": "jaxa_p"}).sort_values("p_combined")[
        ["Symbol", "i4_p", "jaxa_p", "z_i4", "z_jaxa", "z_combined", "p_combined", "fdr_combined"]]

    val = pd.DataFrame({"metric": [
        "n_i4_de_genes", "n_jaxa6_de_genes", "n_overlap_genes", "n_both_de", "sign_concordance_rate",
        "sign_concordance_p", "logFC_pearson_r", "logFC_pearson_p", "logFC_spearman_rho",
        "logFC_spearman_p", "fisher_odds_ratio", "fisher_p", "n_meta_significant_fdr05",
        "n_ml_genes_in_jaxa_de"], "value": [
        len(i4_genes), len(jaxa_de), n, both, k / n, binomtest(k, n, 0.5, alternative="greater").pvalue,
        pr[0], pr[1], sr[0], sr[1], odds, fisher_p, int((st.fdr_combined < 0.05).sum()),
        len(set(ml_genes) & jaxa_de)]})
    cols = ["Symbol", "i4_logFC", "jaxa_logFC", "jaxa_adjP", "i4_sign", "jaxa_sign", "concordant"]
    return con[cols], st, val


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    from _common import DATA, OUT
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--jaxa-dge", required=True, help="JAXA6 DGE table: Symbol, logFC, adj-P-Val")
    a = ap.parse_args()
    data, out = Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    de = pd.read_csv(data / "mineral_pathway_de_combined.csv")
    i4 = de[(de.omics == "RNA-seq") & (de.comparison == "I4-FP1")].rename(
        columns={"gene_symbol": "Symbol", "log2FC": "i4_logFC", "pvalue": "i4_p"})[["Symbol", "i4_logFC", "i4_p"]]
    jx = read_table(a.jaxa_dge).rename(columns={"logFC": "jaxa_logFC", "adj-P-Val": "jaxa_adjP"})
    ml = pd.read_csv(data / "ml_selected_features_v2.csv").gene

    con, st, val = analyse(i4, jx[["Symbol", "jaxa_logFC", "jaxa_adjP"]], ml)
    con.to_csv(out / "jaxa6_concordance_table.csv", index=False)
    st.to_csv(out / "jaxa6_stouffer_meta.csv", index=False)
    val.to_csv(out / "jaxa6_validation_results.csv", index=False)
    print(val.to_string(index=False))


if __name__ == "__main__":
    main()
