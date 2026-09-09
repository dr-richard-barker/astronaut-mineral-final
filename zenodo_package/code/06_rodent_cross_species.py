#!/usr/bin/env python3
"""
06_rodent_cross_species.py — Process rodent RNA-seq data and compute
cross-species concordance with astronaut mineral-pathway DE genes.
"""
import pandas as pd, numpy as np, os, sys
from scipy import stats
import mygene

PROC = "/mnt/results/data"
RAW = "/mnt/shared-workspace/shared/osdr_raw"

# ── Load OSD-47 normalized counts ──
print("Loading OSD-47 rodent liver normalized counts...")
counts_file = f"{RAW}/OSD-47_GLDS-47_rna_seq_Normalized_Counts_GLbulkRNAseq.csv"
rodent_counts = pd.read_csv(counts_file, index_col=0)
print(f"  Shape: {rodent_counts.shape}")

# Groups: BSL (basal), FLT (flight), GC (ground control)
flt_cols = [c for c in rodent_counts.columns if "FLT" in c]
gc_cols = [c for c in rodent_counts.columns if "GC" in c]
print(f"  FLT: {flt_cols}")
print(f"  GC: {gc_cols}")

# ── Flight vs Ground comparison ──
print("\nComputing FLT vs GC differential expression...")
flt_data = rodent_counts[flt_cols].values
gc_data = rodent_counts[gc_cols].values

pseudo = 1.0
flt_mean = np.log2(flt_data + pseudo).mean(axis=1)
gc_mean = np.log2(gc_data + pseudo).mean(axis=1)
log2fc = flt_mean - gc_mean

# T-test per gene
pvalues = np.ones(len(rodent_counts))
for i in range(len(rodent_counts)):
    try:
        _, p = stats.ttest_ind(flt_data[i, :], gc_data[i, :], equal_var=False)
        if not np.isnan(p):
            pvalues[i] = p
    except:
        pass

from statsmodels.stats.multitest import multipletests
_, padj, _, _ = multipletests(pvalues, method='fdr_bh')

rodent_de = pd.DataFrame({
    "gene_id": rodent_counts.index,
    "log2FC": log2fc,
    "pvalue": pvalues,
    "padj": padj,
})
# Relaxed: |log2FC| > 0.5, p < 0.05 (raw)
sig_relaxed = (rodent_de["log2FC"].abs() > 0.5) & (rodent_de["pvalue"] < 0.05)
print(f"  Significant (|log2FC|>0.5, p<0.05): {sig_relaxed.sum()}")
print(f"  Significant (|log2FC|>1, padj<0.05): {((rodent_de['log2FC'].abs()>1) & (rodent_de['padj']<0.05)).sum()}")

# ── Map mouse genes to human symbols via mygene homologene ──
print("\nMapping mouse genes to human orthologs...")
gene_ids = list(rodent_counts.index)
stripped = [g.split(".")[0] for g in gene_ids]

mg = mygene.MyGeneInfo()
# Query mouse genes for homologene data
results = mg.querymany(
    stripped,
    scopes="ensembl.gene",
    fields="symbol,homologene",
    species="mouse",
    verbose=False,
    returnall=False,
)

# Build mapping: mouse stripped ID -> human NCBI gene ID (from homologene, taxid 9606)
mouse_to_human_ncbi = {}
for hit in results:
    query = hit.get("query")
    if not query:
        continue
    homologene = hit.get("homologene")
    if not homologene:
        continue
    genes_list = homologene.get("genes", [])
    for entry in genes_list:
        if isinstance(entry, list) and len(entry) == 2:
            taxid, gene_id = entry
            if taxid == 9606:  # Human
                mouse_to_human_ncbi[query] = str(gene_id)
                break

print(f"  Mouse -> human NCBI gene ID: {len(mouse_to_human_ncbi)}")

# Map human NCBI gene IDs to symbols
human_ncbi_ids = list(set(mouse_to_human_ncbi.values()))
print(f"  Unique human NCBI IDs to map: {len(human_ncbi_ids)}")

# Batch query for human symbols
human_results = mg.querymany(
    human_ncbi_ids,
    scopes="entrezgene",
    fields="symbol",
    species="human",
    verbose=False,
    returnall=False,
)

human_ncbi_to_sym = {}
for hit in human_results:
    query = hit.get("query")
    symbol = hit.get("symbol", "")
    if query and symbol:
        human_ncbi_to_sym[query] = symbol

print(f"  Human NCBI -> symbol: {len(human_ncbi_to_sym)}")

# Build final mapping: mouse stripped ID -> human symbol
final_map = {}
for mouse_id, human_ncbi in mouse_to_human_ncbi.items():
    human_sym = human_ncbi_to_sym.get(human_ncbi, "")
    if human_sym:
        final_map[mouse_id] = human_sym

print(f"  Final mouse -> human symbol: {len(final_map)}")

# Add human symbols to rodent DE
rodent_de["mouse_id_stripped"] = rodent_de["gene_id"].apply(lambda x: x.split(".")[0])
rodent_de["human_symbol"] = rodent_de["mouse_id_stripped"].map(final_map)
rodent_de_mapped = rodent_de[rodent_de["human_symbol"].notna() & (rodent_de["human_symbol"] != "")].copy()
print(f"  Rodent DE with human symbols: {len(rodent_de_mapped)}")

# Save
rodent_de_mapped.to_csv(f"{PROC}/rodent_liver_de_human_mapped.csv", index=False)
print(f"  Saved rodent_liver_de_human_mapped.csv")

# ── Cross-species concordance ──
print("\n=== Cross-species concordance ===")
astro_de = pd.read_csv(f"{PROC}/mineral_pathway_de_combined.csv")
crosswalk = pd.read_csv(f"{PROC}/mineral_pathway_crosswalk.csv")
gene_to_minerals = dict(zip(crosswalk["gene_symbol"], crosswalk["minerals"]))

astro_genes = set(astro_de["gene_symbol"].unique())
rodent_genes = set(rodent_de_mapped["human_symbol"].unique())
shared_genes = astro_genes & rodent_genes
print(f"  Astronaut DE genes: {len(astro_genes)}")
print(f"  Rodent genes (mapped): {len(rodent_genes)}")
print(f"  Shared genes: {len(shared_genes)}")

# For ALL mineral-pathway genes (not just DE), check concordance
mineral_genes_all = set(crosswalk["gene_symbol"].dropna())
shared_mineral = mineral_genes_all & rodent_genes
print(f"  Shared mineral-pathway genes: {len(shared_mineral)}")

# Compute concordance for shared mineral-pathway genes
concordance = []
for gene in shared_mineral:
    # Astronaut: mean log2FC across comparisons (if DE)
    astro_rows = astro_de[astro_de["gene_symbol"] == gene]
    astro_lfc = astro_rows["log2FC"].mean() if len(astro_rows) > 0 else 0.0
    astro_sig = len(astro_rows) > 0

    # Rodent: log2FC
    rodent_rows = rodent_de_mapped[rodent_de_mapped["human_symbol"] == gene]
    if len(rodent_rows) == 0:
        continue
    rodent_row = rodent_rows.iloc[0]
    rodent_lfc = rodent_row["log2FC"]
    rodent_p = rodent_row["pvalue"]
    rodent_padj = rodent_row["padj"]

    same_dir = (astro_lfc > 0 and rodent_lfc > 0) or (astro_lfc < 0 and rodent_lfc < 0)
    minerals = gene_to_minerals.get(gene, "")
    if isinstance(minerals, float):
        minerals = ""

    concordance.append({
        "gene_symbol": gene,
        "astronaut_log2FC": astro_lfc,
        "astronaut_DE": astro_sig,
        "rodent_log2FC": rodent_lfc,
        "rodent_pvalue": rodent_p,
        "rodent_padj": rodent_padj,
        "same_direction": same_dir,
        "minerals": minerals,
    })

conc_df = pd.DataFrame(concordance)

# Overall concordance (all mineral genes)
n_concordant = conc_df["same_direction"].sum()
n_total = len(conc_df)
print(f"\n  All mineral-pathway genes: {n_concordant}/{n_total} concordant ({100*n_concordant/n_total:.1f}%)")

from scipy.stats import binomtest
p_binom = binomtest(n_concordant, n_total, 0.5, alternative='greater').pvalue
print(f"  Binomial test p-value: {p_binom:.4f}")

corr, p_corr = stats.spearmanr(conc_df["astronaut_log2FC"], conc_df["rodent_log2FC"])
print(f"  Spearman correlation: r={corr:.3f}, p={p_corr:.4f}")

# Concordance for astronaut DE genes only
de_conc = conc_df[conc_df["astronaut_DE"]]
if len(de_conc) > 0:
    n_de_conc = de_conc["same_direction"].sum()
    print(f"\n  Astronaut DE genes only: {n_de_conc}/{len(de_conc)} concordant ({100*n_de_conc/len(de_conc):.1f}%)")

# Concordance for rodent significant genes
rodent_sig = conc_df[conc_df["rodent_pvalue"] < 0.05]
if len(rodent_sig) > 0:
    n_rs_conc = rodent_sig["same_direction"].sum()
    print(f"  Rodent p<0.05 genes: {n_rs_conc}/{len(rodent_sig)} concordant ({100*n_rs_conc/len(rodent_sig):.1f}%)")

# Mineral-specific concordance
print("\n=== Mineral-specific concordance ===")
for mineral in ["Iron", "Calcium", "Zinc", "Copper", "Selenium", "Magnesium", "Potassium", "Sodium", "Phosphorus", "Manganese"]:
    mineral_genes = set()
    for _, row in conc_df.iterrows():
        if mineral in str(row["minerals"]):
            mineral_genes.add(row["gene_symbol"])
    if len(mineral_genes) < 2:
        continue
    sub = conc_df[conc_df["gene_symbol"].isin(mineral_genes)]
    n_conc = sub["same_direction"].sum()
    print(f"  {mineral}: {n_conc}/{len(sub)} concordant ({100*n_conc/len(sub):.0f}%)")

# Show top concordant genes
print("\n  Top concordant genes (astronaut DE + rodent p<0.1):")
both_sig = conc_df[conc_df["astronaut_DE"] & (conc_df["rodent_pvalue"] < 0.1)]
print(both_sig.sort_values("rodent_pvalue").to_string(index=False))

# Save
conc_df.to_csv(f"{PROC}/cross_species_concordance.csv", index=False)
print(f"\n  Saved cross_species_concordance.csv")
print("Done.")
