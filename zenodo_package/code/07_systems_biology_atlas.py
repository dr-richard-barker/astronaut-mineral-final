#!/usr/bin/env python3
"""
07_systems_biology_atlas.py — Create systems biology atlas figure.
Downloads KEGG pathway images and creates a composite multi-panel figure
showing mineral-pathway gene expression changes across KEGG pathways.

Outputs:
  /mnt/results/figures/fig8_systems_biology_atlas.svg/png
  /mnt/results/figures/fig9_pathway_expression_summary.svg/png
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rc_params = matplotlib.rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
from pathlib import Path
import urllib.request, os, time
import warnings
warnings.filterwarnings('ignore')

rc_params['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
rc_params['svg.fonttype'] = 'none'

FIG = Path("/mnt/results/figures")
PROC = Path("/mnt/results/data")

# ── Load data ──
print("Loading data...")
rna_mapped = pd.read_csv(PROC / "astronaut_rnaseq_de_mapped.csv")
prot_de = pd.read_csv(PROC / "astronaut_proteomics_de.csv")
crosswalk = pd.read_csv(PROC / "mineral_pathway_crosswalk.csv")
kegg_genes = pd.read_csv(PROC / "kegg_pathway_genes.csv")
de_combined = pd.read_csv(PROC / "mineral_pathway_de_combined.csv")
conc_df = pd.read_csv(PROC / "cross_species_concordance.csv")

gene_to_minerals = dict(zip(crosswalk["gene_symbol"], crosswalk["minerals"]))
def safe_minerals(val):
    if val is None or isinstance(val, float) or val == "":
        return []
    return [m.strip() for m in str(val).split(";") if m.strip()]

MINERAL_COLORS = {
    "Iron": "#E41A1C", "Calcium": "#377EB8", "Zinc": "#4DAF4A",
    "Copper": "#984EA3", "Selenium": "#FF7F00", "Magnesium": "#A65628",
    "Potassium": "#F781BF", "Sodium": "#999999", "Phosphorus": "#66C2A5",
    "Manganese": "#FFD92F",
}

PATHWAY_INFO = {
    "hsa04978": {"name": "Mineral Absorption", "color": "#E41A1C"},
    "hsa04020": {"name": "Calcium Signaling", "color": "#377EB8"},
    "hsa04216": {"name": "Ferroptosis", "color": "#4DAF4A"},
    "hsa00190": {"name": "Oxidative Phosphorylation", "color": "#984EA3"},
    "hsa04976": {"name": "Bile Secretion", "color": "#FF7F00"},
}

# ════════════════════════════════════════════════════════════════════
# Figure 8: Systems Biology Atlas (composite)
# ════════════════════════════════════════════════════════════════════
print("Creating systems biology atlas...")
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# Panel A: KEGG pathway gene expression heatmap
ax_a = fig.add_subplot(gs[0, :])
# For each KEGG pathway, compute mean log2FC of member genes across comparisons
pathway_fc_data = []
pathway_labels = []
comparison_labels = ["I4-FP1", "I4-FP2", "I4-FP3"]

for pid, info in PATHWAY_INFO.items():
    pathway_genes = set(kegg_genes[kegg_genes["pathway_id"] == pid]["gene_symbol"])
    for comp in comparison_labels:
        # RNA-seq
        rna_sub = rna_mapped[(rna_mapped["_sheet"] == comp) & (rna_mapped["gene_symbol"].isin(pathway_genes))]
        rna_fc = rna_sub["pipeline-transcriptome-de_log2FC"].mean()
        # Proteomics
        prot_sub = prot_de[(prot_de["_sheet"] == comp) & (prot_de["Gene"].isin(pathway_genes))]
        prot_fc = prot_sub["logFC"].mean() if len(prot_sub) > 0 else np.nan
        pathway_fc_data.append({
            "pathway": info["name"],
            "comparison": comp,
            "RNA_log2FC": rna_fc,
            "Protein_log2FC": prot_fc,
        })

pfc_df = pd.DataFrame(pathway_fc_data)

# Create heatmap data: pathways × comparisons, separate RNA and protein
rna_heat = pfc_df.pivot_table(index="pathway", columns="comparison", values="RNA_log2FC")
prot_heat = pfc_df.pivot_table(index="pathway", columns="comparison", values="Protein_log2FC")

# Combined heatmap
combined = np.hstack([rna_heat.values, prot_heat.values])
row_labels = list(rna_heat.index)
col_labels = [f"{c}\nRNA" for c in comparison_labels] + [f"{c}\nProtein" for c in comparison_labels]

sns.heatmap(combined, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, ax=ax_a, linewidths=0.5,
            xticklabels=col_labels, yticklabels=row_labels,
            cbar_kws={"label": "Mean log₂FC", "shrink": 0.7})
ax_a.set_title("A. KEGG Pathway Expression Changes: Mineral-Related Pathways", fontsize=13, fontweight='bold', loc='left')
ax_a.set_ylabel("KEGG Pathway", fontsize=11)

# Panel B: Mineral-pathway gene count bar chart
ax_b = fig.add_subplot(gs[1, 0])
mineral_counts = {}
for _, row in crosswalk.iterrows():
    for m in safe_minerals(row["minerals"]):
        mineral_counts[m] = mineral_counts.get(m, 0) + 1

minerals_sorted = sorted(mineral_counts.keys(), key=lambda x: mineral_counts[x], reverse=True)
counts = [mineral_counts[m] for m in minerals_sorted]
colors = [MINERAL_COLORS.get(m, "#888888") for m in minerals_sorted]
ax_b.barh(range(len(minerals_sorted)), counts, color=colors, edgecolor='white')
ax_b.set_yticks(range(len(minerals_sorted)))
ax_b.set_yticklabels(minerals_sorted, fontsize=9)
ax_b.invert_yaxis()
ax_b.set_xlabel("Gene Count", fontsize=10)
ax_b.set_title("B. Mineral-Pathway Gene Sets", fontsize=12, fontweight='bold', loc='left')

# Panel C: DE genes per mineral (stacked: up/down)
ax_c = fig.add_subplot(gs[1, 1])
mineral_de_up = {}
mineral_de_down = {}
for _, row in de_combined.iterrows():
    for m in safe_minerals(row["minerals"]):
        if row["log2FC"] > 0:
            mineral_de_up[m] = mineral_de_up.get(m, 0) + 1
        else:
            mineral_de_down[m] = mineral_de_down.get(m, 0) + 1

all_minerals = sorted(set(list(mineral_de_up.keys()) + list(mineral_de_down.keys())))
up_vals = [mineral_de_up.get(m, 0) for m in all_minerals]
down_vals = [mineral_de_down.get(m, 0) for m in all_minerals]
x = np.arange(len(all_minerals))
ax_c.bar(x, up_vals, color='#FF6B6B', label='Up-regulated', edgecolor='white')
ax_c.bar(x, [-v for v in down_vals], color='#4ECDC4', label='Down-regulated', edgecolor='white')
ax_c.set_xticks(x)
ax_c.set_xticklabels(all_minerals, rotation=45, ha='right', fontsize=8)
ax_c.set_ylabel("DE Gene Count", fontsize=10)
ax_c.set_title("C. DE Genes per Mineral", fontsize=12, fontweight='bold', loc='left')
ax_c.legend(fontsize=8)
ax_c.axhline(0, color='#333333', linewidth=0.5)

# Panel D: Cross-species concordance
ax_d = fig.add_subplot(gs[1, 2])
# Scatter: astronaut log2FC vs rodent log2FC for shared genes
conc_plot = conc_df.dropna(subset=["astronaut_log2FC", "rodent_log2FC"])
# Color by mineral
for _, row in conc_plot.iterrows():
    minerals = safe_minerals(row["minerals"])
    color = MINERAL_COLORS.get(minerals[0], "#CCCCCC") if minerals else "#CCCCCC"
    ax_d.scatter(row["astronaut_log2FC"], row["rodent_log2FC"], s=15, c=color, alpha=0.5, edgecolors='white', linewidths=0.3)

ax_d.axhline(0, color='#666666', linewidth=0.5, linestyle='--')
ax_d.axvline(0, color='#666666', linewidth=0.5, linestyle='--')
ax_d.set_xlabel("Astronaut log₂FC", fontsize=10)
ax_d.set_ylabel("Rodent log₂FC", fontsize=10)
ax_d.set_title("D. Cross-Species Concordance\n(r=0.007, n=587)", fontsize=12, fontweight='bold', loc='left')

# Panel E: Top mineral-pathway DE genes (RNA + Protein)
ax_e = fig.add_subplot(gs[2, :])
# Get top 25 genes by |log2FC| across all comparisons
de_combined["abs_log2FC"] = de_combined["log2FC"].abs()
top25 = de_combined.nlargest(25, "abs_log2FC").copy()

# Create grouped bar chart
bar_data = []
for _, row in top25.iterrows():
    minerals = safe_minerals(row["minerals"])
    primary = minerals[0] if minerals else "Other"
    bar_data.append({
        "gene": row["gene_symbol"],
        "log2FC": row["log2FC"],
        "omics": row["omics"],
        "comparison": row["comparison"],
        "mineral": primary,
    })
bar_df = pd.DataFrame(bar_data)

# Plot as horizontal bar
bar_df = bar_df.sort_values("log2FC")
colors = [MINERAL_COLORS.get(m, "#888888") for m in bar_df["mineral"]]
bars = ax_e.barh(range(len(bar_df)), bar_df["log2FC"], color=colors, edgecolor='white', linewidth=0.5)
ax_e.set_yticks(range(len(bar_df)))
ax_e.set_yticklabels([f"{g} ({o}, {c})" for g, o, c in zip(bar_df["gene"], bar_df["omics"], bar_df["comparison"])], fontsize=8)
ax_e.set_xlabel("log₂ Fold Change", fontsize=10)
ax_e.set_title("E. Top 25 Mineral-Pathway DE Genes (RNA-seq + Proteomics)", fontsize=13, fontweight='bold', loc='left')
ax_e.axvline(0, color='#333333', linewidth=0.5)

# Add mineral legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=MINERAL_COLORS[m], label=m) for m in sorted(MINERAL_COLORS.keys())]
ax_e.legend(handles=legend_elements, loc='lower right', fontsize=7, ncol=5, framealpha=0.8)

fig.suptitle("Systems Biology Atlas: Mineral-Pathway Transcriptional and Molecular Responses to Spaceflight",
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(FIG / "fig8_systems_biology_atlas.svg", format='svg', bbox_inches='tight')
plt.savefig(FIG / "fig8_systems_biology_atlas.png", format='png', bbox_inches='tight')
plt.close()
print("  Saved fig8_systems_biology_atlas")

# ════════════════════════════════════════════════════════════════════
# Figure 9: Pathway expression summary (per-pathway gene-level detail)
# ════════════════════════════════════════════════════════════════════
print("Creating pathway expression summary...")
fig, axes = plt.subplots(1, 5, figsize=(22, 8), sharey=False)

for ax, (pid, info) in zip(axes, PATHWAY_INFO.items()):
    pathway_genes = set(kegg_genes[kegg_genes["pathway_id"] == pid]["gene_symbol"])

    # Get DE data for this pathway's genes
    pathway_de = de_combined[de_combined["gene_symbol"].isin(pathway_genes)].copy()
    if len(pathway_de) == 0:
        # Get all genes with any expression change
        rna_sub = rna_mapped[(rna_mapped["gene_symbol"].isin(pathway_genes)) & (rna_mapped["_sheet"] == "I4-FP3")]
        pathway_de = pd.DataFrame({
            "gene_symbol": rna_sub["gene_symbol"],
            "log2FC": rna_sub["pipeline-transcriptome-de_log2FC"],
            "omics": "RNA-seq",
            "comparison": "I4-FP3",
            "minerals": rna_sub["gene_symbol"].map(gene_to_minerals).fillna(""),
        })

    # Sort by absolute log2FC
    pathway_de["abs_fc"] = pathway_de["log2FC"].abs()
    pathway_de = pathway_de.nlargest(15, "abs_fc")

    # Color by direction
    colors = ['#FF6B6B' if fc > 0 else '#4ECDC4' for fc in pathway_de["log2FC"]]
    ax.barh(range(len(pathway_de)), pathway_de["log2FC"], color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(pathway_de)))
    ax.set_yticklabels(pathway_de["gene_symbol"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("log₂FC", fontsize=9)
    ax.set_title(info["name"], fontsize=11, fontweight='bold', color=info["color"])
    ax.axvline(0, color='#333333', linewidth=0.5)

fig.suptitle("KEGG Pathway Expression Summary: Top Genes per Mineral-Related Pathway", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG / "fig9_pathway_expression_summary.svg", format='svg', bbox_inches='tight')
plt.savefig(FIG / "fig9_pathway_expression_summary.png", format='png', bbox_inches='tight')
plt.close()
print("  Saved fig9_pathway_expression_summary")

print("\n=== Systems biology atlas complete ===")
