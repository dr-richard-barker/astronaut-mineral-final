#!/usr/bin/env python3
"""
05_visualizations.py — Create all data-driven visualizations for the manuscript.
Saves SVG + PNG to /mnt/results/figures/

Figures:
  1. fig1_heatmap_mineral_de.svg     — Heatmap of mineral-pathway DE genes
  2. fig2_volcano_rnaseq.svg         — Volcano plot of RNA-seq DE (mineral genes highlighted)
  3. fig3_barplot_importance.svg     — Bar plot of permutation importance (K/Na regression)
  4. fig4_sankey_omics_minerals.svg  — Sankey: omics layer → minerals → KEGG pathways
  5. fig5_circos_mineral_pathway.svg — Circos: mineral-pathway gene connections
  6. fig6_knowledge_graph.svg        — Knowledge graph of mineral-gene-pathway network
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rc_params = matplotlib.rcParams
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Config ──
rc_params['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
rc_params['svg.fonttype'] = 'none'  # Keep SVG text editable
rc_params['figure.dpi'] = 150
rc_params['savefig.dpi'] = 300

FIG = Path("/mnt/results/figures")
FIG.mkdir(exist_ok=True)
PROC = Path("/mnt/results/data")

# Colorblind-friendly palette
MINERAL_COLORS = {
    "Iron": "#E41A1C", "Calcium": "#377EB8", "Zinc": "#4DAF4A",
    "Copper": "#984EA3", "Selenium": "#FF7F00", "Magnesium": "#A65628",
    "Potassium": "#F781BF", "Sodium": "#999999", "Phosphorus": "#66C2A5",
    "Manganese": "#FFD92F",
}

# ── Load data ──
print("Loading data...")
de_combined = pd.read_csv(PROC / "mineral_pathway_de_combined.csv")
crosswalk = pd.read_csv(PROC / "mineral_pathway_crosswalk.csv")
rna_mapped = pd.read_csv(PROC / "astronaut_rnaseq_de_mapped.csv")
prot_de = pd.read_csv(PROC / "astronaut_proteomics_de.csv")
clf_imp = pd.read_csv(PROC / "permutation_importance_classification.csv")
k_imp = pd.read_csv(PROC / "permutation_importance_potassium.csv")
na_imp = pd.read_csv(PROC / "permutation_importance_sodium.csv")
mineral_combined = pd.read_csv(PROC / "astronaut_minerals_combined.csv")

gene_to_minerals = dict(zip(crosswalk["gene_symbol"], crosswalk["minerals"]))

def safe_split_minerals(val):
    """Safely split minerals string, handling NaN/float."""
    if val is None or isinstance(val, float) or val == "":
        return []
    return [m.strip() for m in str(val).split(";") if m.strip()]

def primary_mineral(val):
    """Get first mineral from a minerals string."""
    minerals = safe_split_minerals(val)
    return minerals[0] if minerals else ""

# ════════════════════════════════════════════════════════════════════
# Figure 1: Heatmap of mineral-pathway DE genes
# ════════════════════════════════════════════════════════════════════
print("Creating heatmap...")
# Select top DE genes by |log2FC| across all comparisons
de_combined["abs_log2FC"] = de_combined["log2FC"].abs()
top_de = de_combined.nlargest(40, "abs_log2FC").copy()

# Pivot: gene × comparison
heatmap_data = top_de.pivot_table(
    index="gene_symbol", columns="comparison", values="log2FC", aggfunc='first'
)
# Sort by mean absolute log2FC
heatmap_data = heatmap_data.loc[heatmap_data.abs().mean(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(10, 12))
sns.heatmap(
    heatmap_data, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
    vmin=-2.5, vmax=2.5, ax=ax, linewidths=0.5, linecolor='white',
    cbar_kws={"label": "log₂ Fold Change", "shrink": 0.6},
)
ax.set_title("Mineral-Pathway Differential Expression\nTop 40 Genes Across Flight Comparisons", fontsize=13)
ax.set_xlabel("Flight Comparison", fontsize=11)
ax.set_ylabel("Gene Symbol", fontsize=11)

# Add mineral annotations on y-axis
yticklabels = ax.get_yticklabels()
for label in yticklabels:
    gene = label.get_text()
    minerals = gene_to_minerals.get(gene, "")
    primary_m = primary_mineral(minerals)
    color = MINERAL_COLORS.get(primary_m, "#333333")
    label.set_color(color)

plt.tight_layout()
plt.savefig(FIG / "fig1_heatmap_mineral_de.svg", format='svg', bbox_inches='tight')
plt.savefig(FIG / "fig1_heatmap_mineral_de.png", format='png', bbox_inches='tight')
plt.close()
print("  Saved fig1_heatmap_mineral_de")

# ════════════════════════════════════════════════════════════════════
# Figure 2: Volcano plot of RNA-seq DE (mineral genes highlighted)
# ════════════════════════════════════════════════════════════════════
print("Creating volcano plot...")
# Use I4-FP3 (longest follow-up) pipeline-transcriptome-de
fp3 = rna_mapped[rna_mapped["_sheet"] == "I4-FP3"].copy()
fp3["log2FC"] = fp3["pipeline-transcriptome-de_log2FC"]
fp3["pvalue"] = fp3["pipeline-transcriptome-de_p-value"]
fp3["neg_log10_p"] = -np.log10(fp3["pvalue"].clip(lower=1e-300))
fp3["sig"] = (fp3["log2FC"].abs() > 0.5) & (fp3["pvalue"] < 0.05)

fig, ax = plt.subplots(figsize=(10, 8))

# Non-mineral genes (grey)
non_mineral = fp3[~fp3["is_mineral_gene"]]
ax.scatter(non_mineral["log2FC"], non_mineral["neg_log10_p"], s=3, alpha=0.2, c='#CCCCCC', rasterized=True)

# Mineral genes (colored by mineral)
mineral_genes = fp3[fp3["is_mineral_gene"]].copy()
mineral_genes["primary_mineral"] = mineral_genes["gene_symbol"].map(
    lambda g: primary_mineral(gene_to_minerals.get(g))
)
for mineral, color in MINERAL_COLORS.items():
    subset = mineral_genes[mineral_genes["primary_mineral"] == mineral]
    if len(subset) > 0:
        ax.scatter(subset["log2FC"], subset["neg_log10_p"], s=20, alpha=0.8, c=color, label=mineral, edgecolors='white', linewidths=0.3)

# Label significant mineral genes
sig_mineral = mineral_genes[mineral_genes["sig"]]
for _, row in sig_mineral.iterrows():
    ax.annotate(row["gene_symbol"], (row["log2FC"], row["neg_log10_p"]),
                fontsize=7, ha='left', va='bottom', xytext=(3, 3), textcoords='offset points')

ax.axhline(-np.log10(0.05), color='#666666', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axvline(0.5, color='#666666', linestyle='--', linewidth=0.8, alpha=0.5)
ax.axvline(-0.5, color='#666666', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel("log₂ Fold Change (R+1/R+45/R+82/R+194 vs Pre-flight)", fontsize=11)
ax.set_ylabel("-log₁₀(p-value)", fontsize=11)
ax.set_title("Volcano Plot: RNA-seq Differential Expression\nMineral-Pathway Genes Highlighted (I4-FP3)", fontsize=13)
ax.legend(title="Mineral", fontsize=8, title_fontsize=9, loc='upper left', framealpha=0.8)
plt.tight_layout()
plt.savefig(FIG / "fig2_volcano_rnaseq.svg", format='svg', bbox_inches='tight')
plt.savefig(FIG / "fig2_volcano_rnaseq.png", format='png', bbox_inches='tight')
plt.close()
print("  Saved fig2_volcano_rnaseq")

# ════════════════════════════════════════════════════════════════════
# Figure 3: Bar plot of permutation importance
# ════════════════════════════════════════════════════════════════════
print("Creating bar plot of permutation importance...")
fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=False)

for ax, imp_df, title in [
    (axes[0], k_imp, "Potassium Regression"),
    (axes[1], na_imp, "Sodium Regression"),
]:
    top20 = imp_df.head(20).copy()
    top20["primary_mineral"] = top20["minerals"].apply(
        lambda x: primary_mineral(x) or "Other"
    )
    colors = [MINERAL_COLORS.get(m, "#AAAAAA") for m in top20["primary_mineral"]]
    bars = ax.barh(range(len(top20)), top20["importance_mean"], color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["gene"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation Importance (ΔR²)", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.axvline(0, color='#333333', linewidth=0.5)

fig.suptitle("Permutation Importance: Key Genes Predicting Serum Mineral Levels", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIG / "fig3_barplot_importance.svg", format='svg', bbox_inches='tight')
plt.savefig(FIG / "fig3_barplot_importance.png", format='png', bbox_inches='tight')
plt.close()
print("  Saved fig3_barplot_importance")

# ════════════════════════════════════════════════════════════════════
# Figure 4: Sankey diagram (omics → minerals → pathways)
# ════════════════════════════════════════════════════════════════════
print("Creating Sankey diagram...")
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath

# Build flow data: omics layer → mineral → KEGG pathway
# Count significant DE genes per omics-mineral combination
flows = []
for _, row in de_combined.iterrows():
    minerals = row["minerals"]
    for mineral in safe_split_minerals(minerals):
        flows.append({"omics": row["omics"], "mineral": mineral, "gene": row["gene_symbol"]})

flow_df = pd.DataFrame(flows)
# Count per omics→mineral
omics_mineral = flow_df.groupby(["omics", "mineral"]).size().reset_index(name="count")

# Get KEGG pathway associations from crosswalk
kegg_pathway_genes = pd.read_csv(PROC / "kegg_pathway_genes.csv")
pathway_names = {
    "hsa04978": "Mineral\nAbsorption",
    "hsa04020": "Calcium\nSignaling",
    "hsa04216": "Ferroptosis",
    "hsa00190": "Oxidative\nPhosphorylation",
    "hsa04976": "Bile\nSecretion",
}

# Count mineral→pathway connections (from crosswalk)
mineral_pathway_counts = []
for _, row in crosswalk.iterrows():
    minerals = row["minerals"]
    pathways = row["kegg_pathways"]
    for mineral in safe_split_minerals(minerals):
        for pathway in safe_split_minerals(pathways):
            pid = pathway.split(":")[0]
            mineral_pathway_counts.append({"mineral": mineral, "pathway": pid})

mp_df = pd.DataFrame(mineral_pathway_counts)
mineral_pathway = mp_df.groupby(["mineral", "pathway"]).size().reset_index(name="count")

# Create Sankey using matplotlib
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 3)
ax.set_ylim(0, 20)
ax.axis('off')

# Define node positions
omics_nodes = {"RNA-seq": (0.5, 16), "Proteomics": (0.5, 12)}
minerals_list = sorted(flow_df["mineral"].unique())
mineral_y = np.linspace(18, 2, len(minerals_list))
mineral_nodes = {m: (1.5, y) for m, y in zip(minerals_list, mineral_y)}

pathways_list = sorted(mp_df["pathway"].unique())
pathway_y = np.linspace(17, 3, len(pathways_list))
pathway_nodes = {p: (2.5, y) for p, y in zip(pathways_list, pathway_y)}

# Draw nodes
for name, (x, y) in omics_nodes.items():
    ax.scatter(x, y, s=200, c='#333333', zorder=5)
    ax.text(x - 0.15, y, name, ha='right', va='center', fontsize=10, fontweight='bold')

for name, (x, y) in mineral_nodes.items():
    color = MINERAL_COLORS.get(name, "#888888")
    ax.scatter(x, y, s=150, c=color, zorder=5)
    ax.text(x, y, name, ha='center', va='center', fontsize=8, fontweight='bold', color='white')

for pid, (x, y) in pathway_nodes.items():
    ax.scatter(x, y, s=150, c='#4DAF4A', zorder=5)
    ax.text(x + 0.15, y, pathway_names.get(pid, pid), ha='left', va='center', fontsize=8)

# Draw flows: omics → mineral
max_count_om = omics_mineral["count"].max() if len(omics_mineral) > 0 else 1
for _, row in omics_mineral.iterrows():
    src = omics_nodes[row["omics"]]
    dst = mineral_nodes[row["mineral"]]
    width = 0.5 + 2.0 * row["count"] / max_count_om
    color = MINERAL_COLORS.get(row["mineral"], "#888888")
    ax.annotate("", xy=dst, xytext=src,
                arrowprops=dict(arrowstyle='->', color=color, alpha=0.4, lw=width))

# Draw flows: mineral → pathway
max_count_mp = mineral_pathway["count"].max() if len(mineral_pathway) > 0 else 1
for _, row in mineral_pathway.iterrows():
    if row["mineral"] not in mineral_nodes or row["pathway"] not in pathway_nodes:
        continue
    src = mineral_nodes[row["mineral"]]
    dst = pathway_nodes[row["pathway"]]
    width = 0.3 + 1.5 * row["count"] / max_count_mp
    color = MINERAL_COLORS.get(row["mineral"], "#888888")
    ax.annotate("", xy=dst, xytext=src,
                arrowprops=dict(arrowstyle='->', color=color, alpha=0.3, lw=width))

ax.set_title("Sankey: Omics Layer → Mineral Association → KEGG Pathway", fontsize=14, pad=20)
ax.text(0.5, 19.5, "Omics Layer", ha='center', fontsize=11, fontweight='bold')
ax.text(1.5, 19.5, "Mineral", ha='center', fontsize=11, fontweight='bold')
ax.text(2.5, 19.5, "KEGG Pathway", ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(FIG / "fig4_sankey_omics_minerals.svg", format='svg', bbox_inches='tight')
plt.savefig(FIG / "fig4_sankey_omics_minerals.png", format='png', bbox_inches='tight')
plt.close()
print("  Saved fig4_sankey_omics_minerals")

# ════════════════════════════════════════════════════════════════════
# Figure 5: Circos plot (mineral-pathway gene connections)
# ════════════════════════════════════════════════════════════════════
print("Creating Circos plot...")
# Build adjacency: minerals (outer) → genes (inner) → KEGG pathways (outer)
# Use crosswalk data
circos_edges = []
for _, row in crosswalk.iterrows():
    gene = row["gene_symbol"]
    minerals = row["minerals"]
    pathways = row["kegg_pathways"]
    for m in safe_split_minerals(minerals):
            circos_edges.append(("mineral", m, "gene", gene))
    for p in safe_split_minerals(pathways):
            pid = p.split(":")[0] if ":" in p else p
            circos_edges.append(("gene", gene, "pathway", pid))

# Get unique nodes
mineral_nodes_c = sorted(set(e[1] for e in circos_edges if e[0] == "mineral"))
gene_nodes_c = sorted(set(e[1] for e in circos_edges if e[0] == "gene"))
pathway_nodes_c = sorted(set(e[3] for e in circos_edges if e[2] == "pathway"))

# Filter genes to those with DE signal for clarity
de_genes = set(de_combined["gene_symbol"].unique())
gene_nodes_c = [g for g in gene_nodes_c if g in de_genes or g in set(clf_imp.head(20)["gene"]) or g in set(k_imp.head(20)["gene"])]
# Rebuild edges with filtered genes
circos_edges_f = [e for e in circos_edges if e[1] in gene_nodes_c or e[3] in gene_nodes_c]

# Draw circos with matplotlib
fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_rlim(0, 1.2)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
ax.spines['polar'].set_visible(False)

# Arrange nodes around circle
all_nodes = mineral_nodes_c + gene_nodes_c + pathway_nodes_c
n_total = len(all_nodes)
angles = np.linspace(0, 2*np.pi, n_total, endpoint=False)
node_angles = {n: a for n, a in zip(all_nodes, angles)}

# Draw node arcs
for i, node in enumerate(all_nodes):
    a = angles[i]
    arc_width = 2*np.pi / n_total * 0.8
    if node in mineral_nodes_c:
        color = MINERAL_COLORS.get(node, "#888888")
        r_outer, r_inner = 1.0, 0.95
    elif node in pathway_nodes_c:
        color = '#4DAF4A'
        r_outer, r_inner = 1.0, 0.95
    else:
        color = '#333333'
        r_outer, r_inner = 0.7, 0.65

    theta = np.linspace(a - arc_width/2, a + arc_width/2, 20)
    ax.fill_between(theta, r_inner, r_outer, color=color, alpha=0.8)

    # Labels
    label_r = 1.08 if node in mineral_nodes_c or node in pathway_nodes_c else 0.55
    label_text = pathway_names.get(node, node) if node in pathway_nodes_c else node
    ax.text(a, label_r, label_text, ha='center', va='center', fontsize=6,
            rotation=np.degrees(a) - 90 if np.pi/2 < a < 3*np.pi/2 else np.degrees(a) + 90,
            rotation_mode='anchor')

# Draw edges
for edge in circos_edges_f:
    if edge[0] == "mineral":
        src, dst = edge[1], edge[3]
        if src in node_angles and dst in node_angles:
            a1, a2 = node_angles[src], node_angles[dst]
            color = MINERAL_COLORS.get(src, "#888888")
            # Bezier-like curve
            t = np.linspace(0, 1, 30)
            r_curve = 0.85 - 0.15 * np.sin(np.pi * t)
            theta_curve = a1 + (a2 - a1) * t
            ax.plot(theta_curve, r_curve, color=color, alpha=0.15, linewidth=0.5)
    elif edge[2] == "pathway":
        src, dst = edge[1], edge[3]
        if src in node_angles and dst in node_angles:
            a1, a2 = node_angles[src], node_angles[dst]
            color = '#4DAF4A'
            t = np.linspace(0, 1, 30)
            r_curve = 0.55 + 0.1 * np.sin(np.pi * t)
            theta_curve = a1 + (a2 - a1) * t
            ax.plot(theta_curve, r_curve, color=color, alpha=0.1, linewidth=0.5)

ax.set_title("Circos: Mineral–Gene–Pathway Network\n(DE genes and top ML features)", fontsize=13, pad=30)
plt.tight_layout()
plt.savefig(FIG / "fig5_circos_mineral_pathway.svg", format='svg', bbox_inches='tight')
plt.savefig(FIG / "fig5_circos_mineral_pathway.png", format='png', bbox_inches='tight')
plt.close()
print("  Saved fig5_circos_mineral_pathway")

# ════════════════════════════════════════════════════════════════════
# Figure 6: Knowledge graph
# ════════════════════════════════════════════════════════════════════
print("Creating knowledge graph...")
import networkx as nx

G = nx.Graph()

# Add nodes
for mineral in mineral_nodes_c:
    G.add_node(mineral, node_type="mineral", color=MINERAL_COLORS.get(mineral, "#888888"))
for gene in gene_nodes_c:
    minerals = gene_to_minerals.get(gene, "")
    primary = primary_mineral(minerals)
    G.add_node(gene, node_type="gene", color=MINERAL_COLORS.get(primary, "#333333"))
for pathway in pathway_nodes_c:
    G.add_node(pathway, node_type="pathway", color="#4DAF4A")

# Add edges
for edge in circos_edges_f:
    if edge[0] == "mineral":
        G.add_edge(edge[1], edge[3], edge_type="mineral_gene")
    elif edge[2] == "pathway":
        G.add_edge(edge[1], edge[3], edge_type="gene_pathway")

# Layout
pos = nx.spring_layout(G, k=1.5/np.sqrt(len(G)), iterations=100, seed=42)

fig, ax = plt.subplots(figsize=(16, 14))

# Draw edges
mineral_gene_edges = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "mineral_gene"]
gene_pathway_edges = [(u, v) for u, v, d in G.edges(data=True) if d["edge_type"] == "gene_pathway"]

nx.draw_networkx_edges(G, pos, edgelist=mineral_gene_edges, alpha=0.2, edge_color='#888888', ax=ax)
nx.draw_networkx_edges(G, pos, edgelist=gene_pathway_edges, alpha=0.15, edge_color='#4DAF4A', style='dashed', ax=ax)

# Draw nodes by type
for node_type, node_size in [("mineral", 400), ("pathway", 350), ("gene", 80)]:
    nodes = [n for n in G.nodes if G.nodes[n]["node_type"] == node_type]
    colors = [G.nodes[n]["color"] for n in nodes]
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, node_size=node_size, alpha=0.8, ax=ax)

# Labels for minerals and pathways only (genes too many)
mineral_labels = {n: n for n in G.nodes if G.nodes[n]["node_type"] == "mineral"}
pathway_labels = {n: pathway_names.get(n, n) for n in G.nodes if G.nodes[n]["node_type"] == "pathway"}
nx.draw_networkx_labels(G, pos, {**mineral_labels, **pathway_labels}, font_size=8, font_weight='bold', ax=ax)

# Gene labels for top DE genes only
top_genes = set(de_combined.nlargest(15, "abs_log2FC")["gene_symbol"])
gene_labels = {n: n for n in G.nodes if G.nodes[n]["node_type"] == "gene" and n in top_genes}
nx.draw_networkx_labels(G, pos, gene_labels, font_size=6, ax=ax)

ax.set_title("Knowledge Graph: Mineral–Gene–Pathway Network\nfrom Astronaut Multi-Omics Spaceflight Data", fontsize=14)
ax.axis('off')
plt.tight_layout()
plt.savefig(FIG / "fig6_knowledge_graph.svg", format='svg', bbox_inches='tight')
plt.savefig(FIG / "fig6_knowledge_graph.png", format='png', bbox_inches='tight')
plt.close()
print("  Saved fig6_knowledge_graph")

# ════════════════════════════════════════════════════════════════════
# Figure 7: Mineral trajectory plot (serum minerals over time)
# ════════════════════════════════════════════════════════════════════
print("Creating mineral trajectory plot...")
timepoint_order = ["L-92", "L-44", "L-3", "R+1", "R+45", "R+82", "R+194"]
mineral_combined["timepoint_order"] = mineral_combined["timepoint"].map({t: i for i, t in enumerate(timepoint_order)})
mineral_combined = mineral_combined.sort_values(["SUBJECT_ID", "timepoint_order"])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
targets = [("CALCIUM", "Calcium (mg/dL)"), ("POTASSIUM", "Potassium (mmol/L)"),
           ("SODIUM", "Sodium (mmol/L)"), ("HEMOGLOBIN", "Hemoglobin (g/dL, Fe proxy)")]
subject_colors = {"C001": "#E41A1C", "C002": "#377EB8", "C003": "#4DAF4A", "C004": "#984EA3"}

for ax, (target, label) in zip(axes.flat, targets):
    for subj in mineral_combined["SUBJECT_ID"].unique():
        sub = mineral_combined[mineral_combined["SUBJECT_ID"] == subj].sort_values("timepoint_order")
        ax.plot(sub["timepoint"], sub[target], 'o-', color=subject_colors[subj], label=subj, markersize=5, linewidth=1.5)
    ax.set_ylabel(label, fontsize=10)
    ax.set_xlabel("Timepoint", fontsize=10)
    ax.axvspan(-0.5, 2.5, alpha=0.08, color='#CCCCCC', label='Pre-flight')
    ax.axvline(3.5, color='#FF9400', linestyle='--', alpha=0.5, label='Flight')
    ax.tick_params(axis='x', rotation=45)

axes[0, 0].legend(fontsize=8, loc='upper right')
fig.suptitle("Serum Mineral Trajectories: Inspiration4 Astronauts\nPre-flight (L-*) to Post-flight (R+*)", fontsize=14)
plt.tight_layout()
plt.savefig(FIG / "fig7_mineral_trajectories.svg", format='svg', bbox_inches='tight')
plt.savefig(FIG / "fig7_mineral_trajectories.png", format='png', bbox_inches='tight')
plt.close()
print("  Saved fig7_mineral_trajectories")

print("\n=== All visualizations complete ===")
print(f"Saved to: {FIG}")
