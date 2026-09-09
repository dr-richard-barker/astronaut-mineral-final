#!/usr/bin/env python3
"""
08_supplementary_outputs.py
Generates supplementary visualizations and DE table for the mineral-pathway
spaceflight multi-omics analysis.

Outputs:
  - Figure S1: GSEA enrichment dot plot (preranked GSEA, 5 runs x 15 gene sets)
  - Figure S2: Time-resolved mineral-pathway activation heatmap (2 panels)
  - supp_table_de_wide.csv: Wide-format DE table (50 genes x all stats)
  - supp_table_de_long.csv: Long-format DE table (71 rows, enhanced)
  - gsea_results.csv: Full GSEA output
  - ssgsea_scores.csv: ssGSEA pathway scores per sample
  - pathway_mean_log2fc.csv: Pathway-level mean log2FC across comparisons
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rc_params = matplotlib.rcParams
import seaborn as sns
from pathlib import Path

rc_params['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
rc_params['svg.fonttype'] = 'none'

warnings.filterwarnings('ignore', category=FutureWarning)

# ── Paths ──
DATA_DIR = Path('/mnt/results/data')
FIG_DIR = Path('/mnt/results/figures')
ZENODO_DIR = Path('/mnt/results/zenodo_package')
MANUSCRIPT_FIG_DIR = Path('/mnt/results/manuscript/figures')
for d in [FIG_DIR, MANUSCRIPT_FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Comparison labels ──
COMPARISON_LABELS = {
    'I4-FP1': 'Acute (R+1)',
    'I4-FP2': 'Subacute (R+1 to R+82)',
    'I4-FP3': 'Long-term (R+1 to R+194)',
}


# ============================================================
# PART 1: Load data
# ============================================================
def load_data():
    """Load all required data files."""
    print("Loading data files...")

    # RNA-seq DE (mapped with gene symbols)
    rna_de = pd.read_csv(DATA_DIR / 'astronaut_rnaseq_de_mapped.csv')
    print(f"  RNA-seq DE: {rna_de.shape}")

    # Proteomics DE
    prot_de = pd.read_csv(DATA_DIR / 'astronaut_proteomics_de.csv')
    print(f"  Proteomics DE: {prot_de.shape}")

    # RNA-seq counts
    counts = pd.read_csv(DATA_DIR / 'astronaut_rnaseq_counts.csv', index_col=0)
    print(f"  RNA-seq counts: {counts.shape}")

    # ENSEMBL to symbol mapping
    ens_map = pd.read_csv(DATA_DIR / 'ensembl_to_symbol.csv')
    print(f"  ENSEMBL mapping: {ens_map.shape}")

    # Mineral pathway DE combined
    de_combined = pd.read_csv(DATA_DIR / 'mineral_pathway_de_combined.csv')
    print(f"  DE combined: {de_combined.shape}")

    # Crosswalk (gene -> mineral -> KEGG pathway)
    crosswalk = pd.read_csv(DATA_DIR / 'mineral_pathway_crosswalk.csv')
    print(f"  Crosswalk: {crosswalk.shape}")

    # GMT gene sets
    gmt_path = DATA_DIR / 'mineral_pathway_genesets.gmt'
    gene_sets = {}
    with open(gmt_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                name = parts[0]
                desc = parts[1]
                genes = [g for g in parts[2:] if g]
                gene_sets[name] = genes
    print(f"  GMT gene sets: {len(gene_sets)} sets")

    return rna_de, prot_de, counts, ens_map, de_combined, crosswalk, gene_sets


# ============================================================
# PART 2: GSEA Preranked Enrichment
# ============================================================
def run_gsea(rna_de, prot_de, gene_sets):
    """Run preranked GSEA for each omics x comparison."""
    import gseapy

    print("\n=== Running preranked GSEA ===")
    all_results = []

    # --- RNA-seq: sign(log2FC) x -log10(p) from pipeline-transcriptome-de ---
    for comp in ['I4-FP1', 'I4-FP2', 'I4-FP3']:
        print(f"  RNA-seq {comp}...")
        sub = rna_de[rna_de['_sheet'] == comp].copy()

        # Use pipeline-transcriptome-de columns (real fold changes)
        lfc_col = 'pipeline-transcriptome-de_log2FC'
        pval_col = 'pipeline-transcriptome-de_p-value'

        # Filter: drop rows with NaN or zero in both columns
        sub = sub.dropna(subset=[lfc_col, pval_col, 'gene_symbol'])
        sub = sub[(sub[lfc_col] != 0) | (sub[pval_col] != 0)]
        sub = sub[sub['gene_symbol'].notna() & (sub['gene_symbol'] != '')]

        # Compute ranking metric: sign(log2FC) x -log10(p-value)
        sub['rank_metric'] = np.sign(sub[lfc_col]) * (-np.log10(sub[pval_col].clip(lower=1e-300)))

        # Aggregate duplicate symbols (take max abs rank)
        sub = sub.groupby('gene_symbol')['rank_metric'].apply(
            lambda x: x.loc[x.abs().idxmax()]
        ).reset_index()

        # Create ranking Series
        ranking = pd.Series(
            sub['rank_metric'].values,
            index=sub['gene_symbol'].values
        ).sort_values(ascending=False)

        # Drop NaN and inf
        ranking = ranking.replace([np.inf, -np.inf], np.nan).dropna()

        print(f"    Ranked genes: {len(ranking)}")

        # Run prerank GSEA
        try:
            out_dir = f'/workspace/gsea_out/rna_{comp}'
            os.makedirs(out_dir, exist_ok=True)
            result = gseapy.prerank(
                rnk=ranking,
                gene_sets=gene_sets,
                outdir=out_dir,
                permutation_num=1000,
                seed=42,
                no_plot=True,
                verbose=False,
            )
            # Extract results
            res_df = result.res2d.copy()
            res_df['omics'] = 'RNA-seq'
            res_df['comparison'] = comp
            all_results.append(res_df)
            print(f"    Significant (FDR<0.25): {(res_df['FDR q-val'] < 0.25).sum()}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # --- Proteomics: limma t-statistic ---
    for comp in ['I4-FP1', 'I4-FP2']:
        print(f"  Proteomics {comp}...")
        # The proteomics file has _sheet column - need to identify it
        # Column 7 (0-indexed) is _sheet based on earlier inspection
        sheet_col = None
        for col in prot_de.columns:
            if prot_de[col].astype(str).str.contains('I4-FP').any():
                sheet_col = col
                break

        if sheet_col is None:
            # Try positional - column index 7
            sheet_col = prot_de.columns[7]

        sub = prot_de[prot_de[sheet_col] == comp].copy()

        # Use t-statistic as ranking
        if 't' in sub.columns:
            rank_col = 't'
        elif 'T' in sub.columns:
            rank_col = 'T'
        else:
            print(f"    No t-statistic column found, using logFC x sign(p)")
            sub['rank_metric'] = np.sign(sub['logFC']) * (-np.log10(sub['P.Value'].clip(lower=1e-300)))
            rank_col = 'rank_metric'

        gene_col = 'Gene' if 'Gene' in sub.columns else 'gene_symbol'
        sub = sub.dropna(subset=[rank_col, gene_col])
        sub = sub[sub[gene_col].notna() & (sub[gene_col] != '')]

        # Aggregate duplicates
        sub = sub.groupby(gene_col)[rank_col].apply(
            lambda x: x.loc[x.abs().idxmax()]
        ).reset_index()

        ranking = pd.Series(
            sub[rank_col].values,
            index=sub[gene_col].values
        ).sort_values(ascending=False)

        ranking = ranking.replace([np.inf, -np.inf], np.nan).dropna()

        print(f"    Ranked genes: {len(ranking)}")

        try:
            out_dir = f'/workspace/gsea_out/prot_{comp}'
            os.makedirs(out_dir, exist_ok=True)
            result = gseapy.prerank(
                rnk=ranking,
                gene_sets=gene_sets,
                outdir=out_dir,
                permutation_num=1000,
                seed=42,
                no_plot=True,
                verbose=False,
            )
            res_df = result.res2d.copy()
            res_df['omics'] = 'Proteomics'
            res_df['comparison'] = comp
            all_results.append(res_df)
            print(f"    Significant (FDR<0.25): {(res_df['FDR q-val'] < 0.25).sum()}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Combine all results
    if all_results:
        gsea_all = pd.concat(all_results, ignore_index=True)
        # Clean up column names
        gsea_all.to_csv(DATA_DIR / 'gsea_results.csv', index=False)
        print(f"\n  GSEA results saved: {gsea_all.shape}")
        return gsea_all
    else:
        print("  WARNING: No GSEA results generated")
        return pd.DataFrame()


def plot_gsea_dotplot(gsea_results):
    """Generate GSEA enrichment dot plot (Figure S1)."""
    print("\n=== Generating GSEA dot plot (Figure S1) ===")

    if gsea_results.empty:
        print("  No results to plot")
        return

    # Prepare data
    df = gsea_results.copy()

    # Clean Term names for display
    term_map = {
        'MINERAL_Calcium': 'Calcium',
        'MINERAL_Copper': 'Copper',
        'MINERAL_Iron': 'Iron',
        'MINERAL_Magnesium': 'Magnesium',
        'MINERAL_Manganese': 'Manganese',
        'MINERAL_Phosphorus': 'Phosphorus',
        'MINERAL_Potassium': 'Potassium',
        'MINERAL_Selenium': 'Selenium',
        'MINERAL_Sodium': 'Sodium',
        'MINERAL_Zinc': 'Zinc',
        'KEGG_hsa00190': 'Oxidative phosphorylation',
        'KEGG_hsa04020': 'Calcium signaling',
        'KEGG_hsa04216': 'Ferroptosis',
        'KEGG_hsa04976': 'Bile secretion',
        'KEGG_hsa04978': 'Mineral absorption',
    }

    df['term_label'] = df['Term'].map(term_map).fillna(df['Term'])

    # Create comparison label
    df['comp_label'] = df['omics'] + ' ' + df['comparison'].map(COMPARISON_LABELS).fillna(df['comparison'])

    # Convert NES and FDR to numeric
    df['NES'] = pd.to_numeric(df['NES'], errors='coerce')
    df['FDR q-val'] = pd.to_numeric(df['FDR q-val'], errors='coerce')

    # Replace FDR=0 with small value for log scale
    df['neg_log10_fdr'] = -np.log10(df['FDR q-val'].clip(lower=1e-10))

    # Sort comparisons in logical order
    comp_order = [
        'RNA-seq Acute (R+1)',
        'RNA-seq Subacute (R+1 to R+82)',
        'RNA-seq Long-term (R+1 to R+194)',
        'Proteomics Acute (R+1)',
        'Proteomics Subacute (R+1 to R+82)',
    ]
    df['comp_label'] = pd.Categorical(df['comp_label'], categories=comp_order, ordered=True)

    # Sort gene sets: minerals first (alphabetical), then KEGG
    mineral_terms = ['Calcium', 'Copper', 'Iron', 'Magnesium', 'Manganese',
                     'Phosphorus', 'Potassium', 'Selenium', 'Sodium', 'Zinc']
    kegg_terms = ['Oxidative phosphorylation', 'Calcium signaling', 'Ferroptosis',
                  'Bile secretion', 'Mineral absorption']
    term_order = mineral_terms + kegg_terms
    df['term_label'] = pd.Categorical(df['term_label'], categories=term_order, ordered=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Dot plot
    scatter = ax.scatter(
        df['term_label'].astype(str).values,
        df['comp_label'].astype(str).values,
        s=df['NES'].abs() * 40 + 20,  # dot size by |NES|
        c=df['neg_log10_fdr'].values,
        cmap='YlOrRd',
        edgecolors='black',
        linewidth=0.5,
        alpha=0.85,
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('-log10(FDR q-val)', fontsize=10)

    # Add significance markers
    for _, row in df.iterrows():
        if pd.notna(row['FDR q-val']) and row['FDR q-val'] < 0.25:
            ax.text(
                str(row['term_label']),
                str(row['comp_label']),
                '*',
                ha='center', va='center',
                fontsize=8, fontweight='bold', color='black'
            )

    ax.set_xlabel('Gene Set', fontsize=11)
    ax.set_ylabel('Comparison', fontsize=11)
    ax.set_title('GSEA: Mineral Pathway Enrichment Across Spaceflight Recovery',
                 fontsize=12, fontweight='bold', pad=12)

    # Rotate x labels
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)

    # Add horizontal line between RNA-seq and Proteomics
    ax.axhline(y=2.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Add vertical line between mineral and KEGG sets
    ax.axvline(x=9.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Add section labels
    ax.text(4, -0.8, 'Mineral-specific', ha='center', fontsize=8, style='italic', color='gray')
    ax.text(12, -0.8, 'KEGG Pathway', ha='center', fontsize=8, style='italic', color='gray')

    plt.tight_layout()

    # Save
    for ext in ['svg', 'png']:
        fig.savefig(FIG_DIR / f'figS1_gsea_enrichment.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: figS1_gsea_enrichment.svg/png")


# ============================================================
# PART 3: Time-Resolved Heatmap (Figure S2)
# ============================================================
def run_ssgsea(counts, ens_map, gene_sets):
    """Run ssGSEA on RNA-seq counts."""
    import gseapy

    print("\n=== Running ssGSEA ===")

    # Map ENSEMBL IDs to gene symbols
    ens_map_dict = dict(zip(ens_map['ensembl_id'], ens_map['gene_symbol']))
    # Also try with version numbers
    counts.index = counts.index.astype(str)

    # Create symbol column
    symbols = []
    for idx in counts.index:
        sym = ens_map_dict.get(idx, None)
        if sym is None:
            # Try without version
            base = idx.split('.')[0]
            sym = ens_map_dict.get(base, None)
        symbols.append(sym)

    counts_sym = counts.copy()
    counts_sym['gene_symbol'] = symbols
    counts_sym = counts_sym.dropna(subset=['gene_symbol'])
    counts_sym = counts_sym[counts_sym['gene_symbol'] != '']

    # Aggregate duplicate symbols (sum counts)
    counts_sym = counts_sym.groupby('gene_symbol').sum(numeric_only=True)
    # Ensure index is string type
    counts_sym.index = counts_sym.index.astype(str)
    print(f"  Mapped genes: {counts_sym.shape[0]} unique symbols, {counts_sym.shape[1]} samples")

    # ssGSEA expects genes as index, samples as columns (NOT transposed)
    expr_matrix = counts_sym  # genes x samples

    # Run ssGSEA
    try:
        out_dir = '/workspace/ssgsea_out'
        os.makedirs(out_dir, exist_ok=True)
        result = gseapy.ssgsea(
            data=expr_matrix,
            gene_sets=gene_sets,
            outdir=out_dir,
            seed=42,
            no_plot=True,
            verbose=False,
        )

        # Get scores
        scores = result.res2d.copy()
        # Pivot: Name (gene set) x Sample
        score_pivot = scores.pivot_table(
            index='Name', columns='Sample', values='NES', aggfunc='first'
        )
        print(f"  ssGSEA scores: {score_pivot.shape}")
        score_pivot.to_csv(DATA_DIR / 'ssgsea_scores.csv')
        return score_pivot
    except Exception as e:
        print(f"  ssGSEA ERROR: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def compute_pathway_mean_log2fc(rna_de, prot_de, gene_sets):
    """Compute mean log2FC for each gene set at each comparison."""
    print("\n=== Computing pathway-level mean log2FC ===")

    results = []

    # RNA-seq
    for comp in ['I4-FP1', 'I4-FP2', 'I4-FP3']:
        sub = rna_de[rna_de['_sheet'] == comp].copy()
        sub = sub.dropna(subset=['gene_symbol', 'pipeline-transcriptome-de_log2FC'])
        sub = sub[sub['gene_symbol'] != '']

        # Aggregate duplicates (take mean log2FC)
        sub = sub.groupby('gene_symbol')['pipeline-transcriptome-de_log2FC'].mean()

        for set_name, genes in gene_sets.items():
            member_genes = [g for g in genes if g in sub.index]
            if len(member_genes) > 0:
                mean_lfc = sub.loc[member_genes].mean()
            else:
                mean_lfc = np.nan
            results.append({
                'gene_set': set_name,
                'comparison': f'RNA {comp}',
                'mean_log2FC': mean_lfc,
                'n_genes_in_set': len(member_genes),
            })

    # Proteomics
    sheet_col = None
    for col in prot_de.columns:
        if prot_de[col].astype(str).str.contains('I4-FP').any():
            sheet_col = col
            break
    if sheet_col is None:
        sheet_col = prot_de.columns[7]

    gene_col = 'Gene' if 'Gene' in prot_de.columns else 'gene_symbol'

    for comp in ['I4-FP1', 'I4-FP2']:
        sub = prot_de[prot_de[sheet_col] == comp].copy()
        sub = sub.dropna(subset=[gene_col, 'logFC'])
        sub = sub[sub[gene_col] != '']
        sub = sub.groupby(gene_col)['logFC'].mean()

        for set_name, genes in gene_sets.items():
            member_genes = [g for g in genes if g in sub.index]
            if len(member_genes) > 0:
                mean_lfc = sub.loc[member_genes].mean()
            else:
                mean_lfc = np.nan
            results.append({
                'gene_set': set_name,
                'comparison': f'Prot {comp}',
                'mean_log2FC': mean_lfc,
                'n_genes_in_set': len(member_genes),
            })

    df = pd.DataFrame(results)
    # Pivot for heatmap
    pivot = df.pivot_table(index='gene_set', columns='comparison', values='mean_log2FC')
    pivot.to_csv(DATA_DIR / 'pathway_mean_log2fc.csv')
    print(f"  Pathway mean log2FC: {pivot.shape}")
    return pivot


def plot_time_resolved_heatmap(ssgsea_scores, pathway_lfc):
    """Generate two-panel time-resolved heatmap (Figure S2)."""
    print("\n=== Generating time-resolved heatmap (Figure S2) ===")

    # Term labels
    term_map = {
        'MINERAL_Calcium': 'Calcium',
        'MINERAL_Copper': 'Copper',
        'MINERAL_Iron': 'Iron',
        'MINERAL_Magnesium': 'Magnesium',
        'MINERAL_Manganese': 'Manganese',
        'MINERAL_Phosphorus': 'Phosphorus',
        'MINERAL_Potassium': 'Potassium',
        'MINERAL_Selenium': 'Selenium',
        'MINERAL_Sodium': 'Sodium',
        'MINERAL_Zinc': 'Zinc',
        'KEGG_hsa00190': 'Oxidative phosphorylation',
        'KEGG_hsa04020': 'Calcium signaling',
        'KEGG_hsa04216': 'Ferroptosis',
        'KEGG_hsa04976': 'Bile secretion',
        'KEGG_hsa04978': 'Mineral absorption',
    }

    mineral_terms = ['Calcium', 'Copper', 'Iron', 'Magnesium', 'Manganese',
                     'Phosphorus', 'Potassium', 'Selenium', 'Sodium', 'Zinc']
    kegg_terms = ['Oxidative phosphorylation', 'Calcium signaling', 'Ferroptosis',
                  'Bile secretion', 'Mineral absorption']
    term_order = mineral_terms + kegg_terms

    # ── Panel A: ssGSEA per-sample ──
    if not ssgsea_scores.empty:
        # Rename rows
        ssgsea_scores.index = [term_map.get(t, t) for t in ssgsea_scores.index]
        # Reorder
        available = [t for t in term_order if t in ssgsea_scores.index]
        ssgsea_scores = ssgsea_scores.loc[available]

        # Parse sample names for annotations
        samples = list(ssgsea_scores.columns)
        timepoints = []
        subjects = []
        for s in samples:
            parts = s.split('_')
            if len(parts) == 2:
                subjects.append(parts[0])
                timepoints.append(parts[1])
            else:
                subjects.append('')
                timepoints.append(s)

        # Sort by timepoint then subject
        tp_order = ['L-92', 'L-44', 'L-3', 'R+1']
        sort_idx = sorted(range(len(samples)),
                         key=lambda i: (tp_order.index(timepoints[i]) if timepoints[i] in tp_order else 99,
                                       subjects[i]))
        ssgsea_scores = ssgsea_scores.iloc[:, sort_idx]
        timepoints = [timepoints[i] for i in sort_idx]
        subjects = [subjects[i] for i in sort_idx]

        # Z-score by row
        ssgsea_z = ssgsea_scores.subtract(ssgsea_scores.mean(axis=1), axis=0)
        std = ssgsea_scores.std(axis=1).replace(0, 1)
        ssgsea_z = ssgsea_z.divide(std, axis=0)
    else:
        ssgsea_z = pd.DataFrame()

    # ── Panel B: Pathway mean log2FC ──
    if not pathway_lfc.empty:
        pathway_lfc.index = [term_map.get(t, t) for t in pathway_lfc.index]
        available = [t for t in term_order if t in pathway_lfc.index]
        pathway_lfc = pathway_lfc.loc[available]

        # Sort columns in logical order
        col_order = ['RNA I4-FP1', 'RNA I4-FP2', 'RNA I4-FP3',
                     'Prot I4-FP1', 'Prot I4-FP2']
        available_cols = [c for c in col_order if c in pathway_lfc.columns]
        pathway_lfc = pathway_lfc[available_cols]
    else:
        pathway_lfc = pd.DataFrame()

    # ── Create figure ──
    n_rows = 15
    fig_height = max(8, n_rows * 0.45 + 3)

    if not ssgsea_z.empty and not pathway_lfc.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, fig_height),
                                       gridspec_kw={'width_ratios': [2.5, 1.2], 'wspace': 0.35})

        # Panel A
        tp_colors = {'L-92': '#4E79A7', 'L-44': '#59A14F', 'L-3': '#F28E2B', 'R+1': '#E15759'}
        subj_colors = {'C001': '#B07AA1', 'C002': '#76B7B2', 'C003': '#EDC948', 'C004': '#FF9DA7'}

        # Create annotation colors
        col_colors_tp = [tp_colors.get(tp, '#CCCCCC') for tp in timepoints]
        col_colors_subj = [subj_colors.get(s, '#CCCCCC') for s in subjects]

        # Build annotation DataFrame for clustermap-style row colors
        from matplotlib.colors import ListedColormap

        sns.heatmap(
            ssgsea_z,
            ax=ax1,
            cmap='RdBu_r',
            center=0,
            vmin=-2, vmax=2,
            xticklabels=[f"{s}\n{tp}" for s, tp in zip(subjects, timepoints)],
            yticklabels=True,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'ssGSEA score (z-scored)', 'shrink': 0.6, 'pad': 0.02}
        )
        ax1.set_title('A. ssGSEA Pathway Activity per Sample', fontsize=11, fontweight='bold', pad=8)
        ax1.set_xlabel('Sample (Subject_Timepoint)', fontsize=9)
        ax1.set_ylabel('Gene Set', fontsize=9)
        ax1.tick_params(axis='x', rotation=45, labelsize=7)
        ax1.tick_params(axis='y', labelsize=9)

        # Add timepoint color bar above Panel A
        for i, (tp, color) in enumerate(zip(timepoints, col_colors_tp)):
            ax1.add_patch(plt.Rectangle((i, -0.8), 1, 0.5, facecolor=color, clip_on=False))
        ax1.text(-1.5, -0.55, 'TP:', fontsize=7, ha='right', va='center')

        # Panel B
        sns.heatmap(
            pathway_lfc,
            ax=ax2,
            cmap='RdBu_r',
            center=0,
            vmin=-0.5, vmax=0.5,
            xticklabels=True,
            yticklabels=True,
            linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Mean log2FC', 'shrink': 0.6, 'pad': 0.02}
        )
        ax2.set_title('B. Pathway Mean log2FC\nAcross Recovery Windows', fontsize=11, fontweight='bold', pad=8)
        ax2.set_xlabel('Comparison', fontsize=9)
        ax2.set_ylabel('')
        ax2.tick_params(axis='x', rotation=45, labelsize=8)
        ax2.tick_params(axis='y', labelsize=9)

        # Add separator line between mineral and KEGG
        ax1.axhline(y=10, color='black', linewidth=1.5, linestyle='--')
        ax2.axhline(y=10, color='black', linewidth=1.5, linestyle='--')

    elif not ssgsea_z.empty:
        fig, ax1 = plt.subplots(figsize=(10, fig_height))
        sns.heatmap(ssgsea_z, ax=ax1, cmap='RdBu_r', center=0, vmin=-2, vmax=2,
                    xticklabels=True, yticklabels=True, linewidths=0.5,
                    cbar_kws={'label': 'ssGSEA score (z-scored)', 'shrink': 0.6})
        ax1.set_title('A. ssGSEA Pathway Activity per Sample', fontsize=11, fontweight='bold')

    elif not pathway_lfc.empty:
        fig, ax2 = plt.subplots(figsize=(8, fig_height))
        sns.heatmap(pathway_lfc, ax=ax2, cmap='RdBu_r', center=0, vmin=-0.5, vmax=0.5,
                    xticklabels=True, yticklabels=True, linewidths=0.5,
                    cbar_kws={'label': 'Mean log2FC', 'shrink': 0.6})
        ax2.set_title('B. Pathway Mean log2FC Across Recovery Windows', fontsize=11, fontweight='bold')

    plt.tight_layout()

    for ext in ['svg', 'png']:
        fig.savefig(FIG_DIR / f'figS2_time_resolved_heatmap.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: figS2_time_resolved_heatmap.svg/png")


# ============================================================
# PART 4: Supplementary DE Tables
# ============================================================
def build_supplementary_tables(de_combined, crosswalk):
    """Build wide and long format supplementary DE tables."""
    print("\n=== Building supplementary DE tables ===")

    # ── Long format (enhanced) ──
    long_df = de_combined.copy()

    # Add KEGG pathways from crosswalk
    cw_dict = crosswalk.set_index('gene_symbol')[['kegg_pathways', 'minerals']].to_dict('index')

    long_df['kegg_pathways'] = long_df['gene_symbol'].map(
        lambda g: cw_dict.get(g, {}).get('kegg_pathways', '')
    )
    # Use minerals from crosswalk if missing in de_combined
    long_df['minerals'] = long_df.apply(
        lambda row: row['minerals'] if pd.notna(row['minerals']) and row['minerals'] != ''
        else cw_dict.get(row['gene_symbol'], {}).get('minerals', ''),
        axis=1
    )

    # Add direction
    long_df['direction'] = long_df['log2FC'].apply(
        lambda x: 'up' if x > 0 else ('down' if x < 0 else 'neutral')
    )

    # Add comparison label
    long_df['comparison_label'] = long_df['comparison'].map(COMPARISON_LABELS).fillna(long_df['comparison'])

    # Add significance flag (|log2FC| > 0.5, p < 0.05)
    long_df['is_significant'] = (
        (long_df['log2FC'].abs() > 0.5) & (long_df['pvalue'] < 0.05)
    )

    # Reorder columns
    long_df = long_df[[
        'gene_symbol', 'omics', 'comparison', 'comparison_label',
        'log2FC', 'pvalue', 'adj_pvalue', 'minerals', 'kegg_pathways',
        'direction', 'is_significant'
    ]]

    long_df.to_csv(DATA_DIR / 'supp_table_de_long.csv', index=False)
    print(f"  Long format: {long_df.shape} -> supp_table_de_long.csv")

    # ── Wide format ──
    # Get all unique genes
    all_genes = sorted(long_df['gene_symbol'].unique())

    # Build wide table
    wide_rows = []
    for gene in all_genes:
        row = {'gene_symbol': gene}

        # Get mineral and pathway info
        gene_rows = long_df[long_df['gene_symbol'] == gene]
        row['minerals'] = gene_rows['minerals'].dropna().iloc[0] if gene_rows['minerals'].notna().any() else ''
        row['kegg_pathways'] = gene_rows['kegg_pathways'].dropna().iloc[0] if gene_rows['kegg_pathways'].notna().any() else ''

        # Fill in stats for each omics x comparison
        for omics in ['RNA-seq', 'Proteomics']:
            omics_label = 'RNA' if omics == 'RNA-seq' else 'Prot'
            for comp in ['I4-FP1', 'I4-FP2', 'I4-FP3']:
                if omics == 'Proteomics' and comp == 'I4-FP3':
                    continue  # No proteomics FP3
                comp_label = comp.replace('I4-', '')
                match = gene_rows[(gene_rows['omics'] == omics) & (gene_rows['comparison'] == comp)]
                if len(match) > 0:
                    m = match.iloc[0]
                    row[f'{omics_label}_{comp_label}_log2FC'] = m['log2FC']
                    row[f'{omics_label}_{comp_label}_pvalue'] = m['pvalue']
                    row[f'{omics_label}_{comp_label}_adj_pvalue'] = m['adj_pvalue']
                else:
                    row[f'{omics_label}_{comp_label}_log2FC'] = np.nan
                    row[f'{omics_label}_{comp_label}_pvalue'] = np.nan
                    row[f'{omics_label}_{comp_label}_adj_pvalue'] = np.nan

        # Direction summary
        lfcs = []
        for col in wide_rows if wide_rows else []:
            pass
        gene_lfcs = [row.get(f'{lab}_{comp}_log2FC') for lab in ['RNA', 'Prot']
                     for comp in ['FP1', 'FP2', 'FP3']
                     if f'{lab}_{comp}_log2FC' in row and pd.notna(row.get(f'{lab}_{comp}_log2FC'))]
        if gene_lfcs:
            signs = [np.sign(x) for x in gene_lfcs]
            if all(s > 0 for s in signs):
                row['direction'] = 'up'
            elif all(s < 0 for s in signs):
                row['direction'] = 'down'
            else:
                row['direction'] = 'mixed'
        else:
            row['direction'] = ''

        # Count significant
        sig_count = 0
        for lab in ['RNA', 'Prot']:
            for comp in ['FP1', 'FP2', 'FP3']:
                lfc_key = f'{lab}_{comp}_log2FC'
                pval_key = f'{lab}_{comp}_pvalue'
                if lfc_key in row and pd.notna(row[lfc_key]) and pval_key in row and pd.notna(row[pval_key]):
                    if abs(row[lfc_key]) > 0.5 and row[pval_key] < 0.05:
                        sig_count += 1
        row['n_significant'] = sig_count

        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows)

    # Reorder columns
    col_order = ['gene_symbol', 'minerals', 'kegg_pathways', 'direction', 'n_significant']
    for lab in ['RNA', 'Prot']:
        for comp in ['FP1', 'FP2', 'FP3']:
            if lab == 'Prot' and comp == 'FP3':
                continue
            col_order.extend([f'{lab}_{comp}_log2FC', f'{lab}_{comp}_pvalue', f'{lab}_{comp}_adj_pvalue'])

    wide_df = wide_df[[c for c in col_order if c in wide_df.columns]]
    wide_df.to_csv(DATA_DIR / 'supp_table_de_wide.csv', index=False)
    print(f"  Wide format: {wide_df.shape} -> supp_table_de_wide.csv")

    return long_df, wide_df


# ============================================================
# PART 5: Copy to Zenodo and manuscript
# ============================================================
def copy_outputs():
    """Copy outputs to Zenodo package and manuscript directory."""
    print("\n=== Copying outputs ===")

    import shutil

    # Figures to Zenodo
    for ext in ['svg', 'png']:
        for fig_name in ['figS1_gsea_enrichment', 'figS2_time_resolved_heatmap']:
            src = FIG_DIR / f'{fig_name}.{ext}'
            if src.exists():
                shutil.copy(src, ZENODO_DIR / 'figures' / src.name)
                print(f"  Copied {fig_name}.{ext} to Zenodo")

    # Figures to manuscript
    for fig_name in ['figS1_gsea_enrichment', 'figS2_time_resolved_heatmap']:
        src = FIG_DIR / f'{fig_name}.png'
        if src.exists():
            shutil.copy(src, MANUSCRIPT_FIG_DIR / src.name)
            print(f"  Copied {fig_name}.png to manuscript")

    # Data to Zenodo
    for data_file in ['gsea_results.csv', 'ssgsea_scores.csv',
                      'pathway_mean_log2fc.csv', 'supp_table_de_wide.csv',
                      'supp_table_de_long.csv']:
        src = DATA_DIR / data_file
        if src.exists():
            shutil.copy(src, ZENODO_DIR / 'data' / data_file)
            print(f"  Copied {data_file} to Zenodo")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("SUPPLEMENTARY OUTPUTS GENERATION")
    print("=" * 60)

    # Load data
    rna_de, prot_de, counts, ens_map, de_combined, crosswalk, gene_sets = load_data()

    # Part 1: GSEA
    gsea_results = run_gsea(rna_de, prot_de, gene_sets)
    plot_gsea_dotplot(gsea_results)

    # Part 2: Time-resolved heatmap
    ssgsea_scores = run_ssgsea(counts, ens_map, gene_sets)
    pathway_lfc = compute_pathway_mean_log2fc(rna_de, prot_de, gene_sets)
    plot_time_resolved_heatmap(ssgsea_scores, pathway_lfc)

    # Part 3: Supplementary tables
    long_df, wide_df = build_supplementary_tables(de_combined, crosswalk)

    # Part 4: Copy outputs
    copy_outputs()

    print("\n" + "=" * 60)
    print("ALL SUPPLEMENTARY OUTPUTS COMPLETE")
    print("=" * 60)
    print(f"\nFigures: {FIG_DIR}")
    print(f"Data: {DATA_DIR}")
    print(f"Zenodo: {ZENODO_DIR}")
