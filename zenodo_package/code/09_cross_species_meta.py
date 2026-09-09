#!/usr/bin/env python3
"""
09_cross_species_meta.py — Multi-study cross-species meta-analytic concordance
between astronaut mineral-pathway DE genes and rodent spaceflight transcriptomics.

Processes 10 liver, 12 muscle, and 6 kidney rodent datasets from NASA OSDR.
Computes per-study concordance (Spearman r, directional %), then combines
via Fisher z random-effects meta-analysis and concordance proportions.

Outputs:
  - Figure S3: Forest plot (per-study r + pooled estimates by tissue)
  - Figure S4: Tissue × pathway heatmap (concordance %)
  - Figure S5: Scatter plot (faceted by tissue)
  - cross_species_meta_analysis.csv (per-study + pooled statistics)
  - cross_species_tissue_pathway.csv (tissue × mineral pathway concordance)
  - mouse_to_human_orthologs.csv (ortholog mapping cache)
"""

import os, sys, re, glob, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, binomtest, norm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rc_params = matplotlib.rcParams
import seaborn as sns
from pathlib import Path
import shutil

rc_params['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
rc_params['svg.fonttype'] = 'none'
warnings.filterwarnings('ignore', category=FutureWarning)

DATA_DIR = Path('/mnt/results/data')
FIG_DIR = Path('/mnt/results/figures')
ZENODO_DIR = Path('/mnt/results/zenodo_package')
MANUSCRIPT_FIG_DIR = Path('/mnt/results/manuscript/figures')
RAW_DIR = Path('/mnt/shared-workspace/shared/osdr_raw')
for d in [FIG_DIR, MANUSCRIPT_FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Study registry ──
STUDIES = {
    # Liver
    'OSD-47':   {'tissue': 'liver', 'subtype': ''},
    'OSD-48':   {'tissue': 'liver', 'subtype': ''},
    'OSD-137':  {'tissue': 'liver', 'subtype': ''},
    'OSD-168':  {'tissue': 'liver', 'subtype': ''},
    'OSD-173':  {'tissue': 'liver', 'subtype': ''},
    'OSD-242':  {'tissue': 'liver', 'subtype': ''},
    'OSD-245':  {'tissue': 'liver', 'subtype': ''},
    'OSD-686':  {'tissue': 'liver', 'subtype': ''},
    'OSD-379':  {'tissue': 'liver', 'subtype': ''},
    'OSD-463':  {'tissue': 'liver', 'subtype': ''},
    # Muscle
    'OSD-99':   {'tissue': 'muscle', 'subtype': 'EDL'},
    'OSD-101':  {'tissue': 'muscle', 'subtype': 'gastrocnemius'},
    'OSD-103':  {'tissue': 'muscle', 'subtype': 'quadriceps'},
    'OSD-104':  {'tissue': 'muscle', 'subtype': 'soleus'},
    'OSD-105':  {'tissue': 'muscle', 'subtype': 'tibialis_anterior'},
    'OSD-326':  {'tissue': 'muscle', 'subtype': 'quadriceps'},
    'OSD-419':  {'tissue': 'muscle', 'subtype': 'gastrocnemius'},
    'OSD-576':  {'tissue': 'muscle', 'subtype': 'tibialis_anterior'},
    'OSD-665':  {'tissue': 'muscle', 'subtype': 'EDL'},
    'OSD-666':  {'tissue': 'muscle', 'subtype': 'quadriceps'},
    'OSD-714':  {'tissue': 'muscle', 'subtype': 'soleus'},
    'OSD-770':  {'tissue': 'muscle', 'subtype': 'soleus'},
    # Kidney
    'OSD-102':  {'tissue': 'kidney', 'subtype': ''},
    'OSD-163':  {'tissue': 'kidney', 'subtype': ''},
    'OSD-253':  {'tissue': 'kidney', 'subtype': ''},
    'OSD-462':  {'tissue': 'kidney', 'subtype': ''},
    'OSD-513':  {'tissue': 'kidney', 'subtype': ''},
    'OSD-771':  {'tissue': 'kidney', 'subtype': ''},
}


# ============================================================
# STEP 1: Parse rodent DE files
# ============================================================
def find_de_file(study_id):
    """Find the DE file for a given study, handling naming variations."""
    patterns = [
        f"{study_id}_*rna_seq_differential_expression_GLbulkRNAseq.csv",
        f"{study_id}_*rna_seq_differential_expression.csv",
        f"{study_id}_*rna_seq_differential_expression_*.csv",
    ]
    for pat in patterns:
        matches = list(RAW_DIR.glob(pat))
        # Exclude rRNArm versions (prefer non-rRNArm)
        non_rrna = [m for m in matches if 'rRNArm' not in m.name]
        if non_rrna:
            return non_rrna[0]
        if matches:
            return matches[0]
    return None


def extract_contrast_columns(columns):
    """Find the log2FC, p-value, and adj p-value columns for Space Flight vs Ground Control.
    Returns (lfc_col, pval_col, padj_col, needs_negation).
    """
    cols_str = [str(c) for c in columns]

    # Strategy 1: Exact match "Log2fc_(Space Flight)v(Ground Control)"
    for c in cols_str:
        if c == 'Log2fc_(Space Flight)v(Ground Control)':
            pval = c.replace('Log2fc_', 'P.value_')
            padj = c.replace('Log2fc_', 'Adj.p.value_')
            if pval in cols_str and padj in cols_str:
                return c, pval, padj, False

    # Strategy 2: Regex — Space Flight ... v ... Ground Control (flight first)
    # Collect all matching columns, then prefer microgravity (uG) over 1G centrifugation
    candidates = []
    for c in cols_str:
        if c.startswith('Log2fc_') and 'Space Flight' in c and 'Ground Control' in c:
            parts = c.split('v(')
            if len(parts) == 2 and 'Space Flight' in parts[0]:
                pval = c.replace('Log2fc_', 'P.value_')
                padj = c.replace('Log2fc_', 'Adj.p.value_')
                if pval in cols_str and padj in cols_str:
                    candidates.append(c)

    if candidates:
        # Prefer microgravity (uG) condition over 1G centrifugation
        ug_candidates = [c for c in candidates if 'uG' in c or 'microgravity' in c.lower()]
        chosen = ug_candidates[0] if ug_candidates else candidates[0]
        pval = chosen.replace('Log2fc_', 'P.value_')
        padj = chosen.replace('Log2fc_', 'Adj.p.value_')
        return chosen, pval, padj, False

    # Strategy 3: Ground Control v Space Flight — negate
    for c in cols_str:
        if c.startswith('Log2fc_') and 'Space Flight' in c and 'Ground Control' in c:
            parts = c.split('v(')
            if len(parts) == 2 and 'Ground Control' in parts[0] and 'Space Flight' in parts[1]:
                pval = c.replace('Log2fc_', 'P.value_')
                padj = c.replace('Log2fc_', 'Adj.p.value_')
                if pval in cols_str and padj in cols_str:
                    return c, pval, padj, True  # negate

    # Strategy 4: Space Flight vs Basal Control (fallback)
    for c in cols_str:
        if c.startswith('Log2fc_') and 'Space Flight' in c and 'Basal Control' in c:
            parts = c.split('v(')
            if len(parts) == 2 and 'Space Flight' in parts[0]:
                pval = c.replace('Log2fc_', 'P.value_')
                padj = c.replace('Log2fc_', 'Adj.p.value_')
                if pval in cols_str and padj in cols_str:
                    return c, pval, padj, False

    # Strategy 5: Basal Control v Space Flight — negate
    for c in cols_str:
        if c.startswith('Log2fc_') and 'Space Flight' in c and 'Basal Control' in c:
            parts = c.split('v(')
            if len(parts) == 2 and 'Basal Control' in parts[0] and 'Space Flight' in parts[1]:
                pval = c.replace('Log2fc_', 'P.value_')
                padj = c.replace('Log2fc_', 'Adj.p.value_')
                if pval in cols_str and padj in cols_str:
                    return c, pval, padj, True

    # Strategy 6: Space Flight vs Vivarium Control (last resort)
    for c in cols_str:
        if c.startswith('Log2fc_') and 'Space Flight' in c and 'Vivarium' in c:
            parts = c.split('v(')
            if len(parts) == 2 and 'Space Flight' in parts[0]:
                pval = c.replace('Log2fc_', 'P.value_')
                padj = c.replace('Log2fc_', 'Adj.p.value_')
                if pval in cols_str and padj in cols_str:
                    return c, pval, padj, False

    return None, None, None, False


def parse_rodent_de():
    """Parse all rodent DE files and extract Space Flight vs Ground Control contrast."""
    print("\n=== Step 1: Parsing rodent DE files ===")
    all_de = {}
    skipped = []

    for study_id, meta in STUDIES.items():
        de_file = find_de_file(study_id)
        if de_file is None:
            print(f"  {study_id}: NO DE FILE FOUND — skipping")
            skipped.append(study_id)
            continue

        try:
            df = pd.read_csv(de_file, index_col=0)
        except Exception as e:
            print(f"  {study_id}: READ ERROR ({e}) — skipping")
            skipped.append(study_id)
            continue

        # Find contrast columns
        lfc_col, pval_col, padj_col, negate = extract_contrast_columns(df.columns)

        if lfc_col is None:
            print(f"  {study_id}: NO CONTRAST FOUND — skipping")
            skipped.append(study_id)
            continue

        # Extract gene symbol and log2FC
        sym_col = 'SYMBOL' if 'SYMBOL' in df.columns else None
        if sym_col is None:
            print(f"  {study_id}: NO SYMBOL COLUMN — skipping")
            skipped.append(study_id)
            continue

        sub = df[[sym_col, lfc_col, pval_col, padj_col]].copy()
        sub.columns = ['gene_symbol', 'log2FC', 'pvalue', 'padj']
        sub = sub.dropna(subset=['gene_symbol'])
        sub['gene_symbol'] = sub['gene_symbol'].astype(str)
        sub = sub[sub['gene_symbol'] != '']

        if negate:
            sub['log2FC'] = -sub['log2FC']

        # Aggregate duplicate symbols (take mean log2FC)
        sub = sub.groupby('gene_symbol').agg({
            'log2FC': 'mean',
            'pvalue': 'min',
            'padj': 'min',
        }).reset_index()

        all_de[study_id] = sub
        print(f"  {study_id}: {len(sub)} genes, contrast={lfc_col[:60]}{'...' if len(lfc_col)>60 else ''}, negate={negate}")

    print(f"\n  Successfully parsed: {len(all_de)}/{len(STUDIES)} studies")
    print(f"  Skipped: {skipped}")
    return all_de


# ============================================================
# STEP 2: Map mouse genes to human orthologs
# ============================================================
def map_mouse_to_human(all_de):
    """Batch-map mouse gene symbols to human orthologs via mygene homologene."""
    print("\n=== Step 2: Mapping mouse → human orthologs ===")

    cache_file = DATA_DIR / 'mouse_to_human_orthologs.csv'
    if cache_file.exists():
        mapping = pd.read_csv(cache_file)
        print(f"  Loaded cached mapping: {len(mapping)} entries")
        mouse_to_human = dict(zip(mapping['mouse_symbol'], mapping['human_symbol']))
        return mouse_to_human

    # Collect all unique mouse symbols
    all_symbols = set()
    for df in all_de.values():
        all_symbols.update(df['gene_symbol'].unique())
    all_symbols = sorted(all_symbols)
    print(f"  Unique mouse symbols: {len(all_symbols)}")

    import mygene
    mg = mygene.MyGeneInfo()
    results = mg.querymany(
        all_symbols,
        scopes='symbol',
        fields='symbol,homologene',
        species='mouse',
        verbose=False,
        returnall=False,
    )

    # homologene['genes'] is a list of [taxid, gene_id] pairs (lists, not dicts)
    # Step 2a: collect human NCBI gene IDs from homologene
    mouse_to_human_gid = {}  # mouse_symbol -> human_entrez_gene_id
    for hit in results:
        query = hit.get('query')
        if not query:
            continue
        homologene = hit.get('homologene')
        if homologene and 'genes' in homologene:
            for g in homologene['genes']:
                if isinstance(g, list) and len(g) >= 2 and g[0] == 9606:
                    mouse_to_human_gid[query] = str(g[1])
                    break

    print(f"  Found human homologene IDs for {len(mouse_to_human_gid)}/{len(all_symbols)} mouse symbols")

    # Step 2b: query human gene IDs to get human symbols
    human_gids = list(set(mouse_to_human_gid.values()))
    if human_gids:
        human_results = mg.querymany(
            human_gids,
            scopes='entrezgene',
            fields='symbol',
            species='human',
            verbose=False,
            returnall=False,
        )
        gid_to_symbol = {}
        for hit in human_results:
            gid = hit.get('query')
            sym = hit.get('symbol')
            if gid and sym:
                gid_to_symbol[gid] = sym

        mouse_to_human = {}
        for ms, gid in mouse_to_human_gid.items():
            if gid in gid_to_symbol:
                mouse_to_human[ms] = gid_to_symbol[gid]
    else:
        mouse_to_human = {}

    print(f"  Mapped: {len(mouse_to_human)}/{len(all_symbols)} ({100*len(mouse_to_human)/len(all_symbols):.1f}%)")

    # Cache
    mapping_df = pd.DataFrame([
        {'mouse_symbol': k, 'human_symbol': v}
        for k, v in mouse_to_human.items()
    ])
    mapping_df.to_csv(cache_file, index=False)
    print(f"  Cached to {cache_file}")

    return mouse_to_human


# ============================================================
# STEP 3: Compute per-study concordance
# ============================================================
def load_astronaut_de():
    """Load astronaut DE and compute mean log2FC per gene across comparisons."""
    rna_de = pd.read_csv(DATA_DIR / 'astronaut_rnaseq_de_mapped.csv')
    crosswalk = pd.read_csv(DATA_DIR / 'mineral_pathway_crosswalk.csv')

    # Filter to mineral-pathway genes
    mineral_genes = set(crosswalk['gene_symbol'].dropna())
    rna_de = rna_de[rna_de['gene_symbol'].isin(mineral_genes)]

    # Use pipeline-transcriptome-de log2FC
    rna_de = rna_de.dropna(subset=['gene_symbol', 'pipeline-transcriptome-de_log2FC'])
    rna_de = rna_de[rna_de['gene_symbol'] != '']

    # Mean log2FC across comparisons per gene
    astro_lfc = rna_de.groupby('gene_symbol')['pipeline-transcriptome-de_log2FC'].mean()
    return astro_lfc, crosswalk


def compute_concordance(all_de, mouse_to_human, astro_lfc, crosswalk):
    """Compute per-study concordance with astronaut mineral-pathway genes."""
    print("\n=== Step 3: Computing per-study concordance ===")

    # Build gene-to-mineral mapping
    gene_to_mineral = {}
    for _, row in crosswalk.iterrows():
        gene = row['gene_symbol']
        minerals = row['minerals']
        if pd.notna(minerals) and isinstance(minerals, str):
            gene_to_mineral[gene] = minerals

    results = []
    per_gene_data = {}  # for scatter plot

    for study_id, df in all_de.items():
        meta = STUDIES[study_id]
        tissue = meta['tissue']
        subtype = meta['subtype']

        # Map mouse symbols to human
        df = df.copy()
        df['human_symbol'] = df['gene_symbol'].map(mouse_to_human)
        df = df.dropna(subset=['human_symbol'])
        df = df[df['human_symbol'] != '']

        # Intersect with astronaut mineral-pathway genes
        shared = set(df['human_symbol']) & set(astro_lfc.index)
        if len(shared) < 10:
            print(f"  {study_id}: only {len(shared)} shared genes — skipping (min 10)")
            continue

        # Build paired data
        astro_vals = []
        rodent_vals = []
        gene_names = []
        for gene in shared:
            astro_vals.append(astro_lfc[gene])
            rodent_vals.append(df[df['human_symbol'] == gene]['log2FC'].iloc[0])
            gene_names.append(gene)

        astro_vals = np.array(astro_vals)
        rodent_vals = np.array(rodent_vals)

        # Spearman correlation
        r, p_spearman = spearmanr(astro_vals, rodent_vals)

        # Fisher z
        if abs(r) < 0.999:
            z = 0.5 * np.log((1 + r) / (1 - r))
            z_var = 1.0 / (len(shared) - 3)
        else:
            z = np.sign(r) * 3.0  # cap
            z_var = 1.0 / (len(shared) - 3)

        # Directional concordance
        same_dir = np.sum((astro_vals > 0) == (rodent_vals > 0))
        n_total = len(shared)
        conc_pct = 100 * same_dir / n_total

        # Binomial test
        p_binom = binomtest(same_dir, n_total, 0.5, alternative='greater').pvalue

        results.append({
            'study_id': study_id,
            'tissue': tissue,
            'muscle_subtype': subtype,
            'n_shared': n_total,
            'spearman_r': r,
            'spearman_p': p_spearman,
            'fisher_z': z,
            'z_var': z_var,
            'concordance_pct': conc_pct,
            'concordance_n': same_dir,
            'binom_p': p_binom,
        })

        # Store per-gene data for scatter plot
        for i, gene in enumerate(gene_names):
            if gene not in per_gene_data:
                per_gene_data[gene] = {'minerals': gene_to_mineral.get(gene, '')}
            per_gene_data[gene][f'{study_id}_lfc'] = rodent_vals[i]

        print(f"  {study_id} ({tissue}): n={n_total}, r={r:.3f}, conc={conc_pct:.1f}%, binom_p={p_binom:.4f}")

    results_df = pd.DataFrame(results)
    return results_df, per_gene_data


# ============================================================
# STEP 4: Meta-analysis
# ============================================================
def fisher_z_meta_analysis(results_df, group_col='tissue', group_val=None):
    """Run Fisher z random-effects meta-analysis for a subset of studies."""
    if group_val:
        subset = results_df[results_df[group_col] == group_val]
    else:
        subset = results_df

    k = len(subset)
    if k < 2:
        return None

    z_vals = subset['fisher_z'].values
    z_vars = subset['z_var'].values
    w = 1.0 / z_vars  # fixed-effect weights

    # Fixed-effect estimate
    z_fe = np.sum(w * z_vals) / np.sum(w)

    # Heterogeneity
    Q = np.sum(w * (z_vals - z_fe) ** 2)
    df = k - 1
    I2 = max(0, (Q - df) / Q) * 100 if Q > 0 else 0

    # DerSimonian-Laird tau²
    tau2_num = max(0, Q - df)
    tau2_den = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = tau2_num / tau2_den if tau2_den > 0 else 0

    # Random-effects weights
    w_re = 1.0 / (z_vars + tau2)
    z_re = np.sum(w_re * z_vals) / np.sum(w_re)
    se_re = 1.0 / np.sqrt(np.sum(w_re))

    # Convert back to r
    r_re = (np.exp(2 * z_re) - 1) / (np.exp(2 * z_re) + 1)
    r_lo = (np.exp(2 * (z_re - 1.96 * se_re)) - 1) / (np.exp(2 * (z_re - 1.96 * se_re)) + 1)
    r_hi = (np.exp(2 * (z_re + 1.96 * se_re)) - 1) / (np.exp(2 * (z_re + 1.96 * se_re)) + 1)

    # Test for overall effect
    z_test = z_re / se_re
    p_test = 2 * norm.sf(abs(z_test))

    return {
        'group': group_val or 'all',
        'k': k,
        'pooled_r': r_re,
        'r_ci_lower': r_lo,
        'r_ci_upper': r_hi,
        'p_value': p_test,
        'Q': Q,
        'I_squared': I2,
        'tau_squared': tau2,
        'z_pooled': z_re,
        'se_pooled': se_re,
    }


def run_meta_analysis(results_df):
    """Run meta-analysis for each tissue group and muscle subtypes."""
    print("\n=== Step 4: Meta-analysis ===")

    meta_results = []

    # By tissue
    for tissue in ['liver', 'muscle', 'kidney']:
        res = fisher_z_meta_analysis(results_df, 'tissue', tissue)
        if res:
            meta_results.append(res)
            print(f"  {tissue}: pooled r={res['pooled_r']:.3f} "
                  f"[{res['r_ci_lower']:.3f}, {res['r_ci_upper']:.3f}], "
                  f"p={res['p_value']:.4f}, I²={res['I_squared']:.1f}%, "
                  f"τ²={res['tau_squared']:.4f}, Q={res['Q']:.2f}")

    # By muscle subtype
    muscle = results_df[results_df['tissue'] == 'muscle']
    for subtype in muscle['muscle_subtype'].unique():
        if subtype == '':
            continue
        res = fisher_z_meta_analysis(results_df, 'muscle_subtype', subtype)
        if res:
            res['group'] = f'muscle_{subtype}'
            meta_results.append(res)
            print(f"  muscle_{subtype}: pooled r={res['pooled_r']:.3f} "
                  f"[{res['r_ci_lower']:.3f}, {res['r_ci_upper']:.3f}], "
                  f"p={res['p_value']:.4f}, I²={res['I_squared']:.1f}%")

    # All studies combined
    res = fisher_z_meta_analysis(results_df)
    if res:
        res['group'] = 'all_studies'
        meta_results.append(res)
        print(f"  all_studies: pooled r={res['pooled_r']:.3f} "
              f"[{res['r_ci_lower']:.3f}, {res['r_ci_upper']:.3f}], "
              f"p={res['p_value']:.4f}")

    meta_df = pd.DataFrame(meta_results)

    # Concordance proportions per tissue
    print("\n  Concordance proportions per tissue:")
    for tissue in ['liver', 'muscle', 'kidney']:
        sub = results_df[results_df['tissue'] == tissue]
        total_conc = sub['concordance_n'].sum()
        total_n = (sub['n_shared']).sum()
        p = binomtest(total_conc, total_n, 0.5, alternative='greater').pvalue
        print(f"    {tissue}: {total_conc}/{total_n} concordant "
              f"({100*total_conc/total_n:.1f}%), binom_p={p:.4f}")

    return meta_df


# ============================================================
# STEP 5: Tissue × pathway concordance
# ============================================================
def compute_tissue_pathway(all_de, mouse_to_human, astro_lfc, crosswalk):
    """Compute concordance by tissue × mineral pathway."""
    print("\n=== Step 5: Tissue × pathway concordance ===")

    # Build gene-to-mineral mapping (primary mineral)
    gene_to_minerals = {}
    gene_to_pathways = {}
    for _, row in crosswalk.iterrows():
        gene = row['gene_symbol']
        minerals = row['minerals']
        pathways = row['kegg_pathways']
        if pd.notna(minerals) and isinstance(minerals, str):
            gene_to_minerals[gene] = [m.strip() for m in minerals.split(';') if m.strip()]
        if pd.notna(pathways) and isinstance(pathways, str):
            gene_to_pathways[gene] = pathways

    # Mineral list
    minerals = ['Calcium', 'Copper', 'Iron', 'Magnesium', 'Manganese',
                'Phosphorus', 'Potassium', 'Selenium', 'Sodium', 'Zinc']
    kegg_pathways = ['hsa00190:Oxidative_phosphorylation', 'hsa04020:Calcium_signaling',
                     'hsa04216:Ferroptosis', 'hsa04976:Bile_secretion', 'hsa04978:Mineral_absorption']
    all_pathways = minerals + kegg_pathways

    # Tissue groups
    tissue_groups = {
        'liver': [sid for sid, m in STUDIES.items() if m['tissue'] == 'liver'],
        'muscle': [sid for sid, m in STUDIES.items() if m['tissue'] == 'muscle'],
        'kidney': [sid for sid, m in STUDIES.items() if m['tissue'] == 'kidney'],
    }
    # Muscle subtypes
    for subtype in ['soleus', 'gastrocnemius', 'tibialis_anterior', 'EDL', 'quadriceps']:
        tissue_groups[f'muscle_{subtype}'] = [
            sid for sid, m in STUDIES.items()
            if m['tissue'] == 'muscle' and m['subtype'] == subtype
        ]

    results = []

    for tissue_name, study_ids in tissue_groups.items():
        if not study_ids:
            continue

        for pathway in all_pathways:
            # Get genes in this pathway
            if pathway in minerals:
                pathway_genes = set()
                for g, mins in gene_to_minerals.items():
                    if pathway in mins:
                        pathway_genes.add(g)
            else:
                # KEGG pathway
                pathway_genes = set()
                for g, pws in gene_to_pathways.items():
                    if pathway in str(pws):
                        pathway_genes.add(g)

            if len(pathway_genes) < 3:
                continue

            # Compute concordance across studies
            total_conc = 0
            total_n = 0
            for sid in study_ids:
                if sid not in all_de:
                    continue
                df = all_de[sid].copy()
                df['human_symbol'] = df['gene_symbol'].map(mouse_to_human)
                df = df.dropna(subset=['human_symbol'])

                shared = set(df['human_symbol']) & pathway_genes & set(astro_lfc.index)
                if len(shared) < 2:
                    continue

                for gene in shared:
                    astro_v = astro_lfc[gene]
                    rodent_v = df[df['human_symbol'] == gene]['log2FC'].iloc[0]
                    if (astro_v > 0) == (rodent_v > 0):
                        total_conc += 1
                    total_n += 1

            if total_n > 0:
                conc_pct = 100 * total_conc / total_n
                p_binom = binomtest(total_conc, total_n, 0.5, alternative='greater').pvalue
            else:
                conc_pct = np.nan
                p_binom = np.nan

            results.append({
                'tissue': tissue_name,
                'pathway': pathway,
                'n_concordant': total_conc,
                'n_total': total_n,
                'concordance_pct': conc_pct,
                'binom_p': p_binom,
            })

    tp_df = pd.DataFrame(results)
    tp_df.to_csv(DATA_DIR / 'cross_species_tissue_pathway.csv', index=False)
    print(f"  Saved tissue × pathway table: {tp_df.shape}")
    return tp_df


# ============================================================
# STEP 6: Generate figures
# ============================================================
def plot_forest(results_df, meta_df):
    """Generate forest plot (Figure S3)."""
    print("\n=== Generating forest plot (Figure S3) ===")

    # Order studies by tissue
    tissue_order = ['liver', 'muscle', 'kidney']
    tissue_colors = {'liver': '#4E79A7', 'muscle': '#F28E2B', 'kidney': '#59A14F'}

    studies = []
    for tissue in tissue_order:
        sub = results_df[results_df['tissue'] == tissue].sort_values('study_id')
        for _, row in sub.iterrows():
            studies.append(row)
        # Add pooled estimate
        pooled = meta_df[meta_df['group'] == tissue]
        if len(pooled) > 0:
            studies.append(pooled.iloc[0].to_dict() | {'is_pooled': True, 'tissue': tissue})

    n_rows = len(studies)
    fig, ax = plt.subplots(figsize=(10, max(8, n_rows * 0.35 + 2)))

    y_positions = list(range(n_rows))[::-1]

    for i, study in enumerate(studies):
        y = y_positions[i]
        is_pooled = study.get('is_pooled', False)

        if is_pooled:
            r = study['pooled_r']
            r_lo = study['r_ci_lower']
            r_hi = study['r_ci_upper']
            label = f"Pooled ({study['tissue']})"
        else:
            r = study['spearman_r']
            # CI from Fisher z
            z = study['fisher_z']
            se = np.sqrt(study['z_var'])
            r_lo = (np.exp(2 * (z - 1.96 * se)) - 1) / (np.exp(2 * (z - 1.96 * se)) + 1)
            r_hi = (np.exp(2 * (z + 1.96 * se)) - 1) / (np.exp(2 * (z + 1.96 * se)) + 1)
            label = f"  {study['study_id']} (n={int(study['n_shared'])})"

        color = tissue_colors.get(study['tissue'], '#999999')

        # Plot point and CI
        if is_pooled:
            ax.plot([r_lo, r_hi], [y, y], color=color, linewidth=2.5)
            ax.plot(r, y, 'D', color=color, markersize=8, markeredgecolor='black', markeredgewidth=0.8)
        else:
            ax.plot([r_lo, r_hi], [y, y], color=color, linewidth=1.2, alpha=0.7)
            ax.plot(r, y, 'o', color=color, markersize=6, markeredgecolor='black', markeredgewidth=0.5)

        # Label
        ax.text(-0.95, y, label, ha='left', va='center', fontsize=8,
                fontweight='bold' if is_pooled else 'normal')

    # Reference line
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # Tissue separators
    prev_tissue = None
    for i, study in enumerate(studies):
        tissue = study['tissue']
        if prev_tissue and tissue != prev_tissue:
            ax.axhline(y=y_positions[i] + 0.5, color='gray', linewidth=0.5, alpha=0.5)
        prev_tissue = tissue

    ax.set_xlabel('Spearman correlation (astronaut vs rodent log2FC)', fontsize=10)
    ax.set_title('Cross-Species Concordance: Mineral-Pathway Gene Expression\n'
                 'Astronaut Blood vs Rodent Tissue (Space Flight vs Ground Control)',
                 fontsize=11, fontweight='bold', pad=10)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.5, n_rows - 0.5)

    # Add heterogeneity stats for pooled estimates
    for i, study in enumerate(studies):
        if study.get('is_pooled', False):
            stats_text = f"I²={study['I_squared']:.0f}%, Q={study['Q']:.1f}, p={study['p_value']:.3f}"
            ax.text(0.95, y_positions[i], stats_text, ha='right', va='center',
                    fontsize=7, style='italic', color='gray')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t.capitalize()) for t, c in tissue_colors.items()]
    legend_elements.append(plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
                                       markersize=8, label='Pooled (random-effects)'))
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    for ext in ['svg', 'png']:
        fig.savefig(FIG_DIR / f'figS3_forest_cross_species.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved figS3_forest_cross_species.svg/png")


def plot_tissue_pathway_heatmap(tp_df):
    """Generate tissue × pathway heatmap (Figure S4)."""
    print("\n=== Generating tissue × pathway heatmap (Figure S4) ===")

    # Pivot
    pivot = tp_df.pivot_table(index='pathway', columns='tissue', values='concordance_pct')

    # Order pathways
    minerals = ['Calcium', 'Copper', 'Iron', 'Magnesium', 'Manganese',
                'Phosphorus', 'Potassium', 'Selenium', 'Sodium', 'Zinc']
    kegg = ['hsa00190:Oxidative_phosphorylation', 'hsa04020:Calcium_signaling',
            'hsa04216:Ferroptosis', 'hsa04976:Bile_secretion', 'hsa04978:Mineral_absorption']
    pathway_order = [p for p in minerals + kegg if p in pivot.index]

    # Order tissues
    tissue_order = ['liver', 'muscle', 'kidney', 'muscle_soleus', 'muscle_gastrocnemius',
                    'muscle_tibialis_anterior', 'muscle_EDL', 'muscle_quadriceps']
    tissue_labels = ['Liver', 'Muscle (all)', 'Kidney', 'Soleus', 'Gastrocnemius',
                     'Tibialis ant.', 'EDL', 'Quadriceps']
    available_tissues = [t for t in tissue_order if t in pivot.columns]
    available_labels = [tissue_labels[tissue_order.index(t)] for t in available_tissues]

    pivot = pivot.loc[pathway_order, available_tissues]

    # Rename pathway labels
    pathway_labels = {
        'hsa00190:Oxidative_phosphorylation': 'Oxidative phosphorylation',
        'hsa04020:Calcium_signaling': 'Calcium signaling',
        'hsa04216:Ferroptosis': 'Ferroptosis',
        'hsa04976:Bile_secretion': 'Bile secretion',
        'hsa04978:Mineral_absorption': 'Mineral absorption',
    }
    y_labels = [pathway_labels.get(p, p) for p in pathway_order]

    # Significance markers
    sig_pivot = tp_df.pivot_table(index='pathway', columns='tissue', values='binom_p')
    sig_pivot = sig_pivot.loc[pathway_order, available_tissues]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pivot, ax=ax, cmap='RdBu_r', center=50, vmin=30, vmax=70,
        xticklabels=available_labels, yticklabels=y_labels,
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': 'Directional concordance (%)', 'shrink': 0.7},
        annot=True, fmt='.0f', annot_kws={'fontsize': 8},
    )

    # Add significance markers
    for i, pathway in enumerate(pathway_order):
        for j, tissue in enumerate(available_tissues):
            p_val = sig_pivot.loc[pathway, tissue]
            if pd.notna(p_val) and p_val < 0.05:
                ax.text(j + 0.5, i + 0.8, '*', ha='center', va='center',
                        fontsize=10, fontweight='bold', color='black')

    ax.set_title('Tissue-Specific Mineral-Pathway Concordance\n'
                 '(% genes with same directional change: astronaut vs rodent)',
                 fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Tissue', fontsize=10)
    ax.set_ylabel('Pathway', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=9)

    # Separator line between minerals and KEGG
    n_minerals = len([p for p in minerals if p in pathway_order])
    ax.axhline(y=n_minerals, color='black', linewidth=1.5, linestyle='--')

    plt.tight_layout()
    for ext in ['svg', 'png']:
        fig.savefig(FIG_DIR / f'figS4_tissue_pathway_heatmap.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved figS4_tissue_pathway_heatmap.svg/png")


def plot_scatter_faceted(all_de, mouse_to_human, astro_lfc, crosswalk):
    """Generate faceted scatter plot (Figure S5)."""
    print("\n=== Generating scatter plot (Figure S5) ===")

    # Build gene-to-mineral mapping for coloring
    gene_to_mineral = {}
    for _, row in crosswalk.iterrows():
        gene = row['gene_symbol']
        minerals = row['minerals']
        if pd.notna(minerals) and isinstance(minerals, str):
            primary = minerals.split(';')[0].strip()
            gene_to_mineral[gene] = primary

    mineral_colors = {
        'Calcium': '#E15759', 'Copper': '#B07AA1', 'Iron': '#D62728',
        'Magnesium': '#59A14F', 'Manganese': '#9C755F', 'Phosphorus': '#EDC948',
        'Potassium': '#4E79A7', 'Selenium': '#FF9DA7', 'Sodium': '#76B7B2',
        'Zinc': '#F28E2B', '': '#CCCCCC',
    }

    tissues = ['liver', 'muscle', 'kidney']
    tissue_labels = {'liver': 'Liver', 'muscle': 'Muscle', 'kidney': 'Kidney'}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for ax_idx, tissue in enumerate(tissues):
        ax = axes[ax_idx]
        study_ids = [sid for sid, m in STUDIES.items() if m['tissue'] == tissue]

        # Pool rodent log2FC across studies (mean per gene)
        gene_lfcs = {}
        for sid in study_ids:
            if sid not in all_de:
                continue
            df = all_de[sid].copy()
            df['human_symbol'] = df['gene_symbol'].map(mouse_to_human)
            df = df.dropna(subset=['human_symbol'])
            for _, row in df.iterrows():
                gene = row['human_symbol']
                if gene in astro_lfc.index:
                    if gene not in gene_lfcs:
                        gene_lfcs[gene] = []
                    gene_lfcs[gene].append(row['log2FC'])

        # Compute mean rodent log2FC per gene
        genes = []
        astro_vals = []
        rodent_vals = []
        colors = []
        for gene, lfcs in gene_lfcs.items():
            if len(lfcs) == 0:
                continue
            genes.append(gene)
            astro_vals.append(astro_lfc[gene])
            rodent_vals.append(np.mean(lfcs))
            colors.append(mineral_colors.get(gene_to_mineral.get(gene, ''), '#CCCCCC'))

        astro_vals = np.array(astro_vals)
        rodent_vals = np.array(rodent_vals)

        if len(astro_vals) > 2:
            r, p = spearmanr(astro_vals, rodent_vals)
        else:
            r, p = np.nan, np.nan

        # Scatter
        ax.scatter(astro_vals, rodent_vals, c=colors, alpha=0.6, s=30,
                   edgecolors='black', linewidth=0.3)

        # Reference line y=x
        lim = max(abs(astro_vals).max(), abs(rodent_vals).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=0.8, alpha=0.3)

        # Zero lines
        ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.3)

        ax.set_xlabel('Astronaut log2FC', fontsize=9)
        if ax_idx == 0:
            ax.set_ylabel('Rodent log2FC (pooled mean)', fontsize=9)
        ax.set_title(f'{tissue_labels[tissue]}\n(n={len(genes)} genes, '
                     f'ρ={r:.3f}, p={p:.3f})', fontsize=10, fontweight='bold')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend (mineral colors)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=m) for m, c in mineral_colors.items() if m
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=7, title='Mineral', title_fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Cross-Species Mineral-Pathway Gene Expression Concordance',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    for ext in ['svg', 'png']:
        fig.savefig(FIG_DIR / f'figS5_scatter_cross_species.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved figS5_scatter_cross_species.svg/png")


# ============================================================
# STEP 7: Save and copy
# ============================================================
def save_and_copy(results_df, meta_df, tp_df):
    """Save all data files and copy to Zenodo/manuscript."""
    print("\n=== Saving and copying outputs ===")

    # Merge per-study results with pooled estimates
    results_out = results_df.copy()
    # Add tissue-level pooled r
    for tissue in ['liver', 'muscle', 'kidney']:
        pooled = meta_df[meta_df['group'] == tissue]
        if len(pooled) > 0:
            results_out.loc[results_out['tissue'] == tissue, 'pooled_r_tissue'] = pooled.iloc[0]['pooled_r']
            results_out.loc[results_out['tissue'] == tissue, 'pooled_r_ci_lower'] = pooled.iloc[0]['r_ci_lower']
            results_out.loc[results_out['tissue'] == tissue, 'pooled_r_ci_upper'] = pooled.iloc[0]['r_ci_upper']
            results_out.loc[results_out['tissue'] == tissue, 'Q_stat'] = pooled.iloc[0]['Q']
            results_out.loc[results_out['tissue'] == tissue, 'I_squared'] = pooled.iloc[0]['I_squared']
            results_out.loc[results_out['tissue'] == tissue, 'tau_squared'] = pooled.iloc[0]['tau_squared']

    results_out.to_csv(DATA_DIR / 'cross_species_meta_analysis.csv', index=False)
    print(f"  Saved cross_species_meta_analysis.csv: {results_out.shape}")

    meta_df.to_csv(DATA_DIR / 'cross_species_meta_summary.csv', index=False)
    print(f"  Saved cross_species_meta_summary.csv: {meta_df.shape}")

    # Copy to Zenodo
    for fname in ['cross_species_meta_analysis.csv', 'cross_species_meta_summary.csv',
                  'cross_species_tissue_pathway.csv', 'mouse_to_human_orthologs.csv']:
        src = DATA_DIR / fname
        if src.exists():
            shutil.copy(src, ZENODO_DIR / 'data' / fname)
            print(f"  Copied {fname} to Zenodo")

    # Copy figures
    for fig_name in ['figS3_forest_cross_species', 'figS4_tissue_pathway_heatmap',
                     'figS5_scatter_cross_species']:
        for ext in ['svg', 'png']:
            src = FIG_DIR / f'{fig_name}.{ext}'
            if src.exists():
                shutil.copy(src, ZENODO_DIR / 'figures' / src.name)
        # PNG to manuscript
        src_png = FIG_DIR / f'{fig_name}.png'
        if src_png.exists():
            shutil.copy(src_png, MANUSCRIPT_FIG_DIR / src_png.name)
            print(f"  Copied {fig_name}.png to manuscript")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("CROSS-SPECIES META-ANALYTIC CONCORDANCE")
    print("=" * 60)

    # Step 1: Parse rodent DE files
    all_de = parse_rodent_de()

    # Step 2: Map mouse to human
    mouse_to_human = map_mouse_to_human(all_de)

    # Step 3: Load astronaut DE and compute concordance
    astro_lfc, crosswalk = load_astronaut_de()
    print(f"\n  Astronaut mineral-pathway genes: {len(astro_lfc)}")
    results_df, per_gene_data = compute_concordance(all_de, mouse_to_human, astro_lfc, crosswalk)

    # Step 4: Meta-analysis
    meta_df = run_meta_analysis(results_df)

    # Step 5: Tissue × pathway concordance
    tp_df = compute_tissue_pathway(all_de, mouse_to_human, astro_lfc, crosswalk)

    # Step 6: Generate figures
    plot_forest(results_df, meta_df)
    plot_tissue_pathway_heatmap(tp_df)
    plot_scatter_faceted(all_de, mouse_to_human, astro_lfc, crosswalk)

    # Step 7: Save and copy
    save_and_copy(results_df, meta_df, tp_df)

    print("\n" + "=" * 60)
    print("CROSS-SPECIES META-ANALYSIS COMPLETE")
    print("=" * 60)
