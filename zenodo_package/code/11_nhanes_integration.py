#!/usr/bin/env python3
"""
11_nhanes_integration.py
NHANES Population Baseline Integration for Astronaut Mineral/CBC Analysis

Downloads NHANES 2013-2018 (3 cycles) Standard Biochemistry Profile,
Complete Blood Count, and Demographics files from CDC. Pools them into
an age 40-60 reference population (all sexes). Computes z-scores,
percentile ranks, reference interval overlays, and one-sample tests
for 15 matched astronaut clinical variables across 7 timepoints.

Outputs:
  - nhanes_reference_stats.csv
  - astronaut_zscores.csv
  - astronaut_percentile_ranks.csv
  - nhanes_onesample_tests.csv
  - fig_s6_nhanes_trajectories.svg/.png
  - fig_s7_zscore_heatmap.svg/.png
  - fig_s8_percentile_ranks.svg/.png
"""

import urllib.request
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from statsmodels.stats.multitest import multipletests
from matplotlib.lines import Line2D

# ── Configuration ──────────────────────────────────────────────────
DATA_DIR = '/mnt/results/data'
FIG_DIR = '/mnt/results/figures'
RAW_DIR = '/workspace/nhanes_raw'
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

rc_params = matplotlib.rcParams
rc_params['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
rc_params['svg.fonttype'] = 'none'
rc_params['figure.dpi'] = 150

CYCLES = {
    '2013-2014': {'year_dir': '2013', 'suffix': '_H'},
    '2015-2016': {'year_dir': '2015', 'suffix': '_I'},
    '2017-2018': {'year_dir': '2017', 'suffix': '_J'},
}
FILE_TYPES = ['BIOPRO', 'CBC', 'DEMO']
TP_ORDER = ['L-92', 'L-44', 'L-3', 'R+1', 'R+45', 'R+82', 'R+194']

VAR_MAP = {
    'CALCIUM':                ('LBXSCA',    'Total Calcium', 'mg/dL'),
    'POTASSIUM':              ('LBXSKSI',   'Potassium', 'mmol/L'),
    'SODIUM':                 ('LBXSNASI',  'Sodium', 'mmol/L'),
    'CHLORIDE':               ('LBXSCLSI',  'Chloride', 'mmol/L'),
    'CARBON DIOXIDE':         ('LBXSC3SI',  'Bicarbonate', 'mmol/L'),
    'HEMOGLOBIN':             ('LBXHGB',    'Hemoglobin', 'g/dL'),
    'HEMATOCRIT':             ('LBXHCT',    'Hematocrit', '%'),
    'RED BLOOD CELL COUNT':   ('LBXRBCSI',  'RBC count', 'M/uL'),
    'MCH':                    ('LBXMCHSI',  'Mean cell Hgb', 'pg'),
    'MCHC':                   ('LBXMC',     'Mean Cell Hgb Conc', 'g/dL'),
    'MCV':                    ('LBXMCVSI',  'Mean cell volume', 'fL'),
    'WHITE BLOOD CELL COUNT': ('LBXWBCSI',  'WBC count', '10^3/uL'),
    'RDW':                    ('LBXRDW',    'RBC distribution width', '%'),
    'PLATELET COUNT':         ('LBXPLTSI',  'Platelet count', '10^3/uL'),
    'MPV':                    ('LBXMPSI',   'Mean platelet volume', 'fL'),
}

CLINICAL_RANGES = {
    'CALCIUM': (6, 14), 'POTASSIUM': (2, 8), 'SODIUM': (120, 160),
    'CHLORIDE': (80, 120), 'CARBON DIOXIDE': (10, 40),
    'HEMOGLOBIN': (5, 22), 'HEMATOCRIT': (15, 65), 'RED BLOOD CELL COUNT': (2, 8),
    'MCH': (15, 45), 'MCHC': (25, 40), 'MCV': (50, 120),
    'WHITE BLOOD CELL COUNT': (1, 50), 'RDW': (10, 25),
    'PLATELET COUNT': (50, 600), 'MPV': (5, 15),
}

DISPLAY_INFO = {
    'CALCIUM': ('Calcium', 'mg/dL'), 'POTASSIUM': ('Potassium', 'mmol/L'),
    'SODIUM': ('Sodium', 'mmol/L'), 'CHLORIDE': ('Chloride', 'mmol/L'),
    'CARBON DIOXIDE': ('Bicarbonate (CO$_2$)', 'mmol/L'),
    'HEMOGLOBIN': ('Hemoglobin', 'g/dL'), 'HEMATOCRIT': ('Hematocrit', '%'),
    'RED BLOOD CELL COUNT': ('RBC Count', 'M/uL'), 'MCH': ('MCH', 'pg'),
    'MCHC': ('MCHC', 'g/dL'), 'MCV': ('MCV', 'fL'),
    'WHITE BLOOD CELL COUNT': ('WBC Count', '10$^3$/uL'),
    'RDW': ('RDW', '%'), 'PLATELET COUNT': ('Platelet Count', '10$^3$/uL'),
    'MPV': ('MPV', 'fL'),
}

BIOCHEM_VARS = ['CALCIUM', 'POTASSIUM', 'SODIUM', 'CHLORIDE', 'CARBON DIOXIDE']
HEMATOLOGY_VARS = ['HEMOGLOBIN', 'HEMATOCRIT', 'RED BLOOD CELL COUNT', 'MCH', 'MCHC', 'MCV',
                   'WHITE BLOOD CELL COUNT', 'RDW', 'PLATELET COUNT', 'MPV']
ALL_VARS_ORDERED = BIOCHEM_VARS + HEMATOLOGY_VARS

CREW_COLORS = {'C001': '#0279EE', 'C002': '#75A025', 'C003': '#FF9400', 'C004': '#FD9BED'}
TP_COLORS = {
    'L-92': '#0279EE', 'L-44': '#4A9FEE', 'L-3': '#75A025',
    'R+1': '#FF9400', 'R+45': '#E97400', 'R+82': '#CC5500', 'R+194': '#FD9BED',
}


def download_nhanes():
    """Download all 9 NHANES XPT files."""
    for cycle_name, info in CYCLES.items():
        for ftype in FILE_TYPES:
            fname = f'{ftype}{info["suffix"]}'
            outpath = os.path.join(RAW_DIR, f'{fname}.XPT')
            if os.path.exists(outpath) and os.path.getsize(outpath) > 100000:
                continue
            url = f'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{info["year_dir"]}/DataFiles/{fname}.XPT'
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            with open(outpath, 'wb') as f:
                f.write(data)
            print(f'  Downloaded {fname}: {len(data)} bytes')


def build_reference_population():
    """Parse, merge, and filter NHANES data to age 40-60 reference population."""
    all_merged = []
    for cycle_name, info in CYCLES.items():
        suffix = info['suffix']
        demo = pd.read_sas(os.path.join(RAW_DIR, f'DEMO{suffix}.XPT'), format='xport')
        biopro = pd.read_sas(os.path.join(RAW_DIR, f'BIOPRO{suffix}.XPT'), format='xport')
        cbc = pd.read_sas(os.path.join(RAW_DIR, f'CBC{suffix}.XPT'), format='xport')

        demo_cols = ['SEQN', 'RIAGENDR', 'RIDAGEYR']
        biopro_cols = ['SEQN'] + [v[0] for v in VAR_MAP.values() if v[0].startswith('LBXS')]
        cbc_cols = ['SEQN'] + [v[0] for v in VAR_MAP.values() if not v[0].startswith('LBXS')]

        m = demo[demo_cols].merge(biopro[biopro_cols], on='SEQN', how='inner')
        m = m.merge(cbc[cbc_cols], on='SEQN', how='inner')
        m['cycle'] = cycle_name
        all_merged.append(m)

    pooled = pd.concat(all_merged, ignore_index=True)
    ref_pop = pooled[(pooled['RIDAGEYR'] >= 40) & (pooled['RIDAGEYR'] <= 60)].copy()
    return ref_pop


def compute_reference_stats(ref_pop):
    """Compute per-variable, per-sex reference statistics."""
    percentiles = [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]
    ref_rows = []
    for astro_var, (nhanes_code, label, units) in VAR_MAP.items():
        for sex_label, sex_filter in [('All', None), ('Male', 1.0), ('Female', 2.0)]:
            subset = ref_pop if sex_filter is None else ref_pop[ref_pop['RIAGENDR'] == sex_filter]
            vals = subset[nhanes_code].dropna()
            row = {'astronaut_variable': astro_var, 'nhanes_code': nhanes_code,
                   'label': label, 'units': units, 'sex': sex_label,
                   'N': len(vals), 'mean': vals.mean(), 'sd': vals.std(), 'median': vals.median()}
            for p in percentiles:
                row[f'P{p}'] = vals.quantile(p / 100)
            ref_rows.append(row)
    return pd.DataFrame(ref_rows)


def load_and_clean_astronaut():
    """Load astronaut data, merge minerals + CBC, clean outliers."""
    astro_minerals = pd.read_csv(os.path.join(DATA_DIR, 'astronaut_minerals_combined.csv'))
    astro_cbc = pd.read_csv(os.path.join(DATA_DIR, 'astronaut_cbc_full.csv'))
    astro = astro_minerals.merge(
        astro_cbc[['SUBJECT_ID', 'timepoint', 'WHITE BLOOD CELL COUNT', 'RDW', 'PLATELET COUNT', 'MPV']],
        on=['SUBJECT_ID', 'timepoint'], how='outer')
    for var, (lo, hi) in CLINICAL_RANGES.items():
        if var in astro.columns:
            astro.loc[(astro[var] < lo) | (astro[var] > hi), var] = np.nan
    return astro


def compute_zscores_and_percentiles(astro, ref_pop, ref_all):
    """Compute z-scores and empirical percentile ranks."""
    for astro_var, (nhanes_code, label, units) in VAR_MAP.items():
        if astro_var not in astro.columns:
            continue
        ref_mean = ref_all.loc[astro_var, 'mean']
        ref_sd = ref_all.loc[astro_var, 'sd']
        ref_p025 = ref_all.loc[astro_var, 'P2.5']
        ref_p975 = ref_all.loc[astro_var, 'P97.5']

        astro[f'{astro_var}_z'] = (astro[astro_var] - ref_mean) / ref_sd
        nhanes_vals = np.sort(ref_pop[nhanes_code].dropna().values)
        astro[f'{astro_var}_pctile'] = astro[astro_var].apply(
            lambda x: np.nan if pd.isna(x) else np.searchsorted(nhanes_vals, x, side='right') / len(nhanes_vals) * 100)
        astro[f'{astro_var}_outside_ref'] = ((astro[astro_var] < ref_p025) | (astro[astro_var] > ref_p975)).astype(int)
        astro.loc[astro[astro_var].isna(), f'{astro_var}_outside_ref'] = np.nan
    return astro


def run_onesample_tests(astro, ref_all):
    """One-sample t-test + Wilcoxon at each timepoint, FDR-corrected."""
    test_rows = []
    for astro_var, (nhanes_code, label, units) in VAR_MAP.items():
        if astro_var not in astro.columns:
            continue
        ref_mean = ref_all.loc[astro_var, 'mean']
        ref_sd = ref_all.loc[astro_var, 'sd']
        ref_p025 = ref_all.loc[astro_var, 'P2.5']
        ref_p975 = ref_all.loc[astro_var, 'P97.5']
        for tp in TP_ORDER:
            tp_data = astro[astro['timepoint'] == tp][astro_var].dropna()
            n = len(tp_data)
            if n < 2:
                test_rows.append({'variable': astro_var, 'timepoint': tp, 'n': n,
                    'astro_mean': tp_data.mean() if n == 1 else np.nan,
                    'nhanes_mean': ref_mean, 'nhanes_sd': ref_sd,
                    'cohen_d': np.nan, 't_stat': np.nan, 't_pvalue': np.nan,
                    'wilcoxon_stat': np.nan, 'wilcoxon_pvalue': np.nan,
                    'mean_z': np.nan, 'pct_outside_ref': np.nan})
                continue
            t_stat, t_p = stats.ttest_1samp(tp_data, ref_mean)
            try:
                w_stat, w_p = stats.wilcoxon(tp_data - ref_mean)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            cohen_d = (tp_data.mean() - ref_mean) / ref_sd
            outside = ((tp_data < ref_p025) | (tp_data > ref_p975)).sum()
            test_rows.append({'variable': astro_var, 'timepoint': tp, 'n': n,
                'astro_mean': tp_data.mean(), 'astro_sd': tp_data.std(),
                'nhanes_mean': ref_mean, 'nhanes_sd': ref_sd,
                'cohen_d': cohen_d, 't_stat': t_stat, 't_pvalue': t_p,
                'wilcoxon_stat': w_stat, 'wilcoxon_pvalue': w_p,
                'mean_z': cohen_d, 'pct_outside_ref': 100 * outside / n})
    test_df = pd.DataFrame(test_rows)
    valid = test_df['t_pvalue'].notna()
    _, fdr_p, _, _ = multipletests(test_df.loc[valid, 't_pvalue'], method='fdr_bh')
    test_df.loc[valid, 't_fdr'] = fdr_p
    _, fdr_w, _, _ = multipletests(test_df.loc[valid, 'wilcoxon_pvalue'].fillna(1.0), method='fdr_bh')
    test_df.loc[valid, 'wilcoxon_fdr'] = fdr_w
    return test_df


def plot_trajectories(astro, ref_all):
    """Fig S6: trajectory plots with NHANES reference bands."""
    tp_x = list(range(len(TP_ORDER)))
    tp_x_map = {tp: i for i, tp in enumerate(TP_ORDER)}
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes_flat = axes.flatten()
    for idx, var in enumerate(ALL_VARS_ORDERED):
        ax = axes_flat[idx]
        label, units = DISPLAY_INFO[var]
        ax.axhspan(ref_all.loc[var, 'P2.5'], ref_all.loc[var, 'P97.5'], alpha=0.4,
                   color='#ECE9E2', edgecolor='#CCCCCC', linewidth=0.5, zorder=0)
        ax.axhspan(ref_all.loc[var, 'P5'], ref_all.loc[var, 'P95'], alpha=0.2,
                   color='#E9ED4C', edgecolor='none', zorder=0)
        ax.axhline(ref_all.loc[var, 'mean'], color='#999999', linestyle='--', linewidth=0.8, zorder=1)
        for crew in ['C001', 'C002', 'C003', 'C004']:
            cd = astro[astro['SUBJECT_ID'] == crew].sort_values('timepoint')
            cx = [tp_x_map[t] for t in cd['timepoint'] if t in tp_x_map]
            cy = [cd[cd['timepoint'] == t][var].values[0] for t in cd['timepoint'] if t in tp_x_map]
            valid = [(x, y) for x, y in zip(cx, cy) if not np.isnan(y)]
            if valid:
                vx, vy = zip(*valid)
                ax.plot(vx, vy, 'o-', color=CREW_COLORS[crew], markersize=4, linewidth=1.2, alpha=0.7, label=crew)
        mean_vals = [astro[astro['timepoint'] == tp][var].dropna().mean() for tp in TP_ORDER]
        ax.plot(tp_x, mean_vals, 's--', color='#000000', markersize=5, linewidth=2, label='Crew mean', zorder=5)
        ax.axvspan(tp_x_map['R+1'] - 0.3, tp_x_map['R+1'] + 0.3, alpha=0.08, color='#FF9400', zorder=0)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_ylabel(units, fontsize=9)
        ax.set_xticks(tp_x)
        ax.set_xticklabels(TP_ORDER, fontsize=7, rotation=45, ha='right')
        if idx == 0:
            ax.legend(fontsize=7, loc='best', framealpha=0.8)
    for idx in range(len(ALL_VARS_ORDERED), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    fig.suptitle('Astronaut Mineral & CBC Trajectories with NHANES Population Reference Bands\n'
                 '(Age 40-60, All Sexes, Pooled 2013-2018)', fontsize=14, fontweight='bold', y=0.98)
    fig.text(0.5, 0.955, 'Grey band: P2.5-P97.5 | Yellow band: P5-P95 | Dashed line: NHANES mean | '
             'Orange shading: flight (R+1)', ha='center', fontsize=9, color='#666666')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG_DIR, 'fig_s6_nhanes_trajectories.svg'), format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig_s6_nhanes_trajectories.png'), format='png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_zscore_heatmap(astro):
    """Fig S7: z-score heatmap."""
    z_matrix = []
    var_labels = []
    for var in ALL_VARS_ORDERED:
        z_col = f'{var}_z'
        row = [astro[astro['timepoint'] == tp][z_col].dropna().mean()
               if astro[astro['timepoint'] == tp][z_col].notna().any() else np.nan
               for tp in TP_ORDER]
        z_matrix.append(row)
        var_labels.append(DISPLAY_INFO[var][0])
    z_arr = np.array(z_matrix)
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(z_arr, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
    for i in range(z_arr.shape[0]):
        for j in range(z_arr.shape[1]):
            val = z_arr[i, j]
            if np.isnan(val):
                ax.text(j, i, '--', ha='center', va='center', fontsize=8, color='#999999')
            else:
                weight = 'bold' if abs(val) >= 1.96 else 'normal'
                color = 'white' if abs(val) >= 1.5 else 'black'
                ax.text(j, i, f'{val:+.2f}', ha='center', va='center', fontsize=8, fontweight=weight, color=color)
    ax.set_xticks(range(len(TP_ORDER)))
    ax.set_xticklabels(TP_ORDER, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(range(len(var_labels)))
    ax.set_yticklabels(var_labels, fontsize=10)
    ax.axvline(x=2.5, color='#333333', linewidth=1.5, linestyle='-')
    ax.axvline(x=3.5, color='#333333', linewidth=1.5, linestyle='-')
    ax.text(1, -1.2, 'Pre-flight', ha='center', fontsize=9, color='#666666')
    ax.text(3, -1.2, 'Flight', ha='center', fontsize=9, color='#FF9400', fontweight='bold')
    ax.text(5, -1.2, 'Post-flight', ha='center', fontsize=9, color='#666666')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Mean Z-score (vs NHANES)', fontsize=10)
    cbar.ax.axhline(y=1.96, color='black', linewidth=0.8, linestyle='--')
    cbar.ax.axhline(y=-1.96, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Astronaut Z-Score Heatmap vs NHANES Population Reference\n'
                 '(Age 40-60, All Sexes, Pooled 2013-2018)', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_s7_zscore_heatmap.svg'), format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig_s7_zscore_heatmap.png'), format='png', dpi=200, bbox_inches='tight')
    plt.close()


def plot_percentile_ranks(astro):
    """Fig S8: percentile rank dot plot."""
    tp_x_map = {tp: i for i, tp in enumerate(TP_ORDER)}
    var_x_map = {var: i for i, var in enumerate(ALL_VARS_ORDERED)}
    fig, ax = plt.subplots(figsize=(14, 8))
    for _, row in astro.iterrows():
        for var in ALL_VARS_ORDERED:
            pct_col = f'{var}_pctile'
            if pct_col in row.index and not pd.isna(row[pct_col]):
                x = var_x_map[var] + (tp_x_map[row['timepoint']] - 3) * 0.08
                ax.scatter(x, row[pct_col], color=TP_COLORS[row['timepoint']], s=25, alpha=0.6, edgecolors='none', zorder=3)
    ax.axhspan(2.5, 97.5, alpha=0.08, color='#ECE9E2', zorder=0)
    ax.axhspan(5, 95, alpha=0.06, color='#E9ED4C', zorder=0)
    ax.axhline(50, color='#999999', linestyle='--', linewidth=0.8, zorder=1)
    ax.axhline(2.5, color='#CCCCCC', linestyle=':', linewidth=0.5, zorder=1)
    ax.axhline(97.5, color='#CCCCCC', linestyle=':', linewidth=0.5, zorder=1)
    ax.set_xticks(range(len(ALL_VARS_ORDERED)))
    ax.set_xticklabels([DISPLAY_INFO[v][0] for v in ALL_VARS_ORDERED], fontsize=9, rotation=45, ha='right')
    ax.set_ylabel('NHANES Percentile Rank (%)', fontsize=11)
    ax.set_ylim(-2, 102)
    ax.set_xlim(-0.5, len(ALL_VARS_ORDERED) - 0.5)
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=TP_COLORS[tp], markersize=7, label=tp)
                       for tp in TP_ORDER]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8, ncol=4, framealpha=0.8)
    ax.axvline(x=4.5, color='#333333', linewidth=0.8, linestyle='-', alpha=0.3)
    ax.text(2, 101, 'Biochemistry', ha='center', fontsize=9, color='#666666')
    ax.text(9.5, 101, 'Hematology', ha='center', fontsize=9, color='#666666')
    ax.set_title('Astronaut Percentile Ranks vs NHANES Population Reference\n'
                 '(Age 40-60, All Sexes, Pooled 2013-2018)', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig_s8_percentile_ranks.svg'), format='svg', bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig_s8_percentile_ranks.png'), format='png', dpi=200, bbox_inches='tight')
    plt.close()


def main():
    print('=== NHANES Population Baseline Integration ===')

    # Step 1: Download
    print('\n1. Downloading NHANES XPT files...')
    download_nhanes()

    # Step 2: Build reference population
    print('\n2. Building reference population (age 40-60)...')
    ref_pop = build_reference_population()
    print(f'   Reference population: {len(ref_pop)} participants')

    # Step 3: Compute reference stats
    print('\n3. Computing reference statistics...')
    ref_stats = compute_reference_stats(ref_pop)
    ref_stats.to_csv(os.path.join(DATA_DIR, 'nhanes_reference_stats.csv'), index=False)
    ref_all = ref_stats[ref_stats['sex'] == 'All'].set_index('astronaut_variable')

    # Step 4: Load & clean astronaut data
    print('\n4. Loading and cleaning astronaut data...')
    astro = load_and_clean_astronaut()
    print(f'   Astronaut data: {astro.shape}')

    # Step 5: Compute z-scores & percentile ranks
    print('\n5. Computing z-scores and percentile ranks...')
    astro = compute_zscores_and_percentiles(astro, ref_pop, ref_all)
    zscore_table = astro[['SUBJECT_ID', 'timepoint'] + [f'{v}_z' for v in VAR_MAP if f'{v}_z' in astro.columns]]
    zscore_table.to_csv(os.path.join(DATA_DIR, 'astronaut_zscores.csv'), index=False)
    pctile_table = astro[['SUBJECT_ID', 'timepoint'] + [f'{v}_pctile' for v in VAR_MAP if f'{v}_pctile' in astro.columns]]
    pctile_table.to_csv(os.path.join(DATA_DIR, 'astronaut_percentile_ranks.csv'), index=False)

    # Step 6: One-sample tests
    print('\n6. Running one-sample tests with FDR correction...')
    test_df = run_onesample_tests(astro, ref_all)
    test_df.to_csv(os.path.join(DATA_DIR, 'nhanes_onesample_tests.csv'), index=False)
    sig = test_df[test_df['t_fdr'] < 0.05]
    print(f'   Significant (FDR < 0.05): {len(sig)} tests')

    # Step 7: Generate figures
    print('\n7. Generating figures...')
    plot_trajectories(astro, ref_all)
    print('   Saved fig_s6_nhanes_trajectories')
    plot_zscore_heatmap(astro)
    print('   Saved fig_s7_zscore_heatmap')
    plot_percentile_ranks(astro)
    print('   Saved fig_s8_percentile_ranks')

    print('\n=== Done ===')


if __name__ == '__main__':
    main()
