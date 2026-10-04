# Data Dictionary

## Processed Data Files (`data/`)

### Astronaut Data

| File | Description | Rows | Columns |
|------|-------------|------|---------|
| `astronaut_minerals_cmp.csv` | Comprehensive Metabolic Panel mineral data (Ca, K, Na, Cl, CO2) | 28 | 7 |
| `astronaut_minerals_combined.csv` | CMP + CBC combined (minerals + hemoglobin/hematocrit) | 28 | 13 |
| `astronaut_cbc_full.csv` | Complete blood count data | 28 | 9 |
| `astronaut_cbc_iron_proxy.csv` | CBC iron proxy (hemoglobin, hematocrit, RBC) | 28 | 8 |
| `astronaut_rnaseq_de.csv` | RNA-seq differential expression (DESeq2 + pipeline) | 185,556 | 8 |
| `astronaut_rnaseq_de_mapped.csv` | RNA-seq DE with gene symbols mapped | 136,650 | 10 |
| `astronaut_rnaseq_counts.csv` | RNA-seq featureCounts matrix | 61,852 | 16 |
| `astronaut_proteomics_de.csv` | Proteomics differential expression (limma) | 3,285 | 9 |
| `astronaut_metabolomics_de.csv` | Metabolomics differential expression (limma) | 2,624 | 9 |

### Mineral-Pathway Gene Sets

| File | Description | Format |
|------|-------------|--------|
| `mineral_pathway_genesets.gmt` | GMT file with mineral and KEGG pathway gene sets | GMT |
| `mineral_pathway_crosswalk.csv` | Gene → minerals + KEGG pathways mapping | CSV (801 rows) |
| `kegg_pathway_genes.csv` | KEGG pathway → gene mapping | CSV (585 rows) |
| `ensembl_to_symbol.csv` | ENSEMBL ID → gene symbol mapping | CSV (61,852 rows) |

### Analysis Results

| File | Description |
|------|-------------|
| `mineral_pathway_de_combined.csv` | Combined RNA-seq + proteomics mineral-pathway DE genes |
| `ml_classification_results.csv` | ML flight vs ground classification results (LOO-CV) |
| `ml_model_comparison.csv` | Model comparison table (5 classifiers) |
| `ml_regression_results.csv` | Mineral level regression results (4 targets × 4 models) |
| `permutation_importance_classification.csv` | Gene importance for flight/ground classification |
| `permutation_importance_potassium.csv` | Gene importance for potassium regression |
| `permutation_importance_sodium.csv` | Gene importance for sodium regression |
| `rodent_liver_de_human_mapped.csv` | Rodent liver DE with human ortholog mapping |
| `cross_species_concordance.csv` | Cross-species concordance analysis results |

## Cross-mission, deep-learning, JAXA6 and NHANES-iron outputs (`data/`)

These are outputs of the original analysis. The scripts that produced them were lost and are
reconstructed in `code/reconstructed/` (see `FIDELITY.md` there).

**Inputs**

| File | Description |
|------|-------------|
| `harmonized_astronaut_expression.csv` | 779 mineral-pathway genes × 22 samples (I4 log2 CPM; AX-1 log2 TPM) |
| `harmonized_expression_zscore.csv` | Same matrix, z-scored per gene within each study; the DL input |
| `harmonized_astronaut_metadata.csv` | Per sample: study, astronaut ID, time point, flight status, tissue |
| `harmonized_expression_multiomics_filtered.csv` | 15-gene multi-omics consensus subset × 22 samples |
| `multiomics_consensus_genes.csv` | The 15 consensus genes (RNA-seq mineral genes ∩ proteomics p < 0.05) |

**Latent features and training losses**

| File | Description |
|------|-------------|
| `autoencoder_latent_features.csv`, `vae_latent_features.csv`, `transformer_latent_features.csv` | 16-dim latents; each astronaut encoded by the leave-one-astronaut-out (LOAO) model that did not see them |
| `*_latent_features_final.csv` | 16-dim latents from the model trained on all 22 samples |
| `autoencoder_recon_losses.csv`, `vae_recon_losses.csv`, `transformer_recon_losses.csv` | Best training loss per LOAO fold, plus the all-data model (`ALL`) |

**Classification and regression results**

| File | Description |
|------|-------------|
| `dl_classification_results.csv` | Flight vs pre-flight: pooled AUC, fold AUCs, accuracy, F1, permutation null and p (DAE latent vs 779 genes) |
| `dl_classification_roc_data.csv` | Out-of-fold predicted probabilities behind the classification results |
| `dl_regression_results.csv` | Serum Ca, K, Na and hemoglobin regression: r, ρ, R², MAE, permutation p (I4, 16 samples) |
| `dl_regression_predictions.csv` | Out-of-fold predictions behind the regression results |
| `dl_extended_classification_results.csv` | Classification across DAE / VAE / Transformer / Raw_779 / Multiomics_15 and both ensembles |
| `dl_extended_regression_results.csv` | Regression across the five feature spaces (p-values for the latent spaces only) |
| `dl_extended_roc_data.csv` | Out-of-fold probabilities for the extended comparison and ensembles |

**JAXA6 and NHANES iron**

| File | Description |
|------|-------------|
| `jaxa6_concordance_table.csv` | 16 genes in both I4-FP1 DE and JAXA6: log fold changes, directions, concordance |
| `jaxa6_stouffer_meta.csv` | √n-weighted Stouffer meta-analysis of I4 and JAXA6 p-values (13 genes) with BH-FDR |
| `jaxa6_validation_results.csv` | Summary statistics: concordance, binomial, Pearson, Spearman, Fisher, meta-analysis counts |
| `nhanes_iron_cbc_correlations.csv` | 24 Pearson correlations, 4 iron markers × 6 CBC parameters (NHANES, age 40–60) |

## Column Descriptions

### `astronaut_minerals_combined.csv`
- `SUBJECT_ID`: Astronaut identifier (C001-C004)
- `timepoint`: Mission timepoint (L-92, L-44, L-3, R+1, R+45, R+82, R+194)
- `CALCIUM`: Serum calcium (mg/dL)
- `POTASSIUM`: Serum potassium (mmol/L)
- `SODIUM`: Serum sodium (mmol/L)
- `CHLORIDE`: Serum chloride (mmol/L)
- `CARBON DIOXIDE`: Serum CO2 (mmol/L)
- `HEMOGLOBIN`: Hemoglobin (g/dL, iron status proxy)
- `HEMATOCRIT`: Hematocrit (%)
- `RED BLOOD CELL COUNT`: RBC count (×10⁶/μL)
- `MCH`: Mean corpuscular hemoglobin (pg)
- `MCHC`: Mean corpuscular hemoglobin concentration (g/dL)
- `MCV`: Mean corpuscular volume (fL)

### `astronaut_rnaseq_de_mapped.csv`
- `ENSEMBL`: Ensembl gene ID (versioned)
- `_sheet`: Comparison (I4-FP1, I4-FP2, I4-FP3)
- `comparison`: Description of comparison
- `DESeq2_log2FC`: DESeq2 log2 fold change
- `DESeq2_p-value`: DESeq2 raw p-value
- `DESeq2_adjusted p-value`: DESeq2 adjusted p-value (BH)
- `pipeline-transcriptome-de_log2FC`: Pipeline log2 fold change
- `pipeline-transcriptome-de_p-value`: Pipeline raw p-value
- `pipeline-transcriptome-de_adjusted p-value`: Pipeline adjusted p-value
- `gene_symbol`: HGNC gene symbol
- `is_mineral_gene`: Boolean flag for mineral-pathway membership

### Timepoint Definitions
- **L-92**: 92 days before launch (pre-flight)
- **L-44**: 44 days before launch (pre-flight)
- **L-3**: 3 days before launch (pre-flight)
- **R+1**: 1 day after return (post-flight)
- **R+45**: 45 days after return (post-flight)
- **R+82**: 82 days after return (post-flight)
- **R+194**: 194 days after return (post-flight)

### Flight Comparisons
- **I4-FP1**: (R+1) vs (L-92, L-44, L-3) — immediate post-flight
- **I4-FP2**: (R+1, R+45, R+82) vs pre-flight — early recovery
- **I4-FP3**: (R+1, R+45, R+82, R+194) vs pre-flight — full follow-up

## Figures (`figures/`)

| File | Description |
|------|-------------|
| `fig1_heatmap_mineral_de.svg` | Heatmap of top 40 mineral-pathway DE genes |
| `fig2_volcano_rnaseq.svg` | Volcano plot with mineral genes highlighted |
| `fig3_barplot_importance.svg` | Permutation importance for K/Na regression |
| `fig4_sankey_omics_minerals.svg` | Sankey: omics → minerals → KEGG pathways |
| `fig5_circos_mineral_pathway.svg` | Circos: mineral-gene-pathway network |
| `fig6_knowledge_graph.svg` | Knowledge graph of mineral-gene-pathway network |
| `fig7_mineral_trajectories.svg` | Serum mineral trajectories over time |
| `fig8_systems_biology_atlas.svg` | Composite systems biology atlas (5 panels) |
| `fig9_pathway_expression_summary.svg` | Per-pathway gene expression summary |
| `fig1_heatmap_mineral_de.png`, `fig2_volcano_rnaseq.png`, `fig3_barplot_importance.png`, `fig8_systems_biology_atlas.png` | PNG renders of the figures above, as included in `main_integrated.tex` |
| `dl_iron_reference_panel.png` | NHANES reference distributions of serum iron, ferritin and transferrin saturation (age 40–60) |
| `dl_iron_cbc_correlation_heatmap.png` | Pearson correlations between four iron status markers and six CBC parameters (NHANES) |
| `dl_extended_auc_comparison.png` | Pooled leave-one-astronaut-out AUC, flight vs pre-flight, across 7 feature spaces |
| `dl_extended_roc_overlay.png` | ROC curves for the best classifier per architecture, including ensembles |
| `dl_extended_regression_comparison.png` | Best \|r\| for mineral regression per architecture (I4 only) |
| `dl_extended_umap_comparison.png` | UMAP of the DAE, VAE and FT-Transformer 16-dimensional latent spaces |
| `dl_latent_correlation_heatmap.png` | Pairwise correlations between DAE, VAE and FT-Transformer latent dimensions |
| `dl_per_astronaut_auc.png` | Per-astronaut leave-one-out AUC for the best classifier per feature space |
