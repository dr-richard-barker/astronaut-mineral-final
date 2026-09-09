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
