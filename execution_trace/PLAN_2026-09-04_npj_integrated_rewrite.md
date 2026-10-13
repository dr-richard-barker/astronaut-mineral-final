> **Historical record, not current results.** This is the plan the Biomni agent wrote on
> 4 Sep 2026 for the first integrated npj Microgravity rewrite, kept unchanged below for
> provenance. Its numbers and gene assignments are as they stood then. Since then:
> - S100A8/A9/A12 and the other EF-hand Ca²⁺-binding proteins were moved from the
>   Magnesium to the Calcium gene set (commits 3c4ca06, d61c05d), so "Magnesium: S100A8/A9/A12
>   down" and "copper/iron/magnesium" below are superseded.
> - The systems-biology conclusion is now "copper and calcium", not "copper and magnesium".
>
> For current results, see [`manuscript/main_integrated.tex`](../manuscript/main_integrated.tex).

# Plan: Integrated npj Microgravity Manuscript Rewrite

## Summary

Rewrite the manuscript as a single integrated paper in npj Microgravity style (Nature Partner Journal format: Results before Methods, numbered superscript references, concise abstract). The NHANES population baseline (5,748 participants, age 40-60, pooled 2013-2018) serves as the foundational lens: the paper first establishes what "normal" looks like for 15 clinical mineral/hematology markers, then shows how 4 Inspiration4 astronauts deviate from population norms, connects those clinical deviations to transcriptomic mineral-pathway changes, evaluates ML predictive performance, and contextualizes findings with a 28-dataset cross-species meta-analysis. All existing figures are reused (renumbered); no new figures are generated. Output is LaTeX (.tex) compiled to PDF.

## Manuscript Structure

### Title
"Population-referenced multi-omics analysis reveals mineral-pathway dysregulation in spaceflight: integrating NHANES baseline with astronaut and rodent transcriptomic data"

### Abstract (~200 words, unstructured)
NHANES baseline → astronaut clinical deviations (MPV, calcium, CO2) → transcriptomic mineral-pathway response (50 DE genes, copper/iron/magnesium) → ML predictive value (K r=0.50) → cross-species meta-analysis (muscle concordance) → implications.

### Introduction (3 paragraphs, ~500 words)
1. Spaceflight disrupts mineral homeostasis (microgravity bone loss, fluid shifts, radiation). Existing studies lack population context for interpreting n=4 astronaut data.
2. NHANES provides nationally representative reference intervals for clinical lab values. We use it as a baseline to determine which astronaut mineral/hematology deviations are population-significant vs. within normal variation.
3. Study aims: (1) establish NHANES reference baseline, (2) identify population-deviating astronaut markers, (3) characterize mineral-pathway transcriptomic response, (4) evaluate ML predictive value, (5) cross-species meta-analysis across 28 rodent datasets.

### Results (6 sections, ~3000 words)

**Section 1: NHANES population reference for clinical mineral markers**
- 5,748 participants (age 40-60, all sexes, 3 cycles pooled)
- 15 variables mapped: 5 biochemistry (Ca, K, Na, Cl, CO2) + 10 hematology (Hb, Hct, RBC, MCH, MCHC, MCV, WBC, RDW, platelets, MPV)
- Reference intervals (P2.5-P97.5) established
- Table 1: key reference statistics
- Fig 1: z-score heatmap (astronaut deviation from population, 15 vars × 7 timepoints)

**Section 2: Astronaut clinical markers deviate from population norms**
- MPV significantly elevated at all 7 timepoints (FDR < 0.05, Cohen's d = 2.12-2.47; 48% outside P2.5-P97.5)
- Calcium elevated (d = 1.30-1.92 at L-92, L-44, R+82; 18% outside)
- CO2/bicarbonate depressed (d = -1.24 at L-92/R+1; d = -1.78 at R+194; 14% outside)
- RDW below population mean (d = -0.86 to -0.97; 19% outside lower bound)
- Sodium, chloride, MCH, MCV, WBC within normal range (0% outside)
- Data quality: C003 L-92 had 3 corrupted values (MPV=330, RDW=32.4, platelets=12.9), set to NaN
- Fig 2: trajectories with NHANES reference bands (15-panel grid)
- Fig S1: percentile rank dot plot
- Table 2: one-sample test results (6 FDR-significant + 16 large-effect tests)

**Section 3: Mineral-pathway transcriptomic response to spaceflight**
- 801 mineral-pathway genes (10 minerals, 5 KEGG pathways), 784 mapped to astronaut RNA-seq
- 50 unique DE genes (71 entries), |log2FC| > 0.5, p < 0.05
- Copper: APP up (proteomics, log2FC=1.32), LOX/LOXL1 down (proteomics), LOXL2 up (RNA-seq, log2FC=2.10)
- Iron: ALAS2 down (RNA-seq, log2FC=-1.45), HFE up (RNA-seq, log2FC=2.39), GLRX5 down
- Magnesium: S100A8/A9/A12 down (proteomics/RNA-seq), NIPA1 up (log2FC=1.84)
- Calcium: CACNA1D down (RNA-seq, log2FC=-2.14 to -2.25, largest fold change)
- Zinc: MT2A/MT3 down (log2FC=-0.75 to -2.03)
- Potassium: KCNQ2 up (log2FC=1.73), FXYD2 down (proteomics)
- GSEA: oxidative phosphorylation down (NES=-1.63, NOM p=0.002), iron pathway down (NES=-1.53, p=0.037)
- Fig 3: DE heatmap + volcano plot (combined 2-panel)
- Fig S2: GSEA enrichment dot plot
- Fig S3: time-resolved pathway heatmap

**Section 4: Machine learning — mineral-pathway genes predict serum mineral levels**
- Regression (v1): Ridge/SVR with LOO-CV, top-200 variance genes
  - Potassium: r=0.50, R²=0.12 (Ridge)
  - Sodium: r=0.46, R²=0.19 (SVR-Linear)
  - Hemoglobin: r=0.48, R²=-0.62 (SVR-Linear)
  - Calcium: r=-0.02 (no signal)
  - Permutation importance: CAMK4, SLC30A7, SELENOW, SLC11A1, ALPL
- Classification (v2): imbalance-aware nested LOO-CV, 784 genes
  - SVM RBF: AUC=1.0 (perfect ranking) but null distribution non-standard (mean=0.121, p<0.002) — class_weight='balanced' biases null toward specificity
  - ElasticNet (balanced): AUC=0.10, non-significant (p=0.68)
  - ElasticNet (filtered): AUC=0.35, non-significant (p=0.68)
  - 16 genes selected in ≥50% LOO folds: KCNE1 (94%), SCT (94%), SLC26A6 (88%), HFE (88%), SLC11A2 (81%)
  - Iron metabolism genes enriched in selected features (HFE, SLC11A2, ALAS2, PCBP2)
  - Results are discovery-only (n=4 flight vs n=12 ground)
- Fig 4: ML summary (regression performance + stability-selected features bar plot)
- Fig S4: permutation null distribution + confusion matrix + ROC/PR curves
- Table 3: ML model comparison (v1 + v2)

**Section 5: Cross-species meta-analysis — muscle-specific concordance**
- 28 rodent datasets: 10 liver, 12 muscle (6 subtypes), 6 kidney
- Fisher z random-effects meta-analysis
- Muscle pooled r=0.035 (95% CI: 0.012-0.058, p=0.0025)
- Soleus r=0.059 (95% CI: 0.015-0.104, p=0.009) — strongest tissue-specific signal
- Liver: r=-0.010 (p=0.50), Kidney: r=-0.016 (p=0.40) — not significant
- All studies combined: r=0.008 (p=0.41) — heterogeneity I²=31%
- Muscle concordance supports skeletal muscle as primary mineral-pathway sensor in spaceflight
- Fig 5: forest plot + tissue×pathway heatmap (combined 2-panel)
- Fig S5: cross-species scatter plot
- Table 4: meta-analysis summary by tissue

**Section 6: Systems biology integration**
- Mineral-gene-pathway network atlas
- Copper and magnesium pathways most consistently dysregulated across omics layers
- Calcium signaling genes show largest individual fold changes
- Iron metabolism genes enriched in ML-selected features
- Fig 6: systems biology atlas (5-panel composite)
- Fig S6: Sankey + Circos diagrams
- Fig S7: knowledge graph

### Discussion (~800 words)
- Population-referenced framing: NHANES baseline reveals that calcium elevation and CO2 depression are not just within-subject changes but population-deviating — astronauts sit outside normal range even pre-flight
- MPV elevation: persistent across all timepoints, likely crew characteristic rather than spaceflight effect; implications for astronaut selection
- Transcriptomic-clinical concordance: CACNA1D downregulation aligns with calcium elevation; iron pathway (ALAS2, HFE) changes align with RBC/hemoglobin deviations; KCNQ2 upregulation aligns with potassium shifts
- Cross-species: muscle concordance (especially soleus) supports skeletal muscle as mineral sensor; liver/kidney discordance cautions against tissue extrapolation
- ML: predictive value of mineral-pathway genes (K r=0.50) despite n=4; stability-selected iron genes (HFE, SLC11A2) bridge clinical and transcriptomic findings
- Limitations: n=4 (severely underpowered), no causal inference, astronaut demographics unknown (age/sex assumed), NHANES unweighted stats, cross-species confounded by tissue/duration/species differences, ML results discovery-only

### Methods (~1500 words, at end)
1. NHANES data acquisition and reference construction
2. Astronaut data acquisition (OSDR API, OSD-569 through OSD-656)
3. Mineral-pathway gene set construction (801 genes, 10 minerals, 5 KEGG pathways)
4. RNA-seq differential expression (pipeline-transcriptome-de, ENSEMBL→HGNC mapping)
5. Proteomics and metabolomics DE (limma)
6. ML regression (v1): Ridge/SVR, LOO-CV, permutation importance
7. ML classification (v2): class-weighted elastic net, nested LOO-CV, 500-permutation test
8. Cross-species meta-analysis: 28 datasets, Fisher z random-effects, concordance proportions
9. GSEA (fgsea, preranked)
10. Systems biology atlas construction
11. Statistical methods (FDR correction, effect sizes, reference intervals)
12. Reproducibility (Zenodo package, Python 3.11)

### References (~20-25, numbered superscript)
Existing 10 + new: NHANES survey design, CDC NHANES documentation, Fisher z meta-analysis method, Inspiration4 clinical chemistry, spaceflight hematology/anemia, MPV clinical significance, soleus muscle spaceflight atrophy

## Figure Renumbering

| New # | Old name | Content |
|---|---|---|
| Fig 1 | fig_s7_zscore_heatmap | NHANES z-score heatmap (15 vars × 7 timepoints) |
| Fig 2 | fig_s6_nhanes_trajectories | Trajectories with NHANES reference bands (15-panel) |
| Fig 3 | fig1 + fig2 (combined ref) | DE heatmap + volcano plot |
| Fig 4 | fig3_barplot_importance + figS8_selected_features | ML regression importance + stability-selected features |
| Fig 5 | figS3_forest + figS4_tissue_heatmap | Cross-species forest plot + tissue heatmap |
| Fig 6 | fig8_systems_biology_atlas | Systems biology atlas |
| Fig S1 | fig_s8_percentile_ranks | Percentile rank dot plot |
| Fig S2 | figS1_gsea_enrichment | GSEA enrichment dot plot |
| Fig S3 | figS2_time_resolved_heatmap | Time-resolved pathway heatmap |
| Fig S4 | figS6_permutation + figS7_confusion + fig3_roc_pr | ML v2 permutation null + confusion matrix + ROC/PR |
| Fig S5 | figS5_scatter_cross_species | Cross-species scatter |
| Fig S6 | fig4_sankey + fig5_circos | Sankey + Circos |
| Fig S7 | fig6_knowledge_graph | Knowledge graph |
| Fig S8 | fig9_pathway_expression_summary | Pathway expression summary |
| Fig S9 | fig7_mineral_trajectories | Original mineral trajectories (without NHANES bands) |

## Tables

| # | Content | Source file |
|---|---|---|
| Table 1 | NHANES reference statistics (15 vars, age 40-60, all sexes) | nhanes_reference_stats.csv |
| Table 2 | One-sample test results (FDR-sig + large effect) | nhanes_onesample_tests.csv |
| Table 3 | ML model comparison (v1 regression + v2 classification) | ml_model_comparison_v2.csv + ml_regression_results.csv |
| Table 4 | Cross-species meta-analysis summary by tissue | cross_species_meta_summary.csv |

## LaTeX Technical Details

- Document class: article, 11pt, a4paper
- Engine: XeLaTeX (for fontspec/Liberation Sans)
- Packages: fontspec, graphicx, booktabs, hyperref, natbib, amsmath, subcaption
- References: numbered, superscript style (natbib with `super` option)
- Figures: PNG files from /mnt/results/manuscript/figures/ (already copied)
- Compile: xelatex → bibtex → xelatex → xelatex
- Install texlive-xetex + texlive-latex-extra + texlive-fonts-recommended via apt (non-persistent, compile in same session)

## Implementation Steps

1. Install texlive-xetex and required packages via apt
2. Write the complete LaTeX manuscript to /mnt/results/manuscript/main_integrated.tex
3. Copy all needed figure files to the manuscript figures directory (verify they're there)
4. Compile with xelatex (3 passes for references)
5. Verify PDF output exists and has reasonable size
6. Copy final .tex and .pdf to /mnt/results/ and Zenodo package

## Compute Estimate

- LaTeX writing: no compute needed
- apt install texlive: ~2-3 minutes, ~500 MB disk
- xelatex compilation: < 30 seconds (3 passes)
- Total: < 5 minutes on default worker-0

## Assumptions

- All existing figures (SVG/PNG) are sufficient for the manuscript; no new figures needed
- The existing figure files in /mnt/results/manuscript/figures/ are correct and usable
- Astronaut demographics (exact age/sex) remain unknown; paper acknowledges this
- npj Microgravity format: Results before Methods, numbered references, ~5000 words main text
- The paper reports findings honestly, including negative results (ElasticNet non-significant, liver/kidney no concordance, calcium regression no signal)
- References will be expanded from 10 to ~20-25 with NHANES and meta-analysis method citations
