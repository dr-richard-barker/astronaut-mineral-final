# Mineral-Pathway Transcriptional and Molecular Responses to Spaceflight: A Multi-Omics Analysis of Astronaut and Rodent Data

## Overview

This package holds the analysis code, processed data, figures and manuscript for a study that
places astronaut mineral and blood markers in a population context and links them to
mineral-pathway gene expression. Clinical chemistry from the four Inspiration4 (I4) crew is
compared against an NHANES reference population. Mineral-pathway transcriptomics and
proteomics are examined, and the signature is tested across missions (Axiom-1, JAXA CFE) and
across 28 rodent spaceflight datasets.

## Data sources

- **Inspiration4 (NASA OSDR):**
  - OSD-569: whole-blood RNA-seq.
  - OSD-571: plasma proteomics and metabolomics.
  - OSD-575: Comprehensive Metabolic Panel and CBC.
  - OSD-656: urine inflammation panel.
  - The fetch scripts also download OSD-570, OSD-572, OSD-574 and OSD-630; the analyses
    reported in the manuscript do not use them.
- **Axiom-1:** GEO GSE276490 (2 astronauts, 3 time points).
- **JAXA Cell-Free Epigenome study:** OSD-530 (6 astronauts, plasma cell-free RNA-seq).
- **NHANES 2013–2018** (CDC): Standard Biochemistry Profile, CBC, demographics, ferritin,
  and TIBC/transferrin saturation.
- **Rodent:** 28 OSDR RNA-seq datasets (10 liver, 12 skeletal muscle, 6 kidney), listed in
  `09_cross_species_meta.py`.
- **Pathways:** KEGG REST API.

## Code (`code/`)

### Original pipeline

| Script | Purpose |
|--------|---------|
| `01_fetch_osdr_data.py` | Fetch astronaut data from the OSDR API |
| `01b_fetch_osdr_data.py` | Discover and download astronaut and rodent data (revised fetcher) |
| `01c_rodent_download.py` | Screen rodent RNA-seq studies by tissue and download them |
| `03_mineral_pathway_genesets.py` | Build the 10 mineral gene sets and 5 KEGG pathway sets |
| `04_ensembl_to_symbol.py` | Map ENSEMBL IDs to HGNC symbols (mygene.info) |
| `05_visualizations.py` | Data-driven figures 1–7 |
| `06_rodent_cross_species.py` | Rodent liver DE and astronaut–rodent concordance |
| `07_systems_biology_atlas.py` | Systems biology atlas (figures 8–9) |
| `08_supplementary_outputs.py` | GSEA, ssGSEA, pathway mean log2FC, supplementary tables |
| `09_cross_species_meta.py` | 28-dataset cross-species meta-analysis |
| `10_ml_imbalance_reanalysis.py` | Imbalance-aware ML: classification, regression, stability selection |
| `11_nhanes_integration.py` | NHANES reference intervals and astronaut z-scores |

There is no `02` script.

The scripts hard-code the paths of the original compute environment (`/mnt/results`,
`/mnt/shared-workspace`, `/workspace`), so adjust those before running.

### Reconstructed scripts (`code/reconstructed/`)

The code behind the cross-mission deep learning, JAXA6 validation and NHANES iron-status
analyses was lost with its compute sandbox. It was reconstructed in October 2026 from the
Methods, the run log (`execution_trace/` in the project repository) and the deposited outputs.

**The deposited files in `data/` and `figures/` are the original outputs and remain
authoritative.** The reconstructions write only to `reproduced/`.

`code/reconstructed/FIDELITY.md` records, per script, what reproduces the original exactly and
what does not.

| Script | Purpose |
|--------|---------|
| `train_autoencoder.py` | Denoising autoencoder (779→128→64→16), leave-one-astronaut-out (LOAO) |
| `train_vae.py` | β-VAE (β = 0.5) |
| `train_transformer.py` | FT-Transformer (117,707 parameters) |
| `classify_latent.py` | Flight vs pre-flight classification, latent vs 779-gene space |
| `regress_latent.py` | Serum calcium, potassium, sodium and hemoglobin from latent features |
| `eval_all_architectures.py` | DAE / VAE / Transformer / Raw_779 / Multiomics_15 comparison |
| `eval_ensemble.py` | Concatenation and prediction-averaging ensembles |
| `jaxa6_validation.py` | I4 vs JAXA6 concordance, Fisher, Stouffer meta-analysis (requires the JAXA6 DGE table) |
| `make_dl_figures.py`, `make_extended_figures.py`, `make_ensemble_figures.py` | DL figures |
| `download_nhanes_iron.py` | NHANES serum iron, ferritin, TIBC download and reference statistics |
| `compute_iron_correlations.py` | NHANES iron × CBC correlations and figures |

Run order and details: `code/reconstructed/RECONSTRUCTION_NOTE.md`.

## Key findings

1. **Population-deviating markers:** mean platelet volume sat above the NHANES reference at six
   of seven time points. Calcium was elevated and bicarbonate depressed, largely from pre-flight
   onward.
2. **Mineral-pathway expression:** 50 mineral-pathway genes were differentially expressed across
   RNA-seq and proteomics (|log2FC| > 0.5, p < 0.05).
3. **Iron pathway:**
   - ALAS2 was down-regulated, and its JAXA6 cross-mission Stouffer meta-analysis gives
     FDR = 0.021. HFE was up-regulated.
   - In NHANES, hemoglobin is the strongest CBC correlate of serum iron (r = 0.42).
4. **Copper pathway:** APP up-regulated; LOX and LOXL1 down-regulated (proteomics).
5. **Calcium pathway:**
   - The calcium-binding S100 proteins S100A8, S100A9 and S100A12 were down-regulated.
   - CACNA1D was down-regulated (log2FC = −2.25, RNA-seq).
6. **ML regression:** mineral-pathway expression predicts serum potassium (r = 0.50, R² = 0.12)
   and sodium (r = 0.46, R² = 0.19).
7. **Deep learning:**
   - The DAE latent space improves flight vs pre-flight classification (AUC 0.728 vs 0.621 for
     raw expression).
   - The DAE + Transformer prediction-averaging ensemble reaches AUC 0.750.
   - In retraining with the reconstructed code, the DAE result holds (AUC ≈ 0.72). The
     Transformer and ensemble results were not reproduced; see
     `code/reconstructed/FIDELITY.md`.
8. **Cross-species:** across 28 rodent datasets, mineral-pathway concordance is significant in
   skeletal muscle (pooled r = 0.035, p = 0.0025), strongest in soleus (r = 0.059, p = 0.009).
   It is not significant in liver or kidney.

## Software requirements

Python 3.10+; see `requirements.txt`. The reconstructed deep-learning scripts also need
PyTorch, umap-learn and joblib. GSEA uses gseapy.

## License

- **Code:** MIT License
- **Data:** CC-BY 4.0 (derived from public NASA OSDR, GEO and CDC NHANES data)
- **Figures:** CC-BY 4.0

## Citation

Please cite this package and the original NASA OSDR, GEO and NHANES datasets.

## Contact

Richard Barker
