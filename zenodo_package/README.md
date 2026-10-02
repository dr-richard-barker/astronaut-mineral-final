# Mineral-Pathway Transcriptional and Molecular Responses to Spaceflight: A Multi-Omics Analysis of Astronaut and Rodent Data

## Overview

This package contains the complete analysis pipeline, processed data, figures, and tables for a study mining NASA Open Science Data Repository (OSDR) astronaut and rodent multi-omics data to identify evidence of mineral deficiencies linked to transcriptional and molecular responses to spaceflight.

## Data Sources

- **Astronaut data**: NASA OSDR studies OSD-569, OSD-570, OSD-571, OSD-572, OSD-575, OSD-630, OSD-656 (Inspiration4 mission, September 2021)
- **Rodent data**: NASA OSDR studies OSD-47, OSD-48, OSD-99, OSD-100, OSD-574 (Rodent Research missions)
- **Pathway data**: KEGG REST API (https://rest.kegg.jp)

## Pipeline Scripts

| Script | Description |
|--------|-------------|
| `01_fetch_osdr_data.py` | Fetch astronaut data from OSDR API |
| `01b_fetch_osdr_data.py` | Extended astronaut data fetcher |
| `01c_rodent_download.py` | Screen and download rodent RNA-seq studies |
| `03_mineral_pathway_genesets.py` | Build mineral-pathway gene sets from KEGG + curated genes |
| `04_ensembl_to_symbol.py` | Map ENSEMBL gene IDs to HGNC symbols via mygene.info |
| `05_visualizations.py` | Generate all data-driven visualizations |
| `06_rodent_cross_species.py` | Process rodent RNA-seq and compute cross-species concordance |
| `07_systems_biology_atlas.py` | Create systems biology atlas composite figure |

## Key Findings

1. **Mineral-pathway DE genes**: 46 significant mineral-pathway genes differentially expressed across RNA-seq and proteomics (|log2FC| > 0.5, p < 0.05)
2. **Copper pathway dysregulation**: APP up-regulated, LOX/LOXL1 down-regulated in proteomics
3. **Calcium-binding S100 proteins suppressed**: S100A8, S100A9, S100A12 down-regulated
4. **Calcium channel alteration**: CACNA1D down-regulated (-2.14 log2FC) in RNA-seq
5. **ML regression**: Mineral-pathway gene expression predicts serum potassium (r=0.50, R²=0.12) and sodium (r=0.46, R²=0.19) levels
6. **Cross-species concordance**: No significant concordance between astronaut (3-day LEO) and rodent (21-day ISS) mineral-pathway responses (r=0.007)

## Software Requirements

### Python (3.10+)
```
pandas>=2.0
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
networkx>=3.0
mygene>=3.2
statsmodels>=0.14
openpyxl>=3.1
```

### R (4.3+)
```
circlize
ggplot2
```

## License

- **Code**: MIT License
- **Data**: CC-BY 4.0 (derived from NASA OSDR, which is publicly available)
- **Figures**: CC-BY 4.0

## Citation

Please cite both this package and the original NASA OSDR datasets.

## Contact

Richard Barker
