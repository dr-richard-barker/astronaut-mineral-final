# Supplementary Section: Population-Referenced Analysis of Astronaut Clinical Markers

## Methods

To contextualize the 4-astronaut mineral and hematological measurements against a population baseline, we integrated data from the National Health and Nutrition Examination Survey (NHANES), a nationally representative survey of the U.S. civilian population conducted by the CDC National Center for Health Statistics. We pooled three consecutive NHANES cycles (2013-2014, 2015-2016, 2017-2018) to maximize reference population size and stability. From each cycle, we retrieved the Standard Biochemistry Profile (BIOPRO), Complete Blood Count with 5-Part Differential (CBC), and Demographics files, merging on the participant identifier (SEQN). The reference population was restricted to adults aged 40-60 years (all sexes), yielding a pooled sample of 5,748 participants (2,713 male, 3,035 female) — demographically consistent with typical astronaut corps age ranges.

Fifteen clinical variables were mapped between the astronaut dataset and NHANES: five biochemistry analytes (calcium, potassium, sodium, chloride, bicarbonate) and ten hematological parameters (hemoglobin, hematocrit, RBC count, MCH, MCHC, MCV, WBC count, RDW, platelet count, MPV). For each variable, we computed population reference statistics including mean, standard deviation, and percentiles (P1 through P99). Each astronaut measurement was then converted to: (1) a z-score relative to the NHANES age 40-60 reference, (2) an empirical percentile rank, and (3) a binary flag indicating whether the value fell outside the 95% reference interval (P2.5-P97.5). One-sample t-tests and Wilcoxon signed-rank tests were performed at each timepoint to assess whether astronaut group means differed from the NHANES population mean, with Benjamini-Hochberg false discovery rate correction across all 105 tests (15 variables x 7 timepoints). Cohen's d was computed as the effect size, using the NHANES pooled standard deviation as the reference. Three implausible values in one crew member (C003 at L-92: MPV = 330 fL, RDW = 32.4%, platelet count = 12.9 x 10^3/uL) were identified as data entry errors and set to missing prior to analysis.

## Results

Mean platelet volume (MPV) was significantly elevated above the NHANES population mean at six of seven timepoints (Benjamini-Hochberg FDR < 0.05; Cohen's d ranging from 2.12 to 2.47); at L-92, where only three crew were measured, the elevation was smaller and not significant after correction (d = 1.73, FDR = 0.13), with 48.1% of astronaut MPV measurements falling outside the population 95% reference interval. This population deviation was persistent across pre-flight, flight, and post-flight phases, suggesting a crew-level characteristic rather than a spaceflight-induced shift.

Calcium showed large positive effect sizes at multiple timepoints (Cohen's d = 1.30-1.92 at L-92, L-44, and R+82), with 17.9% of measurements outside the reference interval. Bicarbonate (CO2) was consistently below the population mean (d = -1.24 at L-92 and R+1; d = -1.78 at R+194), with 14.3% of measurements outside reference range. RDW was consistently below the population mean (d = -0.86 to -0.97), with 18.5% of measurements outside the lower reference bound. Potassium showed a transient elevation at R+45 (d = 1.79) that did not survive multiple testing correction.

Sodium, chloride, MCH, MCV, and WBC count remained within the population reference range across all timepoints (0% outside P2.5-P97.5), indicating these electrolytes and erythrocyte indices are robust to spaceflight-related perturbation at the group level. Hemoglobin, hematocrit, and RBC count showed modest deviations (3.7-7.4% outside reference) with small effect sizes, consistent with the known mild spaceflight-associated anemia pattern.

## Figures and Tables

- **Figure S6**: Astronaut mineral and CBC trajectories overlaid with NHANES P2.5-P97.5 and P5-P95 reference bands (15-panel grid).
- **Figure S7**: Z-score heatmap (15 variables x 7 timepoints) with diverging colormap; cells with |z| >= 1.96 are bolded.
- **Figure S8**: Percentile rank dot plot showing each astronaut measurement as a dot, colored by timepoint, with reference bands at P2.5, P5, P95, and P97.5.
- **Table S-nhanes-1**: NHANES reference statistics (nhanes_reference_stats.csv) — per-variable, per-sex means, SDs, and percentiles.
- **Table S-nhanes-2**: Astronaut z-scores (astronaut_zscores.csv) — 28 rows x 15 z-score columns.
- **Table S-nhanes-3**: Astronaut percentile ranks (astronaut_percentile_ranks.csv) — 28 rows x 15 percentile columns.
- **Table S-nhanes-4**: One-sample test results (nhanes_onesample_tests.csv) — 105 rows with t-statistics, p-values, FDR-adjusted p-values, Wilcoxon results, Cohen's d, and percent outside reference.

## Caveats

With only 4 astronauts, one-sample tests at each timepoint are severely underpowered. We emphasize effect sizes (Cohen's d) and population deviation patterns over p-values. The FDR correction is applied for completeness, but the analysis is inherently discovery-oriented. Exact age and sex of individual crew members were not available in the provided dataset; the all-sexes age 40-60 reference was used as the primary baseline, with sex-specific reference statistics provided as supplementary material. NHANES uses a complex survey design; we used unweighted statistics for reference interval derivation, which is standard practice for reference range estimation. This integration provides descriptive population context, not causal inference about spaceflight effects.
