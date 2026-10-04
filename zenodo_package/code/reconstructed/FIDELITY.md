# Fidelity of the reconstructed scripts

**Checked:** 4 Oct 2026.

**Environment:** Python 3.9, scikit-learn 1.6.1, PyTorch 2.8.0, umap-learn 0.5.12.

**References for every comparison:**
- the deposited files in `../../data/` (outputs of the original analysis);
- the original run log, `execution_trace/worker-0.ipynb` cells 42–58 in the project repository.

**Status labels used below:**
- **exact:** reproduces the deposited values to floating-point tolerance (≤ 1.2 × 10⁻⁴ unless
  stated).
- **Monte-Carlo:** permutation p-values; they cannot be bit-identical, so agreement is judged
  against the permutation error.
- **approximate:** same method, different numbers.
- **not reproducible:** a required input or detail is missing.

Two runs were made:
- **Run A** feeds the original latent features (`--use-deposited-latents`) to the downstream
  scripts. It tests the reconstructed analysis code.
- **Run B** retrains every model from scratch. It tests whether the published numbers survive
  retraining.

To regenerate:
- Run A: run steps 2–3 of the run order with `--use-deposited-latents`.
- Run B: run the full run order with defaults.
- Seed tables: `seed_sensitivity.py --model dae` and `seed_sensitivity.py --model transformer`.
- Comparisons: against the like-named files in `data/`.

## Per script

| Script | What was compared | Status |
|---|---|---|
| `download_nhanes_iron.py` | N, mean, SD, median, P2.5, P97.5 for serum iron, ferritin, transferrin saturation and TIBC, from a fresh CDC download | **exact** (max \|Δ\| 1.8 × 10⁻¹³; N 5,435 / 2,190 / 1,678 / 1,678) |
| `compute_iron_correlations.py` | 24 iron × CBC Pearson r and pair counts | **exact** (max \|Δr\| 3 × 10⁻¹⁵; all n identical) |
| `classify_latent.py` (Run A) | 176 out-of-fold probabilities; pooled and per-fold AUC, accuracy, F1 | LogReg, SVM-linear, SVM-RBF **exact**. Random forest **approximate** (AUC 0.723 vs 0.728; raw 0.652 vs 0.621) |
| | 1,000-permutation p-values | **Monte-Carlo** (e.g. DAE RF 0.017 vs 0.015) |
| `regress_latent.py` (Run A) | 256 out-of-fold predictions; r, ρ, R², MAE | Ridge, SVR-linear, SVR-RBF **exact** (\|Δr\| ≤ 6 × 10⁻⁶). Random forest **approximate** (\|Δr\| up to 0.13) |
| | permutation p | **Monte-Carlo** (non-RF mean \|Δp\| 0.02) |
| `eval_all_architectures.py` (Run A) | AUC for 4 classifiers × 5 feature spaces; regression r for 4 models × 4 targets × 5 spaces | Every non-RF value **exact**. RF **approximate** |
| | p-values | **Monte-Carlo** (non-RF mean \|Δp\| 0.04) |
| `eval_ensemble.py` (Run A) | concatenation ensemble AUCs; prediction-averaging AUC | Concatenation LogReg/SVM **exact**; RF **approximate** (0.562 vs 0.692). Averaging: **exact** AUC 0.750, and the deposited probabilities equal the mean of DAE-RF and Transformer-SVM-RBF |
| `jaxa6_validation.py` | concordance table; one-sided binomial; Pearson and Spearman; Stouffer z / p / FDR; Fisher counts | **exact** when given the original's JAXA6 values (13 Stouffer genes, max \|Δ\| 5 × 10⁻¹³). Fisher OR 0, p 1.0 matches the logged table (0 / 20 / 243 / 18,476) |
| | the JAXA6 DGE table itself | **not reproducible**: the 26,844-gene table is not among the OSD-530 files on OSDR, and its EDGE pairwise tables do not match the logged values (e.g. ALAS2 logFC −0.795, adj-P 0.279) |
| `train_transformer.py` | parameter count | **exact**: 117,707, as printed by the original run |
| | per-fold losses | **approximate**: 0.35–0.42 vs 0.38–0.45 |
| `train_vae.py` | per-fold losses | **approximate**: 0.46–0.52 vs 0.42–0.69 |
| `train_autoencoder.py` | per-fold losses | **approximate**: 0.27–0.34 vs 0.08–0.12. The original reached its best loss at epoch 300 in every fold. Neither training-loss nor eval-loss early stopping reproduces that, so its training differed in ways the Methods do not record |
| `make_dl_figures.py`, `make_extended_figures.py`, `make_ensemble_figures.py` | figure content | Same data as the deposited figures; styling differs. `dl_extended_regression_comparison` is regenerated from the deposited regression table. `dl_classification_auc_bar` (deposited, not used in the manuscript) has no reconstruction |

The settings pinned by Run A are listed in `_common.py`:
- all features standardised within each training fold;
- class-balanced weights;
- LogReg C = 0.1, linear SVM C = 0.1, RBF SVM C = 1;
- Ridge α = 1, linear SVR C = 0.1, RBF SVR C = 1;
- regression samples with a missing target are dropped from training but still predicted.

## Do the published deep-learning numbers survive retraining? (Run B and seed checks)

The original model weights are lost, so the published latent-space results cannot be regenerated. To
see how much they depend on one training run, `seed_sensitivity.py --model dae` retrains the
autoencoder under 10 seeds and recomputes the headline results (no permutations):

| Result | Published (original latents) | Retrained, 10 seeds: mean ± SD (range) |
|---|---|---|
| DAE + RF flight vs pre-flight AUC | 0.728 | 0.71 ± 0.09 (0.58–0.83) |
| DAE + LogReg AUC | 0.509 | 0.59 ± 0.15 (0.33–0.80) |
| Hemoglobin from DAE latents, SVR-RBF r | −0.915 | −0.85 ± 0.06 (−0.71 to −0.90) |
| Potassium from DAE latents, RF r | −0.748 | −0.52 ± 0.19 (−0.24 to −0.83) |

The same check for the FT-Transformer and the ensemble (5 seeds; autoencoder and Transformer
retrained together):

| Result | Published (original latents) | Retrained, 5 seeds: mean ± SD (range) |
|---|---|---|
| DAE + RF AUC | 0.728 | 0.73 ± 0.11 (0.59–0.83) |
| Transformer + SVM-RBF AUC | 0.661 (p = 0.010) | 0.18 ± 0.08 (0.10–0.30) |
| Transformer, best of 4 classifiers, AUC | 0.661 | 0.67 ± 0.07 (0.59–0.75) |
| Prediction-averaging ensemble AUC | 0.750 (p = 0.008) | 0.52 ± 0.10 (0.41–0.68) |

A single full retrain (Run B, seed 42, full permutations) gave the following:
- DAE + RF: AUC 0.723, p = 0.030.
- Transformer + SVM-RBF: AUC 0.277 (published 0.661, p = 0.010).
- Prediction-averaging ensemble: AUC 0.580, p = 0.108 (published 0.750, p = 0.008).
- Best \|r\| for hemoglobin from DAE / VAE / Transformer latents: 0.74 / 0.85 / 0.76 (published
  0.915 / 0.930 / 0.929).

**Reading.**
- **Survives retraining:** the DAE random-forest classification result (AUC ≈ 0.72–0.73), which
  is typical of retrained models.
- **Optimistic end of the retraining spread:** the published latent-space regression
  correlations.
- **Not reproduced:** the published Transformer + SVM-RBF result (AUC 0.661) and the ensemble
  built on it (AUC 0.750). The reconstructed Transformer instead gives AUCs well below chance
  with that classifier, although its latent space is informative with others (best-of-four
  ≈ 0.67). The reconstruction matches the original only in parameter count, so this cannot tell
  whether the original was trained differently or whether those results are seed-dependent.
  Until the original weights or code are found, treat them as unconfirmed.
- **Fixed by the data, not by training:** the raw-expression and multi-omics baselines, which
  reproduce exactly.

With n = 22 samples from 6 astronauts, results that depend on a learned latent space should be
read with this spread in mind.
