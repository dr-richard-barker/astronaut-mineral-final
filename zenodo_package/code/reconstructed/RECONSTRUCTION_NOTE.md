# Reconstructed scripts: what they are

## What happened

The deep-learning, JAXA6-validation and NHANES iron-status analyses were run by 14 scripts in a
sandbox workspace (`/workspace/`) that was never exported and was later wiped. The session log
(`execution_trace/worker-0.ipynb`, cells 42–58, in the project repository) keeps the commands
and their printed output, but not the script bodies.

The scripts here were first rebuilt by the Biomni agent (4 Oct 2026). They were then
corrected against the manuscript Methods and checked against the deposited outputs (Oct 2026).
They are **reconstructions, not the original code.**

**The files in `../../data/` and `../../figures/` are the outputs of the original analysis and
remain authoritative.** Every script here writes to `../../reproduced/` by default. The modelling
scripts also refuse an output directory equal to `data/`.

## How they were checked

- **Downstream models, fully pinned:**
  - The deposited per-sample out-of-fold predictions (`dl_classification_roc_data.csv`,
    `dl_regression_predictions.csv`) were used to pin the hyperparameters.
  - Run on the deposited latent features, the logistic regression, both SVMs, Ridge and both SVRs
    reproduce those predictions to ≤ 1.2 × 10⁻⁴.
  - Random forests do not reproduce them, because of class-weighted leaves and version-dependent
    random number streams.
- **FT-Transformer:** the documented architecture plus a final LayerNorm and a 16→16 latent
  projection has exactly 117,707 parameters, the count the original run printed. Those two layers
  are inferred from the count.
- **Autoencoder, VAE and Transformer weights** cannot be recovered. Retraining gives different
  latent spaces, and so different downstream numbers.
- **JAXA6:** given the same per-gene inputs, the concordance, binomial, correlation, Fisher and
  Stouffer statistics reproduce the deposited values. The 26,844-gene JAXA6 DGE table the
  original read is not on OSDR, so the script needs it supplied (`--jaxa-dge`).
- **NHANES iron:** the scripts re-download the public CDC files and are compared against the
  deposited reference statistics and correlations.

Per-script results: `FIDELITY.md`.

## Not reconstructed

`eval_classification.py` timed out in the original session (cell 53), produced no deposited
output, and is not described in the manuscript. Its content is unknown, so no reconstruction is
included. The version Biomni supplied (a 5,000-permutation test of one classifier) was invented
rather than recovered.

## Changes from the Biomni versions

- **Input:** the models now use the per-study z-scored matrix
  (`harmonized_expression_zscore.csv`), as the original log shows (input range −2.8 to 3.75).
- **Autoencoder:** BatchNorm, dropout 0.3, weight decay 1e-4 and early stopping (patience 30)
  added, per the Methods.
- **VAE:** β changed from 1.0 to 0.5; dropout 0.3 and weight decay 1e-4 added.
- **FT-Transformer:** rebuilt to the documented architecture (gene-identity embedding, one pre-LN
  GELU layer, 4 heads, FFN 32, dropout 0.5).
- **Classifiers and regressors:** hyperparameters and per-fold standardisation set to those that
  reproduce the original predictions.
- **Permutation tests:**
  - They shuffle training labels within each fold, as the Methods state.
  - Classification p = P(null AUC ≥ AUC).
  - Regression p = P(|null r| ≥ |r|), which fits the original p-values best.
- **Ensemble permutation test:** both classifiers are retrained on shuffled labels, as the
  Methods state.
- **JAXA6:** rebuilt to the logged statistics (one-sided binomial; Fisher's exact over the union
  of symbols; Stouffer restricted to genes with JAXA6 adj-P < 1) and the deposited output schema.
- **Headers:** claims of reproducing the deposited outputs removed.

## Run order

From `code/reconstructed/`, with PyTorch, scikit-learn, umap-learn and joblib installed:

1. `train_autoencoder.py`, `train_vae.py`, `train_transformer.py`
2. `classify_latent.py`, `regress_latent.py`
3. `eval_all_architectures.py`, then `eval_ensemble.py`
4. `jaxa6_validation.py --jaxa-dge <table>`
5. `make_dl_figures.py`, `make_extended_figures.py`, `make_ensemble_figures.py`
6. `download_nhanes_iron.py`, then `compute_iron_correlations.py`
7. Optional: `seed_sensitivity.py --model dae|transformer`, which gives the seed tables in `FIDELITY.md`

Add `--use-deposited-latents` to steps 2–3 to run them on the original latent features instead
of retrained ones.
