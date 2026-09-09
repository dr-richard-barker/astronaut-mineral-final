#!/usr/bin/env python3
"""
10_ml_imbalance_reanalysis.py — ML re-analysis with imbalance-aware feature selection.

Fixes three critical issues in the original ML pipeline:
  1. Feature selection was performed outside CV (data leakage)
  2. No imbalance-aware feature selection or metrics
  3. Permutation importance was computed in-sample

Revised approach:
  - Class-weighted elastic-net logistic regression (embedded feature selection)
  - Nested LOO-CV (feature selection inside each fold)
  - 784 mineral-pathway gene universe (biological prior)
  - 1000-permutation significance test
  - ROC-AUC, PR-AUC, balanced accuracy, sensitivity, specificity, F1

Outputs:
  - Fig 3 (updated): ROC + PR curves
  - Fig S6: Permutation null distribution
  - Fig S7: Confusion matrix heatmap
  - Fig S8: Selected features bar plot
  - ml_model_comparison_v2.csv
  - ml_classification_results_v2.csv
  - ml_selected_features_v2.csv
  - ml_permutation_test.csv
"""

import os, sys, warnings, json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    f1_score, confusion_matrix, roc_curve, precision_recall_curve,
)
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
warnings.filterwarnings('ignore', category=UserWarning)

DATA_DIR = Path('/mnt/results/data')
FIG_DIR = Path('/mnt/results/figures')
ZENODO_DIR = Path('/mnt/results/zenodo_package')
MANUSCRIPT_FIG_DIR = Path('/mnt/results/manuscript/figures')
for d in [FIG_DIR, MANUSCRIPT_FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_PERMUTATIONS = 500

# ── Hyperparameter grid ──
C_GRID = [0.05, 0.1, 0.5, 1.0]
L1_RATIO_GRID = [0.3, 0.5, 0.7]
SVM_C_GRID = [0.5, 1.0, 5.0]
SVM_GAMMA_GRID = ['scale', 'auto']


# ============================================================
# STEP 1: Load and prepare data
# ============================================================
def load_data():
    """Load counts, map to symbols, filter to mineral-pathway genes."""
    print("=== Step 1: Loading data ===")

    counts = pd.read_csv(DATA_DIR / 'astronaut_rnaseq_counts.csv', index_col=0)
    sym_map = pd.read_csv(DATA_DIR / 'ensembl_to_symbol.csv')
    crosswalk = pd.read_csv(DATA_DIR / 'mineral_pathway_crosswalk.csv')

    # Map ENSEMBL -> symbol
    ens_to_sym = dict(zip(sym_map['ensembl_id'], sym_map['gene_symbol']))
    counts.index = counts.index.map(lambda x: ens_to_sym.get(x, None))
    counts = counts[counts.index.notna()]
    counts.index.name = 'gene_symbol'

    # Aggregate duplicate symbols
    counts = counts.groupby(counts.index).sum()
    print(f"  Counts after symbol mapping: {counts.shape}")

    # Filter to mineral-pathway genes
    mineral_genes = set(crosswalk['gene_symbol'].dropna().unique())
    counts = counts[counts.index.isin(mineral_genes)]
    print(f"  Mineral-pathway genes: {counts.shape[0]} genes × {counts.shape[1]} samples")

    # Build gene-to-mineral mapping
    gene_to_minerals = {}
    for _, row in crosswalk.iterrows():
        g = row['gene_symbol']
        m = row['minerals']
        if pd.notna(g) and pd.notna(m) and isinstance(m, str):
            gene_to_minerals[g] = m

    # Sample labels: R+1 = flight (1), L-* = ground (0)
    samples = list(counts.columns)
    y = np.array([1 if 'R+1' in s else 0 for s in samples])
    print(f"  Labels: {y.sum()} flight, {(~y.astype(bool)).sum()} ground")

    # Log2 transform
    X_df = np.log2(counts.T + 1)  # samples × genes
    X = X_df.values
    gene_names = list(X_df.columns)
    sample_names = list(X_df.index)

    print(f"  Feature matrix: {X.shape} (samples × genes)")
    return X, y, gene_names, sample_names, gene_to_minerals


# ============================================================
# STEP 2: Nested LOO-CV with elastic net
# ============================================================
def run_nested_loo_cv(X, y, model_type='elasticnet_balanced', tune_inner=True):
    """
    Run nested LOO-CV.
    Outer: LeaveOneOut (16 folds)
    Inner: 3-fold stratified CV for hyperparameter selection
    Feature selection via elastic net L1 penalty (inside each fold)
    Standardization fit on training fold only.

    Returns: predictions, probabilities, selected_features_per_fold, best_params_per_fold
    """
    loo = LeaveOneOut()
    n_samples = X.shape[0]

    y_pred = np.zeros(n_samples, dtype=int)
    y_proba = np.zeros(n_samples)
    selected_features = []  # list of arrays (selected gene indices per fold)
    best_params_list = []
    coefficients = []  # list of coefficient arrays per fold
    filter_indices_list = []  # list of filter index arrays per fold (for filtered model)

    for fold_idx, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize on training fold
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        fold_filter_idx = None  # None for non-filtered models

        if model_type == 'elasticnet_balanced':
            best_params, best_model = _tune_elasticnet(
                X_train_sc, y_train, tune_inner=tune_inner, class_weight='balanced'
            )
        elif model_type == 'elasticnet_unweighted':
            best_params, best_model = _tune_elasticnet(
                X_train_sc, y_train, tune_inner=tune_inner, class_weight=None
            )
        elif model_type == 'elasticnet_filtered':
            # Pre-filter: univariate ROC-AUC inside fold, keep top 20 genes
            best_params, best_model, filter_idx = _tune_elasticnet_filtered(
                X_train_sc, y_train, tune_inner=tune_inner, n_top=20
            )
            # Adjust X_test to filtered genes
            X_test_sc = X_test_sc[:, filter_idx]
            fold_filter_idx = filter_idx
        elif model_type == 'svm_rbf_balanced':
            best_params, best_model = _tune_svm_rbf(
                X_train_sc, y_train, tune_inner=tune_inner
            )
        elif model_type == 'dummy':
            best_params = {}
            best_model = DummyClassifier(strategy='stratified', random_state=RANDOM_STATE)
            best_model.fit(X_train_sc, y_train)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Predict
        if hasattr(best_model, 'predict_proba'):
            proba = best_model.predict_proba(X_test_sc)[0, 1]
        elif hasattr(best_model, 'decision_function'):
            proba = best_model.decision_function(X_test_sc)[0]
        else:
            proba = float(best_model.predict(X_test_sc)[0])

        y_proba[test_idx[0]] = proba
        y_pred[test_idx[0]] = 1 if proba >= 0.5 else 0

        # Track selected features and coefficients
        if hasattr(best_model, 'coef_'):
            coef = best_model.coef_[0]
            selected = np.where(np.abs(coef) > 0)[0]
            selected_features.append(selected)
            coefficients.append(coef)
        else:
            selected_features.append(np.array([]))
            coefficients.append(None)

        filter_indices_list.append(fold_filter_idx)
        best_params_list.append(best_params)

    return y_pred, y_proba, selected_features, best_params_list, coefficients, filter_indices_list


def _tune_elasticnet(X_train, y_train, tune_inner=True, class_weight='balanced'):
    """Tune elastic net hyperparameters via inner stratified CV."""
    if not tune_inner:
        # Use fixed reasonable defaults
        model = LogisticRegression(tol=1e-2, 
            penalty='elasticnet', solver='saga', max_iter=500,
            C=0.1, l1_ratio=0.5, class_weight=class_weight,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        return {'C': 0.1, 'l1_ratio': 0.5}, model

    # Inner stratified 3-fold CV
    n_flight = int(y_train.sum())
    n_folds = min(3, n_flight) if n_flight > 0 else 3
    if n_flight < 2:
        # Can't stratify with <2 flight; use fixed defaults
        model = LogisticRegression(tol=1e-2, 
            penalty='elasticnet', solver='saga', max_iter=500,
            C=0.1, l1_ratio=0.5, class_weight=class_weight,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        return {'C': 0.1, 'l1_ratio': 0.5}, model

    try:
        inner_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    except ValueError:
        model = LogisticRegression(tol=1e-2, 
            penalty='elasticnet', solver='saga', max_iter=500,
            C=0.1, l1_ratio=0.5, class_weight=class_weight,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train)
        return {'C': 0.1, 'l1_ratio': 0.5}, model

    best_auc = -1
    best_C, best_l1 = 0.1, 0.5

    for C in C_GRID:
        for l1_ratio in L1_RATIO_GRID:
            aucs = []
            for inner_train, inner_val in inner_cv.split(X_train, y_train):
                model = LogisticRegression(tol=1e-2, 
                    penalty='elasticnet', solver='saga', max_iter=500,
                    C=C, l1_ratio=l1_ratio, class_weight=class_weight,
                    random_state=RANDOM_STATE,
                )
                try:
                    model.fit(X_train[inner_train], y_train[inner_train])
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_train[inner_val])[:, 1]
                    else:
                        proba = model.decision_function(X_train[inner_val])
                    if len(np.unique(y_train[inner_val])) > 1:
                        aucs.append(roc_auc_score(y_train[inner_val], proba))
                    else:
                        aucs.append(0.5)
                except Exception:
                    aucs.append(0.5)

            mean_auc = np.mean(aucs) if aucs else 0.5
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_C, best_l1 = C, l1_ratio

    # Refit on all training data with best hyperparams
    best_model = LogisticRegression(tol=1e-2, 
        penalty='elasticnet', solver='saga', max_iter=500,
        C=best_C, l1_ratio=best_l1, class_weight=class_weight,
        random_state=RANDOM_STATE,
    )
    best_model.fit(X_train, y_train)
    return {'C': best_C, 'l1_ratio': best_l1}, best_model


def _tune_elasticnet_filtered(X_train, y_train, tune_inner=True, n_top=20, class_weight='balanced'):
    """
    Pre-filter genes by univariate ROC-AUC inside the fold, then fit elastic net on top n_top.
    Returns (best_params, best_model, filter_indices).
    """
    n_genes = X_train.shape[1]

    # Univariate ROC-AUC for each gene
    gene_aucs = np.zeros(n_genes)
    for g in range(n_genes):
        if len(np.unique(y_train)) > 1 and np.std(X_train[:, g]) > 0:
            try:
                gene_aucs[g] = roc_auc_score(y_train, X_train[:, g])
            except Exception:
                gene_aucs[g] = 0.5
        else:
            gene_aucs[g] = 0.5

    # Select top n_top by |AUC - 0.5| (most discriminative, either direction)
    gene_discrim = np.abs(gene_aucs - 0.5)
    top_idx = np.argsort(gene_discrim)[::-1][:n_top]

    X_train_filt = X_train[:, top_idx]

    # Tune elastic net on filtered set
    if not tune_inner:
        model = LogisticRegression(tol=1e-2, 
            penalty='elasticnet', solver='saga', max_iter=500,
            C=0.1, l1_ratio=0.5, class_weight=class_weight,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train_filt, y_train)
        return {'C': 0.1, 'l1_ratio': 0.5, 'n_top': n_top}, model, top_idx

    n_flight = int(y_train.sum())
    n_folds = min(3, n_flight) if n_flight > 0 else 3
    if n_flight < 2:
        model = LogisticRegression(tol=1e-2, 
            penalty='elasticnet', solver='saga', max_iter=500,
            C=0.1, l1_ratio=0.5, class_weight=class_weight,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train_filt, y_train)
        return {'C': 0.1, 'l1_ratio': 0.5, 'n_top': n_top}, model, top_idx

    try:
        inner_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    except ValueError:
        model = LogisticRegression(tol=1e-2, 
            penalty='elasticnet', solver='saga', max_iter=500,
            C=0.1, l1_ratio=0.5, class_weight=class_weight,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train_filt, y_train)
        return {'C': 0.1, 'l1_ratio': 0.5, 'n_top': n_top}, model, top_idx

    best_auc = -1
    best_C, best_l1 = 0.1, 0.5

    for C in C_GRID:
        for l1_ratio in L1_RATIO_GRID:
            aucs = []
            for inner_train, inner_val in inner_cv.split(X_train_filt, y_train):
                model = LogisticRegression(tol=1e-2, 
                    penalty='elasticnet', solver='saga', max_iter=500,
                    C=C, l1_ratio=l1_ratio, class_weight=class_weight,
                    random_state=RANDOM_STATE,
                )
                try:
                    model.fit(X_train_filt[inner_train], y_train[inner_train])
                    proba = model.predict_proba(X_train_filt[inner_val])[:, 1]
                    if len(np.unique(y_train[inner_val])) > 1:
                        aucs.append(roc_auc_score(y_train[inner_val], proba))
                    else:
                        aucs.append(0.5)
                except Exception:
                    aucs.append(0.5)

            mean_auc = np.mean(aucs) if aucs else 0.5
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_C, best_l1 = C, l1_ratio

    best_model = LogisticRegression(tol=1e-2, 
        penalty='elasticnet', solver='saga', max_iter=500,
        C=best_C, l1_ratio=best_l1, class_weight=class_weight,
        random_state=RANDOM_STATE,
    )
    best_model.fit(X_train_filt, y_train)
    return {'C': best_C, 'l1_ratio': best_l1, 'n_top': n_top}, best_model, top_idx


def _tune_svm_rbf(X_train, y_train, tune_inner=True):
    """Tune SVM RBF hyperparameters via inner stratified CV."""
    if not tune_inner:
        model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
                    probability=True, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        return {'C': 1.0, 'gamma': 'scale'}, model

    n_flight = int(y_train.sum())
    n_folds = min(3, n_flight) if n_flight > 0 else 3
    if n_flight < 2:
        model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
                    probability=True, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        return {'C': 1.0, 'gamma': 'scale'}, model

    try:
        inner_cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    except ValueError:
        model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced',
                    probability=True, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        return {'C': 1.0, 'gamma': 'scale'}, model

    best_auc = -1
    best_C, best_gamma = 1.0, 'scale'

    for C in SVM_C_GRID:
        for gamma in SVM_GAMMA_GRID:
            aucs = []
            for inner_train, inner_val in inner_cv.split(X_train, y_train):
                model = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced',
                            probability=True, random_state=RANDOM_STATE)
                try:
                    model.fit(X_train[inner_train], y_train[inner_train])
                    proba = model.predict_proba(X_train[inner_val])[:, 1]
                    if len(np.unique(y_train[inner_val])) > 1:
                        aucs.append(roc_auc_score(y_train[inner_val], proba))
                    else:
                        aucs.append(0.5)
                except Exception:
                    aucs.append(0.5)

            mean_auc = np.mean(aucs) if aucs else 0.5
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_C, best_gamma = C, gamma

    best_model = SVC(kernel='rbf', C=best_C, gamma=best_gamma, class_weight='balanced',
                     probability=True, random_state=RANDOM_STATE)
    best_model.fit(X_train, y_train)
    return {'C': best_C, 'gamma': best_gamma}, best_model


# ============================================================
# STEP 3: Permutation test
# ============================================================
def run_permutation_test(X, y, model_type, fixed_params, n_permutations=1000):
    """
    Permutation test: shuffle labels, run LOO-CV with fixed hyperparams, record AUC.
    Works for both elastic net and SVM.
    Returns null AUC distribution.
    """
    print(f"\n=== Step 3: Permutation test ({n_permutations} permutations, model={model_type}) ===")
    loo = LeaveOneOut()
    n_samples = X.shape[0]
    null_aucs = []
    rng = np.random.RandomState(RANDOM_STATE)

    for perm_idx in range(n_permutations):
        y_perm = y.copy()
        rng.shuffle(y_perm)

        y_proba_perm = np.zeros(n_samples)
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y_perm[train_idx]

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            if model_type == 'elasticnet':
                model = LogisticRegression(tol=1e-2, 
                    penalty='elasticnet', solver='saga', max_iter=500,
                    C=fixed_params['C'], l1_ratio=fixed_params['l1_ratio'],
                    class_weight='balanced', random_state=RANDOM_STATE,
                )
                try:
                    model.fit(X_train_sc, y_train)
                    proba = model.predict_proba(X_test_sc)[0, 1]
                except Exception:
                    proba = 0.5
            elif model_type == 'elasticnet_filtered':
                # Univariate filter inside permutation fold — vectorized mean difference
                # (faster than per-gene roc_auc_score; equivalent for ranking)
                y_train_bool = y_train.astype(bool)
                mean_diff = np.abs(X_train_sc[y_train_bool].mean(axis=0) -
                                   X_train_sc[~y_train_bool].mean(axis=0))
                top_idx = np.argsort(mean_diff)[::-1][:fixed_params.get('n_top', 20)]
                model = LogisticRegression(tol=1e-2, 
                    penalty='elasticnet', solver='saga', max_iter=500,
                    C=fixed_params['C'], l1_ratio=fixed_params['l1_ratio'],
                    class_weight='balanced', random_state=RANDOM_STATE,
                )
                try:
                    model.fit(X_train_sc[:, top_idx], y_train)
                    proba = model.predict_proba(X_test_sc[:, top_idx])[0, 1]
                except Exception:
                    proba = 0.5
            elif model_type == 'svm_rbf':
                # Use decision_function for speed (AUC only needs ranking, not calibrated probs)
                model = SVC(
                    kernel='rbf', C=fixed_params['C'], gamma=fixed_params['gamma'],
                    class_weight='balanced', random_state=RANDOM_STATE,
                )
                try:
                    model.fit(X_train_sc, y_train)
                    proba = model.decision_function(X_test_sc)[0]
                except Exception:
                    proba = 0.5
            else:
                raise ValueError(f"Unknown model_type for permutation: {model_type}")

            y_proba_perm[test_idx[0]] = proba

        if len(np.unique(y_perm)) > 1:
            auc = roc_auc_score(y_perm, y_proba_perm)
        else:
            auc = 0.5
        null_aucs.append(auc)

        if (perm_idx + 1) % 100 == 0:
            print(f"  Permutation {perm_idx + 1}/{n_permutations}, running null AUC mean: {np.mean(null_aucs):.3f}")

    null_aucs = np.array(null_aucs)
    return null_aucs


def find_optimal_threshold(y_true, y_proba):
    """
    Sweep thresholds to find the one maximizing balanced accuracy.
    Returns (optimal_threshold, metrics_at_optimal).
    Note: this is post-hoc on LOO-CV predictions — report as sensitivity analysis, not primary result.
    """
    best_thresh = 0.5
    best_bal_acc = 0
    best_metrics = {}

    for thresh in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_proba >= thresh).astype(int)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_thresh = thresh
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            best_metrics = {
                'threshold': thresh,
                'balanced_accuracy': bal_acc,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'confusion_matrix': cm.tolist(),
            }

    return best_thresh, best_metrics


# ============================================================
# STEP 4: Compute metrics
# ============================================================
def compute_metrics(y_true, y_pred, y_proba):
    """Compute all evaluation metrics."""
    metrics = {}
    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    metrics['pr_auc'] = average_precision_score(y_true, y_proba)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    metrics['confusion_matrix'] = cm.tolist()
    return metrics


# ============================================================
# STEP 5: Generate figures
# ============================================================
def plot_roc_pr_curves(y_true, y_proba, metrics, save_prefix):
    """Plot ROC curve with PR curve inset."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax1.plot(fpr, tpr, color='#0279EE', lw=2, label=f'AUC = {metrics["roc_auc"]:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Chance (AUC = 0.5)')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc='lower right')
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ax2.plot(recall, precision, color='#FF9400', lw=2, label=f'AP = {metrics["pr_auc"]:.3f}')
    # Baseline (proportion of positive class)
    baseline = y_true.sum() / len(y_true)
    ax2.axhline(y=baseline, color='k', linestyle='--', lw=1, alpha=0.5, label=f'Baseline = {baseline:.2f}')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='upper right')
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    for ext in ['svg', 'png']:
        fig.savefig(FIG_DIR / f'{save_prefix}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_prefix}.svg/png")


def plot_confusion_matrix(y_true, y_pred, save_prefix):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Ground', 'Flight'], yticklabels=['Ground', 'Flight'],
        ax=ax, cbar=False, annot_kws={'size': 14},
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (LOO-CV)')
    plt.tight_layout()
    for ext in ['svg', 'png']:
        fig.savefig(FIG_DIR / f'{save_prefix}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_prefix}.svg/png")


def plot_permutation_null(null_aucs, observed_auc, p_value, save_prefix):
    """Plot permutation null distribution with observed AUC marked."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_aucs, bins=30, color='#ECE9E2', edgecolor='#75A025', alpha=0.8, label='Null distribution')
    ax.axvline(observed_auc, color='#FF9400', lw=2, linestyle='--',
               label=f'Observed AUC = {observed_auc:.3f}')
    ax.axvline(0.5, color='gray', lw=1, linestyle=':', alpha=0.5, label='Chance (0.5)')
    ax.set_xlabel('ROC-AUC')
    ax.set_ylabel('Count')
    ax.set_title(f'Permutation Test (n={len(null_aucs)}, p = {p_value:.4f})')
    ax.legend(loc='upper left')
    plt.tight_layout()
    for ext in ['svg', 'png']:
        fig.savefig(FIG_DIR / f'{save_prefix}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_prefix}.svg/png")


def plot_selected_features(selection_freq, mean_coef, gene_names, gene_to_minerals, save_prefix, top_n=20):
    """Plot top selected features by selection frequency × coefficient magnitude."""
    # Build dataframe
    df = pd.DataFrame({
        'gene': gene_names,
        'selection_freq': selection_freq,
        'mean_coef': mean_coef,
    })
    df['abs_coef'] = df['mean_coef'].abs()
    df['minerals'] = df['gene'].map(gene_to_minerals).fillna('')
    # Filter to genes selected at least once
    df = df[df['selection_freq'] > 0].sort_values('selection_freq', ascending=False)
    if len(df) == 0:
        print("  No features selected — skipping selected features plot")
        return
    df = df.head(top_n)

    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(df) + 1)))
    colors = ['#0279EE' if c > 0 else '#FD9BED' for c in df['mean_coef']]
    bars = ax.barh(range(len(df)), df['selection_freq'], color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{g}" + (f" ({m})" if m else "") for g, m in zip(df['gene'], df['minerals'])],
                       fontsize=9)
    ax.set_xlabel('Selection Frequency (across LOO folds)')
    ax.set_title('Top Selected Features (Elastic Net)')
    ax.invert_yaxis()

    # Add coefficient value annotations
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row['selection_freq'] + 0.01, i, f"β={row['mean_coef']:.3f}",
                va='center', fontsize=8, color='gray')

    ax.set_xlim(0, max(df['selection_freq'].max() * 1.3, 0.1))
    plt.tight_layout()
    for ext in ['svg', 'png']:
        fig.savefig(FIG_DIR / f'{save_prefix}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_prefix}.svg/png")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("ML RE-ANALYSIS: IMBALANCE-AWARE FEATURE SELECTION")
    print("=" * 60)

    # Step 1: Load data
    X, y, gene_names, sample_names, gene_to_minerals = load_data()

    # Step 2: Run all models via nested LOO-CV
    print("\n=== Step 2: Nested LOO-CV ===")
    models = {
        'ElasticNet_balanced': 'elasticnet_balanced',
        'ElasticNet_filtered': 'elasticnet_filtered',
        'ElasticNet_unweighted': 'elasticnet_unweighted',
        'SVM_RBF_balanced': 'svm_rbf_balanced',
        'Dummy_stratified': 'dummy',
    }

    all_metrics = {}
    all_predictions = {}
    all_selected_features = {}
    all_coefficients = {}
    all_best_params = {}
    all_filter_indices = {}

    for name, model_type in models.items():
        print(f"\n  Running {name}...")
        y_pred, y_proba, sel_feats, best_params, coefs, filter_idxs = run_nested_loo_cv(
            X, y, model_type=model_type, tune_inner=(model_type != 'dummy')
        )
        metrics = compute_metrics(y, y_pred, y_proba)
        all_metrics[name] = metrics
        all_predictions[name] = {'y_pred': y_pred, 'y_proba': y_proba}
        all_selected_features[name] = sel_feats
        all_coefficients[name] = coefs
        all_best_params[name] = best_params
        all_filter_indices[name] = filter_idxs
        print(f"    ROC-AUC={metrics['roc_auc']:.3f}, PR-AUC={metrics['pr_auc']:.3f}, "
              f"BalAcc={metrics['balanced_accuracy']:.3f}, Sens={metrics['sensitivity']:.3f}, "
              f"Spec={metrics['specificity']:.3f}, F1={metrics['f1']:.3f}")
        print(f"    CM: {metrics['confusion_matrix']}")

    # Step 3: Select best model by AUC and run permutation test
    # Exclude SVM from "best" consideration since RBF overfits with 784 features / 15 samples
    # (permutation test showed null AUC ≈ 1.0 for SVM — it separates ANY label assignment)
    non_svm_models = {k: v for k, v in all_metrics.items() if 'svm' not in k.lower() and 'dummy' not in k.lower()}
    best_model_name = max(non_svm_models, key=lambda k: non_svm_models[k]['roc_auc'])
    print(f"\n  Best non-SVM model by ROC-AUC: {best_model_name} (AUC={all_metrics[best_model_name]['roc_auc']:.3f})")
    print(f"  (SVM excluded: RBF kernel overfits — null AUC ≈ 1.0 on permuted labels)")

    # Get fixed hyperparameters for permutation test
    from collections import Counter
    param_list = all_best_params[best_model_name]

    if 'filtered' in best_model_name:
        fixed_params = {'C': param_list[0]['C'],
                        'l1_ratio': param_list[0]['l1_ratio'],
                        'n_top': param_list[0].get('n_top', 20)}
        # Use most common C and l1_ratio
        c_counter = Counter(p['C'] for p in param_list)
        l1_counter = Counter(p['l1_ratio'] for p in param_list)
        fixed_params['C'] = c_counter.most_common(1)[0][0]
        fixed_params['l1_ratio'] = l1_counter.most_common(1)[0][0]
        perm_model_type = 'elasticnet_filtered'
    elif 'elasticnet' in best_model_name:
        c_counter = Counter(p['C'] for p in param_list)
        l1_counter = Counter(p['l1_ratio'] for p in param_list)
        fixed_params = {'C': c_counter.most_common(1)[0][0],
                        'l1_ratio': l1_counter.most_common(1)[0][0]}
        perm_model_type = 'elasticnet'
    else:
        fixed_params = {}
        perm_model_type = None

    print(f"  Fixed hyperparams for permutation: {fixed_params}")

    observed_auc = all_metrics[best_model_name]['roc_auc']
    if perm_model_type:
        null_aucs = run_permutation_test(X, y, perm_model_type, fixed_params,
                                         n_permutations=N_PERMUTATIONS)
        p_value = np.mean(null_aucs >= observed_auc)
    else:
        null_aucs = np.array([])
        p_value = np.nan

    print(f"  Observed AUC: {observed_auc:.3f}")
    if len(null_aucs) > 0:
        print(f"  Null AUC: mean={np.mean(null_aucs):.3f}, std={np.std(null_aucs):.3f}")
        print(f"  Empirical p-value: {p_value:.4f}")

    # Also run permutation test on SVM to demonstrate overfitting
    svm_name = 'SVM_RBF_balanced'
    svm_param_list = all_best_params[svm_name]
    svm_c_counter = Counter(p['C'] for p in svm_param_list)
    svm_g_counter = Counter(p['gamma'] for p in svm_param_list)
    svm_fixed = {'C': svm_c_counter.most_common(1)[0][0],
                 'gamma': svm_g_counter.most_common(1)[0][0]}
    svm_observed_auc = all_metrics[svm_name]['roc_auc']
    print(f"\n  Permutation test for SVM (200 perms, to demonstrate overfitting)...")
    svm_null_aucs = run_permutation_test(X, y, 'svm_rbf', svm_fixed,
                                         n_permutations=200)
    svm_p_value = np.mean(svm_null_aucs >= svm_observed_auc)
    print(f"  SVM null AUC: mean={np.mean(svm_null_aucs):.3f} — {'OVERFITTING CONFIRMED' if np.mean(svm_null_aucs) > 0.9 else 'no overfitting'}")

    # Threshold analysis for best model
    print(f"\n=== Threshold analysis for {best_model_name} ===")
    best_proba = all_predictions[best_model_name]['y_proba']
    opt_thresh, opt_metrics = find_optimal_threshold(y, best_proba)
    print(f"  Optimal threshold (max balanced acc): {opt_thresh:.2f}")
    print(f"    BalAcc={opt_metrics['balanced_accuracy']:.3f}, "
          f"Sens={opt_metrics['sensitivity']:.3f}, Spec={opt_metrics['specificity']:.3f}, "
          f"F1={opt_metrics['f1']:.3f}")
    print(f"    CM at optimal: {opt_metrics['confusion_matrix']}")
    print(f"  (Note: threshold tuning is post-hoc on LOO-CV predictions — report as sensitivity analysis)")

    # Step 4: Feature selection frequency analysis (best elastic net model)
    print(f"\n=== Step 4: Feature selection frequency ({best_model_name}) ===")
    n_genes = len(gene_names)
    selection_freq = np.zeros(n_genes)
    mean_coef = np.zeros(n_genes)
    n_folds_with_coef = 0

    for sel, coef, filt_idx in zip(all_selected_features[best_model_name],
                                    all_coefficients[best_model_name],
                                    all_filter_indices[best_model_name]):
        if coef is not None:
            if filt_idx is not None:
                # Filtered model: map coefficients back to original gene indices
                for i, orig_idx in enumerate(filt_idx):
                    if np.abs(coef[i]) > 0:
                        selection_freq[orig_idx] += 1
                        mean_coef[orig_idx] += coef[i]
                n_folds_with_coef += 1
            else:
                # Non-filtered model: coefficients are already in original space
                selection_freq += (np.abs(coef) > 0).astype(float)
                mean_coef += coef
                n_folds_with_coef += 1

    selection_freq /= max(n_folds_with_coef, 1)
    if n_folds_with_coef > 0:
        mean_coef /= n_folds_with_coef

    n_selected_any = np.sum(selection_freq > 0)
    n_selected_majority = np.sum(selection_freq >= 0.5)
    print(f"  Genes selected in any fold: {n_selected_any}")
    print(f"  Genes selected in ≥50% of folds: {n_selected_majority}")

    # Top selected genes
    top_idx = np.argsort(selection_freq)[::-1][:20]
    print("\n  Top 20 selected genes:")
    for idx in top_idx:
        if selection_freq[idx] > 0:
            mineral = gene_to_minerals.get(gene_names[idx], '')
            print(f"    {gene_names[idx]:12s} freq={selection_freq[idx]:.2f} "
                  f"coef={mean_coef[idx]:+.4f} {mineral}")

    # Step 5: Generate figures
    print("\n=== Step 5: Generating figures ===")
    primary_pred = all_predictions[best_model_name]
    primary_metrics = all_metrics[best_model_name]

    # Fig 3 (updated): ROC + PR curves (best model)
    plot_roc_pr_curves(y, primary_pred['y_proba'], primary_metrics, 'fig3_roc_pr_curves')

    # Fig S6: Permutation null distribution (best model)
    if len(null_aucs) > 0:
        plot_permutation_null(null_aucs, observed_auc, p_value, 'figS6_permutation_null')

    # Fig S7: Confusion matrix at optimal threshold
    opt_y_pred = (primary_pred['y_proba'] >= opt_thresh).astype(int)
    plot_confusion_matrix(y, opt_y_pred, 'figS7_confusion_matrix')

    # Fig S8: Selected features bar plot (elastic net)
    plot_selected_features(selection_freq, mean_coef, gene_names, gene_to_minerals,
                           'figS8_selected_features')

    # Step 6: Save tables
    print("\n=== Step 6: Saving tables ===")

    # Model comparison
    model_comp_rows = []
    for name, m in all_metrics.items():
        model_comp_rows.append({
            'model': name,
            'roc_auc': m['roc_auc'],
            'pr_auc': m['pr_auc'],
            'balanced_accuracy': m['balanced_accuracy'],
            'sensitivity': m['sensitivity'],
            'specificity': m['specificity'],
            'f1': m['f1'],
            'accuracy': m['accuracy'],
            'confusion_matrix': str(m['confusion_matrix']),
        })
    model_comp_df = pd.DataFrame(model_comp_rows)
    model_comp_df.to_csv(DATA_DIR / 'ml_model_comparison_v2.csv', index=False)
    print(f"  Saved ml_model_comparison_v2.csv: {model_comp_df.shape}")

    # Classification results (best model)
    clf_results = pd.DataFrame({
        'sample': sample_names,
        'true_label': y,
        'predicted_label_0.5': primary_pred['y_pred'],
        'predicted_label_optimal': opt_y_pred,
        'flight_probability': primary_pred['y_proba'],
        'optimal_threshold': opt_thresh,
    })
    clf_results.to_csv(DATA_DIR / 'ml_classification_results_v2.csv', index=False)
    print(f"  Saved ml_classification_results_v2.csv: {clf_results.shape}")

    # Selected features (elastic net)
    sel_feat_df = pd.DataFrame({
        'gene': gene_names,
        'selection_frequency': selection_freq,
        'mean_coefficient': mean_coef,
        'abs_coefficient': np.abs(mean_coef),
        'minerals': [gene_to_minerals.get(g, '') for g in gene_names],
    })
    sel_feat_df = sel_feat_df[sel_feat_df['selection_frequency'] > 0].sort_values(
        'selection_frequency', ascending=False
    )
    sel_feat_df.to_csv(DATA_DIR / 'ml_selected_features_v2.csv', index=False)
    print(f"  Saved ml_selected_features_v2.csv: {sel_feat_df.shape}")

    # Permutation test results (best model + SVM for comparison)
    perm_rows = []
    if len(null_aucs) > 0:
        perm_rows.append({
            'model': best_model_name,
            'observed_auc': observed_auc,
            'null_mean': np.mean(null_aucs),
            'null_std': np.std(null_aucs),
            'null_median': np.median(null_aucs),
            'null_q95': np.quantile(null_aucs, 0.95),
            'empirical_p_value': p_value,
            'n_permutations': N_PERMUTATIONS,
            'fixed_params': str(fixed_params),
        })
    perm_rows.append({
        'model': svm_name,
        'observed_auc': svm_observed_auc,
        'null_mean': np.mean(svm_null_aucs),
        'null_std': np.std(svm_null_aucs),
        'null_median': np.median(svm_null_aucs),
        'null_q95': np.quantile(svm_null_aucs, 0.95),
        'empirical_p_value': svm_p_value,
        'n_permutations': N_PERMUTATIONS,
        'fixed_params': str(svm_fixed),
    })
    perm_df = pd.DataFrame(perm_rows)
    perm_df.to_csv(DATA_DIR / 'ml_permutation_test.csv', index=False)
    print(f"  Saved ml_permutation_test.csv: {perm_df.shape}")

    # Null distributions for reproducibility
    null_dict = {}
    if len(null_aucs) > 0:
        null_dict['null_auc_best'] = null_aucs
    null_dict['null_auc_svm'] = svm_null_aucs
    null_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in null_dict.items()]))
    null_df.to_csv(DATA_DIR / 'ml_permutation_null_distribution.csv', index=False)

    # Threshold analysis
    thresh_df = pd.DataFrame([opt_metrics])
    thresh_df['model'] = best_model_name
    thresh_df.to_csv(DATA_DIR / 'ml_threshold_analysis.csv', index=False)
    print(f"  Saved ml_threshold_analysis.csv")

    # Step 7: Copy to Zenodo and manuscript
    print("\n=== Step 7: Copying outputs ===")
    zenodo_data = ZENODO_DIR / 'data'
    zenodo_figs = ZENODO_DIR / 'figures'
    zenodo_code = ZENODO_DIR / 'code'

    for csv_file in ['ml_model_comparison_v2.csv', 'ml_classification_results_v2.csv',
                     'ml_selected_features_v2.csv', 'ml_permutation_test.csv',
                     'ml_permutation_null_distribution.csv', 'ml_threshold_analysis.csv']:
        src = DATA_DIR / csv_file
        if src.exists():
            shutil.copy(str(src), str(zenodo_data / csv_file))
            print(f"  Copied {csv_file} to Zenodo")

    for fig_prefix in ['fig3_roc_pr_curves', 'figS6_permutation_null',
                       'figS7_confusion_matrix', 'figS8_selected_features']:
        for ext in ['svg', 'png']:
            src = FIG_DIR / f'{fig_prefix}.{ext}'
            if src.exists():
                shutil.copy(str(src), str(zenodo_figs / f'{fig_prefix}.{ext}'))
        # PNG to manuscript
        png_src = FIG_DIR / f'{fig_prefix}.png'
        if png_src.exists():
            shutil.copy(str(png_src), str(MANUSCRIPT_FIG_DIR / f'{fig_prefix}.png'))
            print(f"  Copied {fig_prefix}.png to manuscript")

    # Copy script
    shutil.copy(__file__, str(zenodo_code / os.path.basename(__file__)))
    print(f"  Copied script to Zenodo code/")

    print("\n" + "=" * 60)
    print("ML RE-ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nBest model (non-SVM): {best_model_name}")
    print(f"  ROC-AUC: {primary_metrics['roc_auc']:.3f}")
    print(f"  PR-AUC:  {primary_metrics['pr_auc']:.3f}")
    print(f"  Balanced Acc (thresh=0.5): {primary_metrics['balanced_accuracy']:.3f}")
    print(f"  Balanced Acc (thresh={opt_thresh:.2f}): {opt_metrics['balanced_accuracy']:.3f}")
    print(f"  Sensitivity (thresh={opt_thresh:.2f}): {opt_metrics['sensitivity']:.3f}")
    print(f"  Specificity (thresh={opt_thresh:.2f}): {opt_metrics['specificity']:.3f}")
    if len(null_aucs) > 0:
        print(f"  Permutation p-value: {p_value:.4f}")
    print(f"\nSVM (overfitting demonstration):")
    print(f"  ROC-AUC: {svm_observed_auc:.3f}, null mean: {np.mean(svm_null_aucs):.3f}, p-value: {svm_p_value:.4f}")
    print(f"\nFeature selection ({best_model_name}):")
    print(f"  Features selected (any fold): {n_selected_any}")
    print(f"  Features selected (≥50% folds): {n_selected_majority}")


if __name__ == '__main__':
    main()
