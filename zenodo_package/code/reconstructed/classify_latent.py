"""Flight vs pre-flight classification on the 16-dim DAE latent space vs the 779-gene
z-scored matrix. LogReg, SVM (RBF, linear) and random forest; LOAO CV over 6
astronauts; 1000-permutation test that shuffles training labels within each fold.

RECONSTRUCTION (Oct 2026) of a lost original; see FIDELITY.md. With
--use-deposited-latents the LogReg and SVM out-of-fold probabilities reproduce
data/dl_classification_roc_data.csv to <= 1e-4; random forest is approximate,
and permutation p-values differ by Monte-Carlo error.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from _common import (auc, clf_fit_predict, dirs, find, io_args, latents, load_meta, loao_clf,
                     p_auc, perm_null, zscore_matrix)

NAMES = {"LogReg": "LogReg", "SVM_rbf": "SVM_rbf", "SVM_linear": "SVM_linear", "RF": "RandomForest"}


def main():
    ap = io_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--n-perm", type=int, default=1000)
    a = ap.parse_args()
    data, out = dirs(a)
    meta, y, astr = load_meta(data)
    spaces = {"latent": latents(find("autoencoder_latent_features.csv", a), meta),
              "raw": zscore_matrix(data, meta)}
    print(f"Latent input: {find('autoencoder_latent_features.csv', a)}")

    rows, roc_rows = [], []
    for fs, X in spaces.items():
        for c, label in NAMES.items():
            oof, lab = loao_clf(X, y, astr, c, labels=True)
            pooled = auc(y, oof)
            folds, accs, f1s = [], [], []
            for ast in sorted(set(astr)):
                te = astr == ast
                folds.append(roc_auc_score(y[te], oof[te]) if len(set(y[te])) == 2 else np.nan)
                accs.append(accuracy_score(y[te], lab[te]))
                f1s.append(f1_score(y[te], lab[te], zero_division=0))
            null = perm_null(auc, X, y, astr, clf_fit_predict(c), a.n_perm)
            p = p_auc(null, pooled)
            rows.append({"feature_space": fs, "classifier": label, "pooled_auc": pooled,
                         "mean_fold_auc": np.nanmean(folds), "mean_acc": np.mean(accs),
                         "mean_f1": np.mean(f1s), "perm_null_mean": null.mean(),
                         "perm_null_std": null.std(), "p_value": p,
                         "fold_aucs": str([round(float(v), 3) for v in folds])})
            roc_rows += [{"feature_space": fs, "classifier": label, "sample_idx": i,
                          "true_label": float(y[i]), "predicted_prob": oof[i], "astronaut": astr[i]}
                         for i in range(len(y))]
            print(f"{fs}/{label}: pooled AUC={pooled:.3f}, perm p={p:.4f}")

    pd.DataFrame(rows).to_csv(out / "dl_classification_results.csv", index=False)
    pd.DataFrame(roc_rows).to_csv(out / "dl_classification_roc_data.csv", index=False)


if __name__ == "__main__":
    main()
