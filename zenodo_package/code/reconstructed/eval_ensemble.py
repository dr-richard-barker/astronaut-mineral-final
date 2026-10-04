"""DAE + FT-Transformer ensembles, appended to the extended classification results.

(1) Concatenation: the two 16-dim latents joined into 32 features, four classifiers,
    LOAO, 500 permutations.
(2) Prediction averaging: mean of the out-of-fold probabilities of DAE + random forest
    and Transformer + SVM-RBF. Per the Methods, the permutation test retrains both
    classifiers within each fold on shuffled training labels and averages them.

Run eval_all_architectures.py first.

RECONSTRUCTION (Oct 2026) of a lost original; see FIDELITY.md.
"""
import argparse

import numpy as np
import pandas as pd

from _common import (CLF_NAMES, auc, clf_fit_predict, dirs, find, io_args, latents, load_meta,
                     loao_clf, p_auc, perm_null)


def main():
    ap = io_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--n-perm", type=int, default=500)
    a = ap.parse_args()
    data, out = dirs(a)
    meta, y, astr = load_meta(data)
    dae = latents(find("autoencoder_latent_features.csv", a), meta)
    tft = latents(find("transformer_latent_features.csv", a), meta)
    res_path, roc_path = out / "dl_extended_classification_results.csv", out / "dl_extended_roc_data.csv"
    res, roc = pd.read_csv(res_path), pd.read_csv(roc_path)
    res = res[~res.feature_space.str.startswith("Ensemble")]
    roc = roc[~roc.feature_space.str.startswith("Ensemble")]

    new, new_roc = [], []
    Xc = np.hstack([dae, tft])
    for c in CLF_NAMES:
        oof = loao_clf(Xc, y, astr, c)
        pooled = auc(y, oof)
        null = perm_null(auc, Xc, y, astr, clf_fit_predict(c), a.n_perm)
        new.append({"feature_space": "Ensemble_concat", "classifier": c, "n_features": 32,
                    "pooled_auc": pooled, "perm_null_mean": null.mean(), "perm_null_std": null.std(),
                    "p_value": p_auc(null, pooled)})
        new_roc += [{"feature_space": "Ensemble_concat", "classifier": c, "sample_idx": i,
                     "true_label": y[i], "predicted_prob": oof[i], "astronaut": astr[i]} for i in range(len(y))]
        print(f"Ensemble_concat/{c}: AUC={pooled:.3f} p={new[-1]['p_value']:.3f}")

    # Prediction averaging. Both feature blocks travel together so a single
    # within-fold label shuffle drives both retrained classifiers.
    rf, svm = clf_fit_predict("RF"), clf_fit_predict("SVM_rbf")

    def avg_fit_predict(X, yy, tr, te):
        return (rf(X[:, :16], yy, tr, te) + svm(X[:, 16:], yy, tr, te)) / 2

    avg = (loao_clf(dae, y, astr, "RF") + loao_clf(tft, y, astr, "SVM_rbf")) / 2
    pooled = auc(y, avg)
    null = perm_null(auc, Xc, y, astr, avg_fit_predict, a.n_perm)
    new.append({"feature_space": "Ensemble_avg", "classifier": "DAE_RF+TFL_SVMrbf", "n_features": 32,
                "pooled_auc": pooled, "perm_null_mean": null.mean(), "perm_null_std": null.std(),
                "p_value": p_auc(null, pooled)})
    new_roc += [{"feature_space": "Ensemble_avg", "classifier": "DAE_RF+TFL_SVMrbf", "sample_idx": i,
                 "true_label": y[i], "predicted_prob": avg[i], "astronaut": astr[i]} for i in range(len(y))]
    print(f"Ensemble_avg: AUC={pooled:.3f} p={new[-1]['p_value']:.3f}")

    pd.concat([res, pd.DataFrame(new)], ignore_index=True).to_csv(res_path, index=False)
    pd.concat([roc, pd.DataFrame(new_roc)], ignore_index=True).to_csv(roc_path, index=False)


if __name__ == "__main__":
    main()
