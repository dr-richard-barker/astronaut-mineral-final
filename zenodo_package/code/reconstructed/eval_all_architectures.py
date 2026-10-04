"""Compares five feature spaces (DAE, VAE, Transformer latents; Raw_779; the 15-gene
multi-omics consensus set) for flight vs pre-flight classification (500 permutations)
and I4 mineral regression (100 permutations), LOAO CV throughout.

As in the deposited data/dl_extended_regression_results.csv, regression permutation
tests are run for the three latent spaces only; Raw_779 and Multiomics_15 get r and
R^2 without a p-value.

RECONSTRUCTION (Oct 2026) of a lost original; see FIDELITY.md. The original run of
this step timed out in the recorded session (execution_trace cell 52); the deposited
results come from a later run that the trace does not show.
"""
import argparse

import numpy as np
import pandas as pd

from _common import (CLF_NAMES, REG_NAMES, TARGETS, auc, clf_fit_predict, dirs, find, i4_targets,
                     io_args, latents, load_meta, loao_clf, loao_reg, p_auc, p_r, perm_null, r_ok,
                     reg_fit_predict, zscore_matrix)


def spaces(data, meta, a):
    s = {name: latents(find(fn, a), meta) for name, fn in
         [("DAE", "autoencoder_latent_features.csv"), ("VAE", "vae_latent_features.csv"),
          ("Transformer", "transformer_latent_features.csv")]}
    s["Raw_779"] = zscore_matrix(data, meta)
    s["Multiomics_15"] = pd.read_csv(data / "harmonized_expression_multiomics_filtered.csv",
                                     index_col=0)[meta.sample_id].T.values
    return s


def main():
    ap = io_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--n-perm-clf", type=int, default=500)
    ap.add_argument("--n-perm-reg", type=int, default=100)
    a = ap.parse_args()
    data, out = dirs(a)
    meta, y, astr = load_meta(data)
    S = spaces(data, meta, a)
    i4, ydf = i4_targets(data, meta)
    idx = meta.reset_index().set_index("sample_id").loc[i4.sample_id, "index"].values
    a4 = i4.astronaut_id.values

    clf_rows, roc_rows, reg_rows = [], [], []
    for fs, X in S.items():
        for c in CLF_NAMES:
            oof = loao_clf(X, y, astr, c)
            pooled = auc(y, oof)
            null = perm_null(auc, X, y, astr, clf_fit_predict(c), a.n_perm_clf)
            clf_rows.append({"feature_space": fs, "classifier": c, "n_features": X.shape[1],
                             "pooled_auc": pooled, "perm_null_mean": null.mean(),
                             "perm_null_std": null.std(), "p_value": p_auc(null, pooled)})
            roc_rows += [{"feature_space": fs, "classifier": c, "sample_idx": i, "true_label": y[i],
                          "predicted_prob": oof[i], "astronaut": astr[i]} for i in range(len(y))]
            print(f"CLF {fs}/{c}: AUC={pooled:.3f} p={clf_rows[-1]['p_value']:.3f}")
        Xi = X[idx]
        for t in TARGETS:
            yv = ydf[t].values.astype(float)
            ok = ~np.isnan(yv)
            for m in REG_NAMES:
                oof = loao_reg(Xi, yv, a4, m)
                r = r_ok(yv, oof)
                row = {"feature_space": fs, "target": t, "model": m, "r_pearson": r,
                       "r2": 1 - np.sum((yv[ok] - oof[ok]) ** 2) / np.sum((yv[ok] - yv[ok].mean()) ** 2),
                       "p_value": np.nan, "perm_null_mean": np.nan, "perm_null_std": np.nan}
                if fs in ("DAE", "VAE", "Transformer"):
                    null = perm_null(r_ok, Xi, yv, a4, reg_fit_predict(m), a.n_perm_reg)
                    row.update(p_value=p_r(null, r), perm_null_mean=null.mean(), perm_null_std=null.std())
                reg_rows.append(row)
            print(f"REG {fs}/{t} done")

    pd.DataFrame(clf_rows).to_csv(out / "dl_extended_classification_results.csv", index=False)
    pd.DataFrame(reg_rows).to_csv(out / "dl_extended_regression_results.csv", index=False)
    pd.DataFrame(roc_rows).to_csv(out / "dl_extended_roc_data.csv", index=False)


if __name__ == "__main__":
    main()
