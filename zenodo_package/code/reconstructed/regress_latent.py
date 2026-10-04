"""Predicts serum calcium, potassium, sodium and hemoglobin from the DAE latent space and
from the 779-gene z-scored matrix (I4 only, 16 samples, LOAO over 4 astronauts).
Ridge, SVR (linear, RBF) and random forest; 1000-permutation test shuffling training
labels within each fold; p = P(|null r| >= |r|).

RECONSTRUCTION (Oct 2026) of a lost original; see FIDELITY.md. With
--use-deposited-latents the Ridge and SVR predictions reproduce
data/dl_regression_predictions.csv to <= 1e-3; random forest is approximate.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _common import (TARGETS, dirs, find, io_args, latents, load_meta, loao_reg, p_r, perm_null,
                     r_ok, reg_fit_predict, zscore_matrix, i4_targets)

NAMES = {"Ridge": "Ridge", "SVR_lin": "SVR_linear", "SVR_rbf": "SVR_rbf", "RF": "RandomForest"}


def main():
    ap = io_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--n-perm", type=int, default=1000)
    a = ap.parse_args()
    data, out = dirs(a)
    meta, _, _ = load_meta(data)
    i4, ydf = i4_targets(data, meta)
    idx = meta.reset_index().set_index("sample_id").loc[i4.sample_id, "index"].values
    spaces = {"latent": latents(find("autoencoder_latent_features.csv", a), meta)[idx],
              "raw": zscore_matrix(data, meta)[idx]}
    astr = i4.astronaut_id.values

    res, preds = [], []
    for fs, X in spaces.items():
        for t in TARGETS:
            yv = ydf[t].values.astype(float)
            ok = ~np.isnan(yv)
            for m, label in NAMES.items():
                oof = loao_reg(X, yv, astr, m)
                r = r_ok(yv, oof)
                null = perm_null(r_ok, X, yv, astr, reg_fit_predict(m), a.n_perm)
                res.append({"feature_space": fs, "target": t, "model": label, "r_pearson": r,
                            "r_spearman": spearmanr(yv[ok], oof[ok])[0],
                            "r2": 1 - np.sum((yv[ok] - oof[ok]) ** 2) / np.sum((yv[ok] - yv[ok].mean()) ** 2),
                            "mae": np.abs(yv[ok] - oof[ok]).mean(), "perm_p_value": p_r(null, r),
                            "perm_null_r_mean": null.mean(), "perm_null_r_std": null.std()})
                if fs == "latent":
                    preds += [{"target": t, "model": label, "sample_id": s, "astronaut_id": s.split("_")[0],
                               "true_value": tv, "predicted": pv}
                              for s, tv, pv in zip(i4.sample_id, yv, oof)]
                print(f"{fs}/{t}/{label}: r={r:.3f}, perm p={res[-1]['perm_p_value']:.3f}")

    pd.DataFrame(res).to_csv(out / "dl_regression_results.csv", index=False)
    pd.DataFrame(preds).to_csv(out / "dl_regression_predictions.csv", index=False)


if __name__ == "__main__":
    main()
