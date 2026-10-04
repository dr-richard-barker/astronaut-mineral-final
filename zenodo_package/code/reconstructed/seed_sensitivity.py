"""Seed sensitivity of the latent-space results (the seed tables in FIDELITY.md).

--model dae          retrains the autoencoder under seeds 0..n-1 and reports, without
                     permutations: DAE+RF and DAE+LogReg flight vs pre-flight AUC,
                     hemoglobin r (SVR-RBF) and potassium r (RF) from the latents.
--model transformer  retrains autoencoder and FT-Transformer together under seeds 0..n-1
                     and reports DAE+RF AUC, Transformer+SVM-RBF AUC, the best of the four
                     classifiers on the Transformer latent, and the prediction-averaging
                     ensemble AUC.

RECONSTRUCTION (Oct 2026) of a lost original; see FIDELITY.md.
"""
import argparse

import numpy as np
import pandas as pd

import train_autoencoder as ta
import train_transformer as tt
from _common import CLF_NAMES, DATA, OUT, auc, i4_targets, load_meta, loao_clf, loao_reg, r_ok, zscore_matrix


def oof_latents(module, X, astr, seed):
    Z = np.zeros((len(X), 16))
    for a in sorted(set(astr)):
        tr, te = astr != a, astr == a
        m, _, _ = module.train(X[tr], seed=seed)
        Z[te] = module.encode(m, X[te])
    return Z


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["dae", "transformer"], default="dae")
    ap.add_argument("--n-seeds", type=int, default=None, help="default: 10 (dae) or 5 (transformer)")
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args()
    n = a.n_seeds or (10 if a.model == "dae" else 5)
    meta, y, astr = load_meta(DATA)
    X = zscore_matrix(DATA, meta).astype(np.float32)
    i4, ydf = i4_targets(DATA, meta)
    idx = meta.reset_index().set_index("sample_id").loc[i4.sample_id, "index"].values
    a4 = i4.astronaut_id.values

    rows = []
    for seed in range(n):
        Zd = oof_latents(ta, X, astr, seed)
        p_rf = loao_clf(Zd, y, astr, "RF")
        row = {"seed": seed, "AUC_DAE_RF": auc(y, p_rf)}
        if a.model == "dae":
            hb, k = (ydf[c].values.astype(float) for c in ("HEMOGLOBIN", "POTASSIUM"))
            row.update(AUC_DAE_LogReg=auc(y, loao_clf(Zd, y, astr, "LogReg")),
                       r_Hb_SVRrbf=r_ok(hb, loao_reg(Zd[idx], hb, a4, "SVR_rbf")),
                       r_K_RF=r_ok(k, loao_reg(Zd[idx], k, a4, "RF")))
        else:
            Zt = oof_latents(tt, X, astr, seed)
            p_t = loao_clf(Zt, y, astr, "SVM_rbf")
            row.update(AUC_TFL_SVMrbf=auc(y, p_t),
                       AUC_TFL_best=max(auc(y, loao_clf(Zt, y, astr, c)) for c in CLF_NAMES),
                       AUC_Ensemble_avg=auc(y, (p_rf + p_t) / 2))
        rows.append(row)
        print({k: round(v, 3) for k, v in row.items()}, flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(f"{a.out_dir}/seed_sensitivity_{a.model}.csv", index=False)
    print(d.drop(columns="seed").describe().loc[["mean", "std", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
