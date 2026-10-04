"""Shared paths, models and cross-validation helpers for the reconstructed scripts.

RECONSTRUCTION (Oct 2026) of lost original code; see FIDELITY.md.
Hyperparameters marked VERIFIED reproduce the original out-of-fold predictions
(data/dl_classification_roc_data.csv, data/dl_regression_predictions.csv) to
<= 1e-4 when run on the deposited latent features. Random forests could not be
matched (class-weighted leaves, version-dependent RNG) and are approximate.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

SEED = 42
PKG = Path(__file__).resolve().parents[2]          # zenodo_package/
DATA = PKG / "data"                                 # deposited original outputs (read-only)
OUT = PKG / "reproduced"                            # everything these scripts write
LF = [f"LF{i}" for i in range(1, 17)]
TARGETS = ["CALCIUM", "POTASSIUM", "SODIUM", "HEMOGLOBIN"]


def io_args(ap):
    ap.add_argument("--data-dir", default=str(DATA), help="deposited inputs (never written)")
    ap.add_argument("--out-dir", default=str(OUT), help="where results are written")
    ap.add_argument("--use-deposited-latents", action="store_true",
                    help="read *_latent_features.csv from --data-dir even if reproduced ones exist")
    return ap


def dirs(a):
    data, out = Path(a.data_dir), Path(a.out_dir)
    if out.resolve() == data.resolve():
        raise SystemExit("--out-dir must differ from --data-dir: the deposited outputs are authoritative")
    out.mkdir(parents=True, exist_ok=True)
    return data, out


def find(name, a):
    """Prefer a reproduced file over the deposited one, unless told otherwise."""
    rep = Path(a.out_dir) / name
    if rep.exists() and not (a.use_deposited_latents and "latent" in name):
        return rep
    return Path(a.data_dir) / name


def load_meta(data):
    meta = pd.read_csv(data / "harmonized_astronaut_metadata.csv")
    y = (meta.flight_status == "post").astype(int).values
    return meta, y, meta.astronaut_id.values


def zscore_matrix(data, meta):
    """Per-study z-scored 779-gene matrix (the autoencoder input and 'Raw_779')."""
    return pd.read_csv(data / "harmonized_expression_zscore.csv", index_col=0)[meta.sample_id].T.values


def latents(path, meta):
    return pd.read_csv(path).set_index("sample_id").loc[meta.sample_id][LF].values


# --- models -----------------------------------------------------------------
def clf(name):
    return {  # VERIFIED: LogReg, SVM_linear, SVM_rbf (with per-fold StandardScaler)
        "LogReg": LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=SEED),
        "SVM_rbf": SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=SEED),
        "SVM_linear": SVC(kernel="linear", C=0.1, probability=True, class_weight="balanced", random_state=SEED),
        "RF": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=SEED),  # approximate
    }[name]


def reg(name):
    return {  # VERIFIED: Ridge, SVR_lin, SVR_rbf (with per-fold StandardScaler)
        "Ridge": Ridge(alpha=1.0),
        "SVR_lin": SVR(kernel="linear", C=0.1, epsilon=0.1),
        "SVR_rbf": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        "RF": RandomForestRegressor(n_estimators=500, random_state=SEED),  # approximate
    }[name]


CLF_NAMES = ["LogReg", "SVM_rbf", "SVM_linear", "RF"]
REG_NAMES = ["Ridge", "SVR_lin", "SVR_rbf", "RF"]


# --- leave-one-astronaut-out ---------------------------------------------------
def loao_clf(X, y_train, astr, name, labels=False):
    """Out-of-fold P(post). Features are standardised within each training fold.
    With labels=True also returns predict() labels (for an SVC these come from the
    decision function, not P >= 0.5; the original accuracy/F1 used predict())."""
    oof, lab = np.zeros(len(y_train)), np.zeros(len(y_train), int)
    for a in sorted(set(astr)):
        tr, te = astr != a, astr == a
        s = StandardScaler().fit(X[tr])
        m = clf(name).fit(s.transform(X[tr]), y_train[tr])
        oof[te] = m.predict_proba(s.transform(X[te]))[:, 1]
        lab[te] = m.predict(s.transform(X[te]))
    return (oof, lab) if labels else oof


def loao_reg(X, y, astr, name):
    """Out-of-fold predictions; samples with a missing target are left out of
    training but still predicted (as in the original)."""
    oof = np.zeros(len(y))
    ok = ~np.isnan(y)
    for a in sorted(set(astr)):
        tr, te = (astr != a) & ok, astr == a
        s = StandardScaler().fit(X[tr])
        oof[te] = reg(name).fit(s.transform(X[tr]), y[tr]).predict(s.transform(X[te]))
    return oof


def _shuffle_within_folds(y, astr, rng):
    """Methods: 'shuffling training labels within each fold'. Each fold's
    training labels are permuted independently; test labels stay true."""
    return {a: rng.permutation(y[astr != a]) for a in sorted(set(astr))}


def _perm_oof(X, y, astr, fit_predict, seed):
    rng = np.random.default_rng(seed)
    shuf = _shuffle_within_folds(y, astr, rng)
    oof = np.zeros(len(y))
    for a in sorted(set(astr)):
        tr, te = astr != a, astr == a
        y_tr = y.copy()
        y_tr[tr] = shuf[a]
        oof[te] = fit_predict(X, y_tr, tr, te)
    return oof


def perm_null(stat, X, y, astr, fit_predict, n_perm, n_jobs=-1):
    """Null distribution of stat(y_true, oof) under within-fold label shuffling."""
    seeds = np.random.SeedSequence(SEED).generate_state(n_perm)
    oofs = Parallel(n_jobs=n_jobs)(delayed(_perm_oof)(X, y, astr, fit_predict, int(s)) for s in seeds)
    return np.array([stat(y, o) for o in oofs])


def clf_fit_predict(name):
    def f(X, y, tr, te):
        s = StandardScaler().fit(X[tr])
        return clf(name).fit(s.transform(X[tr]), y[tr]).predict_proba(s.transform(X[te]))[:, 1]
    return f


def reg_fit_predict(name):
    def f(X, y, tr, te):
        tr = tr & ~np.isnan(y)
        s = StandardScaler().fit(X[tr])
        return reg(name).fit(s.transform(X[tr]), y[tr]).predict(s.transform(X[te]))
    return f


def auc(y, p):
    return roc_auc_score(y, p)


def r_ok(y, p):
    ok = ~np.isnan(y)
    return pearsonr(y[ok], p[ok])[0]


def p_auc(null, obs):          # one-sided: AUC at least as large
    return float(np.mean(null >= obs))


def p_r(null, obs):            # two-sided on |r| (fits the original p-values best)
    return float(np.mean(np.abs(null) >= abs(obs)))


def i4_targets(data, meta):
    i4 = meta[meta.study == "I4"].reset_index(drop=True)
    minr = pd.read_csv(data / "astronaut_minerals_combined.csv")
    ydf = i4.merge(minr, left_on=["astronaut_id", "timepoint"],
                   right_on=["SUBJECT_ID", "timepoint"], how="left")
    return i4, ydf
