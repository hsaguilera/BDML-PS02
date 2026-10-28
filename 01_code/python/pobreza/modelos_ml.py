r"""Modelos de aprendizaje automático con el mismo protocolo de validación cruzada.

Todos los ajustes usan los 5 pliegues compartidos. Dentro de cada pliegue de
entrenamiento se separa un 10% (estratificado) SÓLO para la parada temprana, de modo
que el pliegue de validación nunca influye en el número de árboles (sin fuga). El
modelo de test se re-estima con todo train con ⌈1.1·media(iteraciones)⌉ árboles.

Formulaciones (todas devuelven un puntaje orientado: mayor ⇒ más probable pobre):

* ``clf``        clasificación binaria de Pobre (log-loss).
* ``reg``        regresión de t_h = log Ingtotug_h − log R_h (la brecha frente al ingreso
                 requerido; Pobre ⇔ t_h < 0). Con un modelo de dispersión para log r²
                 (heterocedástico, Harvey 1976) se obtiene P(Pobre) = Φ(−μ̂_h / σ̂_h).
* ``cuantil``    regresión cuantílica de t_h al nivel τ (Koenker y Bassett, 1978); el
                 puntaje es −q̂_τ(t_h). Brown, Ravallion y van de Walle (2018, JDE)
                 muestran que los PMT basados en cuantiles bajos reducen la exclusión.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.model_selection import StratifiedKFold, KFold

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None
try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


BASE_LGB = dict(learning_rate=0.03, num_leaves=31, min_child_samples=100, feature_fraction=0.7,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, min_gain_to_split=0.0,
                max_bin=127, verbose=-1, num_threads=0, cat_smooth=20, cat_l2=10, max_cat_to_onehot=8)


def preparar(df: pd.DataFrame, numericas, categoricas, ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """DataFrame para boosting: numéricas en float32 y categóricas como `category` con las
    categorías de `ref` (train) para que test use la misma codificación."""
    ref = df if ref is None else ref
    X = df[list(numericas)].astype("float32").copy()
    for c in categoricas:
        cats = pd.Index(sorted(ref[c].astype(str).unique()))
        X[c] = pd.Categorical(df[c].astype(str), categories=cats)
    return X


def _split_es(y_es, semilla, frac=0.1, estratificar=True):
    n = len(y_es)
    rng = np.random.default_rng(semilla)
    if estratificar:
        idx_es = []
        for v in np.unique(y_es):
            ii = np.flatnonzero(y_es == v); rng.shuffle(ii); idx_es.append(ii[: max(1, int(frac * len(ii)))])
        es = np.concatenate(idx_es)
    else:
        es = rng.choice(n, int(frac * n), replace=False)
    m = np.zeros(n, bool); m[es] = True
    return ~m, m


def cv_lgb(X: pd.DataFrame, target: np.ndarray, folds, params: dict, objetivo: str = "binary",
           X_test: pd.DataFrame | None = None, estrato=None, max_rounds=5000, paciencia=100,
           semillas=(1,), pesos=None, verbose=True):
    """Validación cruzada de LightGBM. Devuelve dict con oof, test, iteraciones, importancias."""
    p = {**BASE_LGB, **params, "objective": objetivo}
    if objetivo == "binary":
        p.setdefault("metric", "binary_logloss")
    oof = np.zeros(len(X)); te = np.zeros(len(X_test)) if X_test is not None else None
    iters, imp = [], pd.Series(0.0, index=X.columns)
    t0 = time.time()
    est = target if estrato is None else estrato
    for k, (tr, va) in enumerate(folds):
        pr = np.zeros(len(va))
        for s in semillas:
            fit_m, es_m = _split_es(np.asarray(est)[tr] if objetivo == "binary" or estrato is not None else np.zeros(len(tr)),
                                    1000 * s + k, estratificar=True)
            itr, ies = tr[fit_m], tr[es_m]
            w_tr = None if pesos is None else pesos[itr]
            dtr = lgb.Dataset(X.iloc[itr], target[itr], weight=w_tr, free_raw_data=False)
            des = lgb.Dataset(X.iloc[ies], target[ies], reference=dtr)
            m = lgb.train({**p, "seed": s, "bagging_seed": s, "feature_fraction_seed": s}, dtr, max_rounds,
                          valid_sets=[des], callbacks=[lgb.early_stopping(paciencia, verbose=False)])
            iters.append(m.best_iteration)
            pr += m.predict(X.iloc[va], num_iteration=m.best_iteration) / len(semillas)
            imp += pd.Series(m.feature_importance("gain"), index=X.columns) / (len(folds) * len(semillas))
        oof[va] = pr
        if verbose:
            print(f"    pliegue {k}: {m.best_iteration} árboles ({time.time()-t0:.0f}s)")
    if X_test is not None:
        n_final = int(np.ceil(1.1 * np.mean(iters)))
        for s in semillas:
            dall = lgb.Dataset(X, target, weight=pesos)
            m = lgb.train({**p, "seed": s, "bagging_seed": s, "feature_fraction_seed": s}, dall, n_final)
            te += m.predict(X_test) / len(semillas)
    return {"oof": oof, "test": te, "iteraciones": iters, "importancia": imp.sort_values(ascending=False),
            "tiempo_s": time.time() - t0, "params": p}


def cv_xgb(X: pd.DataFrame, y: np.ndarray, folds, params: dict, X_test=None, max_rounds=5000, paciencia=150,
           verbose=True):
    p = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", learning_rate=0.03,
             max_depth=6, min_child_weight=20, subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0,
             max_cat_to_onehot=8, nthread=0)
    p.update(params)
    oof = np.zeros(len(X)); iters = []; t0 = time.time()
    for k, (tr, va) in enumerate(folds):
        fit_m, es_m = _split_es(y[tr], 7 + k)
        dtr = xgb.DMatrix(X.iloc[tr[fit_m]], y[tr[fit_m]], enable_categorical=True)
        des = xgb.DMatrix(X.iloc[tr[es_m]], y[tr[es_m]], enable_categorical=True)
        m = xgb.train(p, dtr, max_rounds, evals=[(des, "es")], early_stopping_rounds=paciencia, verbose_eval=False)
        iters.append(m.best_iteration + 1)
        oof[va] = m.predict(xgb.DMatrix(X.iloc[va], enable_categorical=True), iteration_range=(0, m.best_iteration + 1))
        if verbose:
            print(f"    pliegue {k}: {iters[-1]} árboles ({time.time()-t0:.0f}s)")
    te = None
    if X_test is not None:
        m = xgb.train(p, xgb.DMatrix(X, y, enable_categorical=True), int(np.ceil(1.1 * np.mean(iters))))
        te = m.predict(xgb.DMatrix(X_test, enable_categorical=True))
    return {"oof": oof, "test": te, "iteraciones": iters, "tiempo_s": time.time() - t0, "params": p}


def prob_heterocedastica(X, t, mu_oof, mu_test, folds, params_sigma=None, X_test=None, verbose=False):
    """Modelo de dispersión: log r² ~ LightGBM(x), con r = t − μ̂ (OOF). Devuelve P(Pobre)=Φ(−μ/σ)
    OOF y de test. σ̂² = exp(E[log r²] + 1.27) (corrección: E[log χ²₁] = −1.27)."""
    r2 = np.log(np.maximum((t - mu_oof) ** 2, 1e-6))
    ps = {"num_leaves": 15, "min_child_samples": 300, "learning_rate": 0.03, **(params_sigma or {})}
    res = cv_lgb(X, r2, folds, ps, objetivo="regression", X_test=X_test, verbose=verbose)
    sig_oof = np.sqrt(np.exp(res["oof"] + 1.27)); sig_te = np.sqrt(np.exp(res["test"] + 1.27)) if X_test is not None else None
    # recalibración global de escala: que la media de (r/σ)² sea 1
    c = np.sqrt(np.mean(((t - mu_oof) / sig_oof) ** 2))
    p_oof = ndtr(-mu_oof / (c * sig_oof))
    p_te = ndtr(-mu_test / (c * sig_te)) if X_test is not None else None
    return p_oof, p_te, sig_oof * c, (sig_te * c if sig_te is not None else None)


def stacking_logit(S: pd.DataFrame, y, folds, S_test: pd.DataFrame | None = None, C=1.0):
    """Meta-modelo logit sobre logit(probabilidades) / puntajes estandarizados de los modelos base,
    ajustado con los mismos pliegues (las entradas ya son OOF)."""
    from sklearn.linear_model import LogisticRegression
    Z = S.copy()
    mu, sd = Z.mean(), Z.std().replace(0, 1)
    Z = (Z - mu) / sd
    oof = np.zeros(len(Z))
    for tr, va in folds:
        m = LogisticRegression(C=C, max_iter=2000).fit(Z.iloc[tr], y[tr])
        oof[va] = m.predict_proba(Z.iloc[va])[:, 1]
    mf = LogisticRegression(C=C, max_iter=2000).fit(Z, y)
    te = mf.predict_proba((S_test - mu) / sd)[:, 1] if S_test is not None else None
    return oof, te, pd.Series(mf.coef_[0], index=S.columns)


def logit_seguro(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))
