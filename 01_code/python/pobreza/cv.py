"""Validación cruzada.

* Pliegues estratificados por (Pobre × Dominio): preservan la tasa de pobreza y la
  composición geográfica en cada pliegue, reduciendo la varianza del estimador de F1.
* Validación cruzada *anidada* (Varma y Simon, 2006; Cawley y Talbot, 2010): todo lo
  que "aprende de los datos" (selección de variables, λ, umbral de clasificación)
  se ajusta dentro de los pliegues internos, de modo que el F1 externo es un
  estimador honesto del F1 en test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .metrics import umbral_optimo_f1, f1


def pliegues(y, estratos=None, k=5, semilla=20262, repeticiones=1):
    """Lista de (idx_train, idx_valid). Estratifica por y × estratos (si se dan)."""
    y = np.asarray(y).astype(int)
    if estratos is not None:
        e = pd.Series(np.asarray(estratos)).astype(str) + "_" + pd.Series(y).astype(str)
        # estratos con pocos casos se funden con la etiqueta de y
        vc = e.value_counts()
        e = np.where(e.map(vc) < k, pd.Series(y).astype(str), e)
    else:
        e = y
    out = []
    for r in range(repeticiones):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=semilla + r)
        out += list(skf.split(np.zeros(len(y)), e))
    return out


def id_pliegue(n, folds):
    f = np.full(n, -1)
    for i, (_, va) in enumerate(folds):
        f[va] = i
    return f


def oof_predicciones(ajustar, predecir, X, y, folds, **kw):
    """Predicciones fuera de pliegue. `ajustar(Xtr, ytr, **kw)` -> modelo;
    `predecir(modelo, Xva)` -> probabilidad/puntaje."""
    p = np.full(len(y), np.nan)
    modelos = []
    for tr, va in folds:
        m = ajustar(X[tr], y[tr], **kw)
        p[va] = predecir(m, X[va])
        modelos.append(m)
    return p, modelos


def f1_con_umbral_anidado(y, p, folds):
    """F1 honesto: para cada pliegue, el umbral se elige en los demás pliegues (OOF)
    y se aplica al pliegue retenido. Devuelve (F1 global, F1 por pliegue, umbrales)."""
    y = np.asarray(y).astype(int)
    yhat = np.zeros_like(y)
    umbrales, f1s = [], []
    for tr, va in folds:
        t, _ = umbral_optimo_f1(y[tr], p[tr])
        yhat[va] = (p[va] >= t).astype(int)
        umbrales.append(t); f1s.append(f1(y[va], yhat[va]))
    return f1(y, yhat), np.array(f1s), np.array(umbrales)


def bootstrap_ic(y, p, umbral, estadistico=None, B=500, semilla=1, alfa=.05):
    """IC bootstrap (percentil) del F1 dado un umbral fijo."""
    rng = np.random.default_rng(semilla)
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    est = estadistico or (lambda yy, pp: f1(yy, (pp >= umbral).astype(int)))
    n = len(y)
    vals = np.array([est(y[i], p[i]) for i in (rng.integers(0, n, n) for _ in range(B))])
    return float(np.quantile(vals, alfa / 2)), float(np.quantile(vals, 1 - alfa / 2)), vals
