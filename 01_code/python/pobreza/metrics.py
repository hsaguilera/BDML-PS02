"""Métricas de clasificación, focalización y calibración.

La competencia evalúa con F1 sobre la clase `pobre = 1`:

    F1 = 2·P·R / (P + R),  P = TP/(TP+FP) (precisión),  R = TP/(TP+FN) (sensibilidad).

En el lenguaje de focalización de programas sociales (Coady, Grosh y Hoddinott, 2004):
    error de inclusión (filtración)   = FP / (TP+FP) = 1 − precisión
    error de exclusión (subcobertura) = FN / (TP+FN) = 1 − recall
F1 es la media armónica de (1 − inclusión) y (1 − exclusión).

Umbral óptimo para F1 (Lipton, Elkan y Naryanaswamy, 2014): si las probabilidades
están calibradas, el clasificador F1-óptimo predice 1 cuando p(x) ≥ F1*/2, donde F1*
es el F1 máximo alcanzable. Aquí se elige el umbral empíricamente sobre predicciones
fuera de muestra (out-of-fold) y se verifica la regla teórica.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import xlogy


def confusion(y, yhat, w=None):
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    w = np.ones_like(y, dtype=float) if w is None else np.asarray(w, float)
    tp = float(w[(y == 1) & (yhat == 1)].sum()); fp = float(w[(y == 0) & (yhat == 1)].sum())
    fn = float(w[(y == 1) & (yhat == 0)].sum()); tn = float(w[(y == 0) & (yhat == 0)].sum())
    return tp, fp, fn, tn


def f1(y, yhat, w=None) -> float:
    tp, fp, fn, _ = confusion(y, yhat, w)
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def fbeta(y, yhat, beta=1.0, w=None) -> float:
    tp, fp, fn, _ = confusion(y, yhat, w)
    b2 = beta ** 2
    return 0.0 if tp == 0 else (1 + b2) * tp / ((1 + b2) * tp + b2 * fn + fp)


def auc_roc(y, p, w=None) -> float:
    """AUC-ROC (Mann–Whitney) con pesos opcionales; maneja empates."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    o = np.argsort(p, kind="mergesort"); p, y, w = p[o], y[o], w[o]
    # agrupar empates
    _, idx, inv = np.unique(p, return_index=True, return_inverse=True)
    w1 = np.bincount(inv, w * (y == 1)); w0 = np.bincount(inv, w * (y == 0))
    cum0 = np.cumsum(w0) - w0
    num = (w1 * (cum0 + 0.5 * w0)).sum()
    return float(num / (w1.sum() * w0.sum()))


def pr_auc(y, p) -> float:
    """Área bajo la curva precisión–recall (average precision)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    o = np.argsort(-p, kind="mergesort"); y = y[o]
    tp = np.cumsum(y); k = np.arange(1, len(y) + 1)
    prec = tp / k
    return float((prec * y).sum() / max(y.sum(), 1))


def brier(y, p, w=None) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.average((p - y) ** 2, weights=w))


def log_loss(y, p, w=None, eps=1e-12) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.average(xlogy(y, p) + xlogy(1 - y, 1 - p), weights=w))


def mcc(y, yhat) -> float:
    tp, fp, fn, tn = confusion(y, yhat)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def ece(y, p, bins=20) -> float:
    """Error de calibración esperado (bins por cuantiles)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    q = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    e = 0.0
    for k in range(len(q) - 1):
        m = b == k
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def curva_f1(y, s, n=0) -> pd.DataFrame:
    """F1, precisión y recall para TODOS los umbrales posibles de un puntaje s
    (alto = más probable pobre). Complejidad O(n log n)."""
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    o = np.argsort(-s, kind="mergesort"); ys, ss = y[o], s[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = ys.sum()
    last = np.r_[ss[1:] != ss[:-1], True]  # último índice de cada valor
    tp, fp, thr = tp[last], fp[last], ss[last]
    prec = tp / (tp + fp); rec = tp / P
    f = 2 * tp / (tp + fp + P)
    df = pd.DataFrame({"umbral": thr, "precision": prec, "recall": rec, "f1": f,
                       "tasa_predicha": (tp + fp) / len(y)})
    if n and len(df) > n:
        df = df.iloc[np.linspace(0, len(df) - 1, n).astype(int)]
    return df


def umbral_optimo_f1(y, s) -> tuple[float, float]:
    """Umbral que maximiza F1 (predice 1 si s ≥ umbral) y el F1 alcanzado."""
    c = curva_f1(y, s)
    i = int(c.f1.values.argmax())
    # punto medio entre el umbral óptimo y el siguiente valor (más robusto fuera de muestra)
    t = c.umbral.values[i]
    if i + 1 < len(c):
        t = 0.5 * (t + c.umbral.values[i + 1])
    return float(t), float(c.f1.values[i])


def resumen_clasificacion(y, p, umbral, w=None, nombre="") -> dict:
    """Panel completo de métricas para un vector de probabilidades y un umbral."""
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    yhat = (p >= umbral).astype(int)
    tp, fp, fn, tn = confusion(y, yhat)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    d = {
        "modelo": nombre, "umbral": umbral, "F1": f1(y, yhat), "precision": prec, "recall": rec,
        "especificidad": spec, "exactitud": (tp + tn) / len(y), "exactitud_balanceada": 0.5 * (rec + spec),
        "F2": fbeta(y, yhat, 2), "F0.5": fbeta(y, yhat, .5), "MCC": mcc(y, yhat),
        "AUC_ROC": auc_roc(y, p), "AUC_PR": pr_auc(y, p), "Brier": brier(y, p), "LogLoss": log_loss(y, p),
        "ECE": ece(y, p), "error_inclusion": 1 - prec, "error_exclusion": 1 - rec,
        "tasa_real": y.mean(), "tasa_predicha": yhat.mean(),
    }
    if w is not None:
        d["F1_ponderado(Fex_c)"] = f1(y, yhat, w)
        d["tasa_real_pond"] = float(np.average(y, weights=w)); d["tasa_pred_pond"] = float(np.average(yhat, weights=w))
    return d


def errores_por_brecha(y, yhat, brecha) -> pd.DataFrame:
    """Error de exclusión según la profundidad de la pobreza (brecha = 1 − ingreso/Lp).
    Excluir a un indigente es socialmente más costoso que excluir a alguien cerca de la línea
    (familia FGT de Foster, Greer y Thorbecke, 1984)."""
    y, yhat, g = np.asarray(y).astype(int), np.asarray(yhat).astype(int), np.asarray(brecha, float)
    m = y == 1
    cortes = pd.cut(g[m], [-np.inf, .1, .25, .5, .75, np.inf],
                    labels=["0–10%", "10–25%", "25–50%", "50–75%", ">75%"])
    t = pd.DataFrame({"tramo_brecha": cortes, "excluido": 1 - yhat[m]})
    return t.groupby("tramo_brecha", observed=True).agg(hogares=("excluido", "size"), tasa_exclusion=("excluido", "mean"))


def metricas_por_grupo(df: pd.DataFrame, grupo: str, y="y", p="p", umbral=0.5) -> pd.DataFrame:
    filas = []
    for g, d in df.groupby(grupo, observed=True):
        if d[y].nunique() < 2:
            continue
        r = resumen_clasificacion(d[y], d[p], umbral)
        filas.append({grupo: g, "n": len(d), "tasa_real": r["tasa_real"], "tasa_predicha": r["tasa_predicha"],
                      "F1": r["F1"], "precision": r["precision"], "recall": r["recall"], "AUC_ROC": r["AUC_ROC"]})
    return pd.DataFrame(filas).sort_values("F1")
