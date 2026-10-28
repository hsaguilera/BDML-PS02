r"""Métricas del protocolo v7, siempre en el dominio objetivo 𝒯 (los 24 dominios distintos de Bogotá).

Estimando (memo §1.2 y §2.1). El riesgo que importa es $R_{\mathcal T}$; la validación cruzada ponderada por importancia
con los pesos exactos $w=\mathbb 1\{d\neq B\}/(1-\pi_B)$ equivale a promediar sólo sobre los hogares de 𝒯 de los
pliegues de validación. Para el F1 (un cociente) el estimador es el **F1 agregado** (se suman TP, FP y FN de todos los
pliegues y se calcula un solo cociente), consistente para el F1 poblacional; los F1 por pliegue sólo se usan para errores
estándar y pruebas pareadas (Forman y Scholz, 2010).

Umbral anidado: el umbral del pliegue k es el argmax del F1 en las predicciones fuera de pliegue (OOF) de 𝒯 de los
otros pliegues de la misma partición (consistencia del *plug-in* en dos pasos: Koyejo et al., 2014; Narasimhan, Vaish y
Agarwal, 2014). Convención: se predice «pobre» si s ≥ t, y t es el punto medio entre el puntaje óptimo y el siguiente
puntaje distinto (más robusto fuera de muestra).

Incertidumbre del F1. Linealización (función de influencia) de F = N/D con N = 2E[t], D = E[2t + f + g]:
ψ_i = (2t_i − F(2t_i + f_i + g_i))/D, EE(F̂) ≈ sd(ψ)/√n; y bootstrap percentil de hogares con las predicciones fijas.
Calibración: ACE (ECE con 20 bins de igual masa; Nixon et al., 2019), Brier y log-loss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import xlogy

ROPE_F1 = 0.002          # región de equivalencia práctica en F1 (memo §2.4)
ROPE_LL = 0.0005         # en log-loss
ROPE_BRIER = 0.0002      # análogo en Brier (la diferencia de Brier es ≈ 0,4 de la de log-loss con p ≈ 0,2)
EPS = 1e-6


# ------------------------------------------------------------------------------ pérdidas propias
def recortar(p, eps=EPS):
    return np.clip(np.asarray(p, float), eps, 1 - eps)


def perdida_obs(y, p, tipo: str = "logloss") -> np.ndarray:
    """Pérdida por observación (para comparaciones pareadas y CVC)."""
    y = np.asarray(y, float)
    if tipo == "brier":
        return (np.clip(np.asarray(p, float), 0, 1) - y) ** 2
    q = recortar(p)
    return -(xlogy(y, q) + xlogy(1 - y, 1 - q))


def logloss(y, p, w=None) -> float:
    return float(np.average(perdida_obs(y, p, "logloss"), weights=w))


def brier(y, p, w=None) -> float:
    return float(np.average(perdida_obs(y, p, "brier"), weights=w))


def perdida(y, p, tipo: str = "logloss") -> float:
    return logloss(y, p) if tipo == "logloss" else brier(y, p)


def rope_de(tipo: str) -> float:
    return ROPE_LL if tipo == "logloss" else ROPE_BRIER


# ------------------------------------------------------------------------------ discriminación
def auc_roc(y, s) -> float:
    """AUC-ROC (Mann–Whitney) con empates."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_pr(y, s) -> float:
    """Área bajo la curva precisión–exhaustividad (precisión media), con empates agrupados."""
    thr, tp, fp, P, _ = curva_f1(y, s)
    if P == 0:
        return float("nan")
    prec = tp / np.maximum(tp + fp, 1e-12); rec = tp / P
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def ace(y, p, bins: int = 20) -> float:
    """ECE adaptativo (bins de igual masa)."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    if len(p) == 0:
        return float("nan")
    o = np.argsort(p, kind="mergesort")
    partes = np.array_split(o, min(bins, len(p)))
    return float(sum(len(b) / len(p) * abs(y[b].mean() - p[b].mean()) for b in partes if len(b)))


def tabla_confiabilidad(y, p, bins: int = 20) -> pd.DataFrame:
    y = np.asarray(y, float); p = np.asarray(p, float)
    o = np.argsort(p, kind="mergesort")
    filas = []
    for b in np.array_split(o, bins):
        if len(b) == 0:
            continue
        m = y[b].mean(); n = len(b)
        filas.append({"p_media": p[b].mean(), "frecuencia": m, "n": n, "ee": np.sqrt(max(m * (1 - m), 1e-12) / n)})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------------------ F1 y umbral
def confusion(y, yhat):
    y = np.asarray(y).astype(int); yh = np.asarray(yhat).astype(int)
    tp = int(((y == 1) & (yh == 1)).sum()); fp = int(((y == 0) & (yh == 1)).sum())
    fn = int(((y == 1) & (yh == 0)).sum()); tn = int(((y == 0) & (yh == 0)).sum())
    return tp, fp, fn, tn


def f1(y, yhat) -> float:
    tp, fp, fn, _ = confusion(y, yhat)
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def mcc(y, yhat) -> float:
    tp, fp, fn, tn = confusion(y, yhat)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def curva_f1(y, s, w=None):
    """(umbrales, TP, FP, P, F1) para todos los umbrales distintos de s (orden descendente; se predice 1 si s ≥ umbral)."""
    y = np.asarray(y).astype(float); s = np.asarray(s, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    o = np.argsort(-s, kind="mergesort")
    ys, ss, ws = y[o], s[o], w[o]
    tp = np.cumsum(ws * ys); fp = np.cumsum(ws * (1 - ys)); P = float((ws * ys).sum())
    ult = np.r_[ss[1:] != ss[:-1], True]
    tp, fp, thr = tp[ult], fp[ult], ss[ult]
    f = 2 * tp / np.maximum(tp + fp + P, 1e-12)
    return thr, tp, fp, P, f


def _punto_medio(thr, i):
    return float(0.5 * (thr[i] + thr[i + 1])) if i + 1 < len(thr) else float(thr[i])


def umbral_argmax(y, s, w=None) -> tuple[float, float]:
    """Umbral F1-óptimo (punto medio hacia el siguiente puntaje distinto) y el F1 alcanzado."""
    thr, tp, fp, P, f = curva_f1(y, s, w)
    i = int(np.argmax(f))
    return _punto_medio(thr, i), float(f[i])


def umbral_bootstrap(y, s, B: int = 500, semilla: int = 20262, lote: int = 25) -> dict:
    """Umbral suavizado (memo §5.5): mediana bootstrap del argmax del F1 con B remuestreos de hogares. Ordena una sola
    vez y evalúa cada remuestreo con sumas acumuladas ponderadas (O(n) por remuestreo)."""
    y = np.asarray(y).astype(float); s = np.asarray(s, float); n = len(y)
    o = np.argsort(-s, kind="mergesort"); ys, ss = y[o], s[o]
    ult = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True]); thr = ss[ult]
    rng = np.random.default_rng(semilla)
    ts, fs = [], []
    for b0 in range(0, B, lote):
        m = min(lote, B - b0)
        W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(m)]).astype(np.float32)[:, o]
        TP = np.cumsum(W * ys, axis=1)[:, ult]; FP = np.cumsum(W * (1 - ys), axis=1)[:, ult]
        P = (W * ys).sum(1, keepdims=True)
        F = 2 * TP / np.maximum(TP + FP + P, 1e-12)
        ii = F.argmax(1)
        ts += [_punto_medio(thr, int(i)) for i in ii]; fs += list(F[np.arange(m), ii])
    ts = np.asarray(ts)
    t0, f0 = umbral_argmax(y, s)
    return {"umbral": float(np.median(ts)), "umbral_argmax": t0, "F1_argmax": f0,
            "umbral_p05": float(np.quantile(ts, .05)), "umbral_p95": float(np.quantile(ts, .95)),
            "umbral_ee": float(ts.std(ddof=1)), "B": int(B)}


def meseta(y, s, tolerancias=(0.0005, 0.001)) -> dict:
    """Intervalo de umbrales en que el F1 OOF está a menos de `tol` de su máximo, y las tasas predichas extremas."""
    thr, tp, fp, P, f = curva_f1(y, s)
    n = len(np.asarray(y)); out = {"F1_max": float(f.max())}
    for tol in tolerancias:
        ok = f >= f.max() - tol
        out[f"meseta_{tol}"] = [float(thr[ok].min()), float(thr[ok].max())]
        out[f"tasas_{tol}"] = [float(((tp + fp)[ok]).min() / n), float(((tp + fp)[ok]).max() / n)]
    return out


def influencia_f1(y, yhat) -> tuple[float, float, np.ndarray]:
    """F̂, EE(F̂) por función de influencia y el vector ψ (para diferencias pareadas: EE = sd(ψ_A − ψ_B)/√n)."""
    y = np.asarray(y).astype(int); yh = np.asarray(yhat).astype(int)
    t = ((y == 1) & (yh == 1)).astype(float); f = ((y == 0) & (yh == 1)).astype(float); g = ((y == 1) & (yh == 0)).astype(float)
    D = (2 * t + f + g).mean()
    F = 2 * t.mean() / D if D > 0 else 0.0
    psi = (2 * t - F * (2 * t + f + g)) / D if D > 0 else np.zeros_like(t)
    return float(F), float(psi.std(ddof=1) / np.sqrt(len(psi))), psi


def bootstrap_f1(y, yhat, B: int = 2000, semilla: int = 20262, w=None) -> dict:
    """IC 95 % percentil del F1 (predicciones fijas; remuestreo de hogares). `w`: pesos (post-estratificación)."""
    y = np.asarray(y).astype(int); yh = np.asarray(yhat).astype(int); n = len(y)
    w = np.ones(n) if w is None else np.asarray(w, float)
    T = np.stack([((y == 1) & (yh == 1)) * w, ((y == 0) & (yh == 1)) * w, ((y == 1) & (yh == 0)) * w], 1)
    rng = np.random.default_rng(semilla); vals = np.empty(B)
    for b in range(B):
        c = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        tp, fp, fn = c @ T
        vals[b] = 2 * tp / max(2 * tp + fp + fn, 1e-12)
    return {"IC95": [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))], "EE_boot": float(vals.std(ddof=1))}


# ------------------------------------------------------------------------------ F1 anidado (una partición)
def f1_anidado(y, s, folds, mT) -> dict:
    """F1 agregado anidado en 𝒯 para una partición: umbral del pliegue k = argmax del F1 en las OOF de 𝒯 de los otros
    pliegues. Devuelve el F1 agregado, los F1 por pliegue, los umbrales y la predicción anidada (−1 fuera de 𝒯)."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float); mT = np.asarray(mT, bool)
    yhat = np.full(len(y), -1, dtype=np.int8); f1k, umb, tasa = [], [], []
    for tr, va in folds:
        trT, vaT = tr[mT[tr]], va[mT[va]]
        t, _ = umbral_argmax(y[trT], s[trT])
        yhat[vaT] = (s[vaT] >= t)
        f1k.append(f1(y[vaT], yhat[vaT])); umb.append(t); tasa.append(float(yhat[vaT].mean()))
    return {"F1": f1(y[mT], yhat[mT]), "F1_pliegues": np.array(f1k), "umbrales": np.array(umb), "yhat": yhat,
            "tasa_pliegues": np.array(tasa)}


def por_pliegue(y, s, folds, mT, tipo: str = "logloss") -> np.ndarray:
    """Pérdida propia en 𝒯 ∩ validación, por pliegue."""
    y = np.asarray(y); s = np.asarray(s, float); mT = np.asarray(mT, bool)
    return np.array([perdida(y[va[mT[va]]], s[va[mT[va]]], tipo) for _, va in folds])


def metricas_particion(y, s, folds, mT, es_prob: bool = True, B: int = 2000, semilla: int = 20262,
                       dominio=None, pesos_dominio: dict | None = None, completo: bool = True) -> dict:
    """Panel completo de métricas en 𝒯 para las OOF `s` de UNA partición (semillas ya promediadas)."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float); mT = np.asarray(mT, bool)
    an = f1_anidado(y, s, folds, mT)
    yT, sT, hT = y[mT], s[mT], an["yhat"][mT].astype(int)
    tp, fp, fn, tn = confusion(yT, hT)
    F, ee_psi, _ = influencia_f1(yT, hT)
    out = {"F1_anidado": an["F1"], "EE_psi": ee_psi, "F1_pliegues": an["F1_pliegues"].tolist(),
           "EE_pliegues": float(an["F1_pliegues"].std(ddof=1) / np.sqrt(len(folds))),
           "umbrales_pliegues": an["umbrales"].tolist(), "precision": tp / max(tp + fp, 1), "exhaustividad": tp / max(tp + fn, 1),
           "MCC": mcc(yT, hT), "tasa_predicha_anidada": float(hT.mean()), "tasa_real": float(yT.mean()),
           "AUC_ROC": auc_roc(yT, sT), "AUC_PR": auc_pr(yT, sT), "n_T": int(mT.sum())}
    sp = recortar(sT) if es_prob else np.clip(sT, EPS, 1 - EPS)
    out.update({"LogLoss": logloss(yT, sp), "Brier": brier(yT, np.clip(sT, 0, 1)), "ACE": ace(yT, np.clip(sT, 0, 1)),
                "LogLoss_pliegues": por_pliegue(y, np.clip(s, EPS, 1 - EPS), folds, mT, "logloss").tolist(),
                "Brier_pliegues": por_pliegue(y, np.clip(s, 0, 1), folds, mT, "brier").tolist()})
    if completo:
        out.update({f"bootstrap_{k}": v for k, v in bootstrap_f1(yT, hT, B=B, semilla=semilla).items()})
        out["IC95_F1"] = out.pop("bootstrap_IC95")
        to, fo = umbral_argmax(yT, sT)
        out.update({"F1_oraculo": fo, "umbral_oraculo": to, "brecha_oraculo": fo - an["F1"],
                    "meseta": meseta(yT, sT), "umbral_vs_F_medio": [to, fo / 2]})
        if dominio is not None and pesos_dominio:
            wd = pd.Series(np.asarray(dominio)[mT]).map(pesos_dominio).fillna(1.0).values
            tpw = (wd * ((yT == 1) & (hT == 1))).sum(); fpw = (wd * ((yT == 0) & (hT == 1))).sum()
            fnw = (wd * ((yT == 1) & (hT == 0))).sum()
            out["F1_postestratificado"] = float(2 * tpw / (2 * tpw + fpw + fnw))
    return out


def metricas_rk(y, oofs: dict, folds_por_r: dict, mT, tipo: str = "logloss") -> dict:
    """Vectores por (partición, pliegue) — J = R·K — de F1 anidado y de la pérdida propia, para las pruebas pareadas."""
    F, L, etiq = [], [], []
    for r, s in oofs.items():
        an = f1_anidado(y, s, folds_por_r[r], mT)
        F += list(an["F1_pliegues"])
        L += list(por_pliegue(y, np.clip(s, EPS, 1 - EPS) if tipo == "logloss" else np.clip(s, 0, 1), folds_por_r[r], mT, tipo))
        etiq += [(r, k) for k in range(len(folds_por_r[r]))]
    return {"F1": np.array(F), "perdida": np.array(L), "etiquetas": etiq,
            "F1_por_particion": {r: f1_anidado(y, s, folds_por_r[r], mT)["F1"] for r, s in oofs.items()}}


# ------------------------------------------------------------------------------ prevalencia y composición del test
def prevalencia_esperada_test(y, dom_train, mT, dom_test) -> dict:
    """Prevalencia del test post-estratificada por la composición de dominios del test: Σ_d (n_d^test/n^test)·ȳ_d^𝒯,
    con EE muestral √(Σ_d a_d² ȳ_d(1−ȳ_d)/n_d^𝒯). También devuelve los pesos de post-estratificación por dominio."""
    y = np.asarray(y).astype(float); mT = np.asarray(mT, bool)
    dt = pd.Series(np.asarray(dom_train)[mT]); yy = pd.Series(y[mT])
    g = yy.groupby(dt.values).agg(["mean", "size"])
    a = pd.Series(np.asarray(dom_test)).value_counts(normalize=True)
    g = g.reindex(a.index)
    pi = float((a * g["mean"]).sum())
    ee = float(np.sqrt((a ** 2 * g["mean"] * (1 - g["mean"]) / g["size"]).sum()))
    pesos = (a / (g["size"] / g["size"].sum())).to_dict()
    return {"prevalencia_test": pi, "EE": ee, "prevalencia_T": float(yy.mean()), "pesos_dominio": pesos,
            "pesos_min": float(min(pesos.values())), "pesos_max": float(max(pesos.values()))}


def f1_por_dominio(y, yhat, dominio, B: int = 300, semilla: int = 1) -> pd.DataFrame:
    y = np.asarray(y).astype(int); yh = np.asarray(yhat).astype(int); d = np.asarray(dominio).astype(str)
    filas = []
    for dd in np.unique(d):
        m = d == dd
        tp, fp, fn, _ = confusion(y[m], yh[m])
        bt = bootstrap_f1(y[m], yh[m], B=B, semilla=semilla)
        filas.append({"Dominio": dd, "hogares": int(m.sum()), "tasa_real": float(y[m].mean()), "tasa_predicha": float(yh[m].mean()),
                      "F1": f1(y[m], yh[m]), "IC95_inf": bt["IC95"][0], "IC95_sup": bt["IC95"][1],
                      "precision": tp / max(tp + fp, 1), "exhaustividad": tp / max(tp + fn, 1)})
    return pd.DataFrame(filas).sort_values("F1").reset_index(drop=True)


# ------------------------------------------------------------------------------ calibración dentro del pliegue (memo §5.3)
def calibracion_cruzada(y, s, folds, mT) -> dict:
    """Platt (logística sobre logit(s)) e isotónica (Zadrozny y Elkan, 2002) ajustadas en las OOF de 𝒯 de los OTROS
    pliegues y aplicadas al pliegue retenido (calibración dentro del pliegue). Devuelve las OOF calibradas y el cambio
    en log-loss, Brier, ACE y F1 anidado; se adopta sólo si mejora el F1 anidado más allá de la ROPE (el F1 depende
    sólo del orden) o la log-loss más allá de la suya."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y).astype(int); s = np.clip(np.asarray(s, float), 1e-6, 1 - 1e-6); mT = np.asarray(mT, bool)
    out = {"original": s.copy(), "platt": s.copy(), "isotonica": s.copy()}
    lg = np.log(s / (1 - s))
    for tr, va in folds:
        trT = tr[mT[tr]]
        pl = LogisticRegression(C=1e6).fit(lg[trT, None], y[trT])
        out["platt"][va] = pl.predict_proba(lg[va, None])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(s[trT], y[trT])
        out["isotonica"][va] = iso.predict(s[va])
    res = {}
    base = None
    for k, v in out.items():
        r = {"LogLoss": logloss(y[mT], v[mT]), "Brier": brier(y[mT], v[mT]), "ACE": ace(y[mT], v[mT]),
             "F1_anidado": f1_anidado(y, v, folds, mT)["F1"]}
        if k == "original":
            base = r
        res[k] = r
    for k in ("platt", "isotonica"):
        res[k].update({f"d{m}": res[k][m] - base[m] for m in ("LogLoss", "Brier", "ACE", "F1_anidado")})
        res[k]["adoptar"] = bool(res[k]["dF1_anidado"] > ROPE_F1 or res[k]["dLogLoss"] < -ROPE_LL)
    return res
