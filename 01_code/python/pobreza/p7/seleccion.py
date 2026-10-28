r"""Selección de variables por familia, dentro del pliegue, con la pregunta de la encuesta como unidad (memo §3, §11.1
y Paso 6 del protocolo).

La selección es parte del procedimiento (Ambroise y McLachlan, 2002; Varma y Simon, 2006): el ranking se calcula con
el entrenamiento de cada pliegue externo y el conjunto resultante se evalúa en el pliegue retenido. Con esas
predicciones fuera de pliegue se construye la **curva de parsimonia** (F1 anidado y pérdida propia frente al número de
variables y al costo $c(S)$ en preguntas) y se elige el **conjunto parsimonioso**: el de menor $c(S)$ cuyo F1 no es
peor que el del mejor en más de la ROPE (0,002) y cuya pérdida no es peor en más de su ROPE (0,0005 en log-loss). El
conjunto final se recalcula con todo el entrenamiento aplicando la misma receta.

Métodos por familia (tabla §11.1):
* LPM y Logit: camino lasso (MCO con ℓ1 / logit-ℓ1) → modelos con a lo sumo c preguntas → reajuste sin penalizar
  (post-lasso; *relaxed lasso* con γ = 0); además el soporte en λ_1se (validación cruzada interna) y en el BIC.
* Elastic Net: integrada; el camino con el α elegido da la curva por preguntas; λ_1se para la versión parsimoniosa y
  *complementary pairs stability selection* (Shah y Samworth, 2013) para la estabilidad.
* CART: poda de costo-complejidad (Breiman et al., 1984) con validación cruzada y regla 1-EE (en `busqueda`).
* Random Forest: eliminación recursiva por bloques de pregunta (20 % por paso) con permutación **agrupada** sobre las
  observaciones fuera de bolsa; confirmación con LOCO por bloques (nunca la importancia por impureza: Strobl et al., 2007).
* Boosting: SHAP sumado por pregunta (TreeSHAP exacto; Lundberg et al., 2020) → eliminación por bloques re-ordenando en
  cada paso → LOCO.
* Naive Bayes: mRMR (Peng, Long y Ding, 2005) para ordenar + envoltorio hacia adelante (Bayes selectivo; Langley y
  Sage, 1994) con validación cruzada interna de la log-loss sobre variables discretizadas en el pliegue.
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np
import pandas as pd

from . import comparar as CP
from . import glm as GL
from . import metricas as MT
from .cvmotor import division_es

GRILLA_C = (3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80)


# ------------------------------------------------------------------------------ curva de parsimonia
def tabla_curva(datos, puntos: dict, r: int = 1, tipo: str = "logloss", es_prob: bool = True) -> pd.DataFrame:
    """puntos: {nombre: {"oof": array n, "cols_pliegues": [lista por pliegue], "final": lista o None, "orden": float}}.
    Métricas en 𝒯 y diferencias pareadas frente al mejor (t corregido por pliegue y EE de influencia del ΔF1)."""
    y, mT, folds = datos.y, datos.mT, datos.folds(r)
    filas, rk = [], {}
    for nom, p in puntos.items():
        s = np.asarray(p["oof"], float)
        if np.isnan(s[np.concatenate([va for _, va in folds])][mT[np.concatenate([va for _, va in folds])]]).any():
            continue
        m = MT.metricas_particion(y, s, folds, mT, es_prob=es_prob, completo=False)
        cp = p.get("cols_pliegues") or [p.get("final")] * len(folds)
        nv = float(np.mean([len(c) for c in cp])); nq = float(np.mean([datos.MP.costo(c) for c in cp]))
        fin = p.get("final")
        filas.append({"punto": nom, "orden": p.get("orden", 0), "n_vars": nv, "n_preguntas": nq,
                      "n_vars_final": len(fin) if fin is not None else np.nan,
                      "n_preguntas_final": datos.MP.costo(fin) if fin is not None else np.nan,
                      "F1_anidado": m["F1_anidado"], "EE_psi": m["EE_psi"], "LogLoss": m["LogLoss"], "Brier": m["Brier"],
                      "AUC_ROC": m["AUC_ROC"], "AUC_PR": m["AUC_PR"], "perdida": m["LogLoss"] if tipo == "logloss" else m["Brier"]})
        rk[nom] = {"F1": np.array(m["F1_pliegues"]), "perdida": np.array(m["LogLoss_pliegues"] if tipo == "logloss" else m["Brier_pliegues"]),
                   "etiquetas": [(r, k) for k in range(len(folds))], "yhat": MT.f1_anidado(y, s, folds, mT)["yhat"]}
    t = pd.DataFrame(filas)
    if t.empty:
        return t
    mejor_F = t.F1_anidado.max(); mejor_L = t.perdida.min()
    ref = t.loc[t.perdida.idxmin(), "punto"]
    yT = y[mT]
    for i, row in t.iterrows():
        c = CP.comparar_rk(rk[row.punto], rk[ref], tipo, len(folds), row.punto, ref)
        _, _, pa = MT.influencia_f1(yT, rk[row.punto]["yhat"][mT]); _, _, pb = MT.influencia_f1(yT, rk[ref]["yhat"][mT])
        t.loc[i, "dF1_vs_mejor"] = row.F1_anidado - t.loc[t.punto == ref, "F1_anidado"].iloc[0]
        t.loc[i, "dF1_EE_psi"] = float((pa - pb).std(ddof=1) / np.sqrt(len(pa)))
        t.loc[i, "dF1_EE_NB"] = c["dF1_EE"]
        t.loc[i, "dperdida_vs_mejor"] = row.perdida - mejor_L
        t.loc[i, "dperdida_EE_NB"] = c[f"d{tipo}_EE"]
    t["en_ROPE"] = (t.F1_anidado >= mejor_F - MT.ROPE_F1) & (t.perdida <= mejor_L + MT.rope_de(tipo))
    t["mejor_F1"] = t.F1_anidado == mejor_F
    t["mejor_perdida"] = t.punto == ref
    adm = t[t.en_ROPE].sort_values(["n_preguntas", "n_vars", "F1_anidado"], ascending=[True, True, False])
    t["parsimonioso"] = t.punto == (adm.iloc[0].punto if len(adm) else ref)
    return t.sort_values("orden").reset_index(drop=True)


def cvc_curva(datos, puntos: dict, r=1, tipo="logloss", alfa=0.10, B=300) -> dict:
    """Conjunto de confianza CVC (Lei, 2020) al 90 % sobre los puntos de la curva (pérdidas OOF por hogar de 𝒯)."""
    nombres = list(puntos)
    mT = datos.mT
    L = np.column_stack([MT.perdida_obs(datos.y[mT], np.asarray(puntos[n]["oof"])[mT], tipo) for n in nombres])
    ok = np.all(np.isfinite(L), 0)
    nombres = [n for n, o in zip(nombres, ok) if o]
    res = CP.cvc(L[:, ok], alfa=alfa, B=B)
    return {"conjunto": [nombres[i] for i in res["conjunto"]], "p_valores": dict(zip(nombres, res["p_valores"])), "alfa": alfa}


def jaccard(conjuntos) -> float:
    cs = [set(c) for c in conjuntos]
    pares = [(a, b) for a, b in itertools.combinations(cs, 2)]
    if not pares:
        return float("nan")
    return float(np.mean([len(a & b) / max(len(a | b), 1) for a, b in pares]))


def estabilidad(datos, conjuntos_pliegues, B: int = 500, semilla: int = 1) -> dict:
    """Jaccard medio entre los conjuntos de los pliegues (variables y preguntas), con EE bootstrap de los conjuntos, y
    frecuencia de selección por pregunta."""
    qs = [datos.MP.preguntas_de(c) for c in conjuntos_pliegues]
    rng = np.random.default_rng(semilla); m = len(conjuntos_pliegues)
    bj = []
    for _ in range(B):
        idx = rng.integers(0, m, m)
        if len(set(idx)) > 1:
            bj.append(jaccard([qs[i] for i in sorted(set(idx))]))
    frec = pd.Series([q for c in qs for q in c]).value_counts() / m
    return {"jaccard_variables": jaccard(conjuntos_pliegues), "jaccard_preguntas": jaccard(qs),
            "jaccard_preguntas_EE_boot": float(np.std(bj, ddof=1)) if len(bj) > 2 else float("nan"),
            "frecuencia_preguntas": frec.to_dict(), "n_conjuntos": m}


# ------------------------------------------------------------------------------ LPM y Logit: lasso → post-lasso
def _prep_pliegue(motor, datos, tr, va, cols, splines=False, lam_B=1.0, semilla=0):
    tr_fit = tr if lam_B > 0 else tr[datos.mT[tr]]
    w = None if lam_B in (0, 1) else np.where(datos.bog[tr_fit], lam_B, 1.0)
    Xtr = datos.matriz(cols, tr_fit, train_idx=tr)
    prep = GL.PrepLineal(motor.splines if splines else (), semilla=semilla).fit(Xtr)
    Ztr = prep.transform(Xtr); del Xtr
    Zva = prep.transform(datos.matriz(cols, va, train_idx=tr)) if va is not None else None
    return prep, Ztr, Zva, tr_fit, w


def _lasso_logit_cv_interno(Z, y, w, lambdas, k=3, n_sub=40000, semilla=0, max_activas=None):
    """λ_min y λ_1se del logit-ℓ1 por validación cruzada interna (submuestra estratificada del entrenamiento)."""
    rng = np.random.default_rng(semilla); n = len(y)
    sub = np.sort(rng.choice(n, min(n_sub, n), replace=False))
    Zs, ys = Z[sub], np.asarray(y)[sub]; ws = None if w is None else np.asarray(w)[sub]
    bloque = rng.permutation(len(sub)) % k
    E = np.full((k, len(lambdas)), np.nan)
    for b in range(k):
        trb, vab = bloque != b, bloque == b
        ru = GL.ruta_enet_logit(Zs[trb], ys[trb], None if ws is None else ws[trb], lambdas, 1.0, max_activas=max_activas)
        P = GL.predecir_ruta(Zs[vab], ru)
        L = ru["coef"].shape[1]
        E[b, :L] = [MT.logloss(ys[vab], P[:, l]) for l in range(L)]
    m = np.nanmean(E, 0); ee = np.nanstd(E, 0, ddof=1) / np.sqrt(k)
    i_min = int(np.nanargmin(m)); cand = np.flatnonzero(m <= m[i_min] + ee[i_min])
    return i_min, int(cand.min()), m


def seleccion_lineal(motor, datos, cols, tipo="lpm", r=1, grilla_c=GRILLA_C, splines=False, lam_B=1.0, n_lambda=60,
                     razon=1e-4, max_activas=None, verbose=True) -> dict:
    """Curva de parsimonia del LPM o del Logit: camino lasso en el entrenamiento de cada pliegue, modelos con a lo sumo
    c preguntas reajustados sin penalizar, soporte en λ_1se y en el BIC, y el modelo con todas las variables."""
    y = datos.y; n = len(y)
    folds = datos.folds(r)
    nombres = [f"c ≤ {c}" for c in grilla_c] + ["λ_1se", "BIC", "todas"]
    oof = {nm: np.full(n, np.nan) for nm in nombres}
    cols_pl = {nm: [] for nm in nombres}
    rutas = []
    t0 = time.time()
    for k, (tr, va) in enumerate(list(folds) + [(np.arange(n), None)]):
        final = va is None
        prep, Ztr, Zva, tr_fit, w = _prep_pliegue(motor, datos, tr, va, cols, splines, lam_B, semilla=k)
        ytr = y[tr_fit]; origen = np.array(prep.origen)
        if tipo == "lpm":
            est = GL.EstadisticosMCO.de(Ztr, ytr, w)
            lams = GL.grilla_lambda(GL.lambda_max_mco(est), n_lambda, razon)
            B, b0 = GL.ruta_lasso_mco(est, lams, max_activas=max_activas)
            lams = lams[:B.shape[1]]
            _, _, i_min, i_1se = GL.cv_lasso_mco(Ztr, ytr, w, lams, 5, semilla=k, max_activas=max_activas)
            i_min, i_1se = min(i_min, B.shape[1] - 1), min(i_1se, B.shape[1] - 1)
            nll = np.array([est.sse(b0[l], B[:, l]) / est.sw for l in range(len(lams))])
            df = (B != 0).sum(0)
            bic = len(ytr) * np.log(np.maximum(nll, 1e-12)) + df * np.log(len(ytr))
        else:
            lams = GL.grilla_lambda(GL.lambda_max_logit(Ztr, ytr, w, 1.0), n_lambda, razon)
            ru = GL.ruta_enet_logit(Ztr, ytr, w, lams, 1.0, max_activas=max_activas)
            B, b0, lams = ru["coef"], ru["b0"], ru["lambdas"]
            i_min, i_1se, _ = _lasso_logit_cv_interno(Ztr, ytr, w, lams, semilla=k, max_activas=max_activas)
            i_min, i_1se = min(i_min, B.shape[1] - 1), min(i_1se, B.shape[1] - 1)
            bic = 2 * len(ytr) * ru["nll"] + ru["df"] * np.log(len(ytr))
        i_bic = int(np.argmin(bic))
        rutas.append({"pliegue": k if not final else "todo", "lambdas": lams, "df": (B != 0).sum(0),
                      "costo": [datos.MP.costo(sorted(set(origen[B[:, l] != 0]))) for l in range(B.shape[1])],
                      "i_1se": i_1se, "i_min": i_min, "i_bic": i_bic})
        mods = GL.modelos_por_costo(B, origen, datos.MP, grilla_c)
        soportes = {f"c ≤ {c}": v[1] for c, v in mods.items()}
        soportes["λ_1se"] = np.flatnonzero(B[:, i_1se] != 0)
        soportes["BIC"] = np.flatnonzero(B[:, i_bic] != 0)
        soportes["todas"] = np.arange(Ztr.shape[1])
        for nm, act in soportes.items():
            fuentes = sorted(set(origen[act]))
            cols_pl[nm].append(fuentes)
            if final:
                continue
            if len(act) == 0:
                continue
            if tipo == "lpm":
                b0r, br = GL.mco(est, act)
                oof[nm][va] = np.clip(b0r + GL.zbeta(Zva, br), 1e-4, 1 - 1e-4)
            else:
                b0r, br, cc = GL.logit_newton(Ztr, ytr, w, act)
                oof[nm][va] = GL._sig(b0r + GL.zbeta(Zva, br, cc))
        if verbose:
            print(f"  {motor.corto} selección, pliegue {rutas[-1]['pliegue']}: camino de {B.shape[1]} λ, λ_1se con "
                  f"{len(soportes['λ_1se'])} columnas, BIC con {len(soportes['BIC'])}; {time.time() - t0:.0f} s", flush=True)
        del Ztr, Zva
    puntos = {}
    for i, nm in enumerate(nombres):
        if len(cols_pl[nm]) == len(folds) + 1 and not np.isnan(oof[nm][np.concatenate([va for _, va in folds])]).all():
            puntos[nm] = {"oof": oof[nm], "cols_pliegues": cols_pl[nm][:-1], "final": cols_pl[nm][-1], "orden": i}
    return {"puntos": puntos, "rutas": rutas, "tiempo_s": time.time() - t0}


# ------------------------------------------------------------------------------ Elastic Net: camino por α y CPSS
def caminos_en(motor, datos, cols, alfas, r=1, n_lambda=60, razon=1e-4, lam_B=1.0, splines=False, verbose=True,
               cache=None, max_activas=None) -> dict:
    """Para cada α, el camino del logit con red elástica en cada pliegue con una grilla COMÚN de λ (la de todo el
    entrenamiento, como glmnet) y las OOF para todos los λ (n × L). Es la búsqueda en rejilla α × λ del memo §11.2."""
    y = datos.y; n = len(y); folds = datos.folds(r)
    out = {}
    prep0, Z0, _, tr0, w0 = _prep_pliegue(motor, datos, np.arange(n), None, cols, splines, lam_B)
    t0 = time.time()
    for a in alfas:
        lmax = GL.lambda_max_logit(Z0, y[tr0], w0, max(a, 1e-3))
        lams = GL.grilla_lambda(lmax, n_lambda if a > 0 else max(10, n_lambda // 3), razon if a > 0 else razon * 10)
        P = np.full((n, len(lams)), np.nan); dfs = []; cols_l = []
        for k, (tr, va) in enumerate(folds):
            prep, Ztr, Zva, tr_fit, w = _prep_pliegue(motor, datos, tr, va, cols, splines, lam_B, semilla=k)
            ru = GL.ruta_enet_logit(Ztr, y[tr_fit], w, lams, a, max_activas=max_activas if a > 0 else None)
            P[va, :ru["coef"].shape[1]] = GL.predecir_ruta(Zva, ru)
            dfs.append(ru["df"]); org = np.array(prep.origen)
            cols_l.append([sorted(set(org[ru["coef"][:, l] != 0])) for l in range(ru["coef"].shape[1])])
            del Ztr, Zva
        out[a] = {"lambdas": lams, "oof": P, "df": dfs, "cols": cols_l}
        if verbose:
            print(f"  EN α = {a}: {len(lams)} λ × {len(folds)} pliegues ({time.time() - t0:.0f} s)", flush=True)
    del Z0
    return out


def tabla_en(datos, caminos: dict, r=1) -> pd.DataFrame:
    y, mT, folds = datos.y, datos.mT, datos.folds(r)
    filas = []
    for a, c in caminos.items():
        for l, lam in enumerate(c["lambdas"]):
            s = c["oof"][:, l]
            if np.isnan(s[mT]).any():
                continue
            Lk = MT.por_pliegue(y, s, folds, mT, "logloss")
            filas.append({"l1_ratio": a, "lam": lam, "indice": l, "LogLoss": MT.logloss(y[mT], s[mT]),
                          "LogLoss_EE": float(Lk.std(ddof=1) / np.sqrt(len(Lk))),
                          "LogLoss_pliegues": Lk.tolist(),
                          "n_vars_medio": float(np.mean([len(cl[l]) for cl in c["cols"]])),
                          "n_preguntas_medio": float(np.mean([datos.MP.costo(cl[l]) for cl in c["cols"]]))})
    t = pd.DataFrame(filas)
    # λ_1se por α (el mayor λ con log-loss ≤ mínimo + 1 EE)
    t["lambda_1se"] = False
    for a, g in t.groupby("l1_ratio"):
        i = g.LogLoss.idxmin(); ok = g[g.LogLoss <= g.loc[i, "LogLoss"] + g.loc[i, "LogLoss_EE"]]
        t.loc[ok.lam.idxmax(), "lambda_1se"] = True
    return t


def cpss(motor, datos, cols, params, B: int = 50, tau: float = 0.75, lam_B: float = 1.0, semilla: int = 20262,
         splines=False, verbose=True) -> dict:
    """*Complementary pairs stability selection* (Shah y Samworth, 2013): B pares de mitades disjuntas de 𝒯 (∪ ℬ con
    λ_B), el selector es la red elástica con (α, λ) fijos; Π̂_j = frecuencia de selección. Cota sin supuestos:
    E|Ŝ_τ ∩ L_θ| ≤ θ/(2τ−1)·E|Ŝ_{n/2} ∩ L_θ|, y con θ = q/p, q²/((2τ−1)p) (Meinshausen y Bühlmann, 2010)."""
    n = len(datos.y)
    idx = np.arange(n) if lam_B > 0 else np.flatnonzero(datos.mT)
    X = datos.matriz(cols, idx, train_idx=idx)
    prep = GL.PrepLineal(motor.splines if splines else ()).fit(X)
    Z = prep.transform(X); y = datos.y[idx]; del X
    w = None if lam_B in (0, 1) else np.where(datos.bog[idx], lam_B, 1.0)
    org = np.array(prep.origen); fuentes = sorted(set(org)); pos = {f: i for i, f in enumerate(fuentes)}
    cuenta = np.zeros(len(fuentes)); qs = []
    rng = np.random.default_rng(semilla); t0 = time.time()
    a, lam = float(params.get("l1_ratio", 1.0)), float(params["lam"])
    selector = params.get("selector", "en")
    for b in range(B):
        perm = rng.permutation(len(idx)); h = len(idx) // 2
        for mitad in (perm[:h], perm[h:2 * h]):
            mitad = np.sort(mitad)
            ww = None if w is None else w[mitad]
            if selector == "lpm":
                est = GL.EstadisticosMCO.de(Z[mitad], y[mitad], ww)
                lmax = GL.lambda_max_mco(est)
                lams = np.unique(np.r_[GL.grilla_lambda(lmax, 15, max(lam / lmax, 1e-6)), lam])[::-1] if lam < lmax else np.array([lam])
                Bc, _ = GL.ruta_lasso_mco(est, lams)
                sel = sorted(set(org[Bc[:, -1] != 0]))
            else:
                lmax = GL.lambda_max_logit(Z[mitad], y[mitad], ww, max(a, 1e-3))
                lams = np.unique(np.r_[GL.grilla_lambda(lmax, 15, max(lam / lmax, 1e-6)), lam])[::-1] if lam < lmax else np.array([lam])
                ru = GL.ruta_enet_logit(Z[mitad], y[mitad], ww, lams, a)
                sel = sorted(set(org[ru["coef"][:, -1] != 0]))
            qs.append(len(sel))
            for f in sel:
                cuenta[pos[f]] += 1
        if verbose and (b + 1) % max(1, B // 5) == 0:
            print(f"  CPSS {b + 1}/{B} pares ({time.time() - t0:.0f} s)", flush=True)
    frec = pd.Series(cuenta / (2 * B), index=fuentes).sort_values(ascending=False)
    q = float(np.mean(qs)); p = len(fuentes)
    estables = list(frec.index[frec >= tau])
    return {"frecuencias": frec, "estables": estables, "tau": tau, "q_medio": q, "p": p, "B": B,
            "cota_EV": q * q / ((2 * tau - 1) * p), "preguntas_estables": datos.MP.preguntas_de(estables),
            "tiempo_s": time.time() - t0}


# ------------------------------------------------------------------------------ RFE por bloques de pregunta
def importancia_shap_preguntas(motor, aj, Xs, MP) -> pd.Series:
    phi = motor.shap(aj, Xs)
    Phi, qs = MP.matriz_grupo(phi, list(Xs.columns))
    return pd.Series(np.abs(Phi).mean(0), index=qs)


def importancia_perm_oob(aj, A, y, grupos_idx: dict, n_eval: int = 10000, semilla: int = 0) -> pd.Series:
    """Permutación AGRUPADA sobre las observaciones fuera de bolsa del bosque (A: matriz ya transformada con la que se
    ajustó el bosque): para cada grupo de columnas (las variables que dependen de una pregunta) se permutan juntas las
    filas y se mide el aumento de la log-loss OOB."""
    rf = aj.modelo
    rng = np.random.default_rng(semilla); n = len(y)
    ev = np.sort(rng.choice(n, min(n_eval, n), replace=False))
    Ae = np.ascontiguousarray(A[ev], dtype=np.float32); ye = np.asarray(y)[ev]
    pos = np.full(n, -1); pos[ev] = np.arange(len(ev))
    oobm = []
    for est_idx in rf.estimators_samples_:
        m = np.ones(len(ev), bool)
        ii = pos[est_idx]; m[ii[ii >= 0]] = False
        oobm.append(np.flatnonzero(m))
    def oob_prob(M):
        s = np.zeros(len(ev)); c = np.zeros(len(ev))
        for t, idx in zip(rf.estimators_, oobm):
            if len(idx):
                s[idx] += t.predict_proba(M[idx], check_input=False)[:, 1]; c[idx] += 1
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)
    base = oob_prob(Ae); ok = np.isfinite(base)
    l0 = MT.logloss(ye[ok], base[ok])
    out = {}
    perm = rng.permutation(len(ev))
    for g, cols in grupos_idx.items():
        M = Ae.copy(); M[:, cols] = Ae[perm][:, cols]
        p = oob_prob(M)
        out[g] = MT.logloss(ye[ok], p[ok]) - l0
    return pd.Series(out)


def rfe_preguntas(motor, datos, cols, params, r=1, frac=0.2, q_min=3, semilla=1, lam_B=1.0, n_shap=10000,
                  n_eval_perm=10000, verbose=True, cache=None) -> dict:
    """Eliminación recursiva por bloques de pregunta (RF: permutación agrupada OOB; boosting: SHAP agrupado). En cada
    pliegue (y en todo el entrenamiento, para el conjunto final) se ajusta, se predice el pliegue retenido, se ordenan
    las preguntas y se elimina la fracción `frac` menos importante (los grupos de costo cero nunca se eliminan)."""
    y = datos.y; n = len(y); folds = datos.folds(r)
    MP = datos.MP
    Q0 = [q for q in MP.unidades(cols)]
    pasos_tam = [len(Q0)]
    while pasos_tam[-1] > q_min:
        pasos_tam.append(max(q_min, pasos_tam[-1] - max(1, int(math.ceil(frac * pasos_tam[-1])))))
    oof = {t: np.full(n, np.nan) for t in pasos_tam}
    sets = {t: [] for t in pasos_tam}
    ranking_final = []
    t0 = time.time()
    for k, (tr, va) in enumerate(list(folds) + [(np.arange(n), None)]):
        final = va is None
        Q = list(Q0)
        tr_fit = tr if lam_B > 0 else tr[datos.mT[tr]]
        w_fit = None if lam_B in (0, 1) else np.where(datos.bog[tr_fit], lam_B, 1.0)
        for t in pasos_tam:
            if len(Q) > t:
                Q = Q[:t]
            ck = MP.compatibles(Q, cols)
            sets[t].append(ck)
            Xtr = datos.matriz(ck, tr_fit, train_idx=tr)
            if motor.parada_temprana:
                fit_m, es_m = division_es(y[tr_fit], datos.mT[tr_fit], 7000 + 10 * k + semilla)
                aj = motor.ajustar(Xtr.iloc[fit_m], y[tr_fit][fit_m], None if w_fit is None else w_fit[fit_m], params, semilla,
                                   Xes=Xtr.iloc[es_m], yes=y[tr_fit][es_m])
            elif motor.oob:
                aj = motor.ajustar(Xtr, y[tr_fit], w_fit, params, semilla, oob=False)
            else:
                aj = motor.ajustar(Xtr, y[tr_fit], w_fit, params, semilla)
            if not final:
                oof[t][va] = motor.predecir(aj, datos.matriz(ck, va, train_idx=tr))
            # importancia por pregunta (sólo hogares de 𝒯 del entrenamiento)
            filasT = np.flatnonzero(datos.mT[tr_fit])
            rng = np.random.default_rng(100 + k)
            if motor.oob:
                A = aj.prep.transform(Xtr)
                gi = {q: [j for j, c in enumerate(aj.prep.cols) if q in MP.mapa.get(c, [f"var:{c}"])] for q in Q}
                imp = importancia_perm_oob(aj, A, y[tr_fit], {q: v for q, v in gi.items() if v}, n_eval_perm, semilla=k)
            else:
                sub = rng.choice(filasT, min(n_shap, len(filasT)), replace=False)
                imp = importancia_shap_preguntas(motor, aj, Xtr.iloc[sub], MP)
            imp = imp.reindex(Q).fillna(0.0)
            Q = list(imp.sort_values(ascending=False).index)
            if final:
                ranking_final.append({"tam": t, "ranking": imp.sort_values(ascending=False).to_dict()})
            del Xtr
        if verbose:
            print(f"  RFE {motor.corto} pliegue {'todo' if final else k}: {len(pasos_tam)} pasos ({time.time() - t0:.0f} s)", flush=True)
    puntos = {}
    for i, t in enumerate(pasos_tam):
        puntos[f"{t} preguntas"] = {"oof": oof[t], "cols_pliegues": sets[t][:-1], "final": sets[t][-1], "orden": i}
    return {"puntos": puntos, "pasos": pasos_tam, "ranking_final": ranking_final, "tiempo_s": time.time() - t0}


# ------------------------------------------------------------------------------ Naive Bayes: mRMR + hacia adelante
def _mi(a, b, ka, kb) -> float:
    j = np.bincount(a.astype(np.int64) * kb + b, minlength=ka * kb).reshape(ka, kb).astype(float)
    j /= j.sum()
    pa = j.sum(1, keepdims=True); pb = j.sum(0, keepdims=True)
    with np.errstate(all="ignore"):
        t = j * np.log(j / (pa * pb))
    return float(np.nansum(t))


def mrmr(A, y, K, M: int, n_sub: int = 30000, semilla: int = 0) -> list:
    """Orden mRMR (diferencia): J(X_j) = I(X_j;Y) − (1/|S|) Σ_{s∈S} I(X_j;X_s)."""
    rng = np.random.default_rng(semilla)
    sub = rng.choice(len(y), min(n_sub, len(y)), replace=False)
    A = A[sub].astype(np.int64); y = np.asarray(y)[sub].astype(np.int64)
    p = A.shape[1]
    rel = np.array([_mi(A[:, j], y, K[j], 2) for j in range(p)])
    S = [int(np.argmax(rel))]; red = np.zeros(p)
    while len(S) < min(M, p):
        s = S[-1]
        red += np.array([_mi(A[:, j], A[:, s], K[j], K[s]) if j not in S else 0.0 for j in range(p)])
        J = rel - red / len(S); J[S] = -np.inf
        S.append(int(np.argmax(J)))
    return S


def seleccion_nb(motor, datos, cols, params, r=1, M=60, kmax=25, k_interno=3, lam_B=1.0, verbose=True,
                 criterios=("exactitud", "costo")) -> dict:
    """NB selectivo: en cada pliegue, discretización + mRMR (top M) y búsqueda hacia adelante con validación cruzada
    interna de la log-loss (la contribución de cada variable es aditiva en el logit del NB, así que cada paso del
    envoltorio es una suma de vectores precomputados). Dos caminos: «exactitud» (la variable que más baja la log-loss) y
    «costo» (la que más la baja por unidad de costo marginal en preguntas, memo §3.2); ambos entran a la curva de
    parsimonia y la regla ROPE elige entre todos sus puntos."""
    from .motores import PrepNB
    y = datos.y; n = len(y); folds = datos.folds(r)
    a = float(params.get("alpha", 1.0))
    claves = [(cr, k_) for cr in criterios for k_ in range(1, kmax + 1)]
    oof = {c: np.full(n, np.nan) for c in claves}
    sets = {c: [] for c in claves}
    t0 = time.time()
    for k, (tr, va) in enumerate(list(folds) + [(np.arange(n), None)]):
        final = va is None
        tr_fit = tr if lam_B > 0 else tr[datos.mT[tr]]
        Xtr = datos.matriz(cols, tr_fit, train_idx=tr)
        ytr = y[tr_fit]
        prep = PrepNB(params.get("n_bins", 20), params.get("estrategia", "cuantiles"), semilla=k).fit(Xtr, ytr)
        A = prep.transform(Xtr); K = prep.n_categorias()
        cand = mrmr(A, ytr, K, M, semilla=k)
        rng = np.random.default_rng(k); bl = rng.permutation(len(ytr)) % k_interno
        C, P0, Y = [], [], []
        for b in range(k_interno):
            itr, iva = bl != b, bl == b
            w1, w0 = (ytr[itr] == 1).astype(float), (ytr[itr] == 0).astype(float)
            cols_c = []
            for j in cand:
                c1 = np.bincount(A[itr, j], w1, minlength=K[j]); c0 = np.bincount(A[itr, j], w0, minlength=K[j])
                tab = np.log((c1 + a) / (w1.sum() + a * K[j])) - np.log((c0 + a) / (w0.sum() + a * K[j]))
                cols_c.append(tab[A[iva, j]])
            C.append(np.column_stack(cols_c)); P0.append(np.log(w1.sum() / w0.sum())); Y.append(ytr[iva])
        nombres_c = [prep.cols[j] for j in cand]
        for cr in criterios:
            elegidos, s_ = [], [np.zeros(len(Y[b])) for b in range(k_interno)]
            l_act = np.mean([MT.logloss(Y[b], GL._sig(P0[b] + s_[b])) for b in range(k_interno)])
            for paso in range(min(kmax, len(cand))):
                usadas = [nombres_c[i] for i in elegidos]; c0 = datos.MP.costo(usadas)
                mejor, mejor_v, mejor_l = None, -np.inf, np.inf
                for i in range(len(cand)):
                    if i in elegidos:
                        continue
                    l = np.mean([MT.logloss(Y[b], GL._sig(P0[b] + s_[b] + C[b][:, i])) for b in range(k_interno)])
                    v = (l_act - l) / (1 + datos.MP.costo(usadas + [nombres_c[i]]) - c0) if cr == "costo" else -l
                    if v > mejor_v:
                        mejor, mejor_v, mejor_l = i, v, l
                if mejor is None or (cr == "costo" and mejor_v <= 0):
                    break
                l_act = mejor_l
                elegidos.append(mejor)
                for b in range(k_interno):
                    s_[b] = s_[b] + C[b][:, mejor]
            nombres = [nombres_c[i] for i in elegidos]
            for k_ in range(1, len(nombres) + 1):
                ck = nombres[:k_]
                sets[(cr, k_)].append(ck)
                if not final:
                    aj = motor.ajustar(Xtr[ck], ytr, None, params, 1)
                    oof[(cr, k_)][va] = motor.predecir(aj, datos.matriz(ck, va, train_idx=tr))
            if verbose:
                print(f"  NB selectivo ({cr}) pliegue {'todo' if final else k}: {nombres[:5]}… ({time.time() - t0:.0f} s)", flush=True)
        del Xtr
    puntos = {}
    for i, (cr, k_) in enumerate(claves):
        if len(sets[(cr, k_)]) == len(folds) + 1 and not np.isnan(oof[(cr, k_)][np.concatenate([va for _, va in folds])]).any():
            puntos[f"{cr}: k = {k_}"] = {"oof": oof[(cr, k_)], "cols_pliegues": sets[(cr, k_)][:-1], "final": sets[(cr, k_)][-1],
                                         "orden": i}
    return {"puntos": puntos, "tiempo_s": time.time() - t0}


# ------------------------------------------------------------------------------ LOCO por bloque teórico
def loco_bloques(motor, datos, params, cols, r=1, semillas=(1,), lam_B=1.0, cache_dir=None, reusar=True, verbose=True,
                 tipo="logloss") -> pd.DataFrame:
    """Reajusta sin cada bloque teórico (familias de v7 o bloques A–F) y mide la caída de F1 y de la pérdida, con el
    t corregido por pliegue y el EE por influencia del ΔF1."""
    from .cvmotor import cv
    base = cv(motor, datos, params, cols, lam_B, r, semillas, cache_dir=cache_dir, reusar=reusar)
    folds = datos.folds(r)
    mb = MT.metricas_rk(datos.y, {r: base.oof}, {r: folds}, datos.mT, tipo)
    yb = MT.f1_anidado(datos.y, base.oof, folds, datos.mT)["yhat"][datos.mT]
    filas = []
    for b in sorted({datos.bloque(c) for c in cols}):
        sin = [c for c in cols if datos.bloque(c) != b]
        if not sin or len(sin) == len(cols):
            continue
        res = cv(motor, datos, params, sin, lam_B, r, semillas, cache_dir=cache_dir, reusar=reusar)
        m = MT.metricas_rk(datos.y, {r: res.oof}, {r: folds}, datos.mT, tipo)
        c = CP.comparar_rk(mb, m, tipo, len(folds), "completo", f"sin {b}")
        yb2 = MT.f1_anidado(datos.y, res.oof, folds, datos.mT)["yhat"][datos.mT]
        bp = CP.bootstrap_pareado(datos.y[datos.mT], yb, yb2, B=500)
        filas.append({"bloque": b, "nombre": datos.nombre_bloque(b), "variables_quitadas": len(cols) - len(sin),
                      "preguntas_quitadas": datos.MP.costo(cols) - datos.MP.costo(sin),
                      "caida_F1": c["dF1"], "caida_F1_EE_NB": c["dF1_EE"], "caida_F1_IC95_NB": c["dF1_IC95"],
                      "caida_F1_IC95_boot": bp["IC95_boot"], f"aumento_{tipo}": -c[f"d{tipo}"],
                      f"aumento_{tipo}_EE_NB": c[f"d{tipo}_EE"], "P_rope_F1": c["dF1_P_rope"]})
        if verbose:
            print(f"  LOCO sin bloque {b}: ΔF1 = {-c['dF1']:+.4f} (EE {c['dF1_EE']:.4f}), Δ{tipo} = {-c[f'd{tipo}']:+.5f}", flush=True)
    return pd.DataFrame(filas)
