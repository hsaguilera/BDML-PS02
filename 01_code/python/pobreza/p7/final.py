r"""Modelo final, umbral, CSV verificado, registro común y señal de listo (Pasos 11–12 del protocolo; memo §5.5 y §8.3).

* **Umbral**: mediana bootstrap (B = 500) del argmax del F1 en las OOF de 𝒯 (promediadas por semilla), y la tasa de
  positivos $\hat q$ que implica. Diagnósticos: $\hat t$ frente a $\hat F/2$ (Lipton, Elkan y Naryanaswamy, 2014).
* **Predictor final**: reajuste con el 100 % del entrenamiento (con λ_B). Boosting: S semillas a la tasa final con
  árboles = mediana de las iteraciones de los pliegues (a esa tasa) × 1,2; RF: un bosque de 1.000 árboles cuyo
  umbral se elige en sus propias probabilidades OOB (misma escala); deterministas: un ajuste.
* **Traslado del umbral al test**: boosting → tasa igualada (cuantil 1 − $\hat q$ de los puntajes de test); RF → umbral
  de sus OOB; deterministas → $\hat t$ fijo, verificando que la tasa predicha caiga en $\hat q\pm2\sqrt{\hat q(1-\hat q)/n_{test}}$
  (si no, tasa igualada). Contraste obligatorio: promedio de los K × S modelos de pliegue con $\hat t$ fijo y la
  proporción de hogares de test clasificados distinto.
* **CSV** `id,pobre` en el orden de `sample_submission` (66.168 filas, 0/1), nombre = algoritmo + hiperparámetros +
  `_v7`; registro común (`registro.py`). Nada se sube a Kaggle.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import metricas as MT
from .cvmotor import cv, oob_rf, _limpio


def _sample(ent) -> pd.DataFrame:
    return pd.read_csv(Path(ent.raw) / "sample_submission.csv")


def verificar_csv(ruta, sample: pd.DataFrame, n_esperado: int | None = 66168) -> dict:
    csv = pd.read_csv(ruta)
    assert list(csv.columns) == ["id", "pobre"], csv.columns
    assert len(csv) == len(sample) and (n_esperado is None or len(csv) == n_esperado), len(csv)
    assert (csv.id.values == sample.id.values).all(), "ids u orden distintos de sample_submission"
    assert set(np.unique(csv.pobre)) <= {0, 1}
    return {"filas": int(len(csv)), "tasa_pobres": float(csv.pobre.mean()), "verificado": True}


def escribir_csv(ent, ids_test, yhat, archivo) -> Path:
    sample = _sample(ent)
    te = pd.DataFrame({"id": np.asarray(ids_test), "pobre": np.asarray(yhat).astype(int)})
    sub = sample[["id"]].merge(te, on="id", how="left")
    assert len(sub) == len(sample) and sub.pobre.notna().all(), "ids de test incompletos"
    sub["pobre"] = sub.pobre.astype(int)
    p = Path(ent.envios) / archivo
    sub[["id", "pobre"]].to_csv(p, index=False)
    return p


def umbral_tasa(te, q) -> float:
    """Umbral que deja la tasa de positivos más cercana a q entre los valores observados (con empates —CART, NB— el
    cuantil no puede igualar la tasa exactamente: se elige el valor que más se acerca)."""
    te = np.asarray(te, float)
    v = np.unique(te)[::-1]
    tasas = np.searchsorted(np.sort(-te), -v, side="right") / len(te)        # P(te ≥ v)
    return float(v[int(np.argmin(np.abs(tasas - q)))])


def umbral_y_tasa(datos, oof, B=500) -> dict:
    mT = datos.mT
    ub = MT.umbral_bootstrap(datos.y[mT], oof[mT], B=B)
    q = float((oof[mT] >= ub["umbral"]).mean())
    return {**ub, "q_hat": q}


def modelo_final(motor, datos, params, cols, lam_B, P, ent, nombre: str, version: str = "principal",
                 semillas_cv=(1,), notebook: str = "", cache_dir=None, extra_desc: str = "", reusar=True,
                 registrar: bool = True, verbose: bool = True) -> dict:
    t0 = time.time()
    y, mT = datos.y, datos.mT
    es_boost = motor.parada_temprana
    n = len(y); n_te = len(datos.ids_test)
    p_fin = dict(params)
    if es_boost:
        p_fin["learning_rate"] = P.lr_final
    # 1) OOF de la partición 1 (semillas promediadas) para el umbral y contraste de los modelos de pliegue
    res = cv(motor, datos, p_fin, cols, lam_B, 1, semillas_cv, con_test=True, cache_dir=cache_dir, reusar=reusar)
    oof = res.oof
    # 2) reajuste con el 100 %
    idx = np.arange(n) if lam_B > 0 else np.flatnonzero(mT)
    w = None if lam_B in (0, 1) else np.where(datos.bog[idx], float(lam_B), 1.0)
    X = datos.matriz(cols, idx, train_idx=np.arange(n)); XT = datos.matriz_test(cols)
    info = {}
    if es_boost:
        n_arb = int(math.ceil(1.2 * float(np.median(res.iteraciones))))
        S = P.S_final_boost
        te = np.zeros(n_te); por_semilla = []
        for s in range(1, S + 1):
            aj = motor.ajustar(X, y[idx], w, p_fin, s, n_iter=n_arb)
            ps = motor.predecir(aj, XT); te += ps / S; por_semilla.append(ps)
        info.update({"arboles_finales": n_arb, "semillas_finales": S, "iter_pliegues": [int(i) for i in res.iteraciones]})
        um = umbral_y_tasa(datos, oof, P.B_umbral)
        t_test = umbral_tasa(te, um["q_hat"])
        regla = "tasa igualada"
        # inestabilidad de decisión entre semillas (memo §8.1): fracción de hogares de test cuya clase con UNA semilla
        # (misma regla de tasa igualada) difiere de la del promedio de las S semillas
        y_prom = te >= t_test
        dif = [float(((ps >= umbral_tasa(ps, um["q_hat"])) != y_prom).mean()) for ps in por_semilla]
        info.update({"inestabilidad_semillas_media": float(np.mean(dif)), "inestabilidad_semillas_max": float(np.max(dif))})
    elif motor.oob:
        p_fin["n_estimators"] = P.rf_arboles_final
        aj = motor.ajustar(X, y[idx], w, p_fin, 1, oob=True)
        te = motor.predecir(aj, XT)
        oob = np.full(n, np.nan); oob[idx] = motor.oob_proba(aj)
        ok = mT & np.isfinite(oob)
        ub = MT.umbral_bootstrap(y[ok], oob[ok], B=P.B_umbral)
        um = {**ub, "q_hat": float((oob[ok] >= ub["umbral"]).mean())}
        t_test = um["umbral"]; regla = "umbral en las OOB del bosque final"
        info.update({"arboles_finales": P.rf_arboles_final, "oob_logloss_T": MT.logloss(y[ok], oob[ok]),
                     "oob_nan": int((~np.isfinite(oob[idx])).sum()),
                     "oob_F1_umbral": MT.f1(y[ok], oob[ok] >= um["umbral"])})
        oob_final = oob
    else:
        aj = motor.ajustar(X, y[idx], w, p_fin, 1)
        te = motor.predecir(aj, XT)
        um = umbral_y_tasa(datos, oof, P.B_umbral)
        tasa_fija = float((te >= um["umbral"]).mean())
        banda = 2 * math.sqrt(um["q_hat"] * (1 - um["q_hat"]) / n_te)
        if abs(tasa_fija - um["q_hat"]) <= banda:
            t_test, regla = um["umbral"], "umbral fijo (tasa verificada)"
        else:
            t_test, regla = umbral_tasa(te, um["q_hat"]), "tasa igualada (la tasa con umbral fijo salió de la banda)"
        info["banda_tasa"] = banda
    yhat = (te >= t_test).astype(int)
    # diagnósticos
    pi = datos.prevalencia()
    f_oof = MT.umbral_argmax(y[mT], oof[mT])[1]
    yhat_fijo = (te >= um["umbral"]).astype(int)
    yhat_q = (te >= umbral_tasa(te, um["q_hat"])).astype(int)
    yhat_pl = (res.test >= um["umbral"]).astype(int)
    diag = {"umbral_oof": um["umbral"], "umbral_argmax": um["umbral_argmax"], "umbral_IC90": [um["umbral_p05"], um["umbral_p95"]],
            "F1_oof_max": f_oof, "F_medio": f_oof / 2, "q_hat": um["q_hat"], "umbral_test": t_test, "regla": regla,
            "tasa_predicha_test": float(yhat.mean()), "tasa_fija_test": float(yhat_fijo.mean()),
            "tasa_igualada_test": float(yhat_q.mean()), "dif_reglas": float((yhat_fijo != yhat_q).mean()),
            "media_prob_test": float(np.mean(te)), "prevalencia_esperada_test": pi["prevalencia_test"],
            "media_prob_ok": bool(abs(np.mean(te) - pi["prevalencia_test"]) <= 0.003),
            "tasa_ok": bool(abs(float(yhat.mean()) - um["q_hat"]) <= 0.003),
            "contraste_promedio_pliegues_dif": float((yhat_pl != yhat).mean()),
            "contraste_tasa_promedio_pliegues": float(yhat_pl.mean()), **info}
    # variables efectivamente usadas (CART podado, red elástica): las demás no requieren pregunta alguna
    cols_ef = list(cols)
    if hasattr(motor, "variables_efectivas"):
        try:
            cols_ef = motor.variables_efectivas(aj) or list(cols)
        except Exception:
            cols_ef = list(cols)
    diag["n_vars_efectivas"] = len(cols_ef); diag["n_preguntas_efectivas"] = datos.MP.costo(cols_ef)
    # CSV
    S_txt = P.S_final_boost if es_boost else (P.rf_arboles_final if motor.oob else 1)
    lb = f"{lam_B:g}".replace(".", "_")
    suf = {"principal": "", "parsimoniosa": "_pars", "encuesta_corta": "_corta"}.get(version, f"_{version}")
    archivo = motor.nombre_archivo(p_fin, len(cols_ef), f"_lb{lb}_s{S_txt}{suf}")
    ruta = escribir_csv(ent, datos.ids_test, yhat, archivo)
    ver = verificar_csv(ruta, _sample(ent), ent.extra.get("n_test_esperado", 66168))
    out = {"oof": oof, "test": te, "yhat_test": yhat, "umbral": um, "diagnostico": diag, "archivo": str(ruta),
           "verificacion": ver, "params": p_fin, "cols": list(cols), "cols_efectivas": cols_ef, "lam_B": lam_B, "res_cv": res, "tiempo_s": time.time() - t0}
    if motor.oob:
        out["oob_final"] = oob_final
    if registrar:
        from .. import registro as REG
        desc = (f"{motor.nombre} (clasificación), protocolo v7; versión {version}; {len(cols_ef)} variables, "
                f"{datos.MP.costo(cols_ef)} preguntas; λ_B = {lam_B}; umbral: {regla} ({t_test:.4f}); {motor.describir(p_fin)}. {extra_desc}")
        slug = REG.registrar(ent.rutas_registro(), nombre, oof, te, datos.ids, datos.ids_test, familia=motor.familia,
                             descripcion=desc, n_vars=len(cols_ef),
                             params={**_limpio(p_fin), "lam_B": lam_B, "variables": list(cols), "umbral_test": t_test,
                                     "regla_umbral": regla, "version": version},
                             es_probabilidad=True, tiempo_s=out["tiempo_s"], notebook=notebook, archivo_envio=archivo,
                             permitido_envio=True, motivo_envio=f"clasificación directa con {motor.familia} (lista del enunciado)")
        out["slug"] = slug
    if verbose:
        print(f"→ {nombre}: umbral OOF {um['umbral']:.4f} (q̂ = {um['q_hat']:.4f}); regla «{regla}»; tasa test {yhat.mean():.4f}; "
              f"media prob. test {np.mean(te):.4f} (esperada {pi['prevalencia_test']:.4f}); CSV {Path(ruta).name} "
              f"({out['tiempo_s'] / 60:.1f} min)", flush=True)
    return out


def escribir_listo(ent, pref: str, contenido: dict) -> Path:
    p = Path(ent.tablas) / f"{pref}_LISTO.json"
    p.write_text(json.dumps(_limpio(contenido), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return p
