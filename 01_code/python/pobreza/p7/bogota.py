r"""Diagnósticos de Bogotá (Paso 5.1; memo §1.5): clasificador de dominio, tamaño efectivo de muestra de las razones de
densidad y prueba de transferencia.

* **Clasificador de dominio** (Bickel, Brückner y Scheffer, 2009; validación adversarial como prueba de dos muestras:
  Lopez-Paz y Oquab, 2017): Bogotá frente a las demás cabeceras con (a) todas las variables salvo Dominio/Depto y sus
  derivadas, y (b) sólo socioeconómicas (sin $L_p$, $L_i$ ni nada normalizado por ellas). Con $\hat c(x)=\hat P(B\mid x)$,
  $\hat w(x)=\frac{1-\hat c}{\hat c}\frac{\pi_B}{1-\pi_B}$ y $\mathrm{ESS}=(\sum\hat w)^2/\sum\hat w^2$ sobre Bogotá
  (Cortes, Mansour y Mohri, 2010: la calidad del aprendizaje ponderado depende del segundo momento de los pesos).
* **Transferencia**: con λ_B = 0 las OOF de los hogares de Bogotá provienen de modelos que nunca vieron Bogotá
  (AUC y tasa media frente a la observada); con λ_B = 1, la misma medida «dentro» de Bogotá. Si ambas coinciden, la
  función de etiquetado es compartida (término λ de la cota de Ben-David et al., 2010, pequeño).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import metricas as MT

PATRON_LINEA = re.compile(r"(^|_)(lp|li|linea|lineas|requerido|n_lineas)(_|$)|^Lp$|^Li$|^log_Lp|^u_|sobre_linea|log_linea", re.I)
PATRON_DOMINIO = re.compile(r"^(Dominio|Depto)$|^t_dom_|^t_depto_|^te_depto|^te_dominio|_dom_|rgdom|qdom|^v_resto_urbano$|^v_rural$")


def clasificador_dominio(datos, cols, hilos=4, semilla=1, K=5) -> dict:
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    cab = datos.info["cabecera"].values == 1 if "cabecera" in datos.info else np.ones(len(datos.y), bool)
    filas = np.flatnonzero(cab)
    b = datos.bog[filas].astype(int)
    X = datos.matriz(cols, filas, train_idx=np.arange(len(datos.y)))
    c = np.zeros(len(filas))
    p = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_child_samples=50, feature_fraction=0.7,
             verbose=-1, num_threads=hilos, seed=semilla)
    for tr, va in StratifiedKFold(K, shuffle=True, random_state=semilla).split(np.zeros(len(b)), b):
        rng = np.random.default_rng(semilla); es = rng.random(len(tr)) < 0.15
        dtr = lgb.Dataset(X.iloc[tr[~es]], b[tr[~es]]); des = lgb.Dataset(X.iloc[tr[es]], b[tr[es]], reference=dtr)
        m = lgb.train(p, dtr, 2000, valid_sets=[des], callbacks=[lgb.early_stopping(50, verbose=False)])
        c[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    piB = b.mean()
    cb = np.clip(c[b == 1], 1e-6, 1 - 1e-6)
    w = (1 - cb) / cb * piB / (1 - piB)
    return {"AUC": MT.auc_roc(b, c), "ESS_bogota": float(w.sum() ** 2 / (w ** 2).sum()), "n_bogota": int(b.sum()),
            "mediana_w": float(np.median(w)), "n_variables": len(cols), "hogares": int(len(b))}


def diagnostico(datos, hilos=4) -> dict:
    todas = [c for c in datos.todas if not PATRON_DOMINIO.search(c)]
    socio = [c for c in todas if not PATRON_LINEA.search(c)]
    return {"todas_salvo_dominio": clasificador_dominio(datos, todas, hilos),
            "solo_socioeconomicas": clasificador_dominio(datos, socio, hilos),
            "Lp_disjunto": datos.verificacion.get("b_Lp_bogota_ausente_en_test"),
            "valores_Lp_bogota": datos.verificacion.get("b_valores_Lp_bogota")}


def transferencia(datos, oof_lambda0, oof_lambda1) -> dict:
    b = datos.bog; yb = datos.y[b]
    out = {}
    for nom, s in (("entrenado_sin_bogota", oof_lambda0), ("entrenado_con_bogota", oof_lambda1)):
        s = np.asarray(s)[b]
        out[nom] = {"AUC_en_bogota": MT.auc_roc(yb, s), "media_predicha": float(np.mean(s)), "tasa_observada": float(yb.mean()),
                    "logloss_en_bogota": MT.logloss(yb, s)}
    return out
