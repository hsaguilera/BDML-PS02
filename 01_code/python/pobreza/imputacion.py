r"""Primera etapa: ingreso individual esperado como *variable generada* (Pagan, 1984).

En train se observa el ingreso total de cada persona (`Ingtot`), pero en test no.
Se estima E[Ingtot_i | x_i, c_h] con x_i = variables de persona COMUNES a train y test
(educación, edad, posición ocupacional, cotización, horas, tamaño de empresa,
prestaciones, oficio, dominio, ...) y c_h = contexto observable del hogar (vivienda,
perceptores, formalidad, ...). Luego se agrega por unidad de gasto:

    Î_h = Σ_{i ∈ UG(h)} Ê[Ingtot_i | x_i, c_h],     ẑ_h = log( (Î_h + A_h) / Npersug_h / Lp_h )

donde A_h es el arriendo imputado que MESE suma al ingreso (Ingtotugarr = Ingtotug + A_h),
que se reconstruye con P5090, P5130 y P5100 (ver features.variables_hogar).
ẑ_h es un "índice de ingreso permanente predicho" (como en los PMT; Grosh y Baker, 1995).

Para evitar fuga de información, en train la predicción es FUERA DE PLIEGUE usando
los MISMOS pliegues de hogares que la validación cruzada del modelo final.

Estimador (versión 3): LightGBM con pérdida Poisson (log-link), que estima E[y|x] en
niveles (no requiere retransformación de Duan, admite ceros y es consistente bajo mala
especificación de la varianza; Wooldridge, 2010, cap. 18). Árboles de 127 hojas y parada
temprana sobre un 10% interno del pliegue de entrenamiento (hasta 4000 árboles). La
versión anterior (HistGradientBoosting con tope de 400 iteraciones) quedaba sub-ajustada:
varios pliegues alcanzaban el tope.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUM_PERSONA = ["P6040", "P6210s1", "P6426", "P6800", "P7045"]
CAT_PERSONA = ["P6020", "P6050", "P6090", "P6100", "P6210", "P6240", "Oficio", "P6430", "P6510", "P6545",
               "P6580", "P6585s1", "P6585s2", "P6585s3", "P6585s4", "P6590", "P6600", "P6610", "P6620",
               "P6630s1", "P6630s2", "P6630s3", "P6630s4", "P6630s6", "P6870", "P6920", "P7040", "P7050",
               "P7090", "P7110", "P7120", "P7150", "P7160", "P7310", "P7350", "P7422", "P7472", "P7495",
               "P7500s2", "P7500s3", "P7505", "P7510s1", "P7510s2", "P7510s3", "P7510s5", "P7510s6",
               "P7510s7", "Oc", "Des", "Ina", "Clase"]


# Contexto del hogar (sólo columnas observables en test) que se añade a cada persona: la
# vivienda y la inserción laboral del resto del hogar informan sobre el ingreso permanente
# del individuo (PMT; Grosh y Baker, 1995). Se usa si se pasa `ctx_tr`/`ctx_te`.
CTX_HOGAR = ["log_arriendo_sobre_linea", "log_arriendo", "tenencia", "cuartos", "Nper", "Npersug",
             "cabecera", "n_ocupado", "n_cotiza_pension", "educ_max_adultos", "share_subsidiado",
             "log_Lp", "arriendo_imputado_flag", "personas_por_cuarto", "n_pensionado",
             "n_mincer_ocupado", "asalariados_formales_pc", "n_n_beneficios", "ocupados_pc", "dependencia",
             "jefe_anios_educ", "usufructo_rural", "log_ingreso_requerido"]

PARAMS_LGB = dict(objective="poisson", learning_rate=0.05, num_leaves=127, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=255,
                  verbose=-1, cat_smooth=20, max_cat_to_onehot=8)


def matriz_persona(per: pd.DataFrame, dominios: list[str], ctx: pd.DataFrame | None = None) -> pd.DataFrame:
    X = per[NUM_PERSONA].astype("float32").copy()
    for c in CAT_PERSONA:
        X[c] = per[c].fillna(-1).astype("float32")
    X["Dominio"] = pd.Categorical(per["Dominio"], categories=dominios).codes.astype("float32")
    if ctx is not None:
        C = ctx.set_index("id")[[c for c in CTX_HOGAR if c in ctx.columns]].reindex(per["id"].values)
        for c in C.columns:
            X[f"h_{c}"] = C[c].astype("float32").values
    return X


def _categorizar(Xtr, Xte):
    """Columnas categóricas como `category` con los mismos niveles en train y test."""
    cats = [c for c in CAT_PERSONA + ["Dominio", "h_tenencia"] if c in Xtr.columns]
    for c in cats:
        a = Xtr[c].fillna(-1).astype(int); b = Xte[c].fillna(-1).astype(int)
        niv = sorted(set(a) | set(b))
        Xtr[c] = pd.Categorical(a, categories=niv); Xte[c] = pd.Categorical(b, categories=niv)
    return Xtr, Xte


def ingreso_individual_oof(per_tr: pd.DataFrame, per_te: pd.DataFrame, fold_hogar: pd.Series,
                           tope_q: float = 0.999, semilla: int = 1, verbose=True,
                           ctx_tr: pd.DataFrame | None = None, ctx_te: pd.DataFrame | None = None,
                           params: dict | None = None, max_rounds: int = 4000):
    """Devuelve (ŷ_train_oof, ŷ_test, info). `fold_hogar` mapea id de hogar → pliegue.
    `ctx_tr`/`ctx_te`: tablas de hogar (con `id`) para añadir el contexto CTX_HOGAR."""
    import lightgbm as lgb
    dominios = sorted(set(per_tr.Dominio) | set(per_te.Dominio))
    Xtr = matriz_persona(per_tr, dominios, ctx_tr); Xte = matriz_persona(per_te, dominios, ctx_te)
    Xtr, Xte = _categorizar(Xtr, Xte)
    p = {**PARAMS_LGB, **(params or {}), "seed": semilla}
    y = per_tr["Ingtot"].fillna(0).clip(lower=0).values
    tope = np.quantile(y[y > 0], tope_q)
    y = np.minimum(y, tope)
    f = per_tr["id"].map(fold_hogar).values
    oof = np.zeros(len(y))
    iters = []
    imp = pd.Series(0.0, index=Xtr.columns)
    for k in np.unique(f):
        tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
        es = np.random.default_rng(semilla + int(k)).random(len(tr)) < 0.1
        dtr = lgb.Dataset(Xtr.iloc[tr[~es]], y[tr[~es]])
        des = lgb.Dataset(Xtr.iloc[tr[es]], y[tr[es]], reference=dtr)
        m = lgb.train(p, dtr, max_rounds, valid_sets=[des], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(Xtr.iloc[va], num_iteration=m.best_iteration); iters.append(m.best_iteration)
        imp += pd.Series(m.feature_importance("gain"), index=Xtr.columns)
        if verbose:
            print(f"  pliegue {k}: árboles={m.best_iteration}")
    mfull = lgb.train(p, lgb.Dataset(Xtr, y), int(np.ceil(np.mean(iters) * 1.1)))
    pte = mfull.predict(Xte)
    # R² en niveles y en log1p (fuera de pliegue)
    r2 = 1 - np.sum((y - oof) ** 2) / np.sum((y - y.mean()) ** 2)
    r2l = 1 - np.sum((np.log1p(y) - np.log1p(oof)) ** 2) / np.sum((np.log1p(y) - np.log1p(y).mean()) ** 2)
    info = {"R2_niveles_oof": float(r2), "R2_log1p_oof": float(r2l), "tope": float(tope),
            "iteraciones_pliegues": iters, "modelo_final": mfull,
            "importancia": (imp / imp.sum()).sort_values(ascending=False)}
    return oof, pte, info


def agregar_ingreso(per: pd.DataFrame, yhat: np.ndarray) -> pd.Series:
    """Σ de ingreso esperado de los miembros de la unidad de gasto (excluye P6050 ∈ {6,7,8})."""
    en_ug = ~per["P6050"].isin([6, 7, 8]).values
    return pd.Series(yhat * en_ug, index=per.index).groupby(per["id"].values).sum()


def agregar_detalle(per: pd.DataFrame, yhat: np.ndarray, lp_hogar: pd.Series) -> pd.DataFrame:
    """Agregados adicionales del ingreso esperado por hogar (unidad de gasto):
    máximo individual, del jefe, número de perceptores esperados con ingreso ≥ Lp del hogar y
    concentración (participación del principal perceptor)."""
    en_ug = ~per["P6050"].isin([6, 7, 8]).values
    d = pd.DataFrame({"id": per["id"].values, "y": yhat * en_ug, "jefe": (per["P6050"] == 1).values})
    d["lp"] = d["id"].map(lp_hogar).values
    g = d.groupby("id", sort=False)
    out = pd.DataFrame({"ing_esp_max": g["y"].max(), "ing_esp_suma": g["y"].sum()})
    out["ing_esp_jefe"] = d[d.jefe].groupby("id")["y"].sum()
    out["n_perceptores_esp_lp"] = (d["y"] >= d["lp"]).groupby(d["id"]).sum()
    out["share_principal_perceptor"] = out.ing_esp_max / out.ing_esp_suma.clip(lower=1)
    return out
