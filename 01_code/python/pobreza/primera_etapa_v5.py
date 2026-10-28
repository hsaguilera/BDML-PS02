r"""Primera etapa, versión 5: ingreso individual por componentes con esquema de valla (*hurdle*),
primera etapa distribucional y etapa intermedia del hogar (cuaderno 02b).

Identidad contable verificada en train (100 % de las personas y de los hogares):

    Ingtot_i = Σ_c (I_c + I_c^es),   c ∈ {Impa, Isa, Ie, Imdi, Iof1, Iof2, Iof3h, Iof3i, Iof6}
    Ingtotug_h = Σ_{i ∈ UG(h)} Ingtot_i            (UG excluye P6050 ∈ {6, 7, 8})

donde `I_c^es` es el monto imputado por el DANE (no respuesta) del mismo componente. Cada componente
sólo puede ser positivo si una pregunta de «¿recibió…?» que SÍ está en test lo permite (compuerta);
cuando la compuerta está cerrada el componente es exactamente cero. Las compuertas verificadas son:

    laboral (Impa)            ocupado y P6430 ∉ {6, 7} (no es trabajador sin remuneración)
    segunda actividad (Isa)   ocupado y P7040 = 1
    en especie (Ie)           ocupado asalariado/doméstico/jornalero (P6430 ∈ {1, 2, 3, 8})
    desocupados e inactivos   P7422 = 1 o P7472 = 1
    intereses (Iof1)          P7510s5 ∈ {1, 9}
    pensiones (Iof2)          P7500s2 ∈ {1, 9}
    ayudas de hogares (Iof3h) P7510s1, P7510s2 o P7500s3 ∈ {1, 9} (incluye pensión alimentaria)
    ayudas institucionales    P7510s3 ∈ {1, 9}
    arriendos (Iof6)          P7495 = 1 (P7500s1, la pregunta exacta, no está en test)

Modelo de valla por componente (Cragg, 1971; Duan et al., 1983; Mullahy, 1986):

    P(y_ic > 0 | x) = π_c(x)                 (clasificador LightGBM; = tasa si π ≈ 1)
    log y_ic | y_ic > 0, x ~ N(μ_c(x), s²_c(x))   (media: LightGBM L2; varianza: LightGBM gamma
                                                  sobre residuos al cuadrado, calibrado fuera de muestra)

de donde E[y_ic|x] = π·exp(μ + s²/2) y Var[y_ic|x] = π·exp(2μ + 2s²) − E². El ingreso del hogar es la
suma sobre personas y componentes; su distribución se aproxima por una lognormal con los dos primeros
momentos (Fenton, 1960; Wilkinson) con masa en cero P0 = Π(1 − π) y una correlación intrahogar ρ
(ρ = 0 independencia; ρ = 1 comonotonía), elegida dentro del pliegue. Así se obtiene la
**probabilidad estructural de pobreza** P(Ingtotug_h < R_h), R_h = Npersug·Lp − A_h, en el espíritu de los
mapas de pobreza (Elbers, Lanjouw y Lanjouw, 2003; Molina y Rao, 2010).

Etapa intermedia del hogar: LightGBM de z_h = log(Ingtotugarr/(Npersug·Lp)) con variables del hogar y los
agregados de la primera etapa (fuera de pliegue, mismos pliegues), y un segundo LightGBM de |z − ẑ|
(fuera de pliegue) para el error esperado: P_hogar = Φ(−ẑ/σ̂).

Todo se estima fuera de pliegue con los MISMOS pliegues de hogares de `pliegues.csv`; test con modelos
reentrenados en todo train con ⌈1,1·media⌉ de los árboles de los pliegues.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import features as F
from . import imputacion as IMP

# ------------------------------------------------------------------------------ componentes y compuertas
COMPONENTES = {
    "laboral": ("Impa", "Ingreso monetario de la primera actividad"),
    "segunda": ("Isa", "Ingreso de la segunda actividad"),
    "especie": ("Ie", "Ingreso en especie"),
    "desocupados": ("Imdi", "Ingreso de desocupados e inactivos"),
    "intereses": ("Iof1", "Intereses y dividendos"),
    "pensiones": ("Iof2", "Pensiones y jubilaciones"),
    "ayudas_hogares": ("Iof3h", "Ayudas de otros hogares (país, exterior) y pensión alimentaria"),
    "ayudas_inst": ("Iof3i", "Ayudas institucionales"),
    "arriendos": ("Iof6", "Arriendos"),
}
GRUPOS = {"laboral": ["laboral", "segunda", "especie"], "pensiones": ["pensiones"],
          "transferencias": ["ayudas_hogares", "ayudas_inst"], "capital": ["intereses", "arriendos"],
          "desocupados": ["desocupados"]}
PISO = 1000.0          # piso de los montos positivos (log)


def _si(s: pd.Series) -> pd.Series:
    return s.isin([1, 9])


def compuertas(per: pd.DataFrame) -> pd.DataFrame:
    """Compuertas exactas (sólo columnas presentes en test): True si el componente puede ser > 0."""
    oc = per["Oc"].eq(1)
    pos = per["P6430"]
    g = pd.DataFrame(index=per.index)
    g["laboral"] = oc & ~pos.isin([6, 7])
    g["segunda"] = oc & per["P7040"].eq(1)
    g["especie"] = oc & pos.isin([1, 2, 3, 8])
    g["desocupados"] = _si(per["P7422"]) | _si(per["P7472"])
    g["intereses"] = _si(per["P7510s5"])
    g["pensiones"] = _si(per["P7500s2"])
    g["ayudas_hogares"] = _si(per["P7510s1"]) | _si(per["P7510s2"]) | _si(per["P7500s3"])
    g["ayudas_inst"] = _si(per["P7510s3"])
    g["arriendos"] = per["P7495"].eq(1)
    return g


def objetivos_componentes(per: pd.DataFrame) -> pd.DataFrame:
    """Monto de cada componente = observado + imputado por el DANE (sólo train)."""
    return pd.DataFrame({c: (per[v].fillna(0) + per[v + "es"].fillna(0)).clip(lower=0).values
                         for c, (v, _) in COMPONENTES.items()}, index=per.index)


def verificar_identidad(per: pd.DataFrame, hog: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Tabla por componente (participación en el ingreso, positivos, exactitud de la compuerta) y
    verificación de las identidades Ingtot = Σ componentes e Ingtotug = Σ_UG Ingtot."""
    Y = objetivos_componentes(per)
    G = compuertas(per)
    tot = per["Ingtot"].fillna(0)
    filas = []
    for c, (v, desc) in COMPONENTES.items():
        y, g = Y[c], G[c]
        filas.append({"componente": c, "variable_DANE": f"{v} + {v}es", "descripcion": desc,
                      "participacion_ingreso": y.sum() / tot.sum(), "pct_personas_pos": (y > 0).mean(),
                      "pct_imputado_DANE": per[v + "es"].fillna(0).sum() / max(y.sum(), 1),
                      "compuerta_abierta": int(g.sum()), "tasa_pos_en_compuerta": (y[g] > 0).mean(),
                      "pct_monto_fuera_compuerta": y[~g].sum() / max(y.sum(), 1),
                      "pct_pos_fuera_compuerta": (y[~g] > 0).sum() / max((y > 0).sum(), 1)})
    T = pd.DataFrame(filas)
    ug = ~per["P6050"].isin([6, 7, 8])
    s_ug = (tot * ug).groupby(per["id"]).sum().reindex(hog["id"]).values
    chk = {"Ingtot_igual_suma_componentes": float(np.isclose(tot, Y.sum(1), atol=2).mean()),
           "Ingtotug_igual_suma_UG": float(np.isclose(hog["Ingtotug"].values, s_ug, rtol=1e-6, atol=5).mean()),
           "personas_sin_compuerta_con_ingreso": int(((~G.any(axis=1)) & (tot > 0)).sum())}
    return T, chk


# ------------------------------------------------------------------------------ matriz de persona
CTX_V5 = IMP.CTX_HOGAR + ["arriendo_imputado_mese", "share_arriendo_imputado", "log_amortizacion",
                          "n_salud_contributivo", "n_salud_especial", "n_recibe_pension_jubilacion",
                          "personas_por_dormitorio", "jefe_cotiza_pension", "educ_media_adultos",
                          "n_ocupado_superior", "n_asalariado_formal", "n_menor18", "n_mayor65",
                          "jefe_edad", "jefe_mujer", "hay_pareja"]


def matriz_persona_v5(per: pd.DataFrame, ctx: pd.DataFrame, dominios: list) -> pd.DataFrame:
    """Variables de la persona (comunes a train y test), capital humano (años de educación,
    experiencia potencial de Mincer), horas totales y contexto observable del hogar (prefijo h_)."""
    X = per[IMP.NUM_PERSONA].astype("float32").copy()
    ae = F.anios_educacion(per["P6210"], per["P6210s1"])
    X["anios_educ"] = ae.astype("float32").values
    X["exp_pot"] = (per["P6040"] - ae.fillna(5) - 6).clip(lower=0).astype("float32").values
    X["horas_tot"] = (per["P6800"].fillna(0) + per["P7045"].fillna(0)).astype("float32").values
    for c in IMP.CAT_PERSONA:
        X[c] = per[c].fillna(-1).astype("float32")
    X["Dominio"] = pd.Categorical(per["Dominio"], categories=dominios).codes.astype("float32")
    C = ctx.set_index("id")[[c for c in CTX_V5 if c in ctx.columns]].reindex(per["id"].values)
    for c in C.columns:
        X[f"h_{c}"] = C[c].astype("float32").values
    return X


# ------------------------------------------------------------------------------ LightGBM
BASE = dict(learning_rate=0.05, num_leaves=127, min_child_samples=100, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=255, cat_smooth=20,
            max_cat_to_onehot=8, verbose=-1)


def params_tam(n: int) -> dict:
    """Complejidad según el tamaño de la muestra del submodelo (más regularización si es pequeña)."""
    if n >= 50000:
        return dict(num_leaves=127, min_child_samples=100)
    if n >= 5000:
        return dict(num_leaves=31, min_child_samples=50, lambda_l2=10.0)
    return dict(num_leaves=15, min_child_samples=30, lambda_l2=20.0)


def _fit(lgb, params, X, y, idx, semilla, n_fijo=None, max_rounds=4000, paciencia=100, frac_es=0.1):
    """Ajusta con parada temprana sobre un 10 % interno de `idx` (o con `n_fijo` árboles y todo `idx`)."""
    if n_fijo is not None:
        m = lgb.train(params, lgb.Dataset(X.iloc[idx], y[idx]), int(n_fijo))
        return m, int(n_fijo), idx, idx[:0]
    es = np.random.default_rng(semilla).random(len(idx)) < frac_es
    i_fit, i_es = idx[~es], idx[es]
    dtr = lgb.Dataset(X.iloc[i_fit], y[i_fit])
    des = lgb.Dataset(X.iloc[i_es], y[i_es], reference=dtr)
    m = lgb.train(params, dtr, max_rounds, valid_sets=[des], callbacks=[lgb.early_stopping(paciencia, verbose=False)])
    return m, int(max(m.best_iteration, 1)), i_fit, i_es


def ajustar_componente(lgb, X, yc, gate, tr, hilos, semilla, pm=None, fijo=None):
    """Valla de un componente con las filas `tr`: (1) π(x) = P(y>0 | compuerta, x); (2) μ(x) = E[log y | y>0, x];
    (3) s²(x) = Var[log y | y>0, x] con un LightGBM gamma sobre residuos al cuadrado (en muestra) calibrado con
    los residuos de la submuestra de parada temprana (fuera de muestra). `fijo`: árboles y calibración del
    modelo final (sin parada temprana)."""
    g = tr[gate[tr]]
    pos = yc[g] > 0
    res = {"n_compuerta": int(len(g)), "n_pos": int(pos.sum()), "tasa_pos": float(pos.mean()) if len(g) else 0.0,
           "clf": None, "mu": None}
    if pos.sum() < 30:
        return res
    # (1) clasificador de participación
    if not (pos.mean() > 0.995 or (~pos).sum() < 50):
        pc = {**BASE, **params_tam(len(g)), "objective": "binary", "learning_rate": 0.1, "num_threads": hilos, "seed": semilla}
        lab = (yc > 0).astype("float32")
        # tope de 1.500 árboles: algunas participaciones son casi deterministas (p. ej. especie con P6590–P6620)
        m, n, _, _ = _fit(lgb, pc, X, lab, g, semilla, None if fijo is None else fijo["n_clf"], max_rounds=1500, paciencia=50)
        res["clf"], res["n_clf"] = m, n
    # (2) media del log-monto entre los positivos
    gp = g[pos]
    tope = float(np.quantile(yc[gp], 0.999))
    t = np.log(np.clip(yc, PISO, tope)).astype("float64")
    pmu = {**BASE, **params_tam(len(gp)), **(pm or {}), "objective": "regression", "num_threads": hilos, "seed": semilla}
    m, n, i_fit, i_es = _fit(lgb, pmu, X, t, gp, semilla + 1, None if fijo is None else fijo["n_mu"])
    res.update(mu=m, n_mu=n, tope=tope)
    # (3) varianza del log-monto
    r2 = np.zeros(len(yc))
    r2[i_fit] = (t[i_fit] - m.predict(X.iloc[i_fit], num_iteration=n)) ** 2
    if len(i_es):
        r2[i_es] = (t[i_es] - m.predict(X.iloc[i_es], num_iteration=n)) ** 2
    r2p = r2 + 1e-3
    pv = {**BASE, "objective": "gamma", "learning_rate": 0.05, "num_leaves": 15,
          "min_child_samples": max(30, min(200, len(gp) // 50)), "lambda_l2": 10.0, "num_threads": hilos, "seed": semilla}
    if fijo is None:
        dtr = lgb.Dataset(X.iloc[i_fit], r2p[i_fit]); des = lgb.Dataset(X.iloc[i_es], r2p[i_es], reference=dtr)
        mv = lgb.train(pv, dtr, 2000, valid_sets=[des], callbacks=[lgb.early_stopping(50, verbose=False)])
        nv = int(max(mv.best_iteration, 1))
        cal = float(r2[i_es].mean() / max(mv.predict(X.iloc[i_es], num_iteration=nv).mean(), 1e-6))
        res["r2_mu_es"] = float(1 - ((t[i_es] - m.predict(X.iloc[i_es], num_iteration=n)) ** 2).sum()
                                / max(((t[i_es] - t[i_es].mean()) ** 2).sum(), 1e-9))
    else:
        nv, cal = int(fijo["n_var"]), float(fijo["cal"])
        mv = lgb.train(pv, lgb.Dataset(X.iloc[gp], r2p[gp]), nv)
    res.update(var=mv, n_var=nv, cal=cal)
    return res


def predecir_componente(res, X, filas):
    """(π, μ, s²) en las filas dadas (todas con la compuerta abierta)."""
    n = len(filas)
    if n == 0 or res.get("mu") is None:
        z = np.zeros(n)
        return z, np.full(n, np.log(PISO)), z
    Xf = X.iloc[filas]
    p = res["clf"].predict(Xf, num_iteration=res["n_clf"]) if res["clf"] is not None else np.full(n, res["tasa_pos"])
    mu = res["mu"].predict(Xf, num_iteration=res["n_mu"])
    s2 = np.clip(res["cal"] * res["var"].predict(Xf, num_iteration=res["n_var"]), 1e-3, 9.0)
    return p, mu, s2


def preparar_matrices(per_tr, per_te, ctx_tr, ctx_te):
    dominios = sorted(set(per_tr["Dominio"]) | set(per_te["Dominio"]))
    Xtr = matriz_persona_v5(per_tr, ctx_tr, dominios)
    Xte = matriz_persona_v5(per_te, ctx_te, dominios)
    return IMP._categorizar(Xtr, Xte)


def primera_etapa_oof(per_tr, per_te, Xtr, Xte, fold_hogar: pd.Series, hilos=14, semilla=1, pm: dict | None = None,
                      componentes=None, verbose=True):
    """Primera etapa por componentes, fuera de pliegue. Devuelve (P_tr, P_te, info) con P[c] = matriz (n, 3) de
    (π, μ, s²) por persona (NaN si la compuerta está cerrada)."""
    import lightgbm as lgb
    comps = list(componentes or COMPONENTES)
    Y = objetivos_componentes(per_tr)
    Gtr, Gte = compuertas(per_tr), compuertas(per_te)
    f = per_tr["id"].map(fold_hogar).values
    P_tr = {c: np.full((len(per_tr), 3), np.nan) for c in comps}
    P_te = {c: np.full((len(per_te), 3), np.nan) for c in comps}
    info = {c: {"n_clf": [], "n_mu": [], "n_var": [], "cal": [], "tasa_pos": [], "r2_mu_es": [], "tiempo_s": 0.0} for c in comps}
    t0 = time.time()
    for k in sorted(np.unique(f)):
        tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
        for j, c in enumerate(comps):
            t1 = time.time()
            gate, yc = Gtr[c].values, Y[c].values
            res = ajustar_componente(lgb, Xtr, yc, gate, tr, hilos, semilla + 100 * int(k) + j, pm=(pm or {}).get(c))
            filas = va[gate[va]]
            P_tr[c][filas] = np.column_stack(predecir_componente(res, Xtr, filas))
            for key in ("n_clf", "n_mu", "n_var", "cal", "r2_mu_es"):
                if key in res:
                    info[c][key].append(res[key])
            info[c]["tasa_pos"].append(res["tasa_pos"]); info[c]["tiempo_s"] += time.time() - t1
        if verbose:
            print(f"  pliegue {k}: {time.time() - t0:.0f}s acumulados")
    # modelo final con todo train → test
    todo = np.arange(len(per_tr))
    for j, c in enumerate(comps):
        t1 = time.time()
        I = info[c]
        if not I["n_mu"]:
            continue
        fijo = {"n_clf": int(math.ceil(1.1 * np.mean(I["n_clf"]))) if I["n_clf"] else None,
                "n_mu": int(math.ceil(1.1 * np.mean(I["n_mu"]))), "n_var": int(math.ceil(1.1 * np.mean(I["n_var"]))),
                "cal": float(np.mean(I["cal"]))}
        res = ajustar_componente(lgb, Xtr, Y[c].values, Gtr[c].values, todo, hilos, semilla + 999 + j, pm=(pm or {}).get(c), fijo=fijo)
        filas = np.flatnonzero(Gte[c].values)
        P_te[c][filas] = np.column_stack(predecir_componente(res, Xte, filas))
        I["final"] = fijo; I["tiempo_s"] += time.time() - t1
        if c == comps[0]:
            I["importancia_mu"] = pd.Series(res["mu"].feature_importance("gain"), index=Xtr.columns).sort_values(ascending=False)
    if verbose:
        print(f"  modelos finales: {time.time() - t0:.0f}s acumulados")
    return P_tr, P_te, info


def diagnostico_componentes(per_tr, P_tr, info) -> pd.DataFrame:
    """Calidad fuera de pliegue por componente: AUC de π, R² de μ (log-monto), calibración de s²."""
    from . import metrics as M
    Y = objetivos_componentes(per_tr); G = compuertas(per_tr)
    filas = []
    for c in P_tr:
        g = G[c].values; y = Y[c].values; P = P_tr[c]
        ok = g & np.isfinite(P[:, 0])
        pos = y[ok] > 0
        auc = M.auc_roc(pos.astype(int), P[ok, 0]) if 0 < pos.mean() < 1 else np.nan
        okp = ok & (y > 0)
        t = np.log(np.clip(y[okp], PISO, np.quantile(y[okp], 0.999)))
        mu, s2 = P[okp, 1], P[okp, 2]
        r2 = 1 - ((t - mu) ** 2).sum() / ((t - t.mean()) ** 2).sum()
        filas.append({"componente": c, "n_compuerta": int(g.sum()), "tasa_pos": float(pos.mean()), "AUC_participacion": auc,
                      "R2_log_monto": float(r2), "razon_var_resid_s2": float(((t - mu) ** 2 / s2).mean()),
                      "cobertura_90": float((np.abs(t - mu) <= 1.645 * np.sqrt(s2)).mean()),
                      "arboles_media": float(np.mean(info[c]["n_mu"])) if info[c]["n_mu"] else np.nan,
                      "arboles_clf": float(np.mean(info[c]["n_clf"])) if info[c]["n_clf"] else np.nan,
                      "tiempo_s": info[c]["tiempo_s"]})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------------------ agregación al hogar
def momentos_persona(P: dict) -> tuple[dict, dict]:
    """E[y_ic] y Var[y_ic] de la mezcla valla–lognormal."""
    E, V = {}, {}
    for c, A in P.items():
        p = np.nan_to_num(A[:, 0]); mu = np.nan_to_num(A[:, 1], nan=np.log(PISO)); s2 = np.nan_to_num(A[:, 2])
        e = p * np.exp(mu + s2 / 2)
        m2 = p * np.exp(2 * mu + 2 * s2)
        E[c] = e; V[c] = np.maximum(m2 - e ** 2, 0)
    return E, V


def ingreso_persona(P: dict) -> np.ndarray:
    E, _ = momentos_persona(P)
    return np.sum(list(E.values()), axis=0)


def agregar_hogar_v5(per: pd.DataFrame, P: dict, H: pd.DataFrame, rhos=(0.0,)) -> pd.DataFrame:
    """Agregados del hogar (unidad de gasto) de la primera etapa v5. `H` (una fila por hogar, en el orden
    deseado) debe traer id, Npersug, Lp, ingreso_requerido, arriendo_imputado_mese y log_ingreso_requerido.
    Para cada ρ de `rhos` calcula la probabilidad estructural de pobreza (Fenton–Wilkinson con masa en cero)."""
    en_ug = (~per["P6050"].isin([6, 7, 8])).values.astype(float)
    E, V = momentos_persona(P)
    Ei = np.sum(list(E.values()), axis=0) * en_ug
    Vi = np.sum(list(V.values()), axis=0) * en_ug
    p0i = np.prod([1 - np.nan_to_num(P[c][:, 0]) for c in P], axis=0)
    p0i = np.where(en_ug > 0, p0i, 1.0)
    d = pd.DataFrame({"id": per["id"].values, "E": Ei, "V": Vi, "sd": np.sqrt(Vi),
                      "logp0": np.log(np.clip(p0i, 1e-300, 1)), "ppos": 1 - p0i,
                      "Ejefe": Ei * (per["P6050"].values == 1)})
    for g, cs in GRUPOS.items():
        d["E_" + g] = np.sum([E[c] for c in cs if c in E], axis=0) * en_ug
    agg = {"E": "sum", "V": "sum", "sd": "sum", "logp0": "sum", "ppos": "sum", "Ejefe": "sum",
           **{"E_" + g: "sum" for g in GRUPOS}}
    A = d.groupby("id", sort=False).agg(agg)
    A["Emax"] = d.groupby("id", sort=False)["E"].max()
    A = A.reindex(H["id"].values)
    lin = (H["Npersug"].clip(lower=1) * H["Lp"]).values
    R = H["ingreso_requerido"].values
    Eh, Vh = A["E"].values, A["V"].values
    out = pd.DataFrame(index=H.index)
    out["v5_ing_esp"] = Eh
    out["v5_log_ing_esp"] = np.log(np.maximum(Eh, 1e4))
    out["v5_brecha"] = out["v5_log_ing_esp"].values - H["log_ingreso_requerido"].values
    out["v5_lz"] = np.log(np.maximum(Eh + H["arriendo_imputado_mese"].values, 1e3) / lin)
    out["v5_cv_ing"] = np.sqrt(Vh) / np.maximum(Eh, 1.0)
    out["v5_log_p0"] = A["logp0"].values
    out["v5_n_perceptores_esp"] = A["ppos"].values
    out["v5_perceptores_esp_pc"] = A["ppos"].values / H["Npersug"].clip(lower=1).values
    out["v5_log_ing_max_linea"] = np.log(np.maximum(A["Emax"].values, 1e4) / lin)
    out["v5_share_principal"] = A["Emax"].values / np.maximum(Eh, 1.0)
    out["v5_log_ing_jefe_linea"] = np.log(np.maximum(A["Ejefe"].values, 1e4) / lin)
    for g in GRUPOS:
        out[f"v5_lz_{g}"] = np.log((A["E_" + g].values + 1e4) / lin)
        out[f"v5_share_{g}"] = A["E_" + g].values / np.maximum(Eh, 1.0)
    P0 = np.exp(A["logp0"].values)
    for rho in rhos:
        Vr = (1 - rho) * Vh + rho * A["sd"].values ** 2
        q = np.clip(1 - P0, 1e-12, 1)
        Ep = Eh / q
        Vp = np.maximum((Vr + Eh ** 2) / q - Ep ** 2, 0)
        s2 = np.log1p(Vp / np.maximum(Ep, 1.0) ** 2)
        mu = np.log(np.maximum(Ep, 1.0)) - s2 / 2
        s = np.sqrt(np.maximum(s2, 1e-6))
        lr = np.log(np.maximum(R, 1.0))
        gap = (lr - mu) / s
        prob = P0 + (1 - P0) * norm.cdf(gap)
        prob = np.where(R <= 0, 0.0, np.where(Eh <= 0, 1.0, prob))
        suf = "" if len(rhos) == 1 else f"_rho{rho:g}"
        out["v5_mu_log" + suf] = mu
        out["v5_sd_log" + suf] = s
        out["v5_gap_fw" + suf] = np.where(R <= 0, -10.0, np.clip(gap, -10, 10))
        out["v5_prob_pobre_fw" + suf] = prob
    return out


def elegir_rho_en_pliegues(y, A: pd.DataFrame, folds, rhos) -> tuple[np.ndarray, list]:
    """Elige ρ por AUC de la probabilidad estructural en el entrenamiento de cada pliegue y la aplica al
    pliegue retenido. Devuelve la probabilidad OOF y los ρ elegidos."""
    from . import metrics as M
    prob = np.zeros(len(y)); elegidos = []
    for tr, va in folds:
        aucs = [M.auc_roc(y[tr], A[f"v5_prob_pobre_fw_rho{r:g}"].values[tr]) for r in rhos]
        r = rhos[int(np.argmax(aucs))]; elegidos.append(r)
        prob[va] = A[f"v5_prob_pobre_fw_rho{r:g}"].values[va]
    return prob, elegidos


# ------------------------------------------------------------------------------ búsqueda de hiperparámetros
def busqueda_hiper(Xtr, yc, gate, f, configs: list[dict], hilos=14, k_val=0, semilla=7) -> pd.DataFrame:
    """Búsqueda acotada del LightGBM de la media del log-monto (un componente): entrena con los pliegues ≠ k_val
    (parada temprana interna) y mide R² y log-verosimilitud gaussiana en el pliegue k_val."""
    import lightgbm as lgb
    tr = np.flatnonzero((f != k_val) & gate & (yc > 0)); va = np.flatnonzero((f == k_val) & gate & (yc > 0))
    tope = float(np.quantile(yc[tr], 0.999))
    t = np.log(np.clip(yc, PISO, tope))
    filas = []
    for i, cfg in enumerate(configs):
        t0 = time.time()
        p = {**BASE, **cfg, "objective": "regression", "num_threads": hilos, "seed": semilla}
        m, n, i_fit, i_es = _fit(lgb, p, Xtr, t, tr, semilla, max_rounds=6000)
        pr = m.predict(Xtr.iloc[va], num_iteration=n)
        e = t[va] - pr
        s2 = ((t[i_es] - m.predict(Xtr.iloc[i_es], num_iteration=n)) ** 2).mean()
        ll = float(np.mean(-0.5 * np.log(2 * np.pi * s2) - e ** 2 / (2 * s2)))
        filas.append({"config": i, **cfg, "arboles": n, "R2_val": float(1 - (e ** 2).sum() / ((t[va] - t[va].mean()) ** 2).sum()),
                      "RMSE_val": float(np.sqrt((e ** 2).mean())), "loglik_val": ll, "tiempo_s": time.time() - t0})
        print(f"  config {i}: {cfg} → R² = {filas[-1]['R2_val']:.4f}, árboles = {n} ({filas[-1]['tiempo_s']:.0f}s)")
    return pd.DataFrame(filas)


# ------------------------------------------------------------------------------ etapa intermedia del hogar
PARAMS_HOGAR = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_child_samples=50,
                    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=255,
                    cat_smooth=20, max_cat_to_onehot=8, verbose=-1)


def etapa_hogar(X: pd.DataFrame, z: np.ndarray, folds, X_te: pd.DataFrame, hilos=14, semilla=11, params=None):
    """ẑ_h = E[z_h | x_h] (LightGBM L2) y σ̂_h = √(π/2)·E[|z − ẑ| | x_h] (LightGBM L2 sobre el residuo absoluto
    fuera de pliegue), ambos fuera de pliegue con los mismos pliegues; test con modelos de todo train."""
    import lightgbm as lgb
    p = {**PARAMS_HOGAR, **(params or {}), "num_threads": hilos, "seed": semilla}
    idx_all = np.arange(len(z))

    def _oof(y, pp):
        oof = np.zeros(len(y)); its = []
        for k, (tr, va) in enumerate(folds):
            m, n, _, _ = _fit(lgb, pp, X, y, tr, semilla + k, max_rounds=5000)
            oof[va] = m.predict(X.iloc[va], num_iteration=n); its.append(n)
        nf = int(math.ceil(1.1 * np.mean(its)))
        m, _, _, _ = _fit(lgb, pp, X, y, idx_all, semilla, n_fijo=nf)
        return oof, m.predict(X_te), its, m
    t0 = time.time()
    zh, zh_te, its, mz = _oof(z, p)
    ar = np.abs(z - zh)
    pa = {**p, "num_leaves": 31, "min_child_samples": 100}
    ea, ea_te, its_a, _ = _oof(ar, pa)
    k = math.sqrt(math.pi / 2)
    sig, sig_te = np.clip(k * ea, 0.05, None), np.clip(k * ea_te, 0.05, None)
    imp = pd.Series(mz.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    info = {"arboles_z": its, "arboles_abs": its_a, "R2_z_oof": float(1 - ((z - zh) ** 2).sum() / ((z - z.mean()) ** 2).sum()),
            "tiempo_s": time.time() - t0, "importancia": imp / imp.sum()}
    return zh, zh_te, sig, sig_te, info


# ------------------------------------------------------------------------------ calidad del ingreso predicho
def f1_max_ruido(z: np.ndarray, sigmas, semilla=1) -> pd.DataFrame:
    """Techo de F1 si se observara ẑ = z + N(0, σ²): F1 con el umbral óptimo (TRASPASO §3.7)."""
    rng = np.random.default_rng(semilla)
    y = (z < 0).astype(int); e0 = rng.standard_normal(len(z)); filas = []
    for s in sigmas:
        sc = -(z + s * e0)                                 # mayor = más pobre
        o = np.argsort(-sc); yy = y[o]
        tp = np.cumsum(yy); fp = np.cumsum(1 - yy); fn = y.sum() - tp
        f1 = (2 * tp / np.maximum(2 * tp + fp + fn, 1)).max()
        filas.append({"sigma": s, "F1_max": float(f1)})
    return pd.DataFrame(filas)


def sigma_implicito(f1: float, tabla: pd.DataFrame) -> float:
    """Invierte la curva F1_max(σ) (decreciente) por interpolación lineal."""
    t = tabla.sort_values("F1_max")
    return float(np.interp(f1, t["F1_max"].values, t["sigma"].values))


def calidad_prediccion(nombre: str, D: pd.DataFrame, log_ing_ug: np.ndarray | None = None, zhat: np.ndarray | None = None,
                       score_pobre: np.ndarray | None = None, mask=None) -> dict:
    """R² de log Ingtotug y de z, AUC de la brecha para Pobre y d.e. del error en z (y su versión robusta)."""
    from . import metrics as M
    m = np.ones(len(D), bool) if mask is None else np.asarray(mask, bool)
    z = D["z"].values; y = D["Pobre"].values.astype(int)
    out = {"prediccion": nombre}
    if log_ing_ug is not None:
        li = D["log_ingtotug"].values
        out["R2_log_ingtotug"] = float(1 - ((li - log_ing_ug)[m] ** 2).sum() / ((li[m] - li[m].mean()) ** 2).sum())
        brecha = log_ing_ug - D["log_ingreso_requerido"].values
        out["AUC_brecha"] = float(M.auc_roc(y[m], -brecha[m]))
        if zhat is None:
            zhat = np.log(np.maximum(np.exp(log_ing_ug) + D["arriendo_imputado_mese"].values, 1e3)
                          / (D["Npersug"].clip(lower=1).values * D["Lp"].values))
    if zhat is not None:
        e = (z - zhat)[m]
        out["R2_z"] = float(1 - (e ** 2).sum() / ((z[m] - z[m].mean()) ** 2).sum())
        out["sigma_error_z"] = float(e.std())
        out["sigma_robusta_z"] = float(1.4826 * np.median(np.abs(e - np.median(e))))
        out.setdefault("AUC_brecha", float(M.auc_roc(y[m], -zhat[m])))
    if score_pobre is not None:
        out["AUC_score"] = float(M.auc_roc(y[m], score_pobre[m]))
    return out
