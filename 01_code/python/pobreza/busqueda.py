r"""Búsqueda automática de variables (cuaderno 09).

1. **Importancia e interacciones SHAP.** Un LightGBM base sobre S3 da la importancia E|φ_j|
   (TreeSHAP; Lundberg y Lee, 2017). LightGBM no calcula valores de *interacción* SHAP, así que
   éstos se obtienen con XGBoost (`pred_interactions=True`; Lundberg et al., 2020) ajustado sobre
   las 30 variables principales, en una submuestra de hogares: E|φ_ij| para cada par.
2. **Candidatos** (especificaciones reproducibles, aplicadas igual a train y test):
   productos y cocientes de los pares con mayor interacción; per cápita (÷ Npersug);
   log/raíz/cuadrado de las continuas principales; rango percentil y desviación frente a la
   mediana del Dominio y del Depto (estadísticas de referencia calculadas SÓLO con las X de
   train, nunca con la variable respuesta); indicadores de umbral y conteos de carencias.
   Las transformaciones monótonas de UNA variable (log, raíz, cuadrado, rango global) no cambian
   las particiones de un árbol: se reportan pero no se evalúan por validación cruzada.
3. **Filtro en dos etapas.** (i) |ρ de Spearman| con el residuo OOF y − p̂ del modelo base
   (información que el modelo aún no usa); (ii) selección hacia adelante por lotes con
   validación cruzada de 5 pliegues (LightGBM rápido, lr 0,1, parada temprana interna): un lote se
   acepta si mejora el F1 anidado sin Bogotá o el log-loss en más de 1 EE pareado entre pliegues;
   dentro de un lote aceptado se descartan las variables cuya salida no empeora.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import metrics as M
from .ajuste import Conjunto, cv, resumen, CAT_TODAS

EPS = 1e-6


# ------------------------------------------------------------------------------ especificaciones
class _Cols(dict):
    """Diccionario perezoso de columnas (float) de D más las candidatas ya construidas."""
    def __init__(self, D, previas):
        super().__init__(); self.D, self.previas = D, previas

    def __missing__(self, k):
        if k in self.previas:
            return np.asarray(self.previas[k], float)
        return self.D[k].astype(float).values


def construir(spec: dict, Dref: pd.DataFrame, D: pd.DataFrame, previas: dict | None = None) -> np.ndarray:
    """Aplica una especificación a `D` usando `Dref` (train) para las estadísticas de referencia."""
    t, a, b = spec["tipo"], spec.get("a"), spec.get("b")
    if t == "expr":
        with np.errstate(all="ignore"):
            return np.asarray(eval(spec["expr"], {"np": np}, _Cols(D, previas or {})), float) * np.ones(len(D))
    if t == "recorte":
        lo, hi = np.nanquantile(Dref[a].astype(float), [.01, .99])
        return np.clip(D[a].astype(float).values, lo, hi)
    if t == "cuantiles":
        q = np.unique(np.nanquantile(Dref[a].astype(float), np.linspace(0, 1, 11)[1:-1]))
        return np.searchsorted(q, D[a].astype(float).values).astype(float)
    xa = D[a].astype(float).values if a else None
    xb = D[b].astype(float).values if b else None
    if t == "producto":
        return xa * xb
    if t == "cociente":
        return xa / (np.abs(xb) + spec["delta"])
    if t == "diferencia":
        return xa - xb
    if t == "per_capita":
        return xa / np.maximum(D["Npersug"].astype(float).values, 1)
    if t == "log":
        return np.log1p(np.maximum(xa, 0))
    if t == "raiz":
        return np.sqrt(np.maximum(xa, 0))
    if t == "cuadrado":
        return xa ** 2
    if t in ("rango_grupo", "desv_mediana_grupo"):
        g = spec["grupo"]
        ref = Dref[[g, a]].copy(); ref[a] = ref[a].astype(float)
        out = np.full(len(D), np.nan)
        gD = D[g].astype(str).values; gR = ref[g].astype(str).values
        for lev in np.unique(gD):
            r = np.sort(ref[a].values[(gR == lev)]); r = r[np.isfinite(r)]
            mk = gD == lev
            if len(r) == 0:
                continue
            if t == "rango_grupo":
                out[mk] = np.searchsorted(r, xa[mk], side="right") / len(r)
            else:
                out[mk] = xa[mk] - np.median(r)
        return out
    if t == "indicador":
        return eval(spec["expr"], {"np": np}, {c: D[c].astype(float).values for c in spec["vars"]}).astype(float)
    if t == "conteo":
        return np.sum([eval(e, {"np": np}, {c: D[c].astype(float).values for c in spec["vars"]}) for e in spec["exprs"]], 0).astype(float)
    raise ValueError(t)


def _monotona(spec):
    return spec["tipo"] in ("log", "raiz", "cuadrado", "recorte", "cuantiles")


# ------------------------------------------------------------------------------ candidatas guiadas por la literatura
_OC = "np.maximum(n_ocupado, 1)"
LITERATURA = [
    # (nombre, expresión, familia, motivación)
    ("x_escala_ocde", "1 + 0.5*np.maximum(Nper - n_menor12 - 1, 0) + 0.3*n_menor12", "a. Escalas de equivalencia",
     "Escala OCDE modificada (1; 0,5; 0,3): economías de escala en el consumo del hogar (Deaton, 1997)"),
    ("x_escala_raiz", "np.sqrt(Npersug)", "a. Escalas de equivalencia", "Escala raíz cuadrada del tamaño (OCDE, 2013)"),
    ("x_lz_esp_ocde", "log_ing_esperado_ug - np.log(x_escala_ocde) - np.log(Lp)", "a. Escalas de equivalencia",
     "Ingreso esperado (1.ª etapa) por adulto equivalente frente a la línea"),
    ("x_lz_esp_raiz", "log_ing_esperado_ug - 0.5*np.log(Npersug) - np.log(Lp)", "a. Escalas de equivalencia",
     "Ingreso esperado con escala raíz frente a la línea"),
    ("x_log_req_ocde", "log_ingreso_requerido - np.log(x_escala_ocde)", "a. Escalas de equivalencia",
     "Ingreso requerido por adulto equivalente"),
    ("x_log_arr_ocde_linea", "np.log(np.maximum(arriendo, 1e4)/(Lp*x_escala_ocde))", "a. Escalas de equivalencia",
     "Arriendo (valor de uso de la vivienda) por adulto equivalente frente a la línea: ingreso permanente (Friedman, 1957)"),
    ("x_ing_esp_max_linea", "log_ing_esp_max - np.log(Lp*np.maximum(Npersug, 1))", "a. Escalas de equivalencia",
     "Ingreso esperado del principal perceptor frente a la línea del hogar"),
    ("x_menores_por_adulto", "n_menor18/np.maximum(n_edad18_64, 1)", "b. Dependencia",
     "Razón de dependencia juvenil (ciclo de vida; Modigliani y Brumberg, 1954)"),
    ("x_mayores_por_adulto", "n_mayor65/np.maximum(n_edad18_64, 1)", "b. Dependencia", "Razón de dependencia de vejez"),
    ("x_desocupados_por_pet", "n_desocupado/np.maximum(n_pet, 1)", "b. Dependencia", "Desempleo dentro del hogar"),
    ("x_inactivos_por_pet", "n_inactivo/np.maximum(n_pet, 1)", "b. Dependencia", "Inactividad dentro del hogar"),
    ("x_ocupados_por_ae", "n_ocupado/x_escala_ocde", "b. Dependencia", "Perceptores por adulto equivalente"),
    ("x_perceptores_esp_pc", "n_perceptores_esp_lp/np.maximum(Npersug, 1)", "b. Dependencia",
     "Perceptores esperados (1.ª etapa) por miembro de la unidad de gasto"),
    ("x_brecha_educ_jefe_pareja", "jefe_anios_educ - pareja_educ", "c. Capital humano",
     "Brecha educativa jefe–pareja (emparejamiento selectivo y poder de negociación)"),
    ("x_educ_max_x_cabecera", "educ_max_adultos*cabecera", "c. Capital humano",
     "Retornos a la educación distintos en zona urbana y rural (Mincer, 1974)"),
    ("x_educ_jefe_x_cabecera", "jefe_anios_educ*cabecera", "c. Capital humano", "Educación del jefe × urbano"),
    ("x_educ_max_x_formal", "educ_max_adultos*share_formal_ocupados", "c. Capital humano",
     "Educación × formalidad: el retorno educativo se materializa en el empleo formal (Perry et al., 2007)"),
    ("x_exp_pot_jefe", "np.maximum(jefe_edad - jefe_anios_educ - 6, 0)", "c. Capital humano",
     "Experiencia potencial del jefe (Mincer, 1974)"),
    ("x_log_mincer_jefe", "(0.10*jefe_anios_educ + 0.03*x_exp_pot_jefe - 0.0005*x_exp_pot_jefe**2)*jefe_ocupado", "c. Capital humano",
     "Log-salario de Mincer del jefe ocupado (escolaridad, experiencia y su cuadrado)"),
    ("x_educ_ocup_x_ocupados_pc", "educ_media_ocupados*ocupados_pc", "c. Capital humano",
     "Capital humano de los perceptores × perceptores por miembro"),
    ("x_share_contributivo", "n_salud_contributivo/np.maximum(Nper, 1)", "d. Calidad del empleo",
     "Afiliación contributiva: empleo formal o capacidad de pago"),
    ("x_share_asalariados", f"n_asalariado/{_OC}", "d. Calidad del empleo", "Asalariados frente a cuenta propia (Maloney, 2004)"),
    ("x_share_cuenta_propia", f"n_pos_cuenta_propia/{_OC}", "d. Calidad del empleo", "Cuenta propia: informalidad e ingresos volátiles (Fields, 2011)"),
    ("x_horas_por_ocupado", f"n_horas/{_OC}", "d. Calidad del empleo", "Horas efectivas por ocupado (intensidad laboral)"),
    ("x_share_subempleo", f"n_subempleo_horas/{_OC}", "d. Calidad del empleo", "Subempleo por horas"),
    ("x_share_empresa_grande", f"n_empresa_grande/{_OC}", "d. Calidad del empleo", "Tamaño de empresa: productividad y formalidad"),
    ("x_share_aux_transporte", f"n_subsidio_transporte/{_OC}", "d. Calidad del empleo",
     "Auxilio de transporte = salario ≤ 2 SMMLV (señal legal del nivel salarial)"),
    ("x_share_prima", f"n_prima_servicios/{_OC}", "d. Calidad del empleo", "Prima de servicios: prestaciones de ley (formalidad)"),
    ("x_formales_por_ae", "n_cotiza_pension/x_escala_ocde", "d. Calidad del empleo", "Cotizantes por adulto equivalente"),
    ("x_arriendo_cuarto_linea", "np.log(np.maximum(arriendo, 1e4)/(np.maximum(cuartos, 1)*Lp))", "e. Vivienda",
     "Arriendo por cuarto relativo a la línea: calidad de la vivienda (ley de Schwabe)"),
    ("x_arriendo_persona_linea", "np.log(np.maximum(arriendo, 1e4)/(np.maximum(Nper, 1)*Lp))", "e. Vivienda",
     "Arriendo por persona (todas las personas) relativo a la línea"),
    ("x_propia_x_arriendo", "log_arriendo_sobre_linea*ten_propia_pagada", "e. Vivienda", "Tenencia propia × valor de la vivienda (activo)"),
    ("x_hacinamiento_x_cabecera", "personas_por_cuarto*cabecera", "e. Vivienda", "Hacinamiento urbano vs. rural"),
    ("x_dormitorios_por_cuarto", "dormitorios/np.maximum(cuartos, 1)", "e. Vivienda", "Composición de la vivienda (cuartos no dormitorio)"),
    ("x_jefa_x_menores", "jefe_mujer*n_menor18", "f. Composición", "Jefatura femenina con menores (Buvinić y Gupta, 1997)"),
    ("x_monoparental", "(1 - np.nan_to_num(hay_pareja))*(n_menor18 > 0)", "f. Composición", "Hogar monoparental"),
    ("x_jefa_monoparental", "jefe_mujer*(1 - np.nan_to_num(hay_pareja))*(n_menor18 > 0)", "f. Composición", "Monoparental con jefa"),
    ("x_mayores_sin_pension", "np.maximum(n_mayor65 - n_recibe_pension_jubilacion, 0)", "f. Composición",
     "Adultos mayores sin pensión (vulnerabilidad de vejez)"),
    ("x_unipersonal_mayor", "(Nper == 1)*(n_mayor65 >= 1)", "f. Composición", "Adulto mayor que vive solo"),
    ("x_menores5_por_adulto", "n_menor5/np.maximum(n_edad18_64, 1)", "f. Composición", "Primera infancia frente a adultos (cuidado)"),
    ("x_subsidiado_x_menores", "share_subsidiado*n_menor18", "g. Protección social",
     "Régimen subsidiado (proxy del SISBÉN) × menores: focalización de programas"),
    ("x_subsidiado_x_jefa", "jefe_salud_subsidiado*jefe_mujer", "g. Protección social", "Jefa en régimen subsidiado"),
    ("x_transferencias_pc", "(n_recibe_ayuda_instituciones + n_recibe_ayuda_otros_hogares_pais + n_recibe_remesas_exterior)/np.maximum(Npersug, 1)",
     "g. Protección social", "Ayudas institucionales, de otros hogares y remesas por miembro"),
    ("x_subsidiado_sin_formales", "share_subsidiado*(n_cotiza_pension == 0)", "g. Protección social",
     "Subsidiado y sin cotizantes: exclusión del aseguramiento contributivo"),
    ("x_brecha_esp_x_cabecera", "brecha_esperada*cabecera", "h. Territorio", "Brecha esperada × cabecera (mercados laborales urbanos vs. rurales)"),
    ("x_ocupados_pc_x_cabecera", "ocupados_pc*cabecera", "h. Territorio", "Perceptores × cabecera"),
    ("x_formales_pc_x_cabecera", "formales_pc*cabecera", "h. Territorio", "Formalidad × cabecera"),
    ("x_mincer_req_x_cabecera", "mincer_sobre_requerido*cabecera", "h. Territorio", "Capacidad laboral vs. requerimiento × cabecera"),
]
# Posición del hogar en la distribución de su dominio (estadísticas sólo de X de train, dentro del pliegue)
LIT_GRUPO = [
    ("brecha_esperada", "rango_grupo", "Dominio", "Posición del hogar en la distribución de la brecha esperada de su dominio (Elbers, Lanjouw y Lanjouw, 2003)"),
    ("brecha_esperada", "desv_mediana_grupo", "Dominio", "Brecha esperada frente a la mediana del dominio"),
    ("lz_ingreso_esperado", "rango_grupo", "Dominio", "Rango del ingreso esperado en el dominio"),
    ("educ_max_adultos", "desv_mediana_grupo", "Dominio", "Educación relativa al dominio"),
    ("log_arriendo_sobre_linea", "desv_mediana_grupo", "Dominio", "Valor de la vivienda relativo al dominio"),
    ("mincer_sobre_requerido", "rango_grupo", "Dominio", "Capacidad laboral relativa al dominio"),
    ("ocupados_pc", "desv_mediana_grupo", "Dominio", "Perceptores relativos al dominio"),
    ("log_arriendo_sobre_linea", "desv_mediana_grupo", "Depto", "Valor de la vivienda relativo al departamento"),
]
GRUPALES = ("rango_grupo", "desv_mediana_grupo")


def candidatos_literatura(dat) -> list[dict]:
    """Candidatas guiadas por la literatura de focalización y de determinantes del ingreso (familias a–i)."""
    D = dat.D; gen = set(dat.GEN); specs = []; previas = set()
    for nom, e, fam, mot in LITERATURA:
        usa = any(g in e for g in gen) or any(p in e for p in previas if p in {s["nombre"] for s in specs if s["usa_generadas"]})
        specs.append(dict(nombre=nom, tipo="expr", expr=e, familia=fam, motivacion=mot, origen="literatura", usa_generadas=bool(usa)))
        previas.add(nom)
    for a, tp, g, mot in LIT_GRUPO:
        specs.append(dict(nombre=f"x_{a}__{'rango' if tp == 'rango_grupo' else 'desv'}_{g}", tipo=tp, a=a, grupo=g,
                          familia="h. Territorio", motivacion=mot, origen="literatura", usa_generadas=a in gen))
    for c in ["brecha_esperada", "mincer_sobre_requerido", "log_arriendo_sobre_linea", "ocupados_pc"]:
        specs.append(dict(nombre=f"x_{c}__recorte", tipo="recorte", a=c, familia="i. Transformaciones", origen="literatura",
                          motivacion="Recorte de colas al 1 %–99 % (monótona: invariante para árboles)", usa_generadas=c in gen))
        specs.append(dict(nombre=f"x_{c}__cuantiles", tipo="cuantiles", a=c, familia="i. Transformaciones", origen="literatura",
                          motivacion="Binning en deciles (monótona: invariante para árboles)", usa_generadas=c in gen))
    for s in specs:
        s["monotona_univariada"] = _monotona(s)
    return [s for s in specs if s["tipo"] != "expr" or all(
        (v in D.columns) or v.startswith("x_") or v == "np" for v in __import__("re").findall(r"[A-Za-z_][A-Za-z0-9_]*", s["expr"])
        if not v.startswith(("maximum", "log", "sqrt", "nan_to_num")))]


def fijar_motivacion_automatica(specs):
    """Familia y motivación de las candidatas automáticas (pares SHAP, etc.)."""
    for s in specs:
        s.setdefault("familia", {"par SHAP": "i. Pares de mayor interacción SHAP", "par SHAP (logs)": "i. Pares de mayor interacción SHAP",
                                 "per cápita": "a. Escalas de equivalencia", "transformación monótona": "i. Transformaciones",
                                 "indicador de umbral": "f. Composición", "conteo de carencias": "g. Protección social"}.get(
            s["origen"], "h. Territorio"))
        s.setdefault("motivacion", {
            "producto": f"Interacción SHAP alta entre {s.get('a')} y {s.get('b')}: producto explícito",
            "cociente": f"Interacción SHAP alta: cociente {s.get('a')}/{s.get('b')} (los árboles aproximan cocientes con muchas particiones)",
            "diferencia": f"Diferencia de logaritmos = log del cociente {s.get('a')}/{s.get('b')}",
            "per_capita": "Per cápita (÷ Npersug): la línea es per cápita",
            "rango_grupo": f"Posición relativa en {s.get('grupo')} (estadística de X de train)",
            "desv_mediana_grupo": f"Desviación de la mediana de {s.get('grupo')} (estadística de X de train)",
            "indicador": "Umbral con interpretación (IPM / reglas de focalización)",
            "conteo": "Conteo de carencias tipo IPM (Angulo, Díaz y Pardo, 2016)",
        }.get(s["tipo"], "Transformación monótona (invariante para árboles)"))
    return specs


# ------------------------------------------------------------------------------ SHAP
def importancia_e_interacciones(dat, cols, params_lgb, n_shap=20000, top_vars=30, top_pares=25, hilos=20, semilla=1):
    """Importancia SHAP (LightGBM, todo train) y E|φ_ij| (XGBoost sobre las `top_vars` principales)."""
    import lightgbm as lgb
    import xgboost as xgb
    from .modelos_ml import _split_es
    X, y = dat.X, dat.y
    fit_m, es_m = _split_es(y, 5)
    p = {**params_lgb, "num_threads": hilos, "verbose": -1, "seed": semilla}
    dtr = lgb.Dataset(X.loc[fit_m, cols], y[fit_m]); des = lgb.Dataset(X.loc[es_m, cols], y[es_m], reference=dtr)
    mod = lgb.train(p, dtr, 10000, valid_sets=[des], callbacks=[lgb.early_stopping(100, verbose=False)])
    idx = np.random.default_rng(semilla).choice(len(y), min(n_shap, len(y)), replace=False)
    phi = mod.predict(X.iloc[idx][cols], num_iteration=mod.best_iteration, pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(phi).mean(0), index=cols).sort_values(ascending=False)
    top = list(imp.index[:top_vars])
    px = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", learning_rate=0.1, max_depth=6,
              min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, max_cat_to_onehot=8,
              nthread=hilos, seed=semilla)
    dtr = xgb.DMatrix(X.loc[fit_m, top], y[fit_m], enable_categorical=True)
    des = xgb.DMatrix(X.loc[es_m, top], y[es_m], enable_categorical=True)
    mx = xgb.train(px, dtr, 5000, evals=[(des, "es")], early_stopping_rounds=100, verbose_eval=False)
    idx2 = idx[:min(len(idx), 10000)]
    inter = mx.predict(xgb.DMatrix(X.iloc[idx2][top], enable_categorical=True), pred_interactions=True,
                       iteration_range=(0, mx.best_iteration + 1))[:, :-1, :-1]
    A = np.abs(inter).mean(0)
    filas = [{"a": top[i], "b": top[j], "interaccion_media_abs": 2 * A[i, j]}
             for i in range(len(top)) for j in range(i + 1, len(top))]
    pares = pd.DataFrame(filas).sort_values("interaccion_media_abs", ascending=False).reset_index(drop=True)
    return imp, pares.head(top_pares), pares, A, top


# ------------------------------------------------------------------------------ candidatos
def generar_candidatos(dat, imp: pd.Series, pares: pd.DataFrame, top_vars=30) -> list[dict]:
    D = dat.D
    num = lambda c: c not in CAT_TODAS
    cont = [c for c in imp.index if num(c) and D[c].nunique() > 10][:top_vars]
    specs = []

    def add(**s):
        s["nombre"] = s["nombre"][:60]
        if s["nombre"] not in {x["nombre"] for x in specs} and s["nombre"] not in D.columns:
            specs.append(s)

    for _, r in pares.iterrows():
        a, b = r.a, r.b
        if not (num(a) and num(b)):
            continue
        add(nombre=f"x_{a}__por__{b}", tipo="producto", a=a, b=b, origen="par SHAP")
        for u, v in ((a, b), (b, a)):
            if D[v].min() >= 0 and D[v].nunique() > 2:
                delta = float(max(np.nanquantile(D[v], .05), 1e-3 * (np.nanstd(D[v]) + EPS), EPS))
                add(nombre=f"x_{u}__sobre__{v}", tipo="cociente", a=u, b=v, delta=delta, origen="par SHAP")
        if a.startswith("log") and b.startswith("log"):
            add(nombre=f"x_{a}__menos__{b}", tipo="diferencia", a=a, b=b, origen="par SHAP (logs)")
    for c in cont:
        if (c.startswith("n_") or c in ("cuartos", "dormitorios", "arriendo", "arriendo_imputado_mese")) and c != "Npersug":
            add(nombre=f"x_{c}__pc", tipo="per_capita", a=c, origen="per cápita")
        if D[c].min() >= 0:
            add(nombre=f"x_log_{c}", tipo="log", a=c, origen="transformación monótona")
            add(nombre=f"x_raiz_{c}", tipo="raiz", a=c, origen="transformación monótona")
        add(nombre=f"x_{c}__cuad", tipo="cuadrado", a=c, origen="transformación monótona")
    for c in cont[:20]:
        for g in ("Dominio", "Depto"):
            add(nombre=f"x_{c}__rango_{g}", tipo="rango_grupo", a=c, grupo=g, origen=f"posición relativa en {g}")
            add(nombre=f"x_{c}__desv_{g}", tipo="desv_mediana_grupo", a=c, grupo=g, origen=f"desviación de la mediana de {g}")
    ind = [
        ("x_ind_requerido_no_positivo", "ingreso_requerido <= 0", ["ingreso_requerido"]),
        ("x_ind_personas_cuarto_gt2", "personas_por_cuarto > 2", ["personas_por_cuarto"]),
        ("x_ind_jefe_primaria_o_menos", "jefe_anios_educ <= 5", ["jefe_anios_educ"]),
        ("x_ind_sin_cotizantes_con_ocupados", "(n_ocupado > 0) & (n_cotiza_pension == 0)", ["n_ocupado", "n_cotiza_pension"]),
        ("x_ind_un_ocupado_hogar_grande", "(n_ocupado <= 1) & (Npersug >= 4)", ["n_ocupado", "Npersug"]),
        ("x_ind_mincer_bajo_requerido", "mincer_sobre_requerido < 0", ["mincer_sobre_requerido"]),
        ("x_ind_todos_subsidiados", "share_subsidiado >= 0.999", ["share_subsidiado"]),
        ("x_ind_arriendo_bajo_linea", "arriendo_sobre_linea < 0.15", ["arriendo_sobre_linea"]),
    ]
    for nom, e, vs in ind:
        if all(v in D for v in vs):
            add(nombre=nom, tipo="indicador", expr=e, vars=vs, origen="indicador de umbral")
    exprs = ["hacinamiento_critico > 0", "jefe_anios_educ <= 5", "n_cotiza_pension == 0", "share_subsidiado >= 0.5",
             "ocupados_pc < 0.34", "dependencia >= 2", "ten_arriendo > 0", "n_desocupado > 0"]
    vs = ["hacinamiento_critico", "jefe_anios_educ", "n_cotiza_pension", "share_subsidiado", "ocupados_pc", "dependencia",
          "ten_arriendo", "n_desocupado"]
    if all(v in D for v in vs):
        add(nombre="x_conteo_carencias", tipo="conteo", exprs=exprs, vars=vs, origen="conteo de carencias")
    gen = set(dat.GEN)
    for s in specs:
        s["usa_generadas"] = bool({s.get("a"), s.get("b")} & gen)
        s["monotona_univariada"] = _monotona(s)
    return specs


def materializar(specs, Dref, D) -> pd.DataFrame:
    cols = {}
    for s in specs:
        v = construir(s, Dref, D, cols)
        v = np.where(np.isfinite(v), v, np.nan)
        cols[s["nombre"]] = v.astype("float32")
    return pd.DataFrame(cols, index=D.index)


# ------------------------------------------------------------------------------ filtro 1
def filtro_residuo(C: pd.DataFrame, residuo: np.ndarray, specs) -> pd.DataFrame:
    """|ρ de Spearman| entre cada candidato y el residuo OOF del modelo base."""
    r = pd.Series(residuo).rank().values
    filas = []
    for s in specs:
        x = C[s["nombre"]]
        ok = x.notna().values
        if x.nunique() < 2 or ok.mean() < .5:
            rho = 0.0
        else:
            xr = x[ok].rank().values
            rho = float(np.corrcoef(xr, pd.Series(residuo[ok]).rank().values)[0, 1]) if ok.sum() > 10 else 0.0
        filas.append({"variable": s["nombre"], "familia": s.get("familia", ""), "motivacion": s.get("motivacion", ""),
                      "tipo": s["tipo"], "origen": s["origen"], "usa_generadas": s["usa_generadas"],
                      "monotona_univariada": s["monotona_univariada"], "rho_residuo": rho, "abs_rho": abs(rho),
                      "pct_na": float(1 - ok.mean())})
    return pd.DataFrame(filas).sort_values("abs_rho", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------------------ filtro 2
def transformador_grupal(specs, D, folds):
    """Estadísticas de grupo (rango y desviación frente a la mediana del Dominio/Depto) recalculadas
    DENTRO de cada pliegue: la referencia son sólo las X de los hogares de entrenamiento del pliegue."""
    g = [s for s in specs if s["tipo"] in GRUPALES]
    if not g:
        return None
    por_pl = [materializar(g, D.iloc[tr], D) for tr, _ in folds]

    def transformar(k, tr, X):
        cols = [c for c in por_pl[k].columns if c in X.columns]
        if not cols:
            return X
        X2 = X.copy()
        for c in cols:
            X2[c] = por_pl[k][c].values
        return X2
    return transformar


def _cv_eval(motor, dat, cols, params, transformar=None):
    c = Conjunto("tmp", list(cols))
    r = cv(motor, dat.X, dat.y, dat.folds, params, c, transformar=transformar)
    res = resumen(dat.y, r["oof"], dat.folds, dat.sin_bog)
    f1k = np.array([res[f"F1_pliegue_{k}"] for k in range(len(dat.folds))])
    llk = np.array([M.log_loss(dat.y[va], r["oof"][va]) for _, va in dat.folds])
    return res, f1k, llk, r["tiempo_s"]


def _mejora(f1k_new, llk_new, f1k_ref, llk_ref):
    """Mejora pareada por pliegue: (ΔF1, EE, Δlog-loss, EE, ¿acepta?). Se acepta si una métrica mejora más
    de 1 EE y la otra no empeora más de 1 EE."""
    d1 = f1k_new - f1k_ref; d2 = llk_ref - llk_new           # positivos = mejora
    e1 = d1.std(ddof=1) / np.sqrt(len(d1)); e2 = d2.std(ddof=1) / np.sqrt(len(d2))
    acepta = (d1.mean() > max(e1, 0) and d2.mean() >= -e2) or (d2.mean() > max(e2, 0) and d1.mean() >= -e1)
    return d1.mean(), e1, d2.mean(), e2, bool(acepta)


def cribado_individual(motor, dat, base_cols, candidatos, params, transformar=None, limite_s=None):
    """Ganancia de cada candidata añadida SOLA a la base (CV de 5 pliegues, cambios pareados por pliegue)."""
    t0 = time.time()
    res, f1k, llk, ts = _cv_eval(motor, dat, base_cols, params, transformar)
    filas = []
    for i, v in enumerate(candidatos):
        if limite_s and time.time() - t0 > limite_s:
            print(f"  límite de tiempo: {i} de {len(candidatos)} candidatas evaluadas"); break
        r2, f2, l2, ts = _cv_eval(motor, dat, list(base_cols) + [v], params, transformar)
        d1, e1, d2, e2, ok = _mejora(f2, l2, f1k, llk)
        z = d1 / max(e1, 1e-9) + d2 / max(e2, 1e-9)
        filas.append({"variable": v, "F1": r2["F1_anidado_sinBOG"], "LogLoss": r2["LogLoss"], "dF1": d1, "EE_dF1": e1,
                      "dLogLoss": d2, "EE_dLL": e2, "z_conjunto": z, "supera_1EE": ok, "tiempo_s": ts})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(candidatos)} candidatas ({(time.time() - t0) / 60:.1f} min)")
    return pd.DataFrame(filas), res


def adelante_por_lotes(motor, dat, base_cols, candidatos, params, lote=5, limite_s=None, transformar=None):
    """Selección hacia adelante por lotes con CV de 5 pliegues y regla de 1 EE pareado; poda dentro del lote."""
    t0 = time.time()
    actual = list(base_cols)
    res, f1k, llk, ts = _cv_eval(motor, dat, actual, params, transformar)
    ini = res
    filas = [{"paso": 0, "accion": "base S3", "variables": "", "n_vars": len(actual), "F1": res["F1_anidado_sinBOG"],
              "LogLoss": res["LogLoss"], "dF1": 0, "EE_dF1": 0, "dLogLoss": 0, "EE_dLL": 0, "aceptado": True, "tiempo_s": ts}]
    print(f"  base: F1={res['F1_anidado_sinBOG']:.4f} LL={res['LogLoss']:.5f} ({ts:.0f}s)")
    ganadoras = []
    for i in range(0, len(candidatos), lote):
        if limite_s and time.time() - t0 > limite_s:
            print("  límite de tiempo alcanzado"); break
        L = candidatos[i:i + lote]
        r2, f2, l2, ts = _cv_eval(motor, dat, actual + L, params, transformar)
        d1, e1, d2, e2, ok = _mejora(f2, l2, f1k, llk)
        filas.append({"paso": len(filas), "accion": "añadir lote", "variables": ", ".join(L), "n_vars": len(actual) + len(L),
                      "F1": r2["F1_anidado_sinBOG"], "LogLoss": r2["LogLoss"], "dF1": d1, "EE_dF1": e1, "dLogLoss": d2,
                      "EE_dLL": e2, "aceptado": ok, "tiempo_s": ts})
        print(f"  lote {i // lote + 1}: ΔF1={d1:+.4f} (EE {e1:.4f}) Δll={d2:+.5f} (EE {e2:.5f}) → {'ACEPTA' if ok else 'rechaza'} ({ts:.0f}s)")
        if not ok:
            continue
        quedan = list(L)
        if len(L) > 1:
            for v in L:
                prueba = [c for c in actual + quedan if c != v]
                r3, f3, l3, ts = _cv_eval(motor, dat, prueba, params, transformar)
                dd1, ee1, dd2, ee2, _ = _mejora(f2, l2, f3, l3)   # aporte de v frente a no tenerla
                aporta = (dd1 > 0 and dd2 > -ee2) or (dd2 > 0 and dd1 > -ee1)
                filas.append({"paso": len(filas), "accion": f"poda: quitar {v}", "variables": v, "n_vars": len(prueba),
                              "F1": r3["F1_anidado_sinBOG"], "LogLoss": r3["LogLoss"], "dF1": dd1, "EE_dF1": ee1,
                              "dLogLoss": dd2, "EE_dLL": ee2, "aceptado": aporta, "tiempo_s": ts})
                if not aporta:
                    quedan.remove(v); r2, f2, l2 = r3, f3, l3
        actual += quedan; ganadoras += quedan; res, f1k, llk = r2, f2, l2
        print(f"    quedan {quedan}; F1={res['F1_anidado_sinBOG']:.4f}")
    return ganadoras, pd.DataFrame(filas), ini, res