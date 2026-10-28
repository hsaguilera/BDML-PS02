r"""Versión 6 «permitida» de las variables (cuaderno 02c).

Decisión del usuario (19-oct-2026): **no asumir riesgos**. El enunciado pide la ruta de clasificación y Kaggle
prohíbe datos externos, así que en esta versión quedan **prohibidos**:

* toda variable que provenga de un modelo entrenado con ingresos (primera etapa v3/v5: `brecha_esperada`,
  `lz_ingreso_esperado`, `log_ing_esperado_ug`, todas las `v5_*`, las combinaciones del 09 que las usan) y cualquier
  uso de columnas de ingreso (montos de persona, `Ingtot*`, `Ingpcug`, `Npobres`, `Indigente`), salvo la etiqueta
  `Pobre`;
* las constantes externas (salario mínimo, auxilio de transporte, coeficientes de retornos de la literatura, escalas
  de equivalencia, tasas de cambio…): por eso salen el índice de Mincer de `features.py` (ρ = 0,10; 0,03; 0,0005;
  horas/48) y sus derivadas.

Sí se permite: cualquier transformación de columnas comunes a train y test; el umbral observable
R_h = Npersug·Lp − A_h (arriendo imputado reconstruido con P5090, P5100 y P5130, regla exacta de la GUIA §4.1);
la codificación de categóricas con la etiqueta `Pobre` **fuera de pliegue** (con suavizado y calculada sólo con el
pliegue de entrenamiento) y los clasificadores de la lista del enunciado.

Garantía por construcción: `construir_v6` recibe las tablas crudas completas pero lo PRIMERO que hace es quedarse
sólo con las columnas comunes (`HOG_COMUNES`, `PER_COMUNES`); ninguna función posterior ve una columna de ingreso.
La etiqueta sólo entra en la codificación de objetivo (`Transformador`, `te_global`).

Familias nuevas (prefijo del nombre):
  I. Umbral observable (`u_`, más `ingreso_requerido`, `Lp`, `Li`)
  J. Capacidad de generar ingreso frente al umbral, sin montos (`c_`)
  K. Señales de nivel salarial del cuestionario (`s_`)
  L. Composición y vivienda (`v_`)
  M. Codificación de objetivo fuera de pliegue (`te_`)
  N. Interacciones y transformaciones guiadas por SHAP y posiciones relativas en el dominio (`x_`)
"""
from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from . import features as F

# ------------------------------------------------------------------------------ columnas
HOG_COMUNES = ["id", "Clase", "Dominio", "P5000", "P5010", "P5090", "P5100", "P5130", "P5140", "Nper", "Npersug",
               "Li", "Lp", "Fex_c", "Depto", "Fex_dpto"]
PER_COMUNES = list(F.COLS_PERSONA_COMUNES)
ETIQUETA = "Pobre"

# Prohibidas explícitas (además de TODAS las columnas de persona que sólo están en train, que se listan con
# `prohibidas_persona(tp, tep)`).
HOG_PROHIBIDAS = ["Ingtotug", "Ingtotugarr", "Ingpcug", "Indigente", "Npobres", "Nindigentes"]
OBJETIVO_DERIVADAS = ["z", "log_ingtotug", "brecha", "Indigente"]          # columnas objetivo de las tablas v3–v5
CONSTANTES_EXTERNAS = {
    "n_mincer_ocupado": "Índice de Mincer con coeficientes de la literatura (ρ = 0,10; β1 = 0,03; β2 = 0,0005) y horas/48",
    "log_mincer_pc_linea": "Derivada del índice de Mincer (coeficientes externos)",
    "mincer_sobre_requerido": "Derivada del índice de Mincer (coeficientes externos)",
    "x_escala_ocde": "Escala de equivalencia OCDE modificada (coeficientes 0,5 y 0,3 externos)",
}
# patrones de nombre que delatan una variable generada o de ingreso (auditoría de nombres)
PATRONES_PROHIBIDOS = [r"^v5_", r"^lz_", r"brecha_esp", r"ing_esp", r"log_ing_esp", r"perceptores_esp",
                       r"share_principal", r"mincer", r"ocde", r"ingtot", r"ingpcug", r"indigente", r"npobres",
                       r"nindigentes", r"^z$", r"^brecha$", r"^log_ingtotug$", r"estrato", r"cclasnr",
                       r"\bimpa\b", r"\bisa\b", r"\bie\b", r"\bimdi\b", r"iof\d"]

FAMILIAS = {
    "I": "I. Umbral observable",
    "J": "J. Capacidad de ingreso frente al umbral (sin montos)",
    "K": "K. Señales de nivel salarial del cuestionario",
    "L": "L. Composición y vivienda",
    "M": "M. Codificación de objetivo fuera de pliegue",
    "N": "N. Interacciones SHAP y posiciones relativas",
}

# Codificación de objetivo (target encoding) fuera de pliegue: nombre → columnas que forman la clave
TE_ESPECIFICACIONES = {
    "te_oficio_jefe": ["jefe_oficio"],
    "te_oficio_principal": ["principal_oficio"],
    "te_dominio_tenencia": ["Dominio", "tenencia"],
    "te_depto_clase": ["Depto", "cabecera"],
    "te_posicion_jefe": ["jefe_posicion"],
}
TE_SUAVIZADO = 20.0

# Posiciones relativas en el dominio (estadísticas sólo de las X de entrenamiento; ver busqueda.construir)
GRUPO_BASE = [  # las seis variables del 09 que no usan generadas (ya estaban en «S3 + búsqueda 09 sin generadas»)
    dict(nombre="x_share_arriendo_imputado__desv_Depto", tipo="desv_mediana_grupo", a="share_arriendo_imputado", grupo="Depto"),
    dict(nombre="x_share_arriendo_imputado__rango_Depto", tipo="rango_grupo", a="share_arriendo_imputado", grupo="Depto"),
    dict(nombre="x_arriendo_imputado_mese__desv_Dominio", tipo="desv_mediana_grupo", a="arriendo_imputado_mese", grupo="Dominio"),
    dict(nombre="x_share_arriendo_imputado__desv_Dominio", tipo="desv_mediana_grupo", a="share_arriendo_imputado", grupo="Dominio"),
    dict(nombre="x_requerido_sobre_linea__desv_Dominio", tipo="desv_mediana_grupo", a="requerido_sobre_linea", grupo="Dominio"),
    dict(nombre="x_arriendo_imputado_mese__rango_Dominio", tipo="rango_grupo", a="arriendo_imputado_mese", grupo="Dominio"),
]
GRUPO_CAPACIDAD = [  # rango de los índices de capacidad dentro del dominio (familia J)
    dict(nombre="c_hs_linea__rango_Dominio", tipo="rango_grupo", a="c_hs_linea", grupo="Dominio"),
    dict(nombre="c_horas_linea__rango_Dominio", tipo="rango_grupo", a="c_horas_linea", grupo="Dominio"),
    dict(nombre="c_perceptores_linea__rango_Dominio", tipo="rango_grupo", a="c_perceptores_linea", grupo="Dominio"),
]


def comunes(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Sólo las columnas comunes a train y test (falla si falta alguna)."""
    falt = [c for c in cols if c not in df.columns]
    assert not falt, f"faltan columnas comunes: {falt}"
    return df[cols].copy()


def prohibidas_persona(tp: pd.DataFrame, tep: pd.DataFrame) -> list:
    """Columnas de persona que sólo están en train (montos de ingreso, Estrato1, Cclasnr*, …): prohibidas."""
    return [c for c in tp.columns if c not in tep.columns]


# ------------------------------------------------------------------------------ persona
def _es(s, v=1.0):
    return (s == v).astype("float32")


def variables_persona_v6(p: pd.DataFrame) -> pd.DataFrame:
    """Indicadores individuales nuevos (sólo columnas comunes). Las preguntas de prestaciones se hacen a los
    asalariados; las de «¿recibió…?» a todos los mayores de cierta edad (P7495…P7510s7)."""
    d = pd.DataFrame(index=p.index)
    d["id"] = p["id"].values
    edad = p["P6040"].astype(float)
    S = F.anios_educacion(p["P6210"], p["P6210s1"])
    oc = (p["Oc"] == 1)
    d["ug"] = (~p["P6050"].isin([6, 7, 8])).astype("float32")          # unidad de gasto (MESE)
    d["jefe"] = _es(p["P6050"], 1)
    d["mujer"] = _es(p["P6020"], 2)
    d["edad18_64"] = ((edad >= 18) & (edad < 65)).astype("float32")
    d["ocupado"] = oc.astype("float32")
    h = (p["P6800"].fillna(0) + p["P7045"].fillna(0)).where(oc, 0).astype(float)
    d["horas"] = h
    d["S"] = S
    d["exp"] = (edad - S - 6).clip(lower=0)                            # experiencia potencial (edad − años − 6)
    asal = p["P6430"].isin([1, 2]); cot = p["P6920"] == 1
    d["asalariado"] = asal.astype("float32")
    d["formal"] = (oc & cot).astype("float32")
    d["cuenta_propia"] = _es(p["P6430"], 4); d["patron"] = _es(p["P6430"], 5)
    d["pension"] = _es(p["P7500s2"])
    ayudas = p[["P7505", "P7510s1", "P7510s2", "P7510s3"]].eq(1)
    d["ayudas"] = ayudas.any(axis=1).astype("float32")
    d["arriendos"] = _es(p["P7495"]); d["intereses"] = _es(p["P7510s5"]); d["segunda"] = _es(p["P7040"])
    fuentes = pd.concat([oc, p["P7040"].eq(1), p["P7500s2"].eq(1), p["P7500s3"].eq(1), ayudas,
                         p[["P7495", "P7510s5", "P7510s6", "P7510s7", "P7422", "P7472"]].eq(1)], axis=1)
    d["fuentes"] = fuentes.sum(axis=1).astype("float32")
    d["perceptor"] = (fuentes.any(axis=1)).astype("float32")             # perceptor potencial (alguna fuente)
    d["hs"] = h * S.fillna(0)                                            # horas × años de educación
    d["hx"] = h * d["exp"].fillna(0)                                     # horas × experiencia potencial
    d["horas_formal"] = h * cot
    tam = p["P6870"].where(oc)
    d["tam"] = tam; d["h_tam"] = h * tam.fillna(0)
    d["S_oc"] = S.where(oc); d["exp_oc"] = d["exp"].where(oc)
    d["mujer_oc_adulta"] = (d["mujer"] * d["edad18_64"] * d["ocupado"]).astype("float32")
    d["mujer_adulta"] = (d["mujer"] * d["edad18_64"]).astype("float32")
    # señales de nivel salarial (preguntas a asalariados)
    for nom, c in [("aux_alimentacion", "P6585s1"), ("aux_transporte", "P6585s2"), ("subsidio_educativo", "P6585s4"),
                   ("primas", "P6545"), ("bonificaciones", "P6580"), ("prima_servicios", "P6630s1"),
                   ("prima_navidad", "P6630s2"), ("prima_vacaciones", "P6630s3"), ("viaticos", "P6630s4"),
                   ("bonif_anuales", "P6630s6"), ("especie_alimentos", "P6590"), ("especie_vivienda", "P6600"),
                   ("especie_transporte", "P6610"), ("especie_otros", "P6620")]:
        d[nom] = _es(p[c])
    d["formal_sin_aux"] = (asal & cot & (p["P6585s2"] == 2)).astype("float32")   # asalariado formal sin auxilio de transporte
    d["formal_con_aux"] = (asal & cot & (p["P6585s2"] == 1)).astype("float32")
    d["asal_informal"] = (asal & ~cot).astype("float32")
    d["prestaciones"] = d[["prima_servicios", "prima_navidad", "prima_vacaciones"]].sum(axis=1)
    d["ocup_subsidiado"] = (oc & (p["P6100"] == 3)).astype("float32")
    d["ocup_contributivo"] = (oc & (p["P6100"] == 1)).astype("float32")
    d["ocup_especial"] = (oc & (p["P6100"] == 2)).astype("float32")
    d["subempleo_visible"] = ((p["P7090"] == 1) & (p["P7110"] == 1) & (p["P7120"] == 1)).astype("float32")
    d["pariente_extendido"] = p["P6050"].isin([4, 5]).astype("float32")
    d["no_pariente"] = _es(p["P6050"], 9)
    # perceptor principal (sin montos): el ocupado de la unidad de gasto con mayor horas × (1 + años de educación)
    d["score_principal"] = np.where(oc & (d["ug"] == 1), h * (1 + S.fillna(0)), -1.0)
    d["oficio"] = p["Oficio"]; d["posicion"] = p["P6430"]
    return d


def agregar_v6(pv: pd.DataFrame) -> pd.DataFrame:
    """Agrega por hogar sobre la UNIDAD DE GASTO (la que define Ingtotug y Npersug)."""
    u = pv[pv.ug == 1]
    g = u.groupby("id", sort=False)
    sumas = ["perceptor", "fuentes", "ocupado", "asalariado", "formal", "cuenta_propia", "patron", "pension", "ayudas",
             "arriendos", "intereses", "segunda", "horas", "horas_formal", "hs", "hx", "h_tam", "mujer_oc_adulta",
             "mujer_adulta", "aux_alimentacion", "aux_transporte", "subsidio_educativo", "primas", "bonificaciones",
             "prima_vacaciones", "viaticos", "bonif_anuales", "especie_alimentos", "especie_vivienda",
             "especie_transporte", "especie_otros", "formal_sin_aux", "formal_con_aux", "asal_informal", "prestaciones",
             "ocup_subsidiado", "ocup_contributivo", "ocup_especial", "subempleo_visible", "pariente_extendido",
             "no_pariente"]
    A = g[sumas].sum().add_prefix("ug_")
    A["ug_educ_max_oc"] = g["S_oc"].max()
    A["ug_exp_media_oc"] = g["exp_oc"].mean()
    A["ug_tam_medio_oc"] = g["tam"].mean()
    A["ug_tam_max_oc"] = g["tam"].max()
    # jefe
    j = pv[pv.jefe == 1].set_index("id")
    for c in ["exp", "formal_sin_aux", "aux_transporte", "prestaciones", "ocup_subsidiado", "asal_informal"]:
        A[f"jefe6_{c}"] = j[c]
    # perceptor principal
    i = u.groupby("id", sort=False)["score_principal"].idxmax()
    P = u.loc[i.values].set_index("id")
    valido = P["score_principal"] >= 0
    for c, nom in [("horas", "principal_horas"), ("S", "principal_educ"), ("exp", "principal_exp"),
                   ("formal", "principal_formal"), ("jefe", "principal_es_jefe"), ("tam", "principal_tam"),
                   ("oficio", "principal_oficio"), ("posicion", "principal_posicion")]:
        A[nom] = P[c].where(valido)
    return A


# ------------------------------------------------------------------------------ hogar
def familias_hogar(H: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Familias I–L a nivel hogar. Devuelve (nuevas columnas, {columna: letra de familia})."""
    npug = H["Npersug"].clip(lower=1).astype(float)
    R = H["ingreso_requerido"].astype(float)
    nl = R / H["Lp"]                                    # «líneas necesarias»: n_líneas = R_h / Lp
    nlc = nl.clip(lower=0.05)                           # mismo piso que log_ingreso_requerido (R ≥ 0,05·Lp)
    oc1 = H["ug_ocupado"].clip(lower=1)
    asal1 = H["ug_asalariado"].clip(lower=1)
    Z, fam = {}, {}

    def add(nombre, valor, f):
        v = pd.Series(valor, index=H.index).astype(float)
        Z[nombre] = v.replace([np.inf, -np.inf], np.nan).values
        fam[nombre] = f

    # I. Umbral observable (ingreso_requerido, Lp y Li ya están en la tabla; se declaran en la familia)
    add("u_requerido_pc", R / npug, "I")
    add("u_n_lineas", nl, "I")
    add("u_li_sobre_lp", H["Li"] / H["Lp"], "I")
    add("u_arriendo_imp_pc", H["arriendo_imputado_mese"] / npug, "I")
    # J. Capacidad de ingreso frente al umbral (por línea necesaria y por miembro)
    for c in ["perceptor", "ocupado", "asalariado", "formal", "cuenta_propia", "patron", "pension", "ayudas",
              "arriendos", "intereses", "segunda", "horas", "horas_formal", "hs", "hx", "fuentes"]:
        nom = {"perceptor": "perceptores", "ocupado": "ocupados", "asalariado": "asalariados", "formal": "formales",
               "patron": "patronos", "pension": "pensionados"}.get(c, c)
        add(f"c_{nom}_linea", H[f"ug_{c}"] / nlc, "J")
    for c, nom in [("perceptor", "perceptores"), ("fuentes", "fuentes"), ("hs", "hs"), ("asalariado", "asalariados"),
                   ("cuenta_propia", "cuenta_propia"), ("pension", "pensionados"), ("patron", "patronos")]:
        add(f"c_{nom}_pc", H[f"ug_{c}"] / npug, "J")
    add("c_hs", H["ug_hs"], "J")
    add("c_educ_horas", H["ug_hs"] / H["ug_horas"].where(H["ug_horas"] > 0), "J")
    add("c_educ_max_oc", H["ug_educ_max_oc"], "J")
    add("c_exp_media_oc", H["ug_exp_media_oc"], "J")
    add("c_exp_jefe", H["jefe6_exp"], "J")
    add("c_tam_medio_oc", H["ug_tam_medio_oc"], "J")
    add("c_tam_max_oc", H["ug_tam_max_oc"], "J")
    add("c_tam_horas", H["ug_h_tam"] / H["ug_horas"].where(H["ug_horas"] > 0), "J")
    for c in ["principal_horas", "principal_educ", "principal_exp", "principal_formal", "principal_es_jefe", "principal_tam"]:
        add(f"c_{c}", H[c], "J")
    add("c_mujeres_ocupadas_share", H["ug_mujer_oc_adulta"] / H["ug_mujer_adulta"].clip(lower=1), "J")
    add("c_ocupados_menos_lineas", H["ug_ocupado"] - nl, "J")
    # K. Señales de nivel salarial
    for c in ["aux_alimentacion", "subsidio_educativo", "primas", "bonificaciones", "prima_vacaciones", "viaticos",
              "bonif_anuales", "especie_alimentos", "especie_vivienda", "especie_transporte", "especie_otros",
              "formal_sin_aux", "formal_con_aux", "asal_informal", "prestaciones", "ocup_subsidiado",
              "ocup_contributivo", "ocup_especial", "subempleo_visible"]:
        add(f"s_{c}", H[f"ug_{c}"], "K")
    add("s_share_aux_asal", H["ug_aux_transporte"] / asal1, "K")
    add("s_share_formal_sin_aux_oc", H["ug_formal_sin_aux"] / oc1, "K")
    add("s_share_ocup_subsidiado", H["ug_ocup_subsidiado"] / oc1, "K")
    add("s_share_prestaciones_asal", H["ug_prestaciones"] / asal1, "K")
    for c in ["formal_sin_aux", "aux_transporte", "prestaciones", "ocup_subsidiado", "asal_informal"]:
        add(f"s_jefe_{c}", H[f"jefe6_{c}"], "K")
    add("s_formal_sin_aux_linea", H["ug_formal_sin_aux"] / nlc, "K")
    add("s_prestaciones_linea", H["ug_prestaciones"] / nlc, "K")
    # L. Composición y vivienda
    adultos = H["n_edad18_64"].clip(lower=1)
    sin_pareja = 1 - H["hay_pareja"].fillna(0)
    con_menores = (H["n_menor18"] > 0).astype(float)
    arr = H["arriendo"].clip(lower=1e4)
    add("v_cuartos_pc", H["cuartos"] / npug, "L")
    add("v_dormitorios_por_cuarto", H["dormitorios"] / H["cuartos"].clip(lower=1), "L")
    add("v_monoparental", sin_pareja * con_menores, "L")
    add("v_jefa_monoparental", H["jefe_mujer"] * sin_pareja * con_menores, "L")
    add("v_unipersonal_mayor", ((H["Nper"] == 1) & (H["n_mayor65"] >= 1)).astype(float), "L")
    add("v_mayores_sin_pension", (H["n_mayor65"] - H["n_recibe_pension_jubilacion"]).clip(lower=0), "L")
    add("v_menores_por_adulto", H["n_menor18"] / adultos, "L")
    add("v_mayores_por_adulto", H["n_mayor65"] / adultos, "L")
    add("v_menores5_por_adulto", H["n_menor5"] / adultos, "L")
    add("v_desocupados_por_pet", H["n_desocupado"] / H["n_pet"].clip(lower=1), "L")
    add("v_inactivos_por_pet", H["n_inactivo"] / H["n_pet"].clip(lower=1), "L")
    add("v_jefa_x_menores", H["jefe_mujer"] * H["n_menor18"], "L")
    add("v_arriendo_persona_linea", np.log(arr / (H["Nper"].clip(lower=1) * H["Lp"])), "L")
    add("v_arriendo_cuarto_linea", np.log(arr / (H["cuartos"].clip(lower=1) * H["Lp"])), "L")
    add("v_propia_x_arriendo", H["log_arriendo_sobre_linea"] * H["ten_propia_pagada"], "L")
    add("v_brecha_educ_jefe_pareja", H["jefe_anios_educ"] - H["pareja_educ"], "L")
    add("v_superior_por_adulto", H["n_educ_superior"] / (H["n_edad18_64"] + H["n_mayor65"]).clip(lower=1), "L")
    add("v_extendido", H["ug_pariente_extendido"], "L")
    add("v_no_parientes", H["ug_no_pariente"], "L")
    add("v_sin_educacion_por_adulto", H["n_sin_educacion"] / adultos, "L")
    return pd.DataFrame(Z, index=H.index), fam


def construir_v6(th, tp, teh, tep):
    """Tablas v6 de train y test (sin codificación de objetivo ni posiciones relativas, que dependen de la
    partición). Sólo columnas comunes: la restricción se aplica al entrar."""
    hc, pc = comunes(th, HOG_COMUNES), comunes(tp, PER_COMUNES)
    hte, pte = comunes(teh, HOG_COMUNES), comunes(tep, PER_COMUNES)
    Htr = F.construir(hc, pc)
    Hte = F.construir(hte, pte, ref_arriendo=Htr.assign(arriendo=Htr.arriendo.where(Htr.arriendo_imputado_flag == 0)))
    salida, fam = [], {}
    for H, p in ((Htr, pc), (Hte, pte)):
        H = H.drop(columns=[c for c in CONSTANTES_EXTERNAS if c in H.columns])
        A = agregar_v6(variables_persona_v6(p))
        H = H.merge(A, left_on="id", right_index=True, how="left")
        N, fam = familias_hogar(H)
        H = pd.concat([H, N], axis=1)
        H = H.drop(columns=[c for c in H.columns if c.startswith(("ug_", "jefe6_")) or c.startswith("principal_")
                            and c not in ("principal_oficio",)])
        salida.append(H)
    for c in ("ingreso_requerido", "Lp", "Li"):
        fam[c] = "I"
    return salida[0], salida[1], fam


# ------------------------------------------------------------------------------ codificación de objetivo
def _codigos(D: pd.DataFrame, DT: pd.DataFrame | None, cols: list) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Códigos enteros de la clave (faltantes = nivel propio «nan») comunes a train y test."""
    def clave(X):
        return X[cols].astype(str).agg("|".join, axis=1).values if len(cols) > 1 else X[cols[0]].astype(str).values
    k = clave(D) if DT is None else np.concatenate([clave(D), clave(DT)])
    cod, niv = pd.factorize(k)
    if DT is None:
        return cod, None, len(niv)
    return cod[:len(D)], cod[len(D):], len(niv)


def _media_suavizada(cod_fit, y_fit, cod_ap, L, m):
    prior = float(np.mean(y_fit))
    s = np.bincount(cod_fit, weights=y_fit, minlength=L); n = np.bincount(cod_fit, minlength=L)
    return ((s + m * prior) / (n + m))[cod_ap]


def te_global(D, DT, y, fold, especificaciones=None, m=TE_SUAVIZADO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Codificación para las tablas guardadas: en train, fuera de pliegue con los 5 pliegues de `pliegues.csv`
    (la fila del pliegue k se codifica con los otros cuatro); en test, con todo train."""
    esp = especificaciones or TE_ESPECIFICACIONES
    y = np.asarray(y, float); fold = np.asarray(fold)
    Atr, Ate = {}, {}
    for nom, cols in esp.items():
        c_tr, c_te, L = _codigos(D, DT, cols)
        v = np.zeros(len(D))
        for k in np.unique(fold):
            va = fold == k
            v[va] = _media_suavizada(c_tr[~va], y[~va], c_tr[va], L, m)
        Atr[nom] = v.astype("float32")
        Ate[nom] = _media_suavizada(c_tr, y, c_te, L, m).astype("float32")
    return pd.DataFrame(Atr, index=D.index), pd.DataFrame(Ate, index=DT.index)


class Transformador:
    """Recalcula, para cada partición entrenamiento/validación, lo que depende de la muestra de entrenamiento:

    * la **codificación de objetivo**: las filas de entrenamiento `tr` se codifican fuera de pliegue con una
      partición interna de 5 pliegues de `tr` (estratificada por la etiqueta); el resto de filas (validación) con
      todo `tr`. Así ninguna fila ve su propia etiqueta y la validación nunca entra en la codificación;
    * las **posiciones relativas** en el dominio/departamento (sólo X de `tr`, sin etiqueta).

    Interfaz de `ajuste.cv(..., transformar=...)` y de `busqueda._cv_eval`: `transformar(k, tr, X) → X'`."""

    def __init__(self, D, y, especificaciones=None, m=TE_SUAVIZADO, grupo=(), k_interno=5, semilla=20262, max_cache=16):
        from sklearn.model_selection import StratifiedKFold
        self.D, self.y = D, np.asarray(y, float)
        self.esp = dict(especificaciones if especificaciones is not None else TE_ESPECIFICACIONES)
        self.m, self.grupo, self.k, self.semilla = float(m), list(grupo), k_interno, semilla
        self.cod = {}
        for nom, cols in self.esp.items():
            c, _, L = _codigos(D, None, cols)
            self.cod[nom] = (c, L)
        self._skf = StratifiedKFold
        self.cache, self.max_cache = {}, max_cache

    def valores(self, tr) -> dict:
        tr = np.asarray(tr)
        clave = hashlib.md5(tr.tobytes()).hexdigest() + str(len(tr))
        if clave in self.cache:
            return self.cache[clave]
        n = len(self.y); out = {}
        if self.esp:
            internos = list(self._skf(self.k, shuffle=True, random_state=self.semilla).split(np.zeros(len(tr)), self.y[tr]))
            for nom, (c, L) in self.esp_items():
                v = _media_suavizada(c[tr], self.y[tr], c, L, self.m)          # todas las filas con todo tr
                for a, b in internos:                                          # filas de tr: fuera de pliegue interno
                    v[tr[b]] = _media_suavizada(c[tr[a]], self.y[tr[a]], c[tr[b]], L, self.m)
                out[nom] = v.astype("float32")
        if self.grupo:
            from .busqueda import materializar
            G = materializar(self.grupo, self.D.iloc[tr], self.D)
            for c in G.columns:
                out[c] = G[c].values.astype("float32")
        if len(self.cache) >= self.max_cache:
            self.cache.pop(next(iter(self.cache)))
        self.cache[clave] = out
        return out

    def esp_items(self):
        return [(nom, self.cod[nom]) for nom in self.esp]

    def __call__(self, k, tr, X):
        vals = self.valores(tr)
        cols = [c for c in vals if c in X.columns]
        if not cols:
            return X
        X2 = X.copy()
        for c in cols:
            X2[c] = vals[c]
        return X2


def activar(dat, conj: dict):
    """Para los cuadernos de modelos sobre v6: adjunta a un `ajuste.Datos` el transformador por pliegue (pásese
    como `transformar=dat.transformar` a `ajuste.cv`) y las familias como bloques teóricos."""
    from . import ajuste as AJ
    te = conj.get("codificacion_objetivo", {})
    dat.transformar = Transformador(dat.D, dat.y, te.get("especificaciones", TE_ESPECIFICACIONES),
                                    m=te.get("suavizado_m", TE_SUAVIZADO), grupo=conj.get("especificaciones_grupo", []))
    bl = conj.get("bloques", {})
    dat.bloques = lambda: {c: bl.get(c) or AJ.bloque_variable(c, set(), set()) for c in dat.columnas}
    return dat


# ------------------------------------------------------------------------------ auditoría
def matriz_prohibida(th, tp, tep, D5=None, gen5=()) -> pd.DataFrame:
    """Referencia de lo prohibido, alineada con los hogares de `th`: columnas de ingreso del hogar y sus
    transformaciones, cada columna de persona sólo de train agregada por hogar (suma, máximo y valor del jefe),
    las variables generadas de v3/v5 y las de Mincer (constantes externas)."""
    ids = th["id"].values
    P = {}
    for c in HOG_PROHIBIDAS:
        P[f"hogar:{c}"] = th[c].astype(float).values
    lp = th["Lp"].astype(float).values
    ing = th["Ingtotug"].astype(float).values; ipc = th["Ingpcug"].astype(float).values
    P["hogar:log(Ingtotug)"] = np.log(np.maximum(ing, 1.0))
    P["hogar:Ingpcug/Lp"] = ipc / lp
    P["hogar:log(Ingpcug/Lp)"] = np.log(np.maximum(ipc, 1.0) / lp)
    P["hogar:Ingtotug/Npersug"] = ing / th["Npersug"].clip(lower=1).values
    solo_train = prohibidas_persona(tp, tep)
    g = tp[["id"] + solo_train].copy()
    for c in solo_train:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    gg = g.groupby("id", sort=False)
    suma = gg[solo_train].sum(min_count=1).reindex(ids); mx = gg[solo_train].max().reindex(ids)
    jefe = tp.loc[tp.P6050 == 1, ["id"] + solo_train].drop_duplicates("id").set_index("id").reindex(ids)
    for c in solo_train:
        P[f"persona_suma:{c}"] = suma[c].values; P[f"persona_max:{c}"] = mx[c].values
        P[f"persona_jefe:{c}"] = pd.to_numeric(jefe[c], errors="coerce").values
    if D5 is not None:
        assert (D5["id"].values == ids).all()
        for c in list(gen5) + list(CONSTANTES_EXTERNAS) + OBJETIVO_DERIVADAS:
            if c in D5.columns and f"v5:{c}" not in P:
                P[f"v5:{c}"] = pd.to_numeric(D5[c], errors="coerce").astype(float).values
    P = pd.DataFrame(P)
    return P.loc[:, P.std(skipna=True).fillna(0) > 0]


def _corr_pares(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Correlación de Pearson por pares completos entre las columnas de A (n×p) y B (n×q), con faltantes."""
    def prep(M):
        M = M.astype(np.float64)
        mk = np.isfinite(M)
        mu = np.where(mk, M, 0).sum(0) / np.maximum(mk.sum(0), 1)
        sd = np.sqrt(np.where(mk, (M - mu) ** 2, 0).sum(0) / np.maximum(mk.sum(0), 1))
        Z = np.where(mk, (M - mu) / np.where(sd > 0, sd, 1), 0.0)
        return Z, mk.astype(np.float64)
    Za, Ma = prep(A); Zb, Mb = prep(B)
    n = Ma.T @ Mb
    sa = Za.T @ Mb; sb = Ma.T @ Zb
    saa = (Za ** 2).T @ Mb; sbb = Ma.T @ (Zb ** 2); sab = Za.T @ Zb
    num = n * sab - sa * sb
    den = np.sqrt(np.maximum(n * saa - sa ** 2, 0) * np.maximum(n * sbb - sb ** 2, 0))
    with np.errstate(all="ignore"):
        r = np.where((den > 0) & (n >= 30), num / den, 0.0)
    return np.clip(r, -1, 1)


def auditar(Dv6: pd.DataFrame, cols: list, P: pd.DataFrame, y, familias: dict, finales: set,
            umbral: float = 0.99, bloque=1024) -> pd.DataFrame:
    """Para cada columna de v6: máxima |correlación| de Pearson y de Spearman con cualquier columna prohibida
    (pares completos), la prohibida responsable, la correlación con la etiqueta y la revisión de nombre."""
    X = Dv6[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    Xr = X.rank(na_option="keep").values; Pr = P.rank(na_option="keep").values
    rp = _corr_pares(X.values, P.values); rs = _corr_pares(Xr, Pr)
    ry = _corr_pares(X.values, np.asarray(y, float)[:, None])[:, 0]
    filas = []
    for j, c in enumerate(cols):
        ip, is_ = int(np.argmax(np.abs(rp[j]))), int(np.argmax(np.abs(rs[j])))
        nombre_ok = not any(re.search(pt, c.lower()) for pt in PATRONES_PROHIBIDOS)
        mp, ms = float(abs(rp[j, ip])), float(abs(rs[j, is_]))
        filas.append({"variable": c, "familia": familias.get(c, "base"), "en_v6_final": c in finales,
                      "max_abs_pearson_prohibida": mp, "prohibida_pearson": P.columns[ip],
                      "max_abs_spearman_prohibida": ms, "prohibida_spearman": P.columns[is_],
                      "corr_pearson_con_Pobre": float(ry[j]), "nombre_sin_patron_prohibido": nombre_ok,
                      "pct_faltantes": float(X[c].isna().mean())})
    t = pd.DataFrame(filas)
    t["supera_umbral"] = (t.max_abs_pearson_prohibida > umbral) | (t.max_abs_spearman_prohibida > umbral)
    # Revisión de soporte: una correlación > umbral puede deberse sólo a que ambas columnas valen 0 en los mismos
    # hogares porque la prohibida se construyó con una pregunta PERMITIDA como compuerta (p. ej. la primera etapa v5
    # sólo asigna pensión si P7500s2 = 1). Se mide (i) el acuerdo del soporte (≠ 0) y (ii) la correlación dentro del
    # soporte común: si el soporte coincide (≥ 99,9 %) y dentro de él |ρ| < umbral (o la columna v6 es constante ahí,
    # es decir, ES la compuerta), la columna v6 no lleva la información del ingreso, sólo la de la pregunta permitida.
    t["acuerdo_soporte"] = np.nan; t["max_abs_corr_en_soporte"] = np.nan; t["motivo"] = ""
    for i in np.flatnonzero(t.supera_umbral.values):
        c = t.at[i, "variable"]; a = X[c]
        acuerdos, dentro = [], []
        for pcol in {t.at[i, "prohibida_pearson"], t.at[i, "prohibida_spearman"]}:
            b = P[pcol]
            za, zb = a.fillna(0).abs() > 1e-12, b.fillna(0).abs() > 1e-12
            acuerdos.append(float((za.values == zb.values).mean()))
            m = (za & zb).values
            sa, sb = a.values[m], b.values[m]
            if m.sum() < 30 or np.nanstd(sa) == 0:
                dentro.append(0.0)                     # constante en el soporte: es la propia compuerta
            else:
                dentro.append(max(abs(pd.Series(sa).corr(pd.Series(sb))),
                                  abs(pd.Series(sa).corr(pd.Series(sb), method="spearman"))))
        t.at[i, "acuerdo_soporte"] = min(acuerdos); t.at[i, "max_abs_corr_en_soporte"] = max(dentro)
        if min(acuerdos) >= 0.999 and max(dentro) < umbral:
            t.at[i, "motivo"] = ("compuerta común: la prohibida vale 0 exactamente cuando la pregunta permitida lo indica; "
                                 "dentro del soporte la correlación es baja")
    ok_compuerta = t.supera_umbral & (t.motivo != "")
    t["estado"] = np.where(~t.nombre_sin_patron_prohibido | (t.supera_umbral & ~ok_compuerta), "FALLA",
                           np.where(ok_compuerta, "OK (compuerta común)", "OK"))
    return t.sort_values("max_abs_spearman_prohibida", ascending=False).reset_index(drop=True)


def invariancia(th, tp, teh, tep, semilla=7) -> tuple[bool, list]:
    """Prueba de que nada deriva de lo prohibido: se reconstruye v6 con TODAS las columnas no comunes de train
    (ingresos, montos, Estrato1, …) permutadas al azar y se exige que el resultado sea idéntico."""
    rng = np.random.default_rng(semilla)
    th2, tp2 = th.copy(), tp.copy()
    for c in [c for c in th.columns if c not in HOG_COMUNES]:
        th2[c] = th[c].values[rng.permutation(len(th))]
    for c in [c for c in tp.columns if c not in PER_COMUNES]:
        tp2[c] = tp[c].values[rng.permutation(len(tp))]
    A, _, _ = construir_v6(th, tp, teh, tep)
    B, _, _ = construir_v6(th2, tp2, teh, tep)
    distintas = [c for c in A.columns if not A[c].equals(B[c])]
    return (not distintas), distintas
