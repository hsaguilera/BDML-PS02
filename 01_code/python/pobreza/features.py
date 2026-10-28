r"""Construcción de variables a nivel hogar.

Principio rector: SÓLO se usan columnas presentes en train y test (63 de persona y
16 de hogar). Las variables de ingreso (sólo en train) se usan únicamente como
OBJETIVO de modelos auxiliares (variables generadas, Pagan 1984) con predicción
fuera de pliegue para evitar fuga de información.

Bloques teóricos (ver notebook 02 para la justificación y la literatura):

A. Necesidades y escala del hogar  — tamaño, composición demográfica, dependencia
   (la línea de pobreza es per cápita: más miembros ⇒ más ingreso requerido).
B. Capital humano                 — educación del jefe y de los adultos (Mincer, 1974;
   Becker, 1964). Años de educación aproximados desde P6210 y P6210s1.
C. Inserción laboral y formalidad — ocupados por miembro (perceptores), posición
   ocupacional, cotización a pensión, tamaño de empresa, prestaciones legales
   (prima, subsidio de transporte, subsidio familiar), subempleo (Fields, 2011;
   Perry et al., 2007).
D. Protección social y salud      — régimen subsidiado (focalizado por SISBÉN),
   afiliación, pensiones, ayudas institucionales, remesas.
E. Vivienda y activos             — tenencia, cuartos, hacinamiento, arriendo pagado
   o estimado: proxy del ingreso permanente (Friedman, 1957; ley de Schwabe).
F. Territorio                     — Dominio, departamento, clase (urbano/rural),
   línea de pobreza (costo de vida local).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Códigos GEIH (DANE)
# --------------------------------------------------------------------------------------
SI = 1.0
POSICION = {1: "obrero_particular", 2: "obrero_gobierno", 3: "domestico", 4: "cuenta_propia",
            5: "patron", 6: "familiar_sin_rem", 7: "sin_rem_otro_hogar", 8: "jornalero", 9: "otro"}
TENENCIA = {1: "propia_pagada", 2: "propia_pagando", 3: "arriendo", 4: "usufructo",
            5: "posesion_sin_titulo", 6: "otra"}
NIVEL_EDU = {1: "ninguno", 2: "preescolar", 3: "primaria", 4: "secundaria", 5: "media", 6: "superior", 9: "ns"}
PARENTESCO = {1: "jefe", 2: "pareja", 3: "hijo", 4: "nieto", 5: "otro_pariente",
              6: "empleado_domestico", 7: "pensionista", 8: "trabajador", 9: "otro_no_pariente"}
BENEFICIOS = ["P6510", "P6545", "P6580", "P6585s1", "P6585s2", "P6585s3", "P6585s4", "P6590",
              "P6600", "P6610", "P6620", "P6630s1", "P6630s2", "P6630s3", "P6630s4", "P6630s6"]
NO_LABORALES = {"P7495": "arriendos_o_pensiones", "P7500s2": "pension_jubilacion",
                "P7500s3": "pension_alimentaria", "P7505": "otros_ingresos",
                "P7510s1": "ayuda_otros_hogares_pais", "P7510s2": "remesas_exterior",
                "P7510s3": "ayuda_instituciones", "P7510s5": "intereses_dividendos",
                "P7510s6": "cesantias", "P7510s7": "otras_fuentes"}

# Columnas de persona usadas como predictores (comunes a train/test)
COLS_PERSONA_COMUNES = [
    "id", "Orden", "Clase", "Dominio", "P6020", "P6040", "P6050", "P6090", "P6100", "P6210", "P6210s1",
    "P6240", "Oficio", "P6426", "P6430", "P6510", "P6545", "P6580", "P6585s1", "P6585s2", "P6585s3",
    "P6585s4", "P6590", "P6600", "P6610", "P6620", "P6630s1", "P6630s2", "P6630s3", "P6630s4", "P6630s6",
    "P6800", "P6870", "P6920", "P7040", "P7045", "P7050", "P7090", "P7110", "P7120", "P7150", "P7160",
    "P7310", "P7350", "P7422", "P7472", "P7495", "P7500s2", "P7500s3", "P7505", "P7510s1", "P7510s2",
    "P7510s3", "P7510s5", "P7510s6", "P7510s7", "Pet", "Oc", "Des", "Ina", "Fex_c", "Depto", "Fex_dpto"]


def _es(s, v=SI):
    return (s == v).astype("float32")


def anios_educacion(nivel: pd.Series, grado: pd.Series) -> pd.Series:
    """Años de educación aprobados (verificado con la tabla cruzada P6210 × P6210s1, notebook 01):
    ninguno/preescolar → 0; primaria → grado 0–5; secundaria → grado 6–9 (grado 0 = primaria
    completa → 5); media → grado 10–13 (12–13: normalista); superior → 11 + años aprobados de
    educación superior (P6210s1 = 0…16, incluye posgrado)."""
    n, g = nivel.values, grado.values
    a = np.full(len(n), np.nan)
    a[np.isin(n, [1, 2])] = 0
    m = n == 3; a[m] = np.clip(np.nan_to_num(g[m], nan=3), 0, 5)
    m = n == 4; gg = np.nan_to_num(g[m], nan=7); a[m] = np.where(gg < 6, 5, np.clip(gg, 6, 9))
    m = n == 5; gg = np.nan_to_num(g[m], nan=11); a[m] = np.where(gg < 10, 10, np.clip(gg, 10, 13))
    m = n == 6; gg = np.nan_to_num(g[m], nan=3); a[m] = 11 + np.clip(gg, 0, 10)
    return pd.Series(a, index=nivel.index)


def variables_persona(p: pd.DataFrame) -> pd.DataFrame:
    """Indicadores individuales (float32) listos para agregarse por hogar."""
    d = pd.DataFrame(index=p.index)
    d["id"] = p["id"].values
    edad = p["P6040"]
    d["edad"] = edad
    d["mujer"] = _es(p["P6020"], 2)
    d["jefe"] = _es(p["P6050"], 1)
    d["pareja"] = _es(p["P6050"], 2)
    d["en_ug"] = (~p["P6050"].isin([6, 7, 8])).astype("float32")  # unidad de gasto
    d["empleado_domestico_hogar"] = _es(p["P6050"], 6)
    d["pensionista_hogar"] = _es(p["P6050"], 7)
    d["menor5"] = (edad < 5).astype("float32")
    d["menor12"] = (edad < 12).astype("float32")
    d["menor18"] = (edad < 18).astype("float32")
    d["edad18_64"] = ((edad >= 18) & (edad < 65)).astype("float32")
    d["mayor65"] = (edad >= 65).astype("float32")
    d["anios_educ"] = anios_educacion(p["P6210"], p["P6210s1"])
    d["educ_superior"] = _es(p["P6210"], 6)
    d["educ_media_o_mas"] = p["P6210"].isin([5, 6]).astype("float32")
    d["sin_educacion"] = p["P6210"].isin([1, 2]).astype("float32") * (edad >= 15)
    d["pet"] = _es(p["Pet"])
    d["ocupado"] = _es(p["Oc"])
    d["desocupado"] = _es(p["Des"])
    d["inactivo"] = _es(p["Ina"])
    d["estudiante"] = _es(p["P6240"], 3)
    d["oficios_hogar"] = _es(p["P6240"], 4)
    d["incapacitado"] = _es(p["P6240"], 5)
    for k, v in POSICION.items():
        d[f"pos_{v}"] = _es(p["P6430"], k)
    d["asalariado"] = p["P6430"].isin([1, 2]).astype("float32")
    d["independiente"] = p["P6430"].isin([4, 5]).astype("float32")
    d["sin_remuneracion"] = p["P6430"].isin([6, 7]).astype("float32")
    d["cotiza_pension"] = _es(p["P6920"], 1)
    d["pensionado"] = _es(p["P6920"], 3)
    d["ocupado_informal"] = d["ocupado"] * (1 - d["cotiza_pension"])
    d["horas"] = p["P6800"].fillna(0) + p["P7045"].fillna(0)
    d["horas"] = d["horas"].where(d["ocupado"] == 1, 0).astype("float32")
    d["segundo_trabajo"] = _es(p["P7040"])
    d["subempleo_horas"] = _es(p["P7090"])
    d["busca_cambiar"] = _es(p["P7150"])
    d["trabaja_solo"] = _es(p["P6870"], 1)
    d["empresa_grande"] = (p["P6870"] >= 8).astype("float32")   # 51+ trabajadores
    d["empresa_2_10"] = p["P6870"].isin([2, 3, 4]).astype("float32")
    d["antiguedad_meses"] = p["P6426"].where(d["ocupado"] == 1)
    ben = np.column_stack([_es(p[c]).values for c in BENEFICIOS])
    d["n_beneficios"] = ben.sum(1)
    d["prima_servicios"] = _es(p["P6630s1"])
    d["prima_navidad"] = _es(p["P6630s2"])
    d["subsidio_transporte"] = _es(p["P6585s2"])  # legal sólo si salario ≤ 2 SMMLV
    d["subsidio_familiar"] = _es(p["P6585s3"])    # caja de compensación (salario ≤ 4 SMMLV)
    d["horas_extra"] = _es(p["P6510"])
    d["pago_especie"] = p[["P6590", "P6600", "P6610", "P6620"]].eq(1).any(axis=1).astype("float32")
    d["salud_afiliado"] = _es(p["P6090"])
    d["salud_no_afiliado"] = _es(p["P6090"], 2)
    d["salud_contributivo"] = _es(p["P6100"], 1)
    d["salud_especial"] = _es(p["P6100"], 2)
    d["salud_subsidiado"] = _es(p["P6100"], 3)
    for c, nom in NO_LABORALES.items():
        d[f"recibe_{nom}"] = _es(p[c])
    d["desocupado_con_ingreso"] = _es(p["P7422"])
    d["inactivo_con_ingreso"] = _es(p["P7472"])
    d["desocupado_cesante"] = _es(p["P7310"], 2)  # ya había trabajado antes
    # Índice de capacidad de generar ingreso laboral (Mincer, 1974): exp(ρ·S + β1·X − β2·X²)
    # con ρ = 0.10, experiencia potencial X = edad − S − 6, escalado por horas/48. Sólo ocupados.
    S = d["anios_educ"].fillna(5)
    X = (edad - S - 6).clip(lower=0)
    d["mincer_ocupado"] = (d["ocupado"] * np.exp(0.10 * S + 0.03 * X - 0.0005 * X ** 2)
                           * np.minimum(d["horas"], 84) / 48).astype("float32")
    d["ocupado_superior"] = d["ocupado"] * d["educ_superior"]
    d["asalariado_formal"] = d["asalariado"] * d["cotiza_pension"]
    d["horas_formal"] = d["horas"] * d["cotiza_pension"]
    d["patron_o_cp_empresa_grande"] = d["independiente"] * d["empresa_grande"]
    d["oficio"] = p["Oficio"]
    d["posicion"] = p["P6430"]
    d["tam_empresa"] = p["P6870"]
    d["nivel_edu"] = p["P6210"]
    d["regimen_salud"] = p["P6100"]
    d["actividad"] = p["P6240"]
    return d


def agregar_hogar(per: pd.DataFrame) -> pd.DataFrame:
    """Agrega los indicadores individuales al hogar (sumas, proporciones, máximos y
    características del jefe)."""
    g = per.groupby("id", sort=False)
    sumas = ["mujer", "en_ug", "empleado_domestico_hogar", "pensionista_hogar", "menor5", "menor12",
             "menor18", "edad18_64", "mayor65", "educ_superior", "educ_media_o_mas", "sin_educacion",
             "pet", "ocupado", "desocupado", "inactivo", "estudiante", "oficios_hogar", "incapacitado",
             "asalariado", "independiente", "sin_remuneracion", "cotiza_pension", "pensionado",
             "ocupado_informal", "horas", "segundo_trabajo", "subempleo_horas", "busca_cambiar",
             "trabaja_solo", "empresa_grande", "empresa_2_10", "n_beneficios", "prima_servicios",
             "prima_navidad", "subsidio_transporte", "subsidio_familiar", "horas_extra", "pago_especie",
             "salud_afiliado", "salud_no_afiliado", "salud_contributivo", "salud_especial",
             "salud_subsidiado", "desocupado_con_ingreso", "inactivo_con_ingreso", "desocupado_cesante",
             "mincer_ocupado", "ocupado_superior", "asalariado_formal", "horas_formal",
             "patron_o_cp_empresa_grande"] + \
            [c for c in per.columns if c.startswith("pos_") or c.startswith("recibe_")]
    H = g[sumas].sum().add_prefix("n_")
    H["n_personas_reg"] = g.size()
    adultos = per[per.edad >= 18]
    ga = adultos.groupby("id", sort=False)
    H["educ_media_adultos"] = ga["anios_educ"].mean()
    H["educ_max_adultos"] = ga["anios_educ"].max()
    H["educ_min_adultos"] = ga["anios_educ"].min()
    H["edad_media"] = g["edad"].mean()
    oc = per[per.ocupado == 1]
    H["antiguedad_media_oc"] = oc.groupby("id")["antiguedad_meses"].mean()
    H["educ_media_ocupados"] = oc.groupby("id")["anios_educ"].mean()
    # Jefe de hogar
    j = per[per.jefe == 1].set_index("id")
    jc = ["edad", "mujer", "anios_educ", "ocupado", "desocupado", "inactivo", "asalariado", "independiente",
          "cotiza_pension", "pensionado", "horas", "salud_subsidiado", "salud_contributivo", "salud_especial",
          "salud_no_afiliado", "empresa_grande", "trabaja_solo", "antiguedad_meses", "n_beneficios",
          "subempleo_horas", "oficio", "posicion", "tam_empresa", "nivel_edu", "regimen_salud", "actividad",
          "estudiante", "oficios_hogar"]
    H = H.join(j[jc].add_prefix("jefe_"), how="left")
    # pareja del jefe
    H["hay_pareja"] = g["pareja"].max()
    pj = per[per.pareja == 1].groupby("id")
    H["pareja_ocupada"] = pj["ocupado"].max()
    H["pareja_educ"] = pj["anios_educ"].max()
    return H


def _monto(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Limpia montos de vivienda: 98/99 = no sabe/no informa → NA (+ indicador)."""
    ns = s.isin([98, 99]) | ((s > 0) & (s < 1000))
    return s.where(~ns), ns.astype("float32")


def variables_hogar(h: pd.DataFrame) -> pd.DataFrame:
    """Variables a partir de la tabla de hogares (vivienda, escala, territorio)."""
    d = pd.DataFrame(index=h.index)
    d["id"] = h["id"].values
    d["Nper"] = h["Nper"]; d["Npersug"] = h["Npersug"]
    d["cuartos"] = h["P5000"].clip(upper=15); d["dormitorios"] = h["P5010"].clip(upper=10)
    d["personas_por_cuarto"] = h["Nper"] / h["P5000"].clip(lower=1)
    d["personas_por_dormitorio"] = h["Nper"] / h["P5010"].clip(lower=1)
    d["hacinamiento_critico"] = (d["personas_por_dormitorio"] > 3).astype("float32")  # criterio IPM
    for k, v in TENENCIA.items():
        d[f"ten_{v}"] = _es(h["P5090"], k)
    d["tenencia"] = h["P5090"]
    p5140, ns40 = _monto(h["P5140"]); p5130, ns30 = _monto(h["P5130"]); p5100, ns00 = _monto(h["P5100"])
    d["arriendo_pagado"] = p5140
    d["arriendo_estimado"] = p5130
    d["cuota_amortizacion"] = p5100
    d["arriendo"] = p5140.fillna(p5130)  # valor de uso de la vivienda (pagado o imputado)
    d["arriendo_ns"] = np.maximum(ns40, ns30)
    # Arriendo imputado que MESE suma al ingreso (verificado en train, notebooks 02 y 05):
    #   propia pagada (1) y usufructo (4): P5130;  propia pagando (2): max(P5130 − P5100, 0);  resto: 0.
    #   Correcciones verificadas: A = 0 si P5130 > 3.000.000 (tope de MESE, exacto en train) y
    #   A = 0 si la vivienda se está pagando y la cuota P5100 es "no sabe/no informa" (98/99).
    #   En usufructo rural A = 0 en ≈ 1/3 de los casos sin regla observable → bandera.
    ten = h["P5090"].values
    a130, a100 = p5130.fillna(0).values, p5100.fillna(0).values
    a130 = np.where(a130 > 3_000_000, 0.0, a130)
    A = np.where(np.isin(ten, [1, 4]), a130, np.where(ten == 2, np.maximum(a130 - a100, 0), 0.0))
    A = np.where((ten == 2) & (ns00.values == 1), 0.0, A)
    d["arriendo_imputado_mese"] = A
    d["arriendo_estimado_sobre_tope"] = (p5130.fillna(0) > 3_000_000).astype("float32")
    d["usufructo_rural"] = ((h["P5090"] == 4) & (h["Clase"] == 2)).astype("float32")
    d["amortizacion_ns"] = ns00
    d["Lp"] = h["Lp"]; d["Li"] = h["Li"]
    d["log_Lp"] = np.log(h["Lp"])
    d["log_linea_hogar"] = np.log(h["Lp"] * h["Npersug"])
    d["cabecera"] = (h["Clase"] == 1).astype("float32")
    d["Dominio"] = h["Dominio"].values; d["Depto"] = h["Depto"].values
    d["Fex_c"] = h["Fex_c"]; d["log_fex"] = np.log(h["Fex_c"])
    return d


def imputar_arriendo(H: pd.DataFrame, ref: pd.DataFrame | None = None) -> pd.DataFrame:
    """Imputa el arriendo faltante con la mediana por (Dominio × cuartos) del conjunto de
    referencia (train), con respaldo en la mediana por Dominio. Se deja un indicador."""
    ref = H if ref is None else ref
    med1 = ref.groupby(["Dominio", "cuartos"])["arriendo"].median()
    med2 = ref.groupby("Dominio")["arriendo"].median()
    H = H.copy()
    falta = H["arriendo"].isna()
    H["arriendo_imputado_flag"] = falta.astype("float32")
    idx = pd.MultiIndex.from_frame(H.loc[falta, ["Dominio", "cuartos"]])
    v = pd.Series(med1.reindex(idx).values, index=H.index[falta])
    v = v.fillna(H.loc[falta, "Dominio"].map(med2))
    H.loc[falta, "arriendo"] = v
    return H


def derivadas(H: pd.DataFrame) -> pd.DataFrame:
    """Razones y transformaciones con interpretación económica."""
    H = H.copy()
    npug = H["Npersug"].clip(lower=1)
    H["log_arriendo"] = np.log(H["arriendo"].clip(lower=1e4))
    H["log_arriendo_pc"] = np.log(H["arriendo"].clip(lower=1e4) / npug)
    H["arriendo_sobre_linea"] = H["arriendo"] / (H["Lp"] * npug)        # vivienda vs. canasta del hogar
    H["log_arriendo_sobre_linea"] = np.log(H["arriendo_sobre_linea"].clip(lower=1e-3))
    H["log_amortizacion"] = np.log1p(H["cuota_amortizacion"].fillna(0))
    H["ocupados_pc"] = H["n_ocupado"] / npug                              # perceptores por miembro
    H["formales_pc"] = H["n_cotiza_pension"] / npug
    H["tasa_ocupacion_hogar"] = H["n_ocupado"] / H["n_pet"].clip(lower=1)
    H["share_formal_ocupados"] = H["n_cotiza_pension"] / H["n_ocupado"].clip(lower=1)
    H["dependencia"] = (H["n_menor18"] + H["n_mayor65"]) / H["n_edad18_64"].clip(lower=1)
    H["share_menores"] = H["n_menor18"] / H["Nper"]
    H["share_mujeres"] = H["n_mujer"] / H["Nper"]
    H["horas_pc"] = H["n_horas"] / npug
    H["beneficios_por_ocupado"] = H["n_n_beneficios"] / H["n_ocupado"].clip(lower=1)
    H["share_subsidiado"] = H["n_salud_subsidiado"] / H["Nper"]
    H["sin_ocupados"] = (H["n_ocupado"] == 0).astype("float32")
    H["jefe_mujer_sin_pareja"] = H["jefe_mujer"] * (1 - H["hay_pareja"].fillna(0))
    H["recibe_transferencias"] = ((H["n_recibe_ayuda_instituciones"] + H["n_recibe_ayuda_otros_hogares_pais"]) > 0).astype("float32")
    H["recibe_pension"] = (H["n_recibe_pension_jubilacion"] > 0).astype("float32")
    H["recibe_capital"] = ((H["n_recibe_intereses_dividendos"] + H["n_recibe_arriendos_o_pensiones"]) > 0).astype("float32")
    H["jefe_edad2"] = H["jefe_edad"] ** 2 / 100
    H["log_Nper"] = np.log(H["Nper"])
    # Ingreso (sin arriendo imputado) que el hogar necesita para NO ser pobre:
    #   pobre ⇔ Ingtotug < R_h,  R_h = Npersug·Lp − A_h   (A_h = arriendo imputado MESE)
    H["ingreso_requerido"] = H["Npersug"] * H["Lp"] - H["arriendo_imputado_mese"]
    H["no_puede_ser_pobre"] = (H["ingreso_requerido"] <= 0).astype("float32")
    H["log_ingreso_requerido"] = np.log(H["ingreso_requerido"].clip(lower=0.05 * H["Lp"]))
    H["share_arriendo_imputado"] = H["arriendo_imputado_mese"] / (H["Npersug"] * H["Lp"])
    H["n_ug_menos_nper"] = H["Nper"] - H["Npersug"]
    # --- Variables adicionales (notebook 05): proxies de ingreso permanente y su escala ---
    lin = H["Lp"] * npug
    H["log_mincer_pc_linea"] = np.log((H["n_mincer_ocupado"] + 0.05) / npug)       # capacidad laboral por miembro
    H["mincer_sobre_requerido"] = np.log((H["n_mincer_ocupado"] + 0.05) * H["Lp"] / H["ingreso_requerido"].clip(lower=0.05 * H["Lp"]))
    H["asalariados_formales_pc"] = H["n_asalariado_formal"] / npug
    H["horas_formales_pc"] = H["n_horas_formal"] / npug
    H["ocupados_superior_pc"] = H["n_ocupado_superior"] / npug
    H["log_arriendo_por_cuarto"] = np.log(H["arriendo"].clip(lower=1e4) / H["cuartos"].clip(lower=1))
    H["log_ingreso_requerido_pc"] = H["log_ingreso_requerido"] - np.log(npug)
    H["requerido_sobre_linea"] = H["ingreso_requerido"] / lin                       # 1 − share de arriendo imputado
    H["arriendo_x_cabecera"] = H["log_arriendo_sobre_linea"] * H["cabecera"]
    H["arriendo_x_inquilino"] = H["log_arriendo_sobre_linea"] * H["ten_arriendo"]
    H["ocupados_x_formal"] = H["ocupados_pc"] * H["share_formal_ocupados"]
    H["dependientes_por_ocupado"] = (H["Nper"] - H["n_ocupado"]) / H["n_ocupado"].clip(lower=1)
    H["educ_jefe_x_formal"] = H["jefe_anios_educ"].fillna(0) * H["jefe_cotiza_pension"].fillna(0)
    return H


def construir(hog: pd.DataFrame, per: pd.DataFrame, ref_arriendo: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pipeline completo hogar + personas → tabla de modelado (una fila por hogar)."""
    vp = variables_persona(per)
    A = agregar_hogar(vp)
    Hh = variables_hogar(hog)
    H = Hh.merge(A, left_on="id", right_index=True, how="left")
    H = imputar_arriendo(H, ref_arriendo)
    H = derivadas(H)
    return H


def objetivo(hog: pd.DataFrame, piso: float = 0.05) -> pd.DataFrame:
    """Objetivos: Pobre (binario); z = log(Ingpcug / Lp) con piso en log(piso); y
    log_ingtotug = log del ingreso SIN arriendo imputado (piso: piso·Lp), que se compara con
    log_ingreso_requerido en el probit latente."""
    z = np.log(np.maximum(hog["Ingpcug"].values, piso * hog["Lp"].values) / hog["Lp"].values)
    li = np.log(np.maximum(hog["Ingtotug"].values, piso * hog["Lp"].values))
    return pd.DataFrame({"id": hog["id"].values, "Pobre": hog["Pobre"].values.astype(int), "z": z,
                         "log_ingtotug": li,
                         "brecha": np.clip(1 - hog["Ingpcug"].values / hog["Lp"].values, 0, 1),
                         "Indigente": hog["Indigente"].values.astype(int)})
