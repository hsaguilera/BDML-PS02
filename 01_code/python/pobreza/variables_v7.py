r"""Versión 7 de las variables (cuaderno 02d): todas las columnas comunes a train y test, agregadas al hogar,
más las familias construidas con razonamiento económico, el bloque IPM y la codificación con la etiqueta fuera de
pliegue con encogimiento bayesiano empírico jerárquico (beta-binomial).

Reglas del usuario (plan maestro v7, vinculantes)
-------------------------------------------------
1. Sin primera etapa ni nada derivado de modelos entrenados con ingresos (v3/v5 prohibidas como covariables).
2. Se pueden usar las variables de ingreso o monetarias que estén TANTO en train COMO en test (preguntas sí/no de
   recepción de ingresos y prestaciones, montos de vivienda P5100/P5130/P5140, líneas Lp y Li). Lo que sólo está en
   train (montos de ingreso de personas, Ingtot*, Ingpcug, Npobres, Indigente, Estrato1, Cclasnr*…) NO, salvo `Pobre`.
3. Sin datos externos ni constantes externas (coeficientes de Mincer, salario mínimo, escalas de equivalencia):
   sólo definiciones metodológicas documentadas (umbrales del IPM y del hacinamiento crítico).

Garantía por construcción: `construir_v7` recibe las tablas crudas completas pero lo PRIMERO que hace es quedarse con
las columnas comunes (`HOG_COMUNES`, `PER_COMUNES`); ninguna función posterior ve una columna de ingreso. La etiqueta
sólo entra en `CodificadorObjetivo` / `codificar` / `TransformadorV7`, siempre fuera de pliegue.

Familias (letra → prefijo)
--------------------------
B  base permitida: las columnas de `features.py` del conjunto S2 de v5 sin el índice de Mincer.
U  umbral observable y montos de vivienda (`u_`, `Lp`, `Li`, `arriendo_pagado`, …).
I  ingresos observables en ambos conjuntos: preguntas sí/no de recepción (`i_`, `s_`).
C  capacidad de generar ingreso sin montos (`c_`, `principal_*`).
P  bloque IPM (Angulo, Díaz y Pardo, 2016) aproximado con estas columnas (`p_`).
V  composición, vivienda, territorio y protección social (`v_`, `log_fex`).
E  codificación con la etiqueta fuera de pliegue (`te_`).
X  interacciones con motivación económica y guiadas por SHAP (`x_`).
G  agregación exhaustiva de TODAS las columnas comunes de persona (`g_`).
T  transformaciones univariadas, umbrales con sentido económico y dummies (`t_`).

Diccionario (DANE, *ddi-documentation-spanish-608*, MESE 2018): las preguntas sí/no usan 1 = sí, 2 = no y, cuando la
pregunta lo admite, 9 = no sabe/no informa. P6630s*, P7422, P7472, P7495 y P7505 sólo admiten 1/2. P7495 y P7505 son
FILTROS: P7500s2–s3 sólo se preguntan si P7495 = 1 y P7510s* sólo si P7505 = 1 (verificado en los datos). El universo
de las preguntas de prestaciones (P6510…P6630s6) son los asalariados (P6430 ∈ {1, 2, 3, 8}); P7422 se pregunta a los
desocupados cesantes y P7472 a los inactivos. P6090 sólo se observa para personas de 10 años o más en estos datos.
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from . import permitidas as PM

# ------------------------------------------------------------------------------ columnas y prohibiciones
HOG_COMUNES = list(PM.HOG_COMUNES)
PER_COMUNES = list(PM.PER_COMUNES)
ETIQUETA = "Pobre"
HOG_PROHIBIDAS = list(PM.HOG_PROHIBIDAS)
OBJETIVO_DERIVADAS = list(PM.OBJETIVO_DERIVADAS)
CONSTANTES_EXTERNAS = dict(PM.CONSTANTES_EXTERNAS)
PATRONES_PROHIBIDOS = list(PM.PATRONES_PROHIBIDOS)
CAT_TODAS = ["Dominio", "Depto", "jefe_oficio", "jefe_posicion", "jefe_nivel_edu", "jefe_regimen_salud",
             "jefe_actividad", "jefe_tam_empresa", "tenencia"]
NO_NUMERICAS = ["id", "Dominio", "Depto"]

FAMILIAS = {
    "B": "0. Base permitida (S2 de v5 sin Mincer + categóricas)",
    "U": "1. Umbral observable y montos de vivienda",
    "I": "2. Ingresos observables en train y test (preguntas sí/no)",
    "C": "3. Capacidad de ingreso sin montos",
    "P": "4. Bloque IPM (pobreza multidimensional)",
    "V": "5. Composición, vivienda, territorio y protección social",
    "E": "6. Codificación con la etiqueta fuera de pliegue (bayes empírico)",
    "X": "7. Interacciones (económicas y guiadas por SHAP)",
    "G": "8. Agregación exhaustiva de las columnas comunes",
    "T": "9. Transformaciones, umbrales y dummies",
}
ORDEN_FAMILIAS = ["U", "I", "C", "P", "V", "E", "X", "G", "T"]

# Preguntas sí/no de recepción de ingresos y prestaciones presentes en train y test
SI_NO_INGRESO = {
    "P6510": ("horas_extra", "¿El mes pasado recibió ingresos por concepto de horas extras?", "asalariados"),
    "P6545": ("primas", "El mes pasado recibió primas (técnica, de antigüedad, clima, orden público, otras)", "asalariados"),
    "P6580": ("bonificaciones", "¿El mes pasado recibió bonificaciones?", "asalariados"),
    "P6585s1": ("aux_alimentacion", "¿El mes pasado recibió auxilio o subsidio de alimentación?", "asalariados"),
    "P6585s2": ("aux_transporte", "¿El mes pasado recibió auxilio o subsidio de transporte?", "asalariados"),
    "P6585s3": ("subsidio_familiar", "¿El mes pasado recibió subsidio familiar?", "asalariados"),
    "P6585s4": ("subsidio_educativo", "¿El mes pasado recibió subsidio educativo?", "asalariados"),
    "P6590": ("especie_alimentos", "Además del salario en dinero, ¿recibió alimentos como parte de pago?", "asalariados"),
    "P6600": ("especie_vivienda", "Además del salario en dinero, ¿recibió vivienda como parte de pago?", "asalariados"),
    "P6610": ("transporte_empresa", "¿Normalmente utiliza transporte de la empresa para ir a su trabajo?", "asalariados"),
    "P6620": ("especie_otros", "¿Recibió otros ingresos en especie (electrodomésticos, ropa, bonos tipo Sodexo)?", "asalariados"),
    "P6630s1": ("prima_servicios", "En los últimos 12 meses recibió prima de servicios", "asalariados"),
    "P6630s2": ("prima_navidad", "En los últimos 12 meses recibió prima de navidad", "asalariados"),
    "P6630s3": ("prima_vacaciones", "En los últimos 12 meses recibió prima de vacaciones", "asalariados"),
    "P6630s4": ("viaticos", "En los últimos 12 meses recibió viáticos permanentes", "asalariados"),
    "P6630s6": ("bonif_anuales", "En los últimos 12 meses recibió bonificaciones anuales", "asalariados"),
    "P7422": ("desocupado_ingreso", "Desocupados: ¿recibió o ganó el mes pasado ingresos por trabajo?", "desocupados cesantes"),
    "P7472": ("inactivo_ingreso", "Inactivos: ¿recibió o ganó el mes pasado ingresos por trabajo?", "inactivos"),
    "P7495": ("arriendos_o_pensiones", "El mes pasado, ¿recibió pagos por arriendos y/o pensiones? (filtro)", "10 años y más"),
    "P7500s2": ("pension_jubilacion", "¿Recibió pagos por pensiones o jubilaciones (vejez, invalidez, sustitución)?", "P7495 = 1"),
    "P7500s3": ("pension_alimenticia", "¿Recibió pagos por pensión alimenticia (paternidad, divorcio, separación)?", "P7495 = 1"),
    "P7505": ("filtro_doce_meses", "Últimos 12 meses: ¿dinero de otros hogares o instituciones, intereses, dividendos o cesantías? (filtro)", "10 años y más"),
    "P7510s1": ("ayuda_hogares_pais", "Últimos 12 meses: dinero de otros hogares o personas residentes en el país", "P7505 = 1"),
    "P7510s2": ("remesas_exterior", "Últimos 12 meses: dinero de hogares o personas residentes fuera del país", "P7505 = 1"),
    "P7510s3": ("ayudas_instituciones", "Últimos 12 meses: ayudas en dinero de instituciones del país o de fuera", "P7505 = 1"),
    "P7510s5": ("intereses_dividendos", "Últimos 12 meses: intereses, CDT, depósitos, utilidades o dividendos", "P7505 = 1"),
    "P7510s6": ("cesantias", "Últimos 12 meses: cesantías y/o intereses a las cesantías", "P7505 = 1"),
    "P7510s7": ("otras_fuentes", "Últimos 12 meses: dinero de otras fuentes", "P7505 = 1"),
}
SI_NO_OTRAS = {"P7040": "segundo trabajo o negocio la semana pasada", "P7090": "quiere trabajar más horas",
               "P7110": "hizo diligencias para trabajar más horas", "P7120": "disponible para trabajar más horas",
               "P7150": "hizo diligencias para cambiar de trabajo", "P7160": "podría empezar un nuevo trabajo antes de un mes"}
CATEGORICAS_PERSONA = {
    "P6020": ("sexo", [1, 2]), "P6050": ("parentesco", list(range(1, 10))), "P6090": ("afiliado_salud", [1, 2, 9]),
    "P6100": ("regimen_salud", [1, 2, 3, 9]), "P6210": ("nivel_educativo", [1, 2, 3, 4, 5, 6, 9]),
    "P6240": ("actividad", [1, 2, 3, 4, 5, 6]), "P6430": ("posicion", list(range(1, 10))),
    "P6870": ("tam_empresa", list(range(1, 10))), "P6920": ("cotiza_pension", [1, 2, 3]),
    "P7050": ("posicion_segunda", list(range(1, 10))), "P7350": ("posicion_ultimo_desocupado", list(range(1, 10))),
    "P7310": ("busca_primera_vez", [1, 2]),
}
NUMERICAS_PERSONA = {"P6040": "edad", "P6210s1": "grado aprobado", "P6426": "antigüedad en el empleo (meses)",
                     "P6800": "horas normales en el trabajo principal", "P7045": "horas en el segundo trabajo"}
FLAGS_PERSONA = {"Pet": "población en edad de trabajar", "Oc": "ocupado", "Des": "desocupado", "Ina": "inactivo"}

# Variables continuas que reciben transformaciones univariadas (familia T) y posiciones dentro del dominio
TRANSFORMAR = ["ingreso_requerido", "u_n_lineas", "u_requerido_pc", "arriendo", "arriendo_sobre_linea",
               "share_arriendo_imputado", "requerido_sobre_linea", "personas_por_dormitorio", "personas_por_cuarto",
               "dependencia", "ocupados_pc", "formales_pc", "horas_pc", "c_hs_linea", "c_horas_linea",
               "c_perceptores_linea", "i_perceptores_linea", "c_formales_linea", "educ_max_adultos",
               "educ_media_adultos", "jefe_anios_educ", "jefe_edad", "edad_media", "Nper", "Npersug", "cuartos",
               "n_horas", "c_hs", "c_hs_pc", "c_hx_linea", "c_exp_media_oc", "antiguedad_media_oc",
               "p_puntaje", "p_educ_media_15", "share_subsidiado", "share_menores", "u_arriendo_estimado_linea"]
N_GRUPO = 20          # las primeras 20 de TRANSFORMAR reciben rango y decil dentro del Dominio


# ------------------------------------------------------------------------------ utilidades
def comunes(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    falt = [c for c in cols if c not in df.columns]
    assert not falt, f"faltan columnas comunes: {falt}"
    return df[cols].copy()


def prohibidas_persona(tp: pd.DataFrame, tep: pd.DataFrame) -> list:
    return [c for c in tp.columns if c not in tep.columns]


def _f(x) -> np.ndarray:
    return np.asarray(x.values if hasattr(x, "values") else x, dtype=float)


def _div(a, b):
    a, b = _f(a), _f(b)
    with np.errstate(all="ignore"):
        return np.where(b != 0, a / np.where(b != 0, b, 1.0), np.nan)


def _log(x, piso):
    return np.log(np.maximum(_f(x), piso))


class _Agr:
    """Agregación vectorizada persona → hogar (np.ufunc.reduceat) en el orden de la tabla de hogares."""

    def __init__(self, id_persona, id_hogar):
        pos = pd.Index(id_hogar).get_indexer(id_persona)
        if (pos < 0).any():
            raise ValueError("personas sin hogar")
        self.pos = pos
        self.o = np.argsort(pos, kind="stable")
        ps = pos[self.o]
        self.starts = np.r_[0, np.flatnonzero(np.diff(ps)) + 1]
        if len(self.starts) != len(id_hogar):
            raise ValueError("hay hogares sin personas")
        self.nh = len(id_hogar)

    def suma(self, x):
        v = _f(x)[self.o]
        return np.add.reduceat(np.where(np.isnan(v), 0.0, v), self.starts)

    def n(self, cond):
        return self.suma(np.asarray(cond, dtype=float))

    def n_validos(self, x):
        return self.n(~np.isnan(_f(x)))

    def media(self, x):
        s, k = self.suma(x), self.n_validos(x)
        return np.where(k > 0, s / np.maximum(k, 1), np.nan)

    def maximo(self, x):
        v = _f(x)[self.o]
        r = np.maximum.reduceat(np.where(np.isnan(v), -np.inf, v), self.starts)
        return np.where(np.isinf(r), np.nan, r)

    def minimo(self, x):
        v = _f(x)[self.o]
        r = np.minimum.reduceat(np.where(np.isnan(v), np.inf, v), self.starts)
        return np.where(np.isinf(r), np.nan, r)

    def desv(self, x):
        m, m2 = self.media(x), self.media(_f(x) ** 2)
        return np.sqrt(np.maximum(m2 - m ** 2, 0))

    def de(self, x, sel):
        out = np.full(self.nh, np.nan)
        idx = np.flatnonzero(np.asarray(sel, bool))
        out[self.pos[idx]] = _f(x)[idx]
        return out


class _Reg:
    """Registro de columnas nuevas con su familia y su descripción."""

    def __init__(self, n):
        self.n, self.cols, self.fam, self.desc = n, {}, {}, {}

    def add(self, nombre, valor, fam, desc=""):
        if nombre in self.cols:
            raise ValueError(f"columna repetida: {nombre}")
        v = _f(valor) * np.ones(self.n)
        v = np.where(np.isfinite(v), v, np.nan).astype("float32")
        self.cols[nombre], self.fam[nombre], self.desc[nombre] = v, fam, desc

    def frame(self, index):
        return pd.DataFrame(self.cols, index=index)


# ------------------------------------------------------------------------------ contexto por persona
class _Ctx:
    """Variables individuales comunes a todas las familias (sólo columnas comunes)."""

    def __init__(self, p: pd.DataFrame, H: pd.DataFrame):
        self.p = p
        self.A = _Agr(p["id"].values, H["id"].values)
        P6050 = p["P6050"].values
        self.edad = _f(p["P6040"])
        self.S = F.anios_educacion(p["P6210"], p["P6210s1"]).values.astype(float)
        self.ug = ~np.isin(P6050, [6, 7, 8])                      # unidad de gasto (MESE): sin domésticos ni pensionistas
        self.jefe = P6050 == 1
        self.pareja = P6050 == 2
        self.oc = p["Oc"].values == 1
        self.des = p["Des"].values == 1
        self.ina = p["Ina"].values == 1
        self.pet = p["Pet"].values == 1
        h = (p["P6800"].fillna(0) + p["P7045"].fillna(0)).values.astype(float)
        self.horas = np.where(self.oc, h, 0.0)
        self.exp = np.clip(self.edad - np.nan_to_num(self.S, nan=5.0) - 6, 0, None)
        self.formal = self.oc & (p["P6920"].values == 1)
        self.cubierto = self.oc & np.isin(p["P6920"].values, [1, 3])
        self.asal = np.isin(p["P6430"].values, [1, 2])
        self.asal_amplio = np.isin(p["P6430"].values, [1, 2, 3, 8])   # universo de las preguntas de prestaciones
        self.cp = p["P6430"].values == 4
        self.patron = p["P6430"].values == 5
        self.sin_rem = np.isin(p["P6430"].values, [6, 7])
        self.tam = np.where(self.oc, _f(p["P6870"]), np.nan)
        # perceptor principal SIN montos: ocupado de la UG con mayor horas × (1 + años de educación); empates → menor Orden
        score = np.where(self.oc & self.ug, self.horas * (1 + np.nan_to_num(self.S)), -1.0)
        n = len(p)
        o2 = np.lexsort((np.arange(n), -score, self.A.pos))
        ps2 = self.A.pos[o2]
        prim = o2[np.r_[0, np.flatnonzero(np.diff(ps2)) + 1]]
        self.principal = np.zeros(n, bool)
        self.principal[prim[score[prim] >= 0]] = True
        self.npug = np.maximum(_f(H["Npersug"]), 1.0)
        self.nper = np.maximum(_f(H["Nper"]), 1.0)
        R = _f(H["ingreso_requerido"])
        self.nl = R / _f(H["Lp"])                                   # «líneas necesarias» n_ℓ = R_h / Lp
        self.nlc = np.maximum(self.nl, 0.05)                         # piso de log_ingreso_requerido (R ≥ 0,05·Lp)

    def u(self, cond):
        """Conteo sobre la unidad de gasto."""
        return self.A.n(np.asarray(cond, bool) & self.ug)

    def un(self, x, cond=None):
        """Valor restringido a la UG (y a `cond`), NaN fuera."""
        m = self.ug if cond is None else (self.ug & np.asarray(cond, bool))
        return np.where(m, _f(x), np.nan)


def _codigo(x):
    """Recodifica una respuesta sí/no: 1 → 1, 2 → 0, 9/NA → NaN."""
    x = _f(x)
    return np.select([x == 1, x == 2], [1.0, 0.0], np.nan)


# ------------------------------------------------------------------------------ familia U
def _fam_U(reg, H, h, ctx):
    lp, li, npug = _f(H["Lp"]), _f(H["Li"]), ctx.npug
    lin = lp * npug
    A = _f(H["arriendo_imputado_mese"])
    R = _f(H["ingreso_requerido"])
    p5140, ns40 = F._monto(h["P5140"])
    p5130, ns30 = F._monto(h["P5130"])
    p5100, ns00 = F._monto(h["P5100"])
    p5140, p5130, p5100 = _f(p5140), _f(p5130), _f(p5100)
    arr = _f(H["arriendo"])
    reg.add("u_log_n_lineas", np.log(ctx.nlc), "U", "log de las líneas necesarias n_ℓ = R_h/Lp (piso 0,05)")
    reg.add("u_arriendo_imp_lineas", A / lp, "U", "arriendo imputado A_h en número de líneas per cápita")
    reg.add("u_holgura_lineas", np.maximum(-R, 0) / lp, "U", "cuánto excede A_h a la línea del hogar (en líneas); > 0 ⇔ no puede ser pobre")
    reg.add("u_linea_hogar", lin, "U", "línea del hogar Npersug·Lp")
    reg.add("u_lp_menos_li_rel", (lp - li) / lp, "U", "(Lp − Li)/Lp: peso de lo no alimentario en la canasta (dato de la competencia)")
    reg.add("u_p5140_ns", ns40, "U", "P5140 (arriendo pagado) = 98/99 no sabe/no informa")
    reg.add("u_p5130_ns", ns30, "U", "P5130 (arriendo estimado) = 98/99 no sabe/no informa")
    reg.add("u_p5100_ns", ns00, "U", "P5100 (cuota de amortización) = 98/99 no sabe/no informa")
    reg.add("u_arriendo_pagado_linea", p5140 / lin, "U", "arriendo pagado P5140 / (Npersug·Lp)")
    reg.add("u_log_arriendo_pagado_linea", _log(p5140 / lin, 1e-3), "U", "log del arriendo pagado sobre la línea del hogar")
    reg.add("u_arriendo_estimado_linea", p5130 / lin, "U", "arriendo estimado P5130 / (Npersug·Lp)")
    reg.add("u_log_arriendo_estimado_linea", _log(p5130 / lin, 1e-3), "U", "log del arriendo estimado sobre la línea del hogar")
    reg.add("u_cuota_linea", p5100 / lin, "U", "cuota de amortización P5100 / (Npersug·Lp)")
    reg.add("u_cuota_sobre_estimado", _div(p5100, p5130), "U", "cuota de amortización / arriendo estimado (propia pagando)")
    reg.add("u_estimado_menos_cuota_linea", (p5130 - p5100) / lin, "U", "(P5130 − P5100)/línea del hogar: valor neto de uso de la vivienda que se paga")
    reg.add("u_requerido_sobre_arriendo", R / np.maximum(arr, 1e4), "U", "R_h / arriendo (pagado o estimado): cuántos arriendos necesita el hogar")
    reg.add("u_log_req_menos_log_arriendo", _log(R, 1.0) - _log(arr, 1e4), "U", "log R_h − log arriendo (piso de R en 1 peso)")
    ten = _f(h["P5090"])
    reg.add("u_A_anulado_por_regla", (np.isin(ten, [1, 2, 4]) & (A == 0)).astype(float), "U",
            "tenencia con arriendo imputado (1, 2, 4) pero A_h = 0 por las reglas exactas (tope 3.000.000, 98/99)")
    reg.add("u_requerido_sobre_li", R / (li * npug), "U", "R_h / (Npersug·Li): umbral frente a la canasta alimentaria")


# ------------------------------------------------------------------------------ familia I
def _fam_I(reg, H, ctx):
    p, A, ug = ctx.p, ctx.A, ctx.ug
    for q, (nom, txt, univ) in SI_NO_INGRESO.items():
        x = _f(p[q])
        nsi, nresp = ctx.u(x == 1), ctx.u(~np.isnan(x))
        reg.add(f"i_{nom}_si", nsi, "I", f"miembros de la UG con «sí» en {q} ({txt}; universo: {univ})")
        reg.add(f"i_{nom}_ns", ctx.u(x == 9), "I", f"miembros de la UG con «no sabe/no informa» (9) en {q}")
        reg.add(f"i_{nom}_resp", nresp, "I", f"miembros de la UG a quienes se les hizo {q} (universo de la pregunta)")
        reg.add(f"i_{nom}_tasa", np.where(nresp > 0, nsi / np.maximum(nresp, 1), np.nan), "I", f"proporción de «sí» entre quienes respondieron {q}")
        reg.add(f"i_{nom}_pc", nsi / ctx.npug, "I", f"«sí» en {q} por miembro de la UG")
        reg.add(f"i_{nom}_linea", nsi / ctx.nlc, "I", f"«sí» en {q} por línea necesaria n_ℓ")
        reg.add(f"i_{nom}_jefe", A.de(_codigo(x), ctx.jefe), "I", f"respuesta del jefe a {q} (1 sí, 0 no, NaN no aplica/no sabe)")
    g = lambda c: _f(p[c])
    fuentes = {
        "trabajo_remunerado": ctx.oc & ~ctx.sin_rem, "segunda_actividad": g("P7040") == 1,
        "pension": g("P7500s2") == 1, "pension_alimenticia": g("P7500s3") == 1,
        "arriendos_inferidos": (g("P7495") == 1) & (g("P7500s2") != 1) & (g("P7500s3") != 1),
        "ayuda_hogares": g("P7510s1") == 1, "remesas": g("P7510s2") == 1, "instituciones": g("P7510s3") == 1,
        "intereses": g("P7510s5") == 1, "cesantias": g("P7510s6") == 1, "otras": g("P7510s7") == 1,
        "desocupado_ingreso": g("P7422") == 1, "inactivo_ingreso": g("P7472") == 1}
    lab = ["trabajo_remunerado", "segunda_actividad", "desocupado_ingreso", "inactivo_ingreso"]
    nolab = [k for k in fuentes if k not in lab]
    cuentas = {k: ctx.u(v) for k, v in fuentes.items()}
    reg.add("i_arriendos_inferidos_si", cuentas["arriendos_inferidos"], "I",
            "P7495 = 1 sin pensión (P7500s2) ni pensión alimenticia (P7500s3) ⇒ recibió arriendos (P7500s1 sólo está en train)")
    reg.add("i_trabajo_remunerado_n", cuentas["trabajo_remunerado"], "I", "ocupados de la UG que no son trabajadores sin remuneración")
    reg.add("i_fuentes_distintas", sum((cuentas[k] > 0).astype(float) for k in fuentes), "I",
            "número de tipos distintos de fuente de ingreso presentes en la UG (13 tipos)")
    reg.add("i_fuentes_distintas_nolab", sum((cuentas[k] > 0).astype(float) for k in nolab), "I", "tipos distintos de fuentes no laborales")
    tot = sum(cuentas.values())
    M = np.column_stack([fuentes[k] for k in fuentes])
    perc = ctx.u(M.any(1))
    perc_lab = ctx.u(np.column_stack([fuentes[k] for k in lab]).any(1))
    perc_nolab = ctx.u(np.column_stack([fuentes[k] for k in nolab]).any(1))
    reg.add("i_fuentes_total", tot, "I", "suma sobre la UG del número de fuentes de cada persona")
    reg.add("i_perceptores", perc, "I", "perceptores potenciales: miembros de la UG con al menos una fuente")
    reg.add("i_perceptores_lab", perc_lab, "I", "perceptores potenciales de ingreso laboral")
    reg.add("i_perceptores_nolab", perc_nolab, "I", "perceptores potenciales de ingreso no laboral")
    for nom, v in [("i_perceptores", perc), ("i_perceptores_lab", perc_lab), ("i_perceptores_nolab", perc_nolab)]:
        reg.add(f"{nom}_pc", v / ctx.npug, "I", f"{nom} por miembro de la UG")
        reg.add(f"{nom}_linea", v / ctx.nlc, "I", f"{nom} por línea necesaria n_ℓ")
    reg.add("i_fuentes_por_perceptor", tot / np.maximum(perc, 1), "I", "fuentes por perceptor (diversificación)")
    reg.add("i_lineas_por_perceptor", ctx.nl / np.maximum(perc, 1), "I", "líneas necesarias que debe cubrir cada perceptor")
    reg.add("i_dependientes_por_perceptor", (ctx.npug - perc) / np.maximum(perc, 1), "I", "miembros sin fuente por perceptor")
    reg.add("i_sin_fuentes", (perc == 0).astype(float), "I", "ningún miembro de la UG reporta fuente de ingreso")
    reg.add("i_solo_nolab", ((perc_lab == 0) & (perc_nolab > 0)).astype(float), "I", "sólo fuentes no laborales")
    reg.add("i_transferencias_n", cuentas["ayuda_hogares"] + cuentas["remesas"] + cuentas["instituciones"], "I",
            "receptores de ayudas de otros hogares, remesas o ayudas institucionales")
    reg.add("i_capital_n", cuentas["intereses"] + cuentas["arriendos_inferidos"], "I", "receptores de rentas de capital (intereses, arriendos inferidos)")
    reg.add("i_pension_o_capital", ((cuentas["pension"] + cuentas["intereses"] + cuentas["arriendos_inferidos"]) > 0).astype(float), "I",
            "algún miembro con pensión o rentas de capital")
    est = lambda cs: sum((g(c) == 1).astype(float) for c in cs)
    reg.add("i_prestaciones_legales", A.suma(np.where(ug, est(["P6630s1", "P6630s2", "P6630s3"]), 0)), "I",
            "primas de servicios, navidad y vacaciones recibidas en la UG (prestaciones de ley ⇒ formalidad)")
    reg.add("i_extralegales", A.suma(np.where(ug, est(["P6545", "P6580", "P6630s4", "P6630s6", "P6585s1", "P6585s4"]), 0)), "I",
            "primas, bonificaciones, viáticos, auxilio de alimentación y subsidio educativo (beneficios extralegales)")
    reg.add("i_especie", A.suma(np.where(ug, est(["P6590", "P6600", "P6610", "P6620"]), 0)), "I", "pagos en especie recibidos en la UG")
    ns = sum((g(c) == 9).astype(float) for c in SI_NO_INGRESO)
    reg.add("i_ns_total", A.suma(np.where(ug, ns, 0)), "I", "respuestas «no sabe/no informa» (9) a preguntas de ingreso en la UG")
    reg.add("i_ns_algun", (A.suma(np.where(ug, ns, 0)) > 0).astype(float), "I", "alguna respuesta «no sabe/no informa» a preguntas de ingreso")
    detalle = np.column_stack([g(c) == 1 for c in ["P7510s1", "P7510s2", "P7510s3", "P7510s5", "P7510s6", "P7510s7"]]).any(1)
    reg.add("i_filtro12m_sin_detalle", ctx.u((g("P7505") == 1) & ~detalle), "I", "P7505 = 1 sin ninguna fuente detallada en P7510s*")
    asal = ctx.u(ctx.asal_amplio)
    reg.add("i_aux_transporte_por_asal", cuentas_si(ctx, "P6585s2") / np.maximum(asal, 1), "I",
            "asalariados con auxilio de transporte / asalariados (por ley sólo salarios ≤ 2 SMMLV: señal de salario bajo)")
    reg.add("i_prima_servicios_por_asal", cuentas_si(ctx, "P6630s1") / np.maximum(asal, 1), "I", "prima de servicios / asalariados (formalidad)")


def cuentas_si(ctx, q):
    return ctx.u(_f(ctx.p[q]) == 1)


# ------------------------------------------------------------------------------ familia C
def _fam_C(reg, H, ctx):
    p, A = ctx.p, ctx.A
    oc_ug = ctx.oc & ctx.ug
    noc = ctx.u(ctx.oc)
    antig = ctx.un(p["P6426"], ctx.oc)
    reg.add("c_antig_max_oc", A.maximo(antig), "C", "antigüedad máxima en el empleo entre los ocupados de la UG (meses)")
    reg.add("c_antig_min_oc", A.minimo(antig), "C", "antigüedad mínima en el empleo entre los ocupados de la UG")
    reg.add("c_antig_desv_oc", A.desv(antig), "C", "desviación de la antigüedad de los ocupados")
    reg.add("c_horas_por_ocupado", A.suma(np.where(ctx.ug, ctx.horas, 0)) / np.maximum(noc, 1), "C", "horas semanales por ocupado")
    reg.add("c_horas_segunda", A.suma(np.where(oc_ug, np.nan_to_num(_f(p["P7045"])), 0)), "C", "horas en la segunda actividad (UG)")
    reg.add("c_subempleo_n", ctx.u(_f(p["P7090"]) == 1), "C", "ocupados que quieren trabajar más horas (P7090)")
    reg.add("c_busca_cambiar_n", ctx.u(_f(p["P7150"]) == 1), "C", "ocupados que buscaron cambiar de trabajo (P7150)")
    reg.add("c_disponible_n", ctx.u(_f(p["P7160"]) == 1), "C", "podría empezar un nuevo trabajo antes de un mes (P7160)")
    t = _f(p["P6870"])
    for nom, cods, d in [("solo", [1], "trabaja solo"), ("2_10", [2, 3, 4], "2 a 10 personas"),
                         ("11_50", [5, 6, 7], "11 a 50 personas"), ("51mas", [8, 9], "51 o más personas")]:
        reg.add(f"c_tam_{nom}_n", ctx.u(ctx.oc & np.isin(t, cods)), "C", f"ocupados de la UG en empresas de {d} (P6870)")
    S_oc = ctx.un(ctx.S, ctx.oc)
    reg.add("c_educ_min_oc", A.minimo(S_oc), "C", "años de educación mínimos de los ocupados")
    reg.add("c_educ_suma_oc", A.suma(S_oc), "C", "años de educación sumados de los ocupados (capital humano empleado)")
    reg.add("c_exp_max_oc", A.maximo(ctx.un(ctx.exp, ctx.oc)), "C", "experiencia potencial máxima de los ocupados (edad − años − 6)")
    ej = A.de(ctx.exp, ctx.jefe)
    reg.add("c_exp_jefe2", ej ** 2 / 100, "C", "cuadrado de la experiencia potencial del jefe / 100 (concavidad del perfil de ingresos)")
    h = np.where(ctx.ug, ctx.horas, 0.0)
    S0 = np.nan_to_num(ctx.S)
    reg.add("c_hs_formal_linea", A.suma(h * S0 * ctx.formal) / ctx.nlc, "C", "Σ horas × años de educación de los ocupados formales / n_ℓ")
    reg.add("c_h_tam_linea", A.suma(h * np.nan_to_num(ctx.tam)) / ctx.nlc, "C", "Σ horas × tamaño de empresa / n_ℓ")
    reg.add("c_hx_pc", A.suma(h * ctx.exp) / ctx.npug, "C", "Σ horas × experiencia potencial por miembro")
    nform, nasal = ctx.u(ctx.formal), ctx.u(ctx.asal)
    reg.add("c_formales_menos_lineas", nform - ctx.nl, "C", "ocupados formales − líneas necesarias")
    reg.add("c_asalariados_menos_lineas", nasal - ctx.nl, "C", "asalariados − líneas necesarias")
    reg.add("c_lineas_por_ocupado", ctx.nl / np.maximum(noc, 1), "C", "líneas necesarias por ocupado (carga que soporta cada ocupado)")
    reg.add("c_lineas_por_formal", ctx.nl / np.maximum(nform, 1), "C", "líneas necesarias por ocupado formal")
    reg.add("c_jefe_horas_linea", A.de(ctx.horas, ctx.jefe) / ctx.nlc, "C", "horas del jefe / n_ℓ")
    reg.add("c_jefe_y_pareja_ocupados", A.n(ctx.oc & (ctx.jefe | ctx.pareja)), "C", "jefe y pareja ocupados (0–2)")
    pr = ctx.principal
    of = _f(p["Oficio"]); pos = _f(p["P6430"])
    reg.add("principal_oficio", A.de(of, pr), "C", "Oficio (CIUO-68, 2 dígitos) del perceptor principal (sin montos: mayor horas × (1+años))")
    reg.add("principal_oficio_gg", np.floor(A.de(of, pr) / 10), "C", "gran grupo ocupacional (primer dígito) del perceptor principal")
    reg.add("jefe_oficio_gg", np.floor(A.de(of, ctx.jefe) / 10), "C", "gran grupo ocupacional del jefe")
    reg.add("principal_posicion", A.de(pos, pr), "C", "posición ocupacional (P6430) del perceptor principal")
    reg.add("c_principal_antig", A.de(_f(p["P6426"]), pr), "C", "antigüedad del perceptor principal")
    reg.add("c_principal_edad", A.de(ctx.edad, pr), "C", "edad del perceptor principal")
    reg.add("c_principal_mujer", A.de(_f(p["P6020"]) == 2, pr), "C", "perceptor principal mujer")
    reg.add("c_principal_cp", A.de(ctx.cp, pr), "C", "perceptor principal trabaja por cuenta propia")
    reg.add("c_principal_horas_linea", A.de(ctx.horas, pr) / ctx.nlc, "C", "horas del perceptor principal / n_ℓ")
    reg.add("c_principal_hs_linea", A.de(ctx.horas * S0, pr) / ctx.nlc, "C", "horas × años del perceptor principal / n_ℓ")


# ------------------------------------------------------------------------------ familia P (IPM)
PESOS_IPM = {  # (dimensión, peso oficial; Angulo, Díaz y Pardo, 2016)
    "p_bajo_logro": ("educacion", 0.10), "p_analfabetismo": ("educacion", 0.10),
    "p_inasistencia": ("ninez", 0.05), "p_rezago": ("ninez", 0.05), "p_trabajo_infantil": ("ninez", 0.05),
    "p_desempleo": ("trabajo", 0.10), "p_informal": ("trabajo", 0.10),
    "p_sin_salud": ("salud", 0.10), "hacinamiento_critico": ("vivienda", 0.04),
}
IPM_NO_OBSERVABLES = {
    "Barreras de acceso a servicios de cuidado de la primera infancia (niñez y juventud, 0,05)":
        "requiere asistencia de los menores de 5 años a hogar comunitario/jardín y su nutrición: no está en la MESE",
    "Barreras de acceso a servicios de salud dada necesidad (salud, 0,10)":
        "requiere haber tenido un problema de salud y si accedió a atención: no hay preguntas de morbilidad",
    "Sin acceso a fuente de agua mejorada (vivienda, 0,04)": "no hay preguntas de servicios públicos en las tablas",
    "Inadecuada eliminación de excretas (vivienda, 0,04)": "no hay preguntas de servicios públicos en las tablas",
    "Material inadecuado de pisos (vivienda, 0,04)": "no hay preguntas de materiales de la vivienda",
    "Material inadecuado de paredes exteriores (vivienda, 0,04)": "no hay preguntas de materiales de la vivienda",
}


def _fam_P(reg, H, ctx):
    p, A, ug, e, S = ctx.p, ctx.A, ctx.ug, ctx.edad, ctx.S
    m15 = ug & (e >= 15)
    em15 = A.media(np.where(m15, S, np.nan))
    n15 = A.n(m15)
    reg.add("p_educ_media_15", em15, "P", "años de educación promedio de los miembros de 15 años o más")
    d = {}
    d["p_bajo_logro"] = np.where(np.isnan(em15), 0.0, (em15 < 9).astype(float))
    an = m15 & np.isin(_f(p["P6210"]), [1, 2])
    d["p_analfabetismo"] = (A.n(an) > 0).astype(float)
    reg.add("p_share_analfabetismo", A.n(an) / np.maximum(n15, 1), "P", "proporción de ≥ 15 años sin nivel educativo (proxy de analfabetismo)")
    act = _f(p["P6240"])
    m_es = ug & (e <= 16) & ~np.isnan(act)                       # P6240 sólo existe para la PET (≥ 12 urbano, ≥ 10 rural)
    inas = m_es & (act != 3)
    d["p_inasistencia"] = (A.n(inas) > 0).astype(float)
    reg.add("p_share_inasistencia", A.n(inas) / np.maximum(A.n(m_es), 1), "P", "proporción de menores de 17 en la PET cuya actividad principal no es estudiar")
    m717 = ug & (e >= 7) & (e <= 17) & ~np.isnan(S)
    rez = m717 & (S < e - 6)                                    # norma nacional: edad − 6 años aprobados
    rez_h = m717 & (S < e - 7)                                  # con un año de holgura por el mes de la encuesta
    d["p_rezago"] = (A.n(rez) > 0).astype(float)
    reg.add("p_share_rezago", A.n(rez) / np.maximum(A.n(m717), 1), "P", "proporción de 7–17 años con años aprobados < edad − 6")
    reg.add("p_rezago_holgura", (A.n(rez_h) > 0).astype(float), "P", "algún 7–17 con años aprobados < edad − 7 (holgura de un año)")
    ti = ug & (e >= 12) & (e <= 17) & ctx.oc
    d["p_trabajo_infantil"] = (A.n(ti) > 0).astype(float)
    reg.add("p_n_trabajo_infantil", A.n(ti), "P", "ocupados de 12 a 17 años")
    ndes, noc = ctx.u(ctx.des), ctx.u(ctx.oc)
    d["p_desempleo"] = (ndes > 0).astype(float)
    pea = ndes + noc
    inf = ctx.u(ctx.oc & ~ctx.cubierto)
    d["p_informal"] = (inf > 0).astype(float)
    reg.add("p_share_informal_pea", (inf + ndes) / np.maximum(pea, 1), "P", "proporción de la PEA sin empleo formal (ocupados sin cotizar ni pensión + desocupados)")
    reg.add("p_share_desempleo_pea", ndes / np.maximum(pea, 1), "P", "tasa de desempleo del hogar")
    salud = _f(p["P6090"])
    ss = ug & (salud == 2)
    d["p_sin_salud"] = (A.n(ss) > 0).astype(float)
    reg.add("p_share_sin_salud", A.n(ss) / np.maximum(A.n(ug & ~np.isnan(salud)), 1), "P", "proporción sin afiliación a salud (P6090 = 2; ≥ 10 años)")
    hac = _f(H["hacinamiento_critico"])
    for k, v in d.items():
        reg.add(k, v, "P", f"privación IPM: {k[2:].replace('_', ' ')}")
    d["hacinamiento_critico"] = hac
    dims = {}
    for k, (dim, _) in PESOS_IPM.items():
        dims.setdefault(dim, []).append(d[k])
    pd_dim = {dim: np.mean(np.column_stack(v), 1) for dim, v in dims.items()}
    for dim, v in pd_dim.items():
        reg.add(f"p_dim_{dim}", v, "P", f"privación promedio de la dimensión {dim} (indicadores observables)")
    punt = 0.2 * sum(pd_dim.values())
    reg.add("p_n_privaciones", sum(d.values()), "P", "número de privaciones observables (0–9)")
    reg.add("p_n_dimensiones", sum((v > 0).astype(float) for v in pd_dim.values()), "P", "dimensiones con al menos una privación (0–5)")
    reg.add("p_puntaje", punt, "P", "puntaje IPM: cada dimensión pesa 1/5 y su peso se reparte por igual entre los indicadores observables")
    reg.add("p_ipm_pobre", (punt >= 1 / 3 - 1e-9).astype(float), "P", "pobre multidimensional: puntaje ≥ 33 %")
    reg.add("p_intensidad", punt * (punt >= 1 / 3 - 1e-9), "P", "puntaje censurado (Alkire y Foster, 2011): intensidad entre los pobres IPM")
    of = sum(PESOS_IPM[k][1] * d[k] for k in PESOS_IPM)
    reg.add("p_puntaje_oficial", of, "P", "suma de los pesos oficiales de los 9 indicadores observables (máximo 0,69): cota inferior del puntaje oficial")
    reg.add("p_ipm_oficial_33", (of >= 0.33 - 1e-9).astype(float), "P", "cota inferior de la pobreza IPM oficial (puntaje oficial observable ≥ 33 %)")


# ------------------------------------------------------------------------------ familia V
def _fam_V(reg, H, h, ctx):
    p, A, ug, e = ctx.p, ctx.A, ctx.ug, ctx.edad
    reg.add("v_log_fex_dpto", _log(h["Fex_dpto"], 1e-3), "V", "log del factor de expansión departamental (diseño muestral; común a train y test)")
    reg.add("v_fex_c_sobre_dpto", _div(h["Fex_c"], h["Fex_dpto"]), "V", "razón de factores de expansión (diseño muestral)")
    reg_s = _f(p["P6100"])
    adul = ug & (e >= 18)
    reg.add("v_share_subsidiado_adultos", ctx.u(adul & (reg_s == 3)) / np.maximum(ctx.u(adul), 1), "V", "adultos en régimen subsidiado / adultos")
    reg.add("v_share_contributivo", ctx.u(reg_s == 1) / ctx.npug, "V", "miembros en régimen contributivo / miembros")
    reg.add("v_share_especial", ctx.u(reg_s == 2) / ctx.npug, "V", "miembros en régimen especial / miembros")
    nsub, nresp = ctx.u(reg_s == 3), ctx.u(~np.isnan(reg_s))
    reg.add("v_todos_subsidiados", ((nresp > 0) & (nsub == nresp)).astype(float), "V", "todos los afiliados del hogar en régimen subsidiado (focalizado por SISBÉN)")
    reg.add("v_subsidiado_sin_cotizantes", ((nsub > 0) & (ctx.u(ctx.formal) == 0)).astype(float), "V", "algún subsidiado y ningún ocupado cotizante")
    je = A.de(e, ctx.jefe)
    reg.add("v_jefe_mayor65", (je >= 65).astype(float), "V", "jefe de 65 años o más")
    reg.add("v_jefe_menor25", (je < 25).astype(float), "V", "jefe menor de 25 años")
    par = _f(p["P6050"])
    for k, nom in [(3, "hijos"), (4, "nietos"), (5, "otros_parientes")]:
        reg.add(f"v_n_{nom}", ctx.u(par == k), "V", f"número de {nom.replace('_', ' ')} del jefe en la UG")
    for a, b in [(0, 5), (6, 9), (10, 14), (15, 17), (18, 24), (25, 44), (45, 64)]:
        reg.add(f"v_n_edad_{a}_{b}", ctx.u((e >= a) & (e <= b)), "V", f"miembros de {a} a {b} años")
    reg.add("v_cuartos_menos_dormitorios", _f(H["cuartos"]) - _f(H["dormitorios"]), "V", "cuartos que no son dormitorio (sala, comedor…)")
    reg.add("v_dormitorios_pc", _f(H["dormitorios"]) / ctx.nper, "V", "dormitorios por persona")
    reg.add("v_hacinamiento_mitigable", ((_f(H["personas_por_dormitorio"]) > 2) & (_f(H["personas_por_dormitorio"]) <= 3)).astype(float), "V",
            "hacinamiento mitigable: más de 2 y hasta 3 personas por dormitorio")
    reg.add("v_jefa_subsidiada", A.de((_f(p["P6020"]) == 2) & (reg_s == 3), ctx.jefe), "V", "jefa en régimen subsidiado")
    reg.add("v_resto_urbano", (h["Dominio"].values == "RESTO URBANO").astype(float), "V", "dominio «resto urbano» (cabeceras no capitales)")
    reg.add("v_rural", (h["Dominio"].values == "RURAL").astype(float), "V", "dominio rural (centros poblados y rural disperso)")


# ------------------------------------------------------------------------------ familia X (económicas)
def _fam_X(reg, D, ctx):
    g = lambda c: _f(D[c])
    cab = g("cabecera")
    reg.add("x_lineas_menos_ocupados_log", np.log(ctx.nlc) - np.log1p(g("n_ocupado")), "X", "log n_ℓ − log(1 + ocupados): presión del umbral sobre la oferta laboral")
    reg.add("x_hs_linea_x_cabecera", g("c_hs_linea") * cab, "X", "capacidad (Σ h·S / n_ℓ) × cabecera: retornos urbanos distintos")
    reg.add("x_formales_linea_x_cabecera", g("c_formales_linea") * cab, "X", "formales por línea × cabecera")
    reg.add("x_arriendo_linea_x_propia", g("log_arriendo_sobre_linea") * (g("ten_propia_pagada") + g("ten_propia_pagando")), "X",
            "valor de la vivienda (log arriendo/línea) × vivienda propia: activo")
    reg.add("x_arriendo_linea_x_usufructo", g("log_arriendo_sobre_linea") * g("ten_usufructo"), "X", "valor de la vivienda × usufructo")
    reg.add("x_educmax_x_ocupados_pc", g("educ_max_adultos") * g("ocupados_pc"), "X", "educación máxima × ocupados por miembro")
    reg.add("x_educ_oc_x_horas_pc", np.nan_to_num(g("educ_media_ocupados")) * g("horas_pc"), "X", "educación de los ocupados × horas por miembro")
    reg.add("x_subsidiado_x_lineas", g("share_subsidiado") * ctx.nl, "X", "régimen subsidiado × líneas necesarias")
    reg.add("x_ipm_x_cabecera", g("p_puntaje") * cab, "X", "puntaje IPM × cabecera")
    reg.add("x_ipm_x_lineas", g("p_puntaje") * ctx.nl, "X", "puntaje IPM × líneas necesarias")
    reg.add("x_jefa_x_lineas", g("jefe_mujer") * ctx.nl, "X", "jefa × líneas necesarias")
    reg.add("x_dependencia_x_informal", g("dependencia") * g("p_share_informal_pea"), "X", "dependencia × informalidad de la PEA")
    reg.add("x_menores_x_subsidiado", g("n_menor18") * g("share_subsidiado"), "X", "menores × régimen subsidiado (focalización)")
    reg.add("x_rural_x_jornaleros", (1 - cab) * g("n_pos_jornalero"), "X", "jornaleros en zona rural")
    reg.add("x_aux_transporte_x_formales_pc", g("i_aux_transporte_por_asal") * g("formales_pc"), "X", "auxilio de transporte entre asalariados × formales por miembro")
    reg.add("x_perceptores_linea_x_educ_oc", g("i_perceptores_linea") * np.nan_to_num(g("educ_media_ocupados")), "X", "perceptores por línea × educación de los ocupados")
    reg.add("x_requerido_pc_x_cabecera", g("u_requerido_pc") * cab, "X", "umbral per cápita × cabecera")
    reg.add("x_hacinamiento_x_cabecera", g("personas_por_cuarto") * cab, "X", "hacinamiento × cabecera")


# ------------------------------------------------------------------------------ familia G (exhaustiva)
def _fam_G(reg, H, ctx):
    p, A = ctx.p, ctx.A
    nper = ctx.nper
    for c, nom in NUMERICAS_PERSONA.items():
        x = _f(p[c])
        reg.add(f"g_{c}_suma", A.suma(x), "G", f"suma de {nom} ({c}) en el hogar")
        reg.add(f"g_{c}_media", A.media(x), "G", f"media de {nom} ({c})")
        reg.add(f"g_{c}_max", A.maximo(x), "G", f"máximo de {nom} ({c})")
        reg.add(f"g_{c}_min", A.minimo(x), "G", f"mínimo de {nom} ({c})")
        reg.add(f"g_{c}_desv", A.desv(x), "G", f"desviación de {nom} ({c})")
        reg.add(f"g_{c}_n", A.n_validos(x), "G", f"personas con {c} observado")
        reg.add(f"g_{c}_jefe", A.de(x, ctx.jefe), "G", f"{nom} del jefe")
        reg.add(f"g_{c}_principal", A.de(x, ctx.principal), "G", f"{nom} del perceptor principal")
    for c, (nom, niveles) in CATEGORICAS_PERSONA.items():
        x = _f(p[c])
        for k in niveles:
            n = A.n(x == k)
            reg.add(f"g_{c}_{k}_n", n, "G", f"personas con {c} ({nom}) = {k}")
            reg.add(f"g_{c}_{k}_sh", n / nper, "G", f"proporción de personas con {c} = {k}")
        reg.add(f"g_{c}_na_n", A.n(np.isnan(x)), "G", f"personas sin respuesta en {c} (salto del cuestionario)")
        reg.add(f"g_{c}_jefe", A.de(x, ctx.jefe), "G", f"código de {c} del jefe")
        reg.add(f"g_{c}_principal", A.de(x, ctx.principal), "G", f"código de {c} del perceptor principal")
    gg = np.floor(_f(p["Oficio"]) / 10)
    for k in range(10):
        n = A.n(gg == k)
        reg.add(f"g_oficio_gg{k}_n", n, "G", f"ocupados del gran grupo ocupacional {k} (CIUO-68)")
        reg.add(f"g_oficio_gg{k}_sh", n / nper, "G", f"proporción del hogar en el gran grupo ocupacional {k}")
    reg.add("g_oficio_distintos", pd.Series(_f(p["Oficio"])).groupby(A.pos).nunique().reindex(range(A.nh)).fillna(0).values,
            "G", "número de oficios distintos en el hogar")
    for c, txt in SI_NO_OTRAS.items():
        x = _f(p[c])
        nsi, nresp = A.n(x == 1), A.n(~np.isnan(x))
        reg.add(f"g_{c}_si", nsi, "G", f"personas con «sí» en {c} ({txt})")
        reg.add(f"g_{c}_no", A.n(x == 2), "G", f"personas con «no» en {c}")
        reg.add(f"g_{c}_ns", A.n(x == 9), "G", f"personas con «no sabe» en {c}")
        reg.add(f"g_{c}_tasa", np.where(nresp > 0, nsi / np.maximum(nresp, 1), np.nan), "G", f"proporción de «sí» entre quienes respondieron {c}")
        reg.add(f"g_{c}_jefe", A.de(_codigo(x), ctx.jefe), "G", f"respuesta del jefe a {c}")
    for c, txt in FLAGS_PERSONA.items():
        n = A.n(_f(p[c]) == 1)
        reg.add(f"g_{c}_n", n, "G", f"personas: {txt}")
        reg.add(f"g_{c}_sh", n / nper, "G", f"proporción del hogar: {txt}")
    reg.add("g_P5000_crudo", _f(H.get("P5000_crudo", np.nan)), "G", "cuartos (P5000) sin recorte")


# ------------------------------------------------------------------------------ familia T (fija)
def _fam_T(reg, D, dominios, deptos):
    g = lambda c: _f(D[c])
    for lev in dominios:
        reg.add(f"t_dom_{re.sub(r'[^A-Za-z0-9]', '_', lev)}", (D["Dominio"].values == lev).astype(float), "T", f"dummy de Dominio = {lev}")
    for lev in deptos:
        reg.add(f"t_depto_{lev}", (D["Depto"].astype(str).values == lev).astype(float), "T", f"dummy de Depto = {lev}")
    for c, niveles, nom in [("jefe_posicion", range(1, 10), "pos"), ("jefe_nivel_edu", [1, 2, 3, 4, 5, 6, 9], "nivel"),
                            ("jefe_regimen_salud", [1, 2, 3, 9], "reg"), ("jefe_actividad", range(1, 7), "act"),
                            ("jefe_tam_empresa", range(1, 10), "tam"), ("jefe_oficio_gg", range(10), "ofgg")]:
        x = g(c)
        for k in niveles:
            reg.add(f"t_jefe_{nom}_{k}", (x == k).astype(float), "T", f"dummy: {c} = {k}")
    cab = g("cabecera")
    for k, nom in F.TENENCIA.items():
        reg.add(f"t_ten_{nom}_x_cabecera", g(f"ten_{nom}") * cab, "T", f"tenencia {nom} × cabecera")
    umb = [
        ("t_jefe_primaria_o_menos", g("jefe_anios_educ") <= 5, "jefe con primaria o menos (≤ 5 años)"),
        ("t_jefe_bachiller", g("jefe_anios_educ") >= 11, "jefe bachiller (≥ 11 años)"),
        ("t_jefe_superior", g("jefe_anios_educ") >= 12, "jefe con educación superior (≥ 12 años)"),
        ("t_educmax_bachiller", g("educ_max_adultos") >= 11, "algún adulto bachiller"),
        ("t_educmax_superior", g("educ_max_adultos") >= 16, "algún adulto con 5 o más años de educación superior"),
        ("t_un_ocupado_hogar_grande", (g("n_ocupado") <= 1) & (g("Npersug") >= 4), "a lo sumo un ocupado y 4 o más miembros"),
        ("t_sin_cotizantes_con_ocupados", (g("n_ocupado") > 0) & (g("n_cotiza_pension") == 0), "ocupados pero ninguno cotiza"),
        ("t_arriendo_bajo_linea", g("arriendo_sobre_linea") < 0.15, "arriendo < 15 % de la línea del hogar"),
        ("t_arriendo_alto_linea", g("arriendo_sobre_linea") > 0.5, "arriendo > 50 % de la línea del hogar"),
        ("t_hacinamiento_gt2", g("personas_por_cuarto") > 2, "más de 2 personas por cuarto"),
        ("t_lineas_por_ocupado_gt3", g("c_lineas_por_ocupado") > 3, "cada ocupado debe cubrir más de 3 líneas"),
        ("t_dependencia_ge2", g("dependencia") >= 2, "razón de dependencia ≥ 2"),
        ("t_horas_pc_lt20", g("horas_pc") < 20, "menos de 20 horas semanales por miembro"),
        ("t_ocupados_pc_lt_un_tercio", g("ocupados_pc") < 1 / 3, "menos de un ocupado por cada 3 miembros"),
        ("t_requerido_casi_cero", g("requerido_sobre_linea") < 0.25, "el arriendo imputado cubre más del 75 % de la línea"),
        ("t_ipm_2_o_mas", g("p_n_privaciones") >= 2, "dos o más privaciones IPM"),
        ("t_perceptores_menor_lineas", g("i_perceptores") < g("u_n_lineas"), "menos perceptores que líneas necesarias"),
    ]
    for nom, v, d in umb:
        reg.add(nom, np.asarray(v, float), "T", f"umbral: {d}")
    for c in TRANSFORMAR:
        if c not in D.columns:
            continue
        x = g(c)
        pos = np.nanmin(x) >= 0
        if pos:
            reg.add(f"t_log_{c}", np.log1p(np.maximum(x, 0)), "T", f"log(1 + {c})")
            reg.add(f"t_raiz_{c}", np.sqrt(np.maximum(x, 0)), "T", f"√{c}")
            reg.add(f"t_inv_{c}", 1 / (1 + np.maximum(x, 0)), "T", f"1/(1 + {c})")
        reg.add(f"t_cuad_{c}", x ** 2, "T", f"{c}²")


def especificaciones_muestra(D) -> list:
    """Transformaciones que dependen de la muestra de referencia (sólo X, sin etiqueta): recorte 1 %–99 %, rango y decil
    dentro del Dominio (familia T) y las posiciones relativas de v5/v6 (familia X). En las tablas guardadas se calculan
    con las X de todo train; para medir se recalculan dentro de cada pliegue (`TransformadorV7`)."""
    specs = []
    for i, c in enumerate([c for c in TRANSFORMAR if c in D.columns]):
        specs.append(dict(nombre=f"t_rec_{c}", tipo="recorte", a=c, familia="T"))
        if i < N_GRUPO:
            specs.append(dict(nombre=f"t_rgdom_{c}", tipo="rango_grupo", a=c, grupo="Dominio", familia="T"))
            specs.append(dict(nombre=f"t_qdom_{c}", tipo="cuantil_grupo", a=c, grupo="Dominio", familia="T"))
    for s in PM.GRUPO_BASE + PM.GRUPO_CAPACIDAD:
        specs.append({**s, "familia": "X" if s in PM.GRUPO_BASE else "C"})
    return specs


def construir_spec(s, Dref, D):
    from . import busqueda as BU
    if s["tipo"] == "cuantil_grupo":
        a, g = s["a"], s["grupo"]
        xa = D[a].astype(float).values
        out = np.full(len(D), np.nan)
        gD, gR = D[g].astype(str).values, Dref[g].astype(str).values
        ra = Dref[a].astype(float).values
        for lev in np.unique(gD):
            r = np.sort(ra[gR == lev]); r = r[np.isfinite(r)]
            mk = gD == lev
            if len(r):
                out[mk] = np.minimum(np.floor(10 * np.searchsorted(r, xa[mk], side="right") / len(r)), 9)
        return np.where(np.isfinite(xa), out, np.nan)
    return BU.construir(s, Dref, D)


def materializar(specs, Dref, D) -> pd.DataFrame:
    cols = {}
    for s in specs:
        v = np.asarray(construir_spec(s, Dref, D), float)
        cols[s["nombre"]] = np.where(np.isfinite(v), v, np.nan).astype("float32")
    return pd.DataFrame(cols, index=D.index)


# ------------------------------------------------------------------------------ construcción
def _hash(v):
    a = np.asarray(v, dtype=np.float32)
    a = np.where(np.isnan(a), np.float32(-9.87654e30), a)
    return hashlib.md5(a.tobytes()).hexdigest()


def construir_v7(th, tp, teh, tep, verbose=True, depurar=True):
    """Tablas v7 de train y test (sin codificación con la etiqueta). Devuelve (D, DT, META) con META = {familia,
    descripcion, especificaciones_muestra, duplicadas, constantes, base}. Sólo columnas comunes: la restricción se
    aplica al entrar, y `invariancia_v7` prueba que nada depende de lo demás. `depurar=False` conserva constantes y
    duplicados (para el mapa de dependencias en submuestras)."""
    t0 = time.time()
    hc, pc = comunes(th, HOG_COMUNES), comunes(tp, PER_COMUNES)
    hte, pte = comunes(teh, HOG_COMUNES), comunes(tep, PER_COMUNES)
    Htr = F.construir(hc, pc)
    Hte = F.construir(hte, pte, ref_arriendo=Htr.assign(arriendo=Htr.arriendo.where(Htr.arriendo_imputado_flag == 0)))
    dominios = sorted(hc["Dominio"].astype(str).unique())
    deptos = sorted(hc["Depto"].astype(str).unique())
    salida, metas = [], []
    for H0, h, p in ((Htr, hc, pc), (Hte, hte, pte)):
        H0 = H0.drop(columns=[c for c in CONSTANTES_EXTERNAS if c in H0.columns]).reset_index(drop=True)
        h = h.reset_index(drop=True); p = p.reset_index(drop=True)
        assert (H0["id"].values == h["id"].values).all()
        fam = {c: "B" for c in H0.columns if c != "id"}
        desc = {c: "base v5 (features.py, bloques A–F)" for c in fam}
        for c, f_ in [("arriendo_pagado", "U"), ("arriendo_estimado", "U"), ("cuota_amortizacion", "U"), ("Lp", "U"),
                      ("Li", "U"), ("ingreso_requerido", "U"), ("Fex_c", "V"), ("log_fex", "V"), ("n_personas_reg", "G")]:
            if c in fam:
                fam[c] = f_
                desc[c] = {"ingreso_requerido": "R_h = Npersug·Lp − A_h (umbral observable)", "Lp": "línea de pobreza per cápita",
                           "Li": "línea de indigencia per cápita", "Fex_c": "factor de expansión del hogar",
                           "log_fex": "log del factor de expansión"}.get(c, desc[c])
        H0["P5000_crudo"] = _f(h["P5000"])
        # familias de v6 (revisadas): I→U, J→C, K→I, L→V
        A6 = PM.agregar_v6(PM.variables_persona_v6(p))
        H6 = H0.merge(A6, left_on="id", right_index=True, how="left")
        N6, fam6 = PM.familias_hogar(H6)
        mapa6 = {"I": "U", "J": "C", "K": "I", "L": "V"}
        ctx = _Ctx(p, H0)
        reg = _Reg(len(H0))
        for c in N6.columns:
            reg.add(c, N6[c].values, mapa6[fam6[c]], "versión 6 (permitidas.py), revisada")
        _fam_U(reg, H0, h, ctx)
        _fam_I(reg, H0, ctx)
        _fam_C(reg, H0, ctx)
        _fam_P(reg, H0, ctx)
        _fam_V(reg, H0, h, ctx)
        _fam_G(reg, H0, ctx)
        D = pd.concat([H0.drop(columns=["P5000_crudo"]), reg.frame(H0.index)], axis=1)
        reg2 = _Reg(len(D))
        _fam_X(reg2, D, ctx)
        D = pd.concat([D, reg2.frame(D.index)], axis=1)
        reg3 = _Reg(len(D))
        _fam_T(reg3, D, dominios, deptos)
        D = pd.concat([D, reg3.frame(D.index)], axis=1)
        for r in (reg, reg2, reg3):
            fam.update(r.fam); desc.update(r.desc)
        salida.append(D); metas.append((fam, desc))
    D, DT = salida
    fam, desc = metas[0]
    assert list(D.columns) == list(DT.columns), "train y test con columnas distintas"
    specs = especificaciones_muestra(D)
    Mtr, Mte = materializar(specs, D, D), materializar(specs, D, DT)
    D = pd.concat([D, Mtr], axis=1); DT = pd.concat([DT, Mte], axis=1)
    for s in specs:
        fam[s["nombre"]] = s["familia"]
        desc[s["nombre"]] = {"recorte": f"{s['a']} recortada al 1 %–99 % de train",
                             "rango_grupo": f"rango percentil de {s.get('a')} dentro de {s.get('grupo')} (X de train)",
                             "cuantil_grupo": f"decil de {s.get('a')} dentro de {s.get('grupo')} (X de train)",
                             "desv_mediana_grupo": f"{s.get('a')} menos la mediana de {s.get('grupo')} (X de train)"}[s["tipo"]]
    # depuración: columnas constantes en train y duplicados exactos (train y test); se conserva la primera aparición
    num = [c for c in D.columns if c not in NO_NUMERICAS]
    protegidas = set(CAT_TODAS) | {"principal_oficio", "principal_oficio_gg", "jefe_oficio_gg", "cabecera",
                                   "ingreso_requerido", "Lp", "Li", "Npersug", "Nper"} | {c for c, f_ in fam.items() if f_ == "B"}
    constantes = []
    for c in num:
        if not depurar:
            break
        v = D[c].astype(float).values
        if c not in protegidas and (np.isnan(v).all() or np.nanstd(v) == 0):
            constantes.append(c)
    vistos, dup = {}, {}
    base_set = {c for c, f_ in fam.items() if f_ == "B"}
    for c in num:
        if not depurar:
            break
        if c in constantes:
            continue
        k = _hash(np.r_[D[c].astype(float).values, DT[c].astype(float).values])
        if c in base_set:                      # la base (S2 de v5) se conserva íntegra, aunque tenga duplicados
            vistos.setdefault(k, c)
            continue
        if k in vistos:
            prev = vistos[k]
            if c in protegidas and prev not in protegidas:
                dup[prev] = c; vistos[k] = c
            else:
                dup[c] = prev
        else:
            vistos[k] = c
    quitar = set(constantes) | set(dup)
    D = D.drop(columns=[c for c in quitar if c in D.columns]); DT = DT.drop(columns=[c for c in quitar if c in DT.columns])
    specs = [s for s in specs if s["nombre"] not in quitar]
    fam = {c: f_ for c, f_ in fam.items() if c in D.columns}
    desc = {c: d for c, d in desc.items() if c in D.columns}
    base = [c for c in fam if fam[c] == "B"]
    if verbose:
        print(f"construir_v7: {time.time() - t0:.0f} s | train {D.shape} | test {DT.shape} | "
              f"constantes {len(constantes)} | duplicadas {len(dup)}")
    meta = {"familia": fam, "descripcion": desc, "especificaciones_muestra": specs, "duplicadas": dup,
            "constantes": constantes, "base_columnas": base}
    return D, DT, meta


# ------------------------------------------------------------------------------ mapa variable → preguntas
PREGUNTAS = {  # columna común: (descripción DANE abreviada, tipo, costo para c(S))
    "Clase": ("Clase: 1 cabecera, 2 resto (centros poblados y rural disperso)", "territorio (dato administrativo)", 0),
    "Dominio": ("Dominio geográfico (24 áreas metropolitanas, resto urbano, rural)", "territorio (dato administrativo)", 0),
    "Depto": ("Departamento", "territorio (dato administrativo)", 0),
    "Lp": ("Línea de pobreza per cápita del hogar", "línea oficial (la conoce el Estado)", 0),
    "Li": ("Línea de indigencia per cápita del hogar", "línea oficial (la conoce el Estado)", 0),
    "Fex_c": ("Factor de expansión del hogar", "diseño muestral (no es pregunta; no existe fuera de la encuesta)", 0),
    "Fex_dpto": ("Factor de expansión departamental", "diseño muestral (no es pregunta; no existe fuera de la encuesta)", 0),
    "Orden": ("Número de orden de la persona en el hogar", "identificador", 0),
    "Nper": ("Número de personas del hogar (listado de personas)", "hogar: listado", 1),
    "Npersug": ("Número de personas de la unidad de gasto (listado + parentesco)", "hogar: listado", 1),
    "P5000": ("Incluyendo sala-comedor, ¿de cuántos cuartos en total dispone este hogar?", "hogar: vivienda", 1),
    "P5010": ("¿En cuántos de esos cuartos duermen las personas de este hogar?", "hogar: vivienda", 1),
    "P5090": ("La vivienda ocupada por este hogar es: propia pagada, propia pagando, arriendo, usufructo, posesión sin título, otra", "hogar: vivienda", 1),
    "P5100": ("¿Cuánto pagan mensualmente por cuota de amortización?", "hogar: vivienda (monto)", 1),
    "P5130": ("Si tuviera que pagar arriendo por esta vivienda, ¿cuánto estima que tendría que pagar mensualmente?", "hogar: vivienda (monto)", 1),
    "P5140": ("¿Cuánto pagan mensualmente por arriendo?", "hogar: vivienda (monto)", 1),
    "P6020": ("Sexo", "persona: listado", 1), "P6040": ("¿Cuántos años cumplidos tiene?", "persona: listado", 1),
    "P6050": ("Parentesco con el jefe o jefa del hogar", "persona: listado", 1),
    "P6090": ("¿Está afiliado, es cotizante o es beneficiario de alguna entidad de seguridad social en salud?", "persona: salud", 1),
    "P6100": ("Régimen de seguridad social en salud: contributivo, especial, subsidiado", "persona: salud", 1),
    "P6210": ("Nivel educativo más alto alcanzado", "persona: educación", 1), "P6210s1": ("Último año o grado aprobado", "persona: educación", 1),
    "P6240": ("Actividad en la que ocupó la mayor parte del tiempo la semana pasada", "persona: fuerza de trabajo", 1),
    "Oficio": ("¿Qué hace en este trabajo? (oficio, CIUO-68 a 2 dígitos)", "persona: ocupados", 1),
    "P6426": ("¿Cuánto tiempo lleva trabajando en esta empresa de manera continua? (meses)", "persona: ocupados", 1),
    "P6430": ("Posición ocupacional en el trabajo principal", "persona: ocupados", 1),
    "P6800": ("¿Cuántas horas a la semana trabaja normalmente en ese trabajo?", "persona: ocupados", 1),
    "P6870": ("¿Cuántas personas en total tiene la empresa donde trabaja?", "persona: ocupados", 1),
    "P6920": ("¿Está cotizando actualmente a un fondo de pensiones?", "persona: ocupados", 1),
    "P7040": ("¿Tenía la semana pasada otro trabajo o negocio?", "persona: ocupados", 1),
    "P7045": ("¿Cuántas horas trabajó la semana pasada en ese segundo trabajo?", "persona: ocupados", 1),
    "P7050": ("Posición ocupacional en el segundo trabajo", "persona: ocupados", 1),
    "P7090": ("¿Quiere trabajar más horas?", "persona: subempleo", 1), "P7110": ("¿Hizo diligencias para trabajar más horas?", "persona: subempleo", 1),
    "P7120": ("¿Estaba disponible para trabajar más horas?", "persona: subempleo", 1),
    "P7150": ("¿Hizo diligencias para cambiar de trabajo?", "persona: subempleo", 1),
    "P7160": ("¿Podría empezar un nuevo trabajo antes de un mes?", "persona: subempleo", 1),
    "P7310": ("¿Buscó trabajo por primera vez o había trabajado antes?", "persona: desocupados", 1),
    "P7350": ("Posición ocupacional en el último trabajo (desocupados)", "persona: desocupados", 1),
    "Pet": ("Población en edad de trabajar (derivada de la edad y la clase)", "persona: clasificación derivada", 1),
    "Oc": ("Ocupado (derivado del módulo de fuerza de trabajo)", "persona: clasificación derivada", 1),
    "Des": ("Desocupado (derivado del módulo de fuerza de trabajo)", "persona: clasificación derivada", 1),
    "Ina": ("Inactivo (derivado del módulo de fuerza de trabajo)", "persona: clasificación derivada", 1),
}
for _q, (_n, _t, _u) in SI_NO_INGRESO.items():
    PREGUNTAS[_q] = (_t, f"persona: ingresos sí/no ({_u})", 1)
PREGUNTAS_HOGAR_EN_PERSONA = ["Clase", "Dominio", "Depto", "Fex_c", "Fex_dpto"]   # repetidas en la tabla de personas


def _iguales(a, b):
    if a.dtype == object or b.dtype == object:
        return bool((a.astype(str).values == b.astype(str).values).all())
    return bool(np.array_equal(a.astype(float).values, b.astype(float).values, equal_nan=True))


def mapa_preguntas_v7(th, tp, teh, tep, columnas=None, n_hogares=6000, semilla=3, verbose=True):
    r"""Mapa exacto variable → columnas comunes de las que depende, por **rastreo de dependencias**: en una submuestra
    de `n_hogares` hogares de train se reconstruye v7 (sin depurar) permutando al azar UNA columna común cada vez; la
    variable $v$ depende de la columna $c$ si cambia. La permutación de `P6050` se hace sólo entre los no jefes (para
    no romper la unicidad del jefe) y, por construcción, se añade P6050 a toda variable del jefe, de la pareja, del
    perceptor principal o agregada sobre la unidad de gasto. Devuelve {variable: [columnas]}."""
    rng = np.random.default_rng(semilla)
    t0 = time.time()
    hs = th.iloc[np.sort(rng.choice(len(th), min(n_hogares, len(th)), replace=False))].reset_index(drop=True)
    ps = tp[tp["id"].isin(set(hs["id"]))].reset_index(drop=True)
    ht = teh.iloc[:300].reset_index(drop=True); pt = tep[tep["id"].isin(set(ht["id"]))].reset_index(drop=True)
    B0, _, meta0 = construir_v7(hs, ps, ht, pt, verbose=False, depurar=False)
    cols = [c for c in (columnas or B0.columns) if c in B0.columns and c != "id"]
    dep = {c: set() for c in cols}
    candidatas = [("hogar", c) for c in HOG_COMUNES if c != "id"] + [("persona", c) for c in PER_COMUNES if c != "id"]
    for tabla, c in candidatas:
        h2, p2 = hs.copy(), ps.copy()
        if tabla == "hogar":
            h2[c] = hs[c].values[rng.permutation(len(hs))]
        elif c == "P6050":
            nj = np.flatnonzero(ps["P6050"].values != 1)
            v = ps["P6050"].values.copy(); v[nj] = v[nj][rng.permutation(len(nj))]; p2[c] = v
        else:
            p2[c] = ps[c].values[rng.permutation(len(ps))]
        B1, _, _ = construir_v7(h2, p2, ht, pt, verbose=False, depurar=False)
        for v in cols:
            if v not in B1.columns or not _iguales(B0[v], B1[v]):
                dep[v].add(c)
    fam = meta0["familia"]
    ug_fams = {"I", "C", "P"}
    nom_q = {nom: q for q, (nom, _, _) in SI_NO_INGRESO.items()}
    comunes_ = set(HOG_COMUNES) | set(PER_COMUNES)
    for v in cols:
        if fam.get(v) in ug_fams or re.search(r"jefe|pareja|principal|_ug|ug_|perceptor", v):
            dep[v].add("P6050")
        # red de seguridad por nombre: códigos raros que la submuestra no contiene (p. ej. P7050 = 9)
        for q in re.findall(r"P\d{4}(?:s\d+)?|Oficio|Pet|Oc|Des|Ina", v):
            if q in comunes_:
                dep[v].add(q)
        m_ = re.match(r"i_(.+?)_(si|ns|resp|tasa|pc|linea|jefe)$", v)
        if m_ and m_.group(1) in nom_q:
            dep[v].add(nom_q[m_.group(1)])
        if not dep[v]:
            dep[v].add("Nper")                     # conteos de filas del listado de personas
    if verbose:
        print(f"mapa_preguntas_v7: {len(candidatas)} columnas comunes rastreadas sobre {len(hs)} hogares en {time.time() - t0:.0f} s")
    return {v: sorted(s) for v, s in dep.items()}


def mapa_derivadas(mapa, especificaciones, te_especificaciones=None):
    """Extiende el mapa a las variables definidas por especificaciones (SHAP, dependientes de la muestra) y a las
    codificaciones con la etiqueta (preguntas de la clave y del padre)."""
    m = dict(mapa)
    for s in especificaciones:
        fuentes = [s.get("a"), s.get("b"), s.get("grupo")] + list(s.get("vars", []))
        if s.get("tipo") == "expr":
            fuentes += re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s.get("expr", ""))
        if s.get("tipo") == "per_capita":
            fuentes.append("Npersug")
        m[s["nombre"]] = sorted(set().union(*[set(m.get(f, [])) for f in fuentes if f]))
    for nom, e in (te_especificaciones or {}).items():
        m[nom] = sorted(set().union(*[set(m.get(f, [])) for f in e["clave"] + e["padre"]]))
    return m


def costo_preguntas(mapa, cols) -> tuple[int, list]:
    """c(S) = |∪_j Q(j)| contando sólo las columnas con costo 1 (preguntas reales del cuestionario)."""
    qs = sorted(set().union(*[set(mapa.get(c, [])) for c in cols])) if cols else []
    return sum(PREGUNTAS.get(q, ("", "", 1))[2] for q in qs), qs


def invariancia_v7(th, tp, teh, tep, semilla=7, D=None):
    """Prueba de aleatorización: se reconstruye v7 con TODAS las columnas no comunes de train (ingresos, montos,
    Estrato1, la propia etiqueta…) permutadas al azar; si alguna variable dependiera de ellas, cambiaría."""
    rng = np.random.default_rng(semilla)
    th2, tp2 = th.copy(), tp.copy()
    no_h = [c for c in th.columns if c not in HOG_COMUNES]
    no_p = [c for c in tp.columns if c not in PER_COMUNES]
    for c in no_h:
        th2[c] = th[c].values[rng.permutation(len(th))]
    for c in no_p:
        tp2[c] = tp[c].values[rng.permutation(len(tp))]
    if D is None:
        D, _, _ = construir_v7(th, tp, teh, tep, verbose=False)
    B, _, _ = construir_v7(th2, tp2, teh, tep, verbose=False)
    distintas = [c for c in D.columns if c not in B.columns or not D[c].equals(B[c])]
    return (not distintas), distintas, {"hogar_permutadas": no_h, "persona_permutadas": len(no_p)}


# ------------------------------------------------------------------------------ codificación con la etiqueta (bayes empírico)
TE_ESPECIFICACIONES = {
    "te_oficio_jefe": {"clave": ["jefe_oficio"], "padre": ["jefe_oficio_gg"]},
    "te_oficio_principal": {"clave": ["principal_oficio"], "padre": ["principal_oficio_gg"]},
    "te_dominio_tenencia": {"clave": ["Dominio", "tenencia"], "padre": ["Dominio"]},
    "te_depto_clase": {"clave": ["Depto", "cabecera"], "padre": ["Depto"]},
    "te_posicion_jefe": {"clave": ["jefe_posicion"], "padre": []},
}
TE_SIN_BOGOTA = True
TE_K_INTERNO = 5


def _claves(X, cols):
    if not cols:
        return np.zeros(len(X), dtype=object)
    partes = [X[c].astype(str).values for c in cols]
    k = partes[0].astype(object)
    for q in partes[1:]:
        k = k + "|" + q
    return k


def concentracion_betabinomial(s, n, mu, m_min=1e-2, m_max=1e6):
    r"""Estimador de máxima verosimilitud marginal (bayes empírico tipo II) de la concentración m = α + β de la
    beta-binomial: s_g | p_g ~ Bin(n_g, p_g), p_g ~ Beta(m μ_g, m (1 − μ_g)), con medias a priori μ_g dadas.
    log L(m) = Σ_g [ln B(s_g + mμ_g, n_g − s_g + m(1−μ_g)) − ln B(mμ_g, m(1−μ_g))]. La varianza entre grupos implícita
    es τ² = μ(1−μ)/(m + 1). Si la verosimilitud es monótona creciente en m (sin heterogeneidad), m → m_max
    (agrupamiento completo)."""
    from scipy.optimize import minimize_scalar
    from scipy.special import betaln
    s, n, mu = np.asarray(s, float), np.asarray(n, float), np.clip(np.asarray(mu, float), 1e-6, 1 - 1e-6)
    ok = n > 0
    s, n, mu = s[ok], n[ok], mu[ok]
    if len(s) < 3:
        return float(m_max)

    def nll(lm):
        m = np.exp(lm)
        a, b = m * mu, m * (1 - mu)
        return -float(np.sum(betaln(s + a, n - s + b) - betaln(a, b)))
    r = minimize_scalar(nll, bounds=(np.log(m_min), np.log(m_max)), method="bounded", options={"xatol": 1e-4})
    return float(np.exp(r.x))


class CodificadorObjetivo:
    r"""Codificación de categóricas con la etiqueta por **bayes empírico jerárquico beta-binomial**.

    Para una clave c con padre π(c) (p. ej. oficio → gran grupo; Dominio × tenencia → Dominio):
      nivel 0: μ₀ = ȳ del conjunto de ajuste;
      nivel 1: p̂_π = (s_π + m₁ μ₀)/(n_π + m₁), con m₁ por MV marginal (`concentracion_betabinomial`);
      nivel 2: p̂_c = (s_c + m₂ p̂_π(c))/(n_c + m₂), con m₂ por MV marginal con medias a priori p̂_π(c).
    Un nivel no visto recibe la media de su padre (o μ₀). Es la media posterior bajo el modelo jerárquico con
    hiperparámetros estimados (Efron y Morris, 1975; Robbins, 1956); el suavizado m-estimado de Micci-Barreca (2001)
    es el caso m fijo sin padre.

    **Fuera de pliegue.** `codificar(train_idx, aplicar_idx)`: las estadísticas sólo usan `train_idx` (sin Bogotá si
    `excluir_bogota`); las filas de `aplicar_idx` que están en `train_idx` se codifican con una partición interna de
    `k_interno` pliegues (estratificada por la etiqueta), de modo que ninguna fila ve su propia etiqueta; las demás con
    todo `train_idx`. `codificar_test(train_idx)` codifica las filas de `DT`.
    """

    def __init__(self, D, y, DT=None, especificaciones=None, excluir_bogota=TE_SIN_BOGOTA, k_interno=TE_K_INTERNO,
                 semilla=20262, m_fijo=None):
        self.esp = dict(especificaciones or TE_ESPECIFICACIONES)
        self.y = np.asarray(y, float)
        self.n = len(D)
        self.k, self.semilla, self.m_fijo = int(k_interno), int(semilla), m_fijo
        self.excluir_bogota = bool(excluir_bogota)
        bog = D["Dominio"].astype(str).values == "BOGOTA"
        self.ok_stats = ~bog if self.excluir_bogota else np.ones(self.n, bool)
        self.cod, self.codp, self.mapa, self.L, self.Lp = {}, {}, {}, {}, {}
        for nom, e in self.esp.items():
            kc = _claves(D, e["clave"]); kp = _claves(D, e["padre"])
            if DT is not None:
                kc = np.concatenate([kc, _claves(DT, e["clave"])]); kp = np.concatenate([kp, _claves(DT, e["padre"])])
            c, niv = pd.factorize(kc); cp, nivp = pd.factorize(kp)
            mapa = np.zeros(len(niv), int); mapa[c] = cp
            self.cod[nom], self.codp[nom], self.mapa[nom], self.L[nom], self.Lp[nom] = c, cp, mapa, len(niv), len(nivp)
        self.ultimos_m = {}

    def _post(self, nom, fit, objetivo):
        """Media posterior para las filas `objetivo` (índices sobre train+test) con las estadísticas de `fit`."""
        y = self.y[fit]
        c, cp, mapa, L, Lp = self.cod[nom][fit], self.codp[nom][fit], self.mapa[nom], self.L[nom], self.Lp[nom]
        mu0 = float(y.mean())
        if self.esp[nom]["padre"]:
            sp, nn = np.bincount(cp, y, Lp), np.bincount(cp, minlength=Lp).astype(float)
            m1 = self.m_fijo or concentracion_betabinomial(sp, nn, np.full(Lp, mu0))
            p1 = (sp + m1 * mu0) / (nn + m1)
            prior = p1[mapa]
        else:
            m1, prior = None, np.full(L, mu0)
        s, n = np.bincount(c, y, L), np.bincount(c, minlength=L).astype(float)
        m2 = self.m_fijo or concentracion_betabinomial(s, n, prior)
        post = (s + m2 * prior) / (n + m2)
        self.ultimos_m[nom] = {"mu0": mu0, "m_padre": m1, "m_clave": m2,
                               "tau_clave": float(np.sqrt(mu0 * (1 - mu0) / (m2 + 1))),
                               "tau_padre": None if m1 is None else float(np.sqrt(mu0 * (1 - mu0) / (m1 + 1)))}
        return post[self.cod[nom][objetivo]]

    def _internos(self, tr):
        from sklearn.model_selection import StratifiedKFold
        return list(StratifiedKFold(self.k, shuffle=True, random_state=self.semilla).split(np.zeros(len(tr)), self.y[tr]))

    def codificar(self, train_idx, aplicar_idx=None) -> pd.DataFrame:
        tr = np.asarray(train_idx, int)
        ap = np.arange(self.n) if aplicar_idx is None else np.asarray(aplicar_idx, int)
        fit = tr[self.ok_stats[tr]]
        en_tr = np.zeros(self.n, bool); en_tr[tr] = True
        mk = en_tr[ap]
        internos = self._internos(tr) if mk.any() else []
        out = {}
        for nom in self.esp:
            v = self._post(nom, fit, ap)
            if mk.any():
                full = np.full(self.n, np.nan)
                for a, b in internos:
                    fa = tr[a][self.ok_stats[tr[a]]]
                    full[tr[b]] = self._post(nom, fa, tr[b])
                v[mk] = full[ap[mk]]
            out[nom] = v.astype("float32")
        for nom in self.esp:                                   # hiperparámetros con todo el ajuste (para reportar)
            self._post(nom, fit, ap[:1])
        self.hiper = {nom: dict(self.ultimos_m[nom]) for nom in self.esp}
        return pd.DataFrame(out, index=ap)

    def codificar_test(self, train_idx) -> pd.DataFrame:
        tr = np.asarray(train_idx, int)
        fit = tr[self.ok_stats[tr]]
        n_total = len(self.cod[next(iter(self.esp))])
        obj = np.arange(self.n, n_total)
        out = {nom: self._post(nom, fit, obj).astype("float32") for nom in self.esp}
        self.hiper = {nom: dict(self.ultimos_m[nom]) for nom in self.esp}
        return pd.DataFrame(out, index=np.arange(n_total - self.n))


_TABLAS = {}


def _tabla(nombre):
    if nombre not in _TABLAS:
        from .config import Rutas
        from . import io as pio
        _TABLAS[nombre] = pio.leer_tabla(Rutas(), nombre)
    return _TABLAS[nombre]


def codificar(train_idx, aplicar_idx=None, D=None, y=None, DT=None, **kw) -> pd.DataFrame:
    """Interfaz de la fase C: codificación con la etiqueta recalculada con `train_idx` (posiciones en la tabla de
    train v7). `aplicar_idx`: posiciones a codificar (las que estén en `train_idx` se codifican fuera de pliegue
    interno); `aplicar_idx="test"` codifica la tabla de test con todo `train_idx`. Sin `D`, lee
    `00_data/processed/modelo_train_v7.parquet` (y `modelo_test_v7` si se pide test). `kw`: excluir_bogota, k_interno,
    semilla, especificaciones, m_fijo."""
    D = _tabla("modelo_train_v7") if D is None else D
    y = D[ETIQUETA].values if y is None else y
    if isinstance(aplicar_idx, str) and aplicar_idx == "test":
        DT = _tabla("modelo_test_v7") if DT is None else DT
        T = CodificadorObjetivo(D, y, DT=DT, **kw).codificar_test(train_idx)
        T.index = DT.index
        return T
    return CodificadorObjetivo(D, y, **kw).codificar(train_idx, aplicar_idx)


def te_global(D, DT, y, fold, **kw):
    """Codificación de las tablas guardadas: train fuera de pliegue con los pliegues de `pliegues.csv` (la fila del
    pliegue k con los otros cuatro; dentro de esos cuatro no hace falta partición interna porque la fila no está en el
    ajuste); test con todo train. Devuelve (TE_train, TE_test, hiperparámetros por pliegue)."""
    fold = np.asarray(fold)
    cod = CodificadorObjetivo(D, y, DT=DT, **kw)
    Atr = {nom: np.zeros(len(D), "float32") for nom in cod.esp}
    hip = {}
    for k in np.unique(fold):
        va, tr = np.flatnonzero(fold == k), np.flatnonzero(fold != k)
        fit = tr[cod.ok_stats[tr]]
        for nom in cod.esp:
            Atr[nom][va] = cod._post(nom, fit, va)
        hip[int(k)] = {nom: dict(cod.ultimos_m[nom]) for nom in cod.esp}
    te = cod.codificar_test(np.arange(len(D)))
    te.index = DT.index
    hip["todo_train"] = cod.hiper
    return pd.DataFrame(Atr, index=D.index), te, hip


class TransformadorV7:
    """Recalcula dentro de cada partición lo que depende de la muestra de entrenamiento `tr`:
    * la codificación con la etiqueta (`CodificadorObjetivo`: filas de `tr` con partición interna, resto con todo `tr`);
    * las transformaciones dependientes de la muestra (recorte, rango y decil en el dominio, posiciones relativas), con
      las X de `tr` como referencia.
    Interfaz de `ajuste.cv(..., transformar=...)`: `transformar(k, tr, X) → X'`."""

    def __init__(self, D, y, especificaciones=None, excluir_bogota=TE_SIN_BOGOTA, grupo=(), k_interno=TE_K_INTERNO,
                 semilla=20262, max_cache=8, m_fijo=None):
        self.D = D
        self.cod = CodificadorObjetivo(D, y, especificaciones=especificaciones, excluir_bogota=excluir_bogota,
                                       k_interno=k_interno, semilla=semilla, m_fijo=m_fijo) if (especificaciones is None or especificaciones) else None
        self.grupo = list(grupo)
        self.cache, self.max_cache = {}, max_cache
        self.hiper = {}

    def valores(self, tr) -> dict:
        tr = np.asarray(tr)
        clave = hashlib.md5(tr.tobytes()).hexdigest() + str(len(tr))
        if clave in self.cache:
            return self.cache[clave]
        out = {}
        if self.cod is not None:
            T = self.cod.codificar(tr)
            self.hiper[clave] = self.cod.hiper
            out.update({c: T[c].values for c in T.columns})
        if self.grupo:
            G = materializar(self.grupo, self.D.iloc[tr], self.D)
            out.update({c: G[c].values for c in G.columns})
        if len(self.cache) >= self.max_cache:
            self.cache.pop(next(iter(self.cache)))
        self.cache[clave] = out
        return out

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
    """Para los cuadernos de modelos sobre v7: adjunta a un `ajuste.Datos` el transformador por pliegue (pásese como
    `transformar=dat.transformar` a `ajuste.cv`) y las familias como bloques."""
    te = conj.get("codificacion_objetivo", {})
    dat.transformar = TransformadorV7(dat.D, dat.y, te.get("especificaciones", TE_ESPECIFICACIONES),
                                      excluir_bogota=te.get("excluir_bogota", TE_SIN_BOGOTA),
                                      grupo=conj.get("especificaciones_muestra", []))
    bl = conj.get("bloques", {})
    dat.bloques = lambda: {c: bl.get(c, "B") for c in dat.columnas}
    return dat


# ------------------------------------------------------------------------------ Logit regularizado (medición)
class _PrepLogit:
    """Preprocesamiento ajustado SÓLO con el pliegue de entrenamiento: recorte al 0,5 %–99,5 %, imputación por la mediana
    con indicador de faltante, estandarización, B-splines cúbicos (5 nodos en cuantiles) en las continuas elegidas y
    one-hot de las categóricas (niveles con ≥ `min_frec` hogares; el resto se agrupa)."""

    def __init__(self, splines=(), n_nodos=5, grado=3, min_frec=30, n_sub=40000, semilla=0):
        self.splines, self.n_nodos, self.grado, self.min_frec, self.n_sub, self.semilla = list(splines), n_nodos, grado, min_frec, n_sub, semilla

    BLOQUE = 20000

    def fit(self, X):
        from sklearn.preprocessing import SplineTransformer
        self.cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        num = [c for c in X.columns if c not in self.cat]
        A = X[num].to_numpy(dtype=np.float32)                     # float32 para acotar memoria; cálculos por bloques en float64
        n = len(A)
        rng = np.random.default_rng(self.semilla)
        sub = A[rng.choice(n, min(self.n_sub, n), replace=False)].astype(np.float64)
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            lo, hi = np.nanquantile(sub, [0.005, 0.995], axis=0)
            med = np.nanmedian(sub, axis=0)
        recorta = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
        lo, hi = np.where(recorta, lo, -np.inf), np.where(recorta, hi, np.inf)
        med = np.where(np.isfinite(med), med, 0.0)
        s1 = np.zeros(len(num)); s2 = np.zeros(len(num)); nna = np.zeros(len(num))
        for i in range(0, n, self.BLOQUE):
            a = A[i:i + self.BLOQUE].astype(np.float64)
            na_ = np.isnan(a); nna += na_.sum(0)
            b = np.where(na_, med, np.clip(a, lo, hi))
            s1 += b.sum(0); s2 += (b ** 2).sum(0)
        mu = s1 / n
        sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0))
        keep = sd > 1e-9
        self.num = [c for c, k in zip(num, keep) if k]
        self.lo, self.hi, self.med = lo[keep], hi[keep], med[keep]
        self.mu, self.sd = mu[keep], sd[keep]
        na = nna / n
        self.na_cols = [c for c, f in zip(num, na) if 0.001 < f < 0.999]
        if self.na_cols:
            Z = X[self.na_cols].isna().to_numpy(np.float32)
            self.na_mu, self.na_sd = Z.mean(0).astype(np.float64), np.maximum(Z.std(0), 1e-6).astype(np.float64)
        self.spl = {}
        for c in self.splines:
            if c in self.num:
                j = self.num.index(c)
                col = A[:, num.index(c)].astype(np.float64)
                col = np.where(np.isnan(col), self.med[j], np.clip(col, self.lo[j], self.hi[j]))
                st = SplineTransformer(n_knots=self.n_nodos, degree=self.grado, knots="quantile", extrapolation="linear")
                Zs = st.fit_transform(col[:, None])
                self.spl[c] = (j, st, Zs.mean(0), np.maximum(Zs.std(0), 1e-6))
        self.niveles = {}
        for c in self.cat:
            vc = X[c].astype(str).value_counts()
            self.niveles[c] = list(vc.index[vc >= self.min_frec])
        self.p = (len(self.num) + len(self.na_cols) + sum(len(v[2]) for v in self.spl.values())
                  + sum(len(v) + 1 for v in self.niveles.values()))
        return self

    def transform(self, X):
        n = len(X)
        A = X[self.num].to_numpy(dtype=np.float32)
        Z = np.empty((n, self.p), dtype=np.float64)
        k = len(self.num)
        for i in range(0, n, self.BLOQUE):
            a = A[i:i + self.BLOQUE].astype(np.float64)
            Z[i:i + self.BLOQUE, :k] = (np.where(np.isnan(a), self.med, np.clip(a, self.lo, self.hi)) - self.mu) / self.sd
        del A
        if self.na_cols:
            Z[:, k:k + len(self.na_cols)] = (X[self.na_cols].isna().to_numpy(np.float64) - self.na_mu) / self.na_sd
            k += len(self.na_cols)
        for c, (j, st, m, s) in self.spl.items():
            b = Z[:, j] * self.sd[j] + self.mu[j]                     # columna recortada e imputada (antes de estandarizar)
            S = (st.transform(b[:, None]) - m) / s
            Z[:, k:k + S.shape[1]] = S; k += S.shape[1]
        for c in self.cat:
            v = X[c].astype(str).values
            niv = self.niveles[c]
            O = np.column_stack([(v == l) for l in niv] + [~np.isin(v, niv)]).astype(np.float64)
            Z[:, k:k + O.shape[1]] = O - O.mean(0); k += O.shape[1]
        assert k == self.p
        return Z


class MotorLogitV7:
    """Logit con penalización L2 fuerte sobre variables estandarizadas y splines en las 5 continuas principales
    (scikit-learn, lbfgs). Interfaz de motor de `ajuste.cv` sin parada temprana."""
    nombre, corto = "Logit L2", "LOGIT"
    parada_temprana = False
    usa_shap = False
    perdida = "LogLoss_sinBOG"

    def __init__(self, C=0.01, splines=(), max_iter=3000, min_frec=30):
        self.C, self.splines, self.max_iter, self.min_frec = float(C), list(splines), max_iter, min_frec
        self.base = {"C": self.C}

    def ajustar(self, Xtr, ytr, Xes=None, yes=None, params=None, semilla=1, n_arboles=None, paciencia=None, w=None):
        from sklearn.linear_model import LogisticRegression
        p = {**self.base, **(params or {})}
        prep = _PrepLogit(self.splines, min_frec=self.min_frec, semilla=semilla).fit(Xtr)
        Z = prep.transform(Xtr)
        m = LogisticRegression(C=float(p["C"]), solver="lbfgs", max_iter=self.max_iter, tol=1e-5)
        m.fit(Z, ytr, sample_weight=w)
        self.ultimo_n_iter = int(np.max(m.n_iter_))
        return (prep, m), 1

    def n_final(self, iters):
        return 1

    def predecir(self, mod, X, n_iter=None):
        prep, m = mod
        return m.predict_proba(prep.transform(X))[:, 1]


# ------------------------------------------------------------------------------ evaluación y comparación pareada
def evaluar(motor, X, y, folds, sin_bog, cols, params, transformar=None, semillas=(1,)):
    """CV de 5 pliegues con el protocolo común; devuelve métricas, F1 y log-loss sin Bogotá por pliegue, la predicción
    anidada 0/1 y el OOF."""
    from . import ajuste as AJ
    from . import metrics as M
    r = AJ.cv(motor, X, y, folds, params, AJ.Conjunto("tmp", list(cols)), semillas=semillas, transformar=transformar)
    oof = r["oof"]
    res = AJ.resumen(y, oof, folds, sin_bog)
    _, f1k, umb, yhat = AJ.f1_anidado_sb(y, oof, folds, sin_bog)
    llk = np.array([M.log_loss(y[va][sin_bog[va]], np.clip(oof[va][sin_bog[va]], 1e-6, 1 - 1e-6)) for _, va in folds])
    return {"res": res, "f1k": np.asarray(f1k), "llk": llk, "yhat": yhat, "oof": oof, "umbrales": umb,
            "tiempo_s": r["tiempo_s"], "iteraciones": r["iteraciones"]}


def comparar(a: dict, b: dict, y, sin_bog, B=1000, semilla=1) -> dict:
    """Mejora de `b` sobre `a` (positivo = mejor): ΔF1 anidado y Δ log-loss sin Bogotá pareados por pliegue, su EE
    entre pliegues, la prueba t con la varianza corregida de Nadeau y Bengio (2003) para validación cruzada de K
    pliegues (σ̂²·(1/K + n_val/n_ent), gl = K − 1), IC 95 % bootstrap pareado por hogares del ΔF1 global (predicciones
    anidadas fijas) y la regla de 1 EE (una métrica mejora > 1 EE y la otra no empeora > 1 EE)."""
    from scipy import stats
    d1 = b["f1k"] - a["f1k"]; d2 = a["llk"] - b["llk"]
    K = len(d1)
    out = {}
    for nom, d in (("F1", d1), ("LL", d2)):
        m, s = float(d.mean()), float(d.std(ddof=1))
        ee = s / np.sqrt(K)
        s_corr = s * np.sqrt(1 / K + 1 / (K - 1))
        t = m / s_corr if s_corr > 0 else np.inf * np.sign(m)
        out[f"d{nom}"] = m; out[f"EE_d{nom}"] = ee
        out[f"t_NB_{nom}"] = float(t); out[f"p_NB_{nom}"] = float(2 * stats.t.sf(abs(t), K - 1)) if np.isfinite(t) else 0.0
        out[f"pliegues_mejora_{nom}"] = int((d > 0).sum())
    m = np.asarray(sin_bog, bool); yy = np.asarray(y)[m].astype(int)
    ya, yb = a["yhat"][m].astype(int), b["yhat"][m].astype(int)
    rng = np.random.default_rng(semilla)
    def f1(w, yh):
        tp = w @ ((yy == 1) & (yh == 1)); fp = w @ ((yy == 0) & (yh == 1)); fn = w @ ((yy == 1) & (yh == 0))
        return 2 * tp / (2 * tp + fp + fn)
    n = len(yy); vals = np.empty(B)
    for i in range(B):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        vals[i] = f1(w, yb) - f1(w, ya)
    out["IC95_dF1_boot"] = [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]
    e1, e2 = out["EE_dF1"], out["EE_dLL"]
    out["supera_1EE"] = bool((out["dF1"] > e1 and out["dLL"] >= -e2) or (out["dLL"] > e2 and out["dF1"] >= -e1))
    return out


# ------------------------------------------------------------------------------ auditoría
def matriz_prohibida_v7(th, tp, tep):
    """Referencia de lo prohibido alineada con los hogares de `th`, con el tipo de cada columna:
    * `ingreso_hogar`: Ingtotug, Ingtotugarr, Ingpcug, Indigente, Npobres, Nindigentes y transformaciones;
    * `monto_persona`: cada columna de persona sólo de train que es un MONTO (máximo > 12: P6500, P6750, P7070, los
      s1/a1 de cada pregunta, Impa, Isa, Ie, Imdi, Iof*, *es, Ingtot*…), agregada por hogar (suma, máximo y valor del
      jefe; faltante = 0, porque «no se preguntó» significa que la persona no tiene ese ingreso);
    * `estrato`: Estrato1 (del jefe, media y máximo);
    * `seguimiento` (informativa): preguntas de seguimiento sólo de train que NO son montos (¿incluyó este valor…?,
      meses a que corresponde lo recibido P6760, motivos P7140s*, P7500s1, banderas de imputación Cclasnr*).
    Devuelve (P, tipos)."""
    ids = th["id"].values
    P, tipo = {}, {}
    lp = th["Lp"].astype(float).values
    for c in HOG_PROHIBIDAS:
        P[f"hogar:{c}"] = th[c].astype(float).values
    ing = th["Ingtotug"].astype(float).values; ipc = th["Ingpcug"].astype(float).values
    P["hogar:log(Ingtotug)"] = np.log(np.maximum(ing, 1.0)); P["hogar:Ingpcug/Lp"] = ipc / lp
    P["hogar:log(Ingpcug/Lp)"] = np.log(np.maximum(ipc, 1.0) / lp); P["hogar:Ingtotug/Npersug"] = ing / th["Npersug"].clip(lower=1).values
    for k in P:
        tipo[k] = "ingreso_hogar"
    solo = prohibidas_persona(tp, tep)
    g = tp[["id"] + solo].copy()
    for c in solo:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    montos = [c for c in solo if c != "Estrato1" and g[c].max() > 12]
    seg = [c for c in solo if c != "Estrato1" and c not in montos]
    gg = g.groupby("id", sort=False)
    jefe = g[tp["P6050"].values == 1].drop_duplicates("id").set_index("id").reindex(ids)
    s_m = gg[montos].sum().reindex(ids).fillna(0); x_m = gg[montos].max().reindex(ids).fillna(0)
    for c in montos:
        P[f"persona_suma:{c}"] = s_m[c].values; P[f"persona_max:{c}"] = x_m[c].values
        P[f"persona_jefe:{c}"] = jefe[c].fillna(0).values
        for k in ("suma", "max", "jefe"):
            tipo[f"persona_{k}:{c}"] = "monto_persona"
    P["persona_jefe:Estrato1"] = jefe["Estrato1"].values
    P["persona_media:Estrato1"] = gg["Estrato1"].mean().reindex(ids).values
    P["persona_max:Estrato1"] = gg["Estrato1"].max().reindex(ids).values
    for k in ("persona_jefe:Estrato1", "persona_media:Estrato1", "persona_max:Estrato1"):
        tipo[k] = "estrato"
    s_s = gg[seg].sum(min_count=1).reindex(ids); x_s = gg[seg].max().reindex(ids)
    for c in seg:
        P[f"persona_suma:{c}"] = s_s[c].values; P[f"persona_max:{c}"] = x_s[c].values; P[f"persona_jefe:{c}"] = jefe[c].values
        for k in ("suma", "max", "jefe"):
            tipo[f"persona_{k}:{c}"] = "seguimiento"
    P = pd.DataFrame(P)
    keep = P.std(skipna=True).fillna(0) > 0
    P = P.loc[:, keep]
    return P, {k: v for k, v in tipo.items() if k in P.columns}


def _corr_n(A, B):
    """Correlación de Pearson por pares completos y número de pares (n_A × n_B)."""
    def prep(M):
        M = M.astype(np.float64)
        mk = np.isfinite(M)
        return np.where(mk, M, 0.0), mk.astype(np.float64)
    Xa, Ma = prep(A); Xb, Mb = prep(B)
    n = Ma.T @ Mb
    sa, sb = Xa.T @ Mb, Ma.T @ Xb
    saa, sbb, sab = (Xa ** 2).T @ Mb, Ma.T @ (Xb ** 2), Xa.T @ Xb
    num = n * sab - sa * sb
    den = np.sqrt(np.maximum(n * saa - sa ** 2, 0) * np.maximum(n * sbb - sb ** 2, 0))
    with np.errstate(all="ignore"):
        r = np.where(den > 0, num / den, 0.0)
    return np.clip(r, -1, 1), n


def _zrank(X):
    """Rangos centrados por columna (Spearman = Pearson de rangos)."""
    return pd.DataFrame(X).rank(na_option="keep").values


def auditar_v7(D, cols, P, tipos, y, familias, finales, mapa=None, umbral=0.95, umbral_y=0.9, n_min=1000, bloque=100):
    r"""Auditoría de correlación por variable (procesada por bloques de columnas para acotar la memoria):
    * máxima |ρ| de Pearson y de Spearman (pares completos, al menos `n_min` pares) con las columnas PROHIBIDAS
      (`ingreso_hogar`, `monto_persona`, `estrato`) y, aparte e informativa, con las de `seguimiento`;
    * si la máxima supera `umbral`, prueba de **compuerta común**: el monto prohibido $b$ sólo es positivo cuando la
      pregunta permitida lo está ($P(a\neq0\mid b\neq0)\ge 0{,}99$) y dentro del soporte común $|\rho(a,b)|<$ `umbral`
      (o $a$ es constante allí): la variable reproduce el patrón de salto de una pregunta permitida (regla 2), no el
      valor del ingreso;
    * |corr con Pobre| > `umbral_y` sería fuga; nombre sin patrón prohibido; preguntas fuente ⊂ columnas comunes."""
    prohib = [c for c in P.columns if tipos[c] != "seguimiento"]
    seg = [c for c in P.columns if tipos[c] == "seguimiento"]
    Pp, Ps = P[prohib].values.astype(np.float64), P[seg].values.astype(np.float64)
    Ppr, Psr = _zrank(Pp), _zrank(Ps)
    yv = np.asarray(y, float)[:, None]
    comunes_ = set(HOG_COMUNES) | set(PER_COMUNES)
    filas = []
    for i in range(0, len(cols), bloque):
        cs = cols[i:i + bloque]
        Xb = D[cs].apply(pd.to_numeric, errors="coerce").astype(np.float64).values
        Xr = _zrank(Xb)
        res = {}
        for nom, (A, Ar, nombres) in {"prohibida": (Pp, Ppr, prohib), "seguimiento": (Ps, Psr, seg)}.items():
            rp, n_ = _corr_n(Xb, A); rs, _ = _corr_n(Xr, Ar)
            rp = np.where(n_ >= n_min, rp, 0.0); rs = np.where(n_ >= n_min, rs, 0.0)
            res[nom] = (rp, rs, nombres)
        ry, _ = _corr_n(Xb, yv)
        for j, c in enumerate(cs):
            f = {"variable": c, "familia": familias.get(c, "B"), "en_todas_permitidas": c in finales}
            for nom, (rp, rs, nombres) in res.items():
                ip, is_ = int(np.argmax(np.abs(rp[j]))), int(np.argmax(np.abs(rs[j])))
                f[f"max_abs_pearson_{nom}"] = float(abs(rp[j, ip])); f[f"{nom}_pearson"] = nombres[ip]
                f[f"max_abs_spearman_{nom}"] = float(abs(rs[j, is_])); f[f"{nom}_spearman"] = nombres[is_]
            f["corr_pearson_con_Pobre"] = float(ry[j, 0])
            f["nombre_sin_patron_prohibido"] = not any(re.search(pt, c.lower()) for pt in PATRONES_PROHIBIDOS)
            if mapa is not None:
                qs = mapa.get(c)
                f["preguntas_fuente"] = "|".join(qs) if qs is not None else ""
                f["n_preguntas_costo"] = costo_preguntas(mapa, [c])[0] if qs is not None else np.nan
                f["preguntas_comunes"] = bool(qs is not None and set(qs) <= comunes_)
            filas.append(f)
    t = pd.DataFrame(filas)
    t["supera_umbral"] = (t.max_abs_pearson_prohibida > umbral) | (t.max_abs_spearman_prohibida > umbral)
    t["contencion_soporte"] = np.nan; t["max_abs_corr_en_soporte"] = np.nan; t["motivo"] = ""
    for i in np.flatnonzero(t.supera_umbral.values):
        c = t.at[i, "variable"]; a = pd.to_numeric(D[c], errors="coerce").astype(float)
        cont, dentro = [], []
        for pcol in {t.at[i, "prohibida_pearson"], t.at[i, "prohibida_spearman"]}:
            b = P[pcol]
            za, zb = a.fillna(0).abs().values > 1e-12, b.fillna(0).abs().values > 1e-12
            cont.append(float(za[zb].mean()) if zb.any() else 1.0)
            m = za & zb
            sa, sb = a.values[m], b.values[m]
            if m.sum() < 30 or np.nanstd(sa) == 0:
                dentro.append(0.0)
            else:
                dentro.append(max(abs(pd.Series(sa).corr(pd.Series(sb))), abs(pd.Series(sa).corr(pd.Series(sb), method="spearman"))))
        t.at[i, "contencion_soporte"] = min(cont); t.at[i, "max_abs_corr_en_soporte"] = max(dentro)
        if min(cont) >= 0.99 and max(dentro) < umbral:
            t.at[i, "motivo"] = "compuerta común: el monto sólo es positivo cuando la pregunta permitida lo está; ρ baja dentro del soporte"
    ok_c = t.supera_umbral & (t.motivo != "")
    fuga_y = t.corr_pearson_con_Pobre.abs() > umbral_y
    mal_mapa = (~t.preguntas_comunes) if "preguntas_comunes" in t else False
    t["estado"] = np.where(~t.nombre_sin_patron_prohibido | (t.supera_umbral & ~ok_c) | fuga_y | mal_mapa, "FALLA",
                           np.where(ok_c, "OK (compuerta común)", "OK"))
    return t


def ks_train_test(D, DT, cols, mascara_train):
    """Kolmogorov–Smirnov de dos muestras (train sin Bogotá vs test) por variable, con faltantes."""
    from scipy.stats import ks_2samp
    filas = []
    for c in cols:
        a = pd.to_numeric(D.loc[mascara_train, c], errors="coerce").astype(float).values
        b = pd.to_numeric(DT[c], errors="coerce").astype(float).values
        fa, fb = np.isnan(a).mean(), np.isnan(b).mean()
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) > 10 and len(b) > 10 and (np.ptp(a) > 0 or np.ptp(b) > 0):
            r = ks_2samp(a, b, method="asymp")
            ks, pv = float(r.statistic), float(r.pvalue)
        else:
            ks, pv = 0.0, 1.0
        filas.append({"variable": c, "KS_trainSB_vs_test": ks, "p_KS": pv, "pct_na_train": float(fa), "pct_na_test": float(fb),
                      "dif_na": float(fb - fa)})
    return pd.DataFrame(filas)
