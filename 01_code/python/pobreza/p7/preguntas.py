r"""La pregunta de la encuesta como unidad de parsimonia (memo §3.2).

Para la política el costo de un conjunto de variables S no es su número de columnas sino el número de **preguntas**
del cuestionario que hay que hacer: $c(S)=\bigl|\bigcup_{j\in S}Q(j)\bigr|$, con $Q(j)$ las preguntas fuente de la
variable j. Agregar una variable derivada de preguntas ya incluidas cuesta cero. Dominio, departamento, clase
(cabecera), las líneas $L_p$, $L_i$ y el factor de expansión no son preguntas (los conoce el Estado): costo cero.

Si `conjuntos_variables_v7.json` trae `mapa_preguntas` (variable → lista de preguntas), se usa tal cual. Si no, este
módulo construye un mapa **heurístico y documentado** a partir de: (i) los códigos DANE (P5000…P7510s7) que aparecen en
el nombre o en la descripción de la variable; (ii) un diccionario de raíces (indicadores individuales de
`features.py` y nombres cortos de las preguntas de `variables_v7.py`); (iii) herencia recursiva de las variables que la
descripción o el nombre mencionan (p. ej. `t_raiz_share_subsidiado` hereda de `share_subsidiado`). Lo que quede sin
resolver se trata como su propia unidad (costo 1) y se reporta la cobertura. «FT» es el módulo de fuerza de trabajo
(ocupado/desocupado/inactivo) y «P6050» (parentesco) es necesario para identificar al jefe, la pareja y la unidad de
gasto (Npersug).

Asignación de importancias a preguntas. Si la contribución de la variable j sólo existe cuando están TODAS sus
preguntas (un juego de unanimidad $u_{Q(j)}$), su valor de Shapley reparte la contribución por partes iguales entre
las preguntas de $Q(j)$; por eso una importancia aditiva (SHAP, reducción de pérdida) se asigna a cada pregunta como
$\Phi_q=\sum_{j: q\in Q(j)}\phi_j/|Q(j)|$.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

GRATIS = "_gratis"     # pseudo-pregunta de costo cero (diseño muestral y líneas: las conoce el Estado)
FT = "FT"
BENEFICIOS = ["P6510", "P6545", "P6580", "P6585s1", "P6585s2", "P6585s3", "P6585s4", "P6590", "P6600", "P6610", "P6620",
              "P6630s1", "P6630s2", "P6630s3", "P6630s4", "P6630s6"]
ETIQUETAS = {
    "Nper": "número de personas del hogar", "P5000": "número de cuartos", "P5010": "número de dormitorios",
    "P5090": "tenencia de la vivienda", "P5100": "cuota de amortización", "P5130": "arriendo estimado (propietarios)",
    "P5140": "arriendo pagado", "P6020": "sexo", "P6040": "edad", "P6050": "parentesco con el jefe",
    "P6090": "afiliación a salud", "P6100": "régimen de salud", "P6210": "nivel educativo", "P6210s1": "grado aprobado",
    "P6240": "actividad principal", "Oficio": "oficio", "P6426": "antigüedad en el empleo", "P6430": "posición ocupacional",
    "P6800": "horas en el trabajo principal", "P6870": "tamaño de la empresa", "P6920": "cotiza a pensiones",
    "P7040": "segundo trabajo", "P7045": "horas en el segundo trabajo", "P7050": "posición en el segundo trabajo",
    "P7090": "quiere trabajar más horas", "P7110": "diligencias para más horas", "P7120": "disponible para más horas",
    "P7150": "diligencias para cambiar de trabajo", "P7160": "podría empezar antes de un mes",
    "P7310": "busca trabajo por primera vez", "P7350": "posición en el último trabajo (desocupados)",
    FT: "situación laboral (ocupado/desocupado/inactivo)", GRATIS: "diseño y líneas (costo cero)",
}

# indicadores individuales de features.py → preguntas
ATOMOS = {
    "edad": ["P6040"], "mujer": ["P6020"], "jefe": ["P6050"], "pareja": ["P6050"], "en_ug": ["P6050"],
    "empleado_domestico_hogar": ["P6050"], "pensionista_hogar": ["P6050"], "menor5": ["P6040"], "menor12": ["P6040"],
    "menor18": ["P6040"], "edad18_64": ["P6040"], "mayor65": ["P6040"], "anios_educ": ["P6210", "P6210s1"],
    "educ_superior": ["P6210"], "educ_media_o_mas": ["P6210"], "sin_educacion": ["P6210", "P6040"], "pet": ["P6040"],
    "ocupado": [FT], "desocupado": [FT], "inactivo": [FT], "estudiante": ["P6240"], "oficios_hogar": ["P6240"],
    "incapacitado": ["P6240"], "asalariado": ["P6430"], "independiente": ["P6430"], "sin_remuneracion": ["P6430"],
    "cotiza_pension": ["P6920"], "pensionado": ["P6920"], "ocupado_informal": [FT, "P6920"],
    "horas": [FT, "P6800", "P7045"], "segundo_trabajo": ["P7040"], "subempleo_horas": ["P7090"],
    "busca_cambiar": ["P7150"], "trabaja_solo": ["P6870"], "empresa_grande": ["P6870"], "empresa_2_10": ["P6870"],
    "antiguedad_meses": [FT, "P6426"], "antiguedad": [FT, "P6426"], "n_beneficios": BENEFICIOS, "beneficios": BENEFICIOS,
    "prima_servicios": ["P6630s1"], "prima_navidad": ["P6630s2"], "subsidio_transporte": ["P6585s2"],
    "subsidio_familiar": ["P6585s3"], "horas_extra": ["P6510"], "pago_especie": ["P6590", "P6600", "P6610", "P6620"],
    "salud_afiliado": ["P6090"], "salud_no_afiliado": ["P6090"], "salud_contributivo": ["P6090", "P6100"],
    "salud_especial": ["P6090", "P6100"], "salud_subsidiado": ["P6090", "P6100"], "subsidiado": ["P6090", "P6100"],
    "desocupado_con_ingreso": ["P7422"], "inactivo_con_ingreso": ["P7472"], "desocupado_cesante": ["P7310"],
    "ocupado_superior": [FT, "P6210"], "ocupados_superior": [FT, "P6210"], "asalariado_formal": ["P6430", "P6920"],
    "asalariados_formales": ["P6430", "P6920"], "horas_formal": [FT, "P6800", "P7045", "P6920"],
    "horas_formales": [FT, "P6800", "P7045", "P6920"], "patron_o_cp_empresa_grande": ["P6430", "P6870"],
    "oficio": ["Oficio"], "posicion": ["P6430"], "tam_empresa": ["P6870"], "nivel_edu": ["P6210"],
    "regimen_salud": ["P6100"], "actividad": ["P6240"], "pos": ["P6430"], "ocupados": [FT], "formales": [FT, "P6920"],
    "formal": [FT, "P6920"], "cuenta_propia": ["P6430"], "patronos": ["P6430"], "patron": ["P6430"],
    "asalariados": ["P6430"], "pensionados": ["P6920"], "desocupados": [FT], "inactivos": [FT],
    "recibe_arriendos_o_pensiones": ["P7495"], "recibe_pension_jubilacion": ["P7495", "P7500s2"],
    "recibe_pension_alimentaria": ["P7495", "P7500s3"], "recibe_otros_ingresos": ["P7505"],
    "recibe_ayuda_otros_hogares_pais": ["P7505", "P7510s1"], "recibe_remesas_exterior": ["P7505", "P7510s2"],
    "recibe_ayuda_instituciones": ["P7505", "P7510s3"], "recibe_intereses_dividendos": ["P7505", "P7510s5"],
    "recibe_cesantias": ["P7505", "P7510s6"], "recibe_otras_fuentes": ["P7505", "P7510s7"],
    "recibe_transferencias": ["P7505", "P7510s1", "P7510s3"], "recibe_pension": ["P7495", "P7500s2"],
    "recibe_capital": ["P7495", "P7505", "P7510s5"],
}
# vivienda, tamaño y líneas
HOGAR = {
    "Nper": ["Nper"], "Npersug": ["Nper", "P6050"], "npersug": ["Nper", "P6050"], "cuartos": ["P5000"], "cuarto": ["P5000"],
    "dormitorios": ["P5010"], "dormitorio": ["P5010"], "hacinamiento": ["Nper", "P5010"], "tenencia": ["P5090"],
    "ten": ["P5090"], "inquilino": ["P5090"], "usufructo": ["P5090"], "propia": ["P5090"], "propietario": ["P5090"],
    "arriendo_pagado": ["P5090", "P5140"], "arriendo_estimado": ["P5090", "P5130"],
    "cuota_amortizacion": ["P5090", "P5100"], "amortizacion": ["P5090", "P5100"], "cuota": ["P5090", "P5100"],
    "arriendo": ["P5090", "P5130", "P5140"], "vivienda": ["P5090", "P5130", "P5140"],
    "arriendo_imputado": ["P5090", "P5130", "P5140", "P5000"],
    "requerido": ["Nper", "P6050", "P5090", "P5130", "P5140", "P5000"],
    "linea_hogar": ["Nper", "P6050"], "linea": ["Nper", "P6050"], "lineas": ["Nper", "P6050"], "pc": ["Nper", "P6050"],
    "ug": ["Nper", "P6050"], "dependencia": ["P6040"], "dependientes": ["Nper", FT], "menores": ["P6040", "Nper"],
    "mujeres": ["P6020", "Nper"], "pareja": ["P6050"], "jefe": ["P6050"], "principal": ["P6050"], "hijos": ["P6050"],
    "educ": ["P6210", "P6210s1", "P6040"], "edad": ["P6040"], "exp": ["P6040", "P6210", "P6210s1"],
    "experiencia": ["P6040", "P6210", "P6210s1"], "perceptores": [FT, "P7495", "P7505", "P7422", "P7472"],
    "fuentes": [FT, "P7495", "P7505", "P7422", "P7472"], "nolab": ["P7495", "P7505", "P7422", "P7472"],
    "ayudas": ["P7505", "P7510s1", "P7510s3"], "arriendos": ["P7495"], "intereses": ["P7505", "P7510s5"],
    "segunda": ["P7040"], "segundo": ["P7040"], "subempleo": ["P7090"], "ipm": ["P6040", "P6210", "P6210s1", "P6240", FT,
                                                                                  "P6920", "P6090", "Nper", "P5010"],
    "puntaje": ["P6040", "P6210", "P6210s1", "P6240", FT, "P6920", "P6090", "Nper", "P5010"],
    "personas": ["Nper"], "edad2": ["P6040"], "ocupacion": [FT], "hs": [FT, "P6800", "P7045", "P6210", "P6210s1"],
    "hx": [FT, "P6800", "P7045", "P6040", "P6210", "P6210s1"], "antig": [FT, "P6426"], "asal": ["P6430"],
    "informal": [FT, "P6920"], "prestaciones": BENEFICIOS, "extralegales": BENEFICIOS, "ocup": [FT],
    "contributivo": ["P6090", "P6100"], "especial": ["P6090", "P6100"], "especie": ["P6590", "P6600", "P6620"],
    "transporte": ["P6585s2", "P6610"], "aux": ["P6585s1", "P6585s2", "P6585s3", "P6585s4"],
    "analfabetismo": ["P6210", "P6040"], "logro": ["P6210", "P6210s1", "P6040"], "inasistencia": ["P6240", "P6040"],
    "rezago": ["P6210", "P6210s1", "P6040"], "trabajo_infantil": [FT, "P6040"], "desempleo": [FT],
    "sin_salud": ["P6090"], "pea": [FT], "dim_educacion": ["P6210", "P6210s1", "P6040"],
    "dim_ninez": ["P6210", "P6210s1", "P6040", "P6240", FT], "dim_trabajo": [FT, "P6920"],
    "privaciones": ["P6040", "P6210", "P6210s1", "P6240", FT, "P6920", "P6090", "Nper", "P5010"],
    "intensidad": ["P6040", "P6210", "P6210s1", "P6240", FT, "P6920", "P6090", "Nper", "P5010"],
    "dimensiones": ["P6040", "P6210", "P6210s1", "P6240", FT, "P6920", "P6090", "Nper", "P5010"],
    "monoparental": ["P6050"], "jefa": ["P6020", "P6050"], "unipersonal": ["Nper"], "mayores": ["P6040"],
    "menores5": ["P6040"], "adulto": ["P6040"], "superior": ["P6210"], "extendido": ["P6050"], "parientes": ["P6050"],
    "nietos": ["P6050"], "sin_pension": ["P7495", "P7500s2"], "todos_subsidiados": ["P6090", "P6100"],
    "Pet": ["P6040"], "Oc": [FT], "Des": [FT], "Ina": [FT], "jornaleros": ["P6430"], "educmax": ["P6210", "P6210s1", "P6040"],
    "trabajo_remunerado": [FT, "P7422", "P7472"], "transferencias": ["P7505", "P7510s1", "P7510s3"],
    "capital": ["P7495", "P7505", "P7510s5"], "pension": ["P7495", "P7500s2"],
    "n_lineas": ["Nper", "P6050", "P5090", "P5130", "P5140", "P5000"],
    "no_puede_ser_pobre": ["Nper", "P6050", "P5090", "P5130", "P5140", "P5000"],
}
SI_NO_TODAS = ["P6510", "P6545", "P6580", "P6585s1", "P6585s2", "P6585s3", "P6585s4", "P6590", "P6600", "P6610", "P6620",
               "P6630s1", "P6630s2", "P6630s3", "P6630s4", "P6630s6", "P7422", "P7472", "P7495", "P7500s2", "P7500s3", "P7505",
               "P7510s1", "P7510s2", "P7510s3", "P7510s5", "P7510s6", "P7510s7"]
HOGAR.update({"ns_total": SI_NO_TODAS, "ns_algun": SI_NO_TODAS})
GRATIS_TOKENS = {"rural", "resto_urbano", "fex", "dpto"}
GRATIS_NOMBRES = {"Dominio", "Depto", "cabecera", "Clase", "Lp", "Li", "log_Lp", "Fex_c", "log_fex", "Fex_dpto",
                  "u_li_sobre_lp", "u_lp_menos_li_rel", "te_depto_clase"}
GRATIS_PATRONES = [r"^t_dom_", r"^t_depto_", r"^log_fex", r"^t_cabecera$", r"^t_log_Lp$"]
PCODIGO = re.compile(r"P\d{4}(?:s\d{1,2})?(?![0-9])")
VALIDOS = {"P5000", "P5010", "P5090", "P5100", "P5130", "P5140", "P6020", "P6040", "P6050", "P6090", "P6100", "P6210",
           "P6210s1", "P6240", "P6426", "P6430", "P6800", "P6870", "P6920", "P7040", "P7045", "P7050", "P7090", "P7110",
           "P7120", "P7150", "P7160", "P7310", "P7350"} | set(BENEFICIOS) | {
           "P7422", "P7472", "P7495", "P7500s2", "P7500s3", "P7505", "P7510s1", "P7510s2", "P7510s3", "P7510s5", "P7510s6",
           "P7510s7"}


def _codigos(texto: str) -> set:
    return {q for q in PCODIGO.findall(texto) if q in VALIDOS}
TE_CLAVES = {"te_oficio_jefe": ["Oficio", "P6050"], "te_oficio_principal": ["Oficio", "P6050", FT],
             "te_dominio_tenencia": ["P5090"], "te_depto_clase": [], "te_posicion_jefe": ["P6430", "P6050"]}


def _tokens_extra():
    """Nombres cortos de las preguntas definidos en variables_v7 (si el módulo está disponible)."""
    out = {}
    try:
        from .. import variables_v7 as V
        for q, v in V.SI_NO_INGRESO.items():
            out[v[0]] = [q] + (["P7495"] if q.startswith("P7500") else []) + (["P7505"] if q.startswith("P7510") else [])
        for q, (nom, _) in V.CATEGORICAS_PERSONA.items():
            out[nom] = [q]
        for q, d in V.NUMERICAS_PERSONA.items():
            out[{"P6040": "edad", "P6210s1": "grado", "P6426": "antiguedad", "P6800": "horas_principal", "P7045": "horas_segundo"}[q]] = [q]
    except Exception:  # pragma: no cover
        pass
    out.update({"aux_alimentacion": ["P6585s1"], "aux_transporte": ["P6585s2"], "subsidio_educativo": ["P6585s4"],
                "especie_alimentos": ["P6590"], "especie_vivienda": ["P6600"], "transporte_empresa": ["P6610"],
                "especie_otros": ["P6620"], "prima_vacaciones": ["P6630s3"], "viaticos": ["P6630s4"],
                "bonif_anuales": ["P6630s6"], "primas": ["P6545"], "bonificaciones": ["P6580"],
                "disponible": ["P7160"], "diligencias": ["P7110"], "afiliado": ["P6090"], "cotiza": ["P6920"],
                "oficio_gg": ["Oficio"], "ofgg": ["Oficio"], "tam": ["P6870"]})
    return out


def _es_gratis(c: str) -> bool:
    return c in GRATIS_NOMBRES or any(re.search(p, c) for p in GRATIS_PATRONES)


def mapa_heuristico(columnas, descripciones: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """Mapa variable → lista ordenada de preguntas y tabla de diagnóstico (regla usada por variable)."""
    descripciones = descripciones or {}
    raices = {**ATOMOS, **HOGAR, **_tokens_extra()}
    claves = sorted(raices, key=len, reverse=True)
    cols = list(columnas)
    conocidas = set(cols)
    mapa, regla = {}, {}

    def por_tokens(nombre: str) -> set:
        s = _codigos(nombre)
        resto = "_" + nombre + "_"
        for k in claves:
            pat = "_" + k + "_"
            if pat in resto:
                s |= set(raices[k]); resto = resto.replace(pat, "_")
        return s

    for c in cols:
        if _es_gratis(c):
            mapa[c] = set(); regla[c] = "costo cero (diseño o líneas)"; continue
        if c in TE_CLAVES:
            mapa[c] = set(TE_CLAVES[c]); regla[c] = "exacto (codificación con la etiqueta: claves)"; continue
        base = c
        if re.match(r"^(t_)?ten_", c):
            mapa[c] = {"P5090"}; regla[c] = "exacto"; continue
        if c.startswith("v_") and c[2:] in GRATIS_TOKENS | {"resto_urbano", "log_fex_dpto", "fex_c_sobre_dpto"}:
            mapa[c] = set(); regla[c] = "costo cero (diseño o líneas)"; continue
        if c.startswith(("n_", "jefe_", "pareja_")):
            pref, resto = c.split("_", 1)
            if resto in raices:
                q = set(raices[resto]) | ({"P6050"} if pref in ("jefe", "pareja") else set())
                mapa[c] = q; regla[c] = "indicador individual agregado"; continue
        q = por_tokens(base)
        d = descripciones.get(c, "")
        q |= _codigos(d)
        if c.startswith(("c_", "i_")) and "_linea" in c:            # «por línea necesaria» n_ℓ = R_h/Lp (v7)
            q |= set(HOGAR["n_lineas"])
        mapa[c] = q; regla[c] = "códigos y raíces" if q else ""
        if not q and any(("_" + t + "_") in ("_" + c + "_") for t in GRATIS_TOKENS):
            regla[c] = "costo cero (diseño o líneas)"
    # herencia de variables mencionadas en el nombre o la descripción (hasta punto fijo): se buscan las
    # subsecuencias contiguas de partes separadas por «_» de cada identificador del texto
    def mencionadas(c):
        texto = c + " " + descripciones.get(c, "")
        out = set()
        for ident in re.findall(r"[A-Za-z0-9_]+", texto):
            p = ident.split("_")
            for i in range(len(p)):
                for j in range(i + 1, len(p) + 1):
                    s = "_".join(p[i:j])
                    if s != c and len(s) >= 4 and s in conocidas:
                        out.add(s)
        return out

    refs = {c: mencionadas(c) for c in cols}
    for _ in range(5):
        cambio = False
        for c in cols:
            if regla[c].startswith(("costo cero", "indicador individual", "exacto")):
                continue
            for o in refs[c]:
                nuevo = mapa[c] | mapa[o]
                if nuevo != mapa[c]:
                    mapa[c] = nuevo; cambio = True
                    if "hereda" not in regla[c]:
                        regla[c] = (regla[c] + "; " if regla[c] else "") + "hereda"
        if not cambio:
            break
    sin = [c for c in cols if not mapa[c] and regla[c] != "costo cero (diseño o líneas)"]
    for c in sin:
        mapa[c] = {f"var:{c}"}; regla[c] = "sin resolver: unidad propia"
    tabla = pd.DataFrame({"variable": cols, "preguntas": [", ".join(sorted(mapa[c])) for c in cols],
                          "n_preguntas": [len(mapa[c]) for c in cols], "regla": [regla[c] for c in cols]})
    return {c: sorted(mapa[c]) for c in cols}, tabla


class MapaPreguntas:
    """Utilidades sobre el mapa variable → preguntas."""

    def __init__(self, mapa: dict, costo_cero=(), etiquetas: dict | None = None):
        cero = set(costo_cero) | {GRATIS}
        self.mapa = {c: [q for q in qs if q not in cero] for c, qs in mapa.items()}
        self.preguntas = sorted({q for qs in self.mapa.values() for q in qs})
        self.etiquetas = dict(etiquetas or {})

    def costo(self, variables) -> int:
        return len({q for c in variables for q in self.mapa.get(c, [f"var:{c}"])})

    def preguntas_de(self, variables) -> list:
        return sorted({q for c in variables for q in self.mapa.get(c, [f"var:{c}"])})

    def compatibles(self, preguntas, universo) -> list:
        """Variables del universo cuyas preguntas están todas en `preguntas` (las de costo cero siempre entran)."""
        P = set(preguntas)
        return [c for c in universo if set(self.mapa.get(c, [f"var:{c}"])) <= P]

    def grupos(self, universo) -> dict:
        """Pregunta → variables del universo que dependen de ella (grupos solapados, para permutación y LOCO)."""
        g = {}
        for c in universo:
            for q in self.mapa.get(c, [f"var:{c}"]):
                g.setdefault(q, []).append(c)
        return g

    def unidades(self, universo) -> list:
        return sorted({q for c in universo for q in self.mapa.get(c, [f"var:{c}"])})

    def importancia_por_pregunta(self, imp_var: pd.Series) -> pd.Series:
        """Φ_q = Σ_{j: q∈Q(j)} φ_j/|Q(j)| (reparto de Shapley de un juego de unanimidad). Las variables de costo cero
        se acumulan en `_gratis`."""
        out = Counter()
        for c, v in imp_var.items():
            qs = self.mapa.get(c, [f"var:{c}"])
            if not qs:
                out[GRATIS] += float(v)
            for q in qs:
                out[q] += float(v) / len(qs)
        return pd.Series(out).sort_values(ascending=False)

    def matriz_grupo(self, phi: np.ndarray, columnas) -> tuple[np.ndarray, list]:
        """Suma por pregunta de contribuciones aditivas por observación (φ: n × p) → (n × |preguntas|)."""
        qs = self.unidades(columnas) + [GRATIS]
        idx = {q: i for i, q in enumerate(qs)}
        A = np.zeros((len(columnas), len(qs)))
        for j, c in enumerate(columnas):
            m = self.mapa.get(c, [f"var:{c}"])
            if not m:
                A[j, idx[GRATIS]] = 1.0
            for q in m:
                A[j, idx[q]] = 1.0 / len(m)
        return phi @ A, qs

    def etiqueta(self, q: str) -> str:
        if q.startswith("var:"):
            return q[4:]
        if q in self.etiquetas:
            e = self.etiquetas[q]
            return f"{q} ({e[:60]}{'…' if len(e) > 60 else ''})"
        if q in ETIQUETAS:
            return f"{q} ({ETIQUETAS[q]})"
        try:
            from .. import variables_v7 as V
            if q in V.SI_NO_INGRESO:
                return f"{q} ({V.SI_NO_INGRESO[q][0].replace('_', ' ')})"
        except Exception:  # pragma: no cover
            pass
        return q
