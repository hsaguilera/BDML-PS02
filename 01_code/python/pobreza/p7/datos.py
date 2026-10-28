r"""Datos del protocolo v7: tablas, verificaciones de integridad (Paso 0), particiones repetidas (Paso 1),
codificación con la etiqueta dentro del pliegue y mapa variable → preguntas.

Fuentes. `v7`: `modelo_{train,test}_v7.parquet` y `conjuntos_variables_v7.json` (fase B). `respaldo` (mientras v7 no
exista): `modelo_{train,test}.parquet` con el conjunto S2 sin el índice de Mincer (coeficientes externos) ni variables
generadas, más las categóricas; la interfaz es idéntica.

Particiones (memo §2.2 y Paso 1). r = 1: la columna `fold` actual (`pliegues.csv`); r = 2, 3: `StratifiedKFold(5)`
con semillas 20263 y 20264 sobre el estrato Pobre × Dominio de los 164.960 hogares (Bogotá incluida, para cuando
entra al entrenamiento con λ_B > 0; nunca entra a una métrica); r = 4 (semilla 20265): partición **nueva** reservada a
la re-evaluación de la configuración elegida (no se usa en la búsqueda ni en la confirmación). Se guardan en
`pliegues_v7.csv` (columnas `fold_r1` … `fold_r4`) y son las mismas para todos los algoritmos.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import preguntas as PQ

SEMILLAS_PARTICION = {2: 20263, 3: 20264, 4: 20265}
CAT_RESPALDO = ["Dominio", "Depto", "jefe_oficio", "jefe_posicion", "jefe_nivel_edu", "jefe_regimen_salud",
                "jefe_actividad", "jefe_tam_empresa", "tenencia"]
ENCUESTA_CORTA_RESPALDO = ["Nper", "n_menor18", "n_mayor65", "jefe_mujer", "jefe_edad", "jefe_anios_educ",
                           "educ_max_adultos", "n_ocupado", "n_cotiza_pension", "cuartos", "tenencia", "log_arriendo",
                           "n_salud_subsidiado", "Dominio", "cabecera"]
# candidatas a restricción monótona por teoría económica (memo §4.4): signo de ∂η/∂x a lo demás constante
MONOTONAS = {"log_ingreso_requerido": 1, "ocupados_pc": -1, "formales_pc": -1, "educ_max_adultos": -1,
             "jefe_anios_educ": -1, "personas_por_cuarto": 1, "dependencia": 1, "share_subsidiado": 1}


def _huella(*partes) -> str:
    h = hashlib.md5()
    for p in partes:
        h.update(json.dumps(p, sort_keys=True, default=str).encode() if not isinstance(p, (bytes, np.ndarray)) else
                 (p.tobytes() if isinstance(p, np.ndarray) else p))
    return h.hexdigest()[:12]


def estratos(y, dominio, k=5) -> np.ndarray:
    e = pd.Series(np.asarray(dominio).astype(str)) + "_" + pd.Series(np.asarray(y).astype(int)).astype(str)
    vc = e.value_counts()
    return np.where(e.map(vc).values < k, pd.Series(np.asarray(y).astype(int)).astype(str).values, e.values)


def construir_particiones(ids, y, dominio, fold_r1, K=5) -> pd.DataFrame:
    e = estratos(y, dominio, K)
    P = pd.DataFrame({"id": np.asarray(ids), "fold_r1": np.asarray(fold_r1).astype(int)})
    for r, sem in SEMILLAS_PARTICION.items():
        f = np.full(len(e), -1)
        for k, (_, va) in enumerate(StratifiedKFold(K, shuffle=True, random_state=sem).split(np.zeros(len(e)), e)):
            f[va] = k
        P[f"fold_r{r}"] = f
    return P


def preparar(df: pd.DataFrame, columnas, categoricas, ref: pd.DataFrame) -> pd.DataFrame:
    """Numéricas en float32 y categóricas como `category` con los niveles de train (`ref`)."""
    cols = {}
    for c in columnas:
        if c in categoricas:
            cats = pd.Index(sorted(ref[c].astype(str).unique()))
            cols[c] = pd.Categorical(df[c].astype(str), categories=cats)
        else:
            cols[c] = pd.to_numeric(df[c], errors="coerce").astype("float32").values
    return pd.DataFrame(cols, index=df.index)


class Datos7:
    """Tablas, máscaras, particiones y matrices del protocolo v7."""

    def __init__(self, ent, fuente: str | None = None, submuestra: float | None = None, semilla_sub: int = 1,
                 verbose: bool = True):
        import os
        t0 = time.time()
        self.ent = ent
        dd = Path(ent.datos)
        fuente = fuente or os.environ.get("P7_FUENTE", "auto")
        if fuente == "auto":
            fuente = "v7" if (dd / "modelo_train_v7.parquet").exists() and (dd / "conjuntos_variables_v7.json").exists() else "respaldo"
        self.fuente = fuente
        suf = "_v7" if fuente == "v7" else ""
        D = pd.read_parquet(dd / f"modelo_train{suf}.parquet")
        DT = pd.read_parquet(dd / f"modelo_test{suf}.parquet")
        conj = json.load(open(dd / ("conjuntos_variables_v7.json" if fuente == "v7" else "conjuntos_variables.json"), encoding="utf-8"))
        self.conj = conj
        if "fold" not in D.columns:
            P0 = pd.read_csv(dd / "pliegues.csv")
            D = D.merge(P0, on="id", how="left")
        # ---------------------------------------------------------------- columnas
        if fuente == "v7":
            todas = conj.get("todas_permitidas") or conj.get("S3")
            cats = [c for c in conj.get("categoricas", CAT_RESPALDO) if c in todas]
            # LOCO por bloque teórico (A–F de la base y familias de v7) si la fase B lo entrega; si no, las familias
            self.bloques_mapa = dict(conj.get("bloques_teoricos") or conj.get("bloques", {}))
            self.familias = dict(conj.get("bloques", {}))
            self.nombres_bloques = dict(conj.get("nombres_bloques", conj.get("familias_letras", {})))
            te = conj.get("codificacion_objetivo", {})
            self.te_cols = [c for c in te.get("columnas", [c for c in todas if c.startswith("te_")]) if c in todas]
            self.te_spec = te.get("especificaciones")
            self.te_sin_bog = bool(te.get("excluir_bogota", True))
            corta = conj.get("encuesta_corta_v7") or conj.get("encuesta_corta") or ENCUESTA_CORTA_RESPALDO
            desc = conj.get("descripciones", {})
            mapa = conj.get("mapa_preguntas")
        else:
            from ..ajuste import bloque_variable, NOMBRE_BLOQUE
            self.familias = {}
            gen = set(conj.get("S3", [])) - set(conj.get("S2", []))
            todas = [c for c in conj["S2"] if "mincer" not in c.lower() and c not in gen] + \
                    [c for c in CAT_RESPALDO if c not in conj["S2"]]
            cats = [c for c in CAT_RESPALDO if c in todas]
            self.bloques_mapa = {c: bloque_variable(c) for c in todas}
            self.nombres_bloques = dict(NOMBRE_BLOQUE)
            self.te_cols, self.te_spec, self.te_sin_bog = [], None, True
            corta, desc, mapa = ENCUESTA_CORTA_RESPALDO, {}, None
        todas = [c for c in dict.fromkeys(todas) if c in D.columns and c in DT.columns and c not in ("id", "Pobre", "fold")]
        self.todas = todas
        self.categoricas = [c for c in cats if c in todas]
        self.corta = [c for c in corta if c in todas]
        self.descripciones = desc
        info_q = conj.get("preguntas", {}) if isinstance(conj.get("preguntas"), dict) else {}
        cero = {q for q, v in info_q.items() if isinstance(v, dict) and float(v.get("costo", 1)) == 0}
        etiq = {q: v.get("descripcion", "") for q, v in info_q.items() if isinstance(v, dict)}
        if mapa:
            self.mapa_fuente = "conjuntos_variables_v7.json (fase B: rastreo exacto de dependencias)"
            faltan = [c for c in todas if c not in mapa]
            mh, _ = PQ.mapa_heuristico(faltan, desc) if faltan else ({}, None)
            mapa = {c: list(mapa[c]) if c in mapa else mh[c] for c in todas}
            self.tabla_mapa = pd.DataFrame({"variable": todas, "preguntas": [", ".join(mapa[c]) for c in todas],
                                            "n_preguntas_costo": [len([q for q in mapa[c] if q not in cero]) for c in todas],
                                            "regla": ["fase B" if c not in faltan else "heurístico" for c in todas]})
        else:
            self.mapa_fuente = "heurístico (preguntas.py)"
            mapa, self.tabla_mapa = PQ.mapa_heuristico(todas, desc)
        self.MP = PQ.MapaPreguntas(mapa, costo_cero=cero, etiquetas=etiq)
        # ---------------------------------------------------------------- particiones (en la muestra completa)
        self.particiones_df = self._particiones(D)
        # ---------------------------------------------------------------- verificaciones del Paso 0
        self.verificacion = self.verificar(D, DT)
        # ---------------------------------------------------------------- submuestra (sólo pruebas)
        self.submuestra = float(submuestra) if submuestra else None
        if self.submuestra:
            g = D.fold.astype(str) + "_" + D.Pobre.astype(int).astype(str)
            keep = D.groupby(g, group_keys=False).sample(frac=self.submuestra, random_state=semilla_sub).index
            keep = np.sort(keep)
            D = D.loc[keep].reset_index(drop=True)
            self.particiones_df = self.particiones_df.iloc[keep].reset_index(drop=True)
        assert (self.particiones_df.id.values == D.id.values).all()
        self.y = D.Pobre.values.astype(int)
        self.ids, self.ids_test = D.id.values, DT.id.values
        self.dominio = D.Dominio.astype(str).values
        self.dominio_test = DT.Dominio.astype(str).values
        self.mT = self.dominio != "BOGOTA"
        self.bog = ~self.mT
        self.info = D[[c for c in ("id", "Dominio", "Depto", "cabecera", "Nper", "Lp") if c in D.columns]].copy()
        self.info_test = DT[[c for c in ("id", "Dominio", "Depto", "cabecera", "Nper", "Lp") if c in DT.columns]].copy()
        # ---------------------------------------------------------------- matrices
        self.X = preparar(D, todas, self.categoricas, D)
        self.XT = preparar(DT, todas, self.categoricas, D)
        self._cod = None
        if self.te_cols:
            from .. import variables_v7 as V
            kw = {"especificaciones": self.te_spec} if self.te_spec else {}
            self._cod = V.CodificadorObjetivo(D, self.y, DT=DT, excluir_bogota=self.te_sin_bog, **kw)
            self.te_cols = [c for c in self.te_cols if c in self._cod.esp]
        self._te_cache = {}
        # la huella incluye la fecha de las tablas: si la fase B las reescribe, ninguna caché anterior se reutiliza
        mtime = int(os.path.getmtime(dd / f"modelo_train{suf}.parquet"))
        self.huella = _huella(self.fuente, self.submuestra, len(self.y), int(self.y.sum()), self.todas[:50], len(self.todas), mtime)
        self.tiempo_carga_s = time.time() - t0
        del D, DT
        if verbose:
            mil = lambda v: f"{int(v):,}".replace(",", ".")
            print(f"Datos7[{self.fuente}] train {mil(self.X.shape[0])} × {self.X.shape[1]}, test {mil(self.XT.shape[0])} | "
                  f"𝒯 = {mil(self.mT.sum())} hogares, Bogotá = {mil(self.bog.sum())} | {len(self.todas)} variables "
                  f"({len(self.categoricas)} categóricas, {len(self.te_cols)} con la etiqueta) | "
                  f"{len(self.MP.unidades(self.todas))} preguntas (mapa {self.mapa_fuente}) | {self.tiempo_carga_s:.0f} s")

    # ------------------------------------------------------------------ particiones
    def _particiones(self, D) -> pd.DataFrame:
        nombre = "pliegues_v7.csv"
        for d in (Path(self.ent.procesados), Path(self.ent.datos)):
            p = d / nombre
            if p.exists():
                P = pd.read_csv(p)
                if len(P) == len(D) and (P.id.values == D.id.values).all() and (P.fold_r1.values == D.fold.values).all():
                    return P
        P = construir_particiones(D.id.values, D.Pobre.values, D.Dominio.values, D.fold.values)
        import os
        tmp = Path(self.ent.procesados) / f"{nombre}.{os.getpid()}.tmp"
        P.to_csv(tmp, index=False)
        os.replace(tmp, Path(self.ent.procesados) / nombre)          # escritura atómica (varios cuadernos en paralelo)
        return P

    def folds(self, r: int = 1) -> list:
        f = self.particiones_df[f"fold_r{r}"].values
        return [(np.flatnonzero(f != k), np.flatnonzero(f == k)) for k in sorted(np.unique(f))]

    def folds_por_r(self, R) -> dict:
        return {r: self.folds(r) for r in R}

    # ------------------------------------------------------------------ verificaciones (Paso 0)
    def verificar(self, D, DT) -> dict:
        from .. import permitidas as PM
        out = {}
        dom_T = set(D.Dominio[D.Dominio != "BOGOTA"].astype(str)); dom_te = set(DT.Dominio.astype(str))
        out["a_dominios_test_en_T"] = dom_te <= dom_T
        if "Lp" in D.columns and "Lp" in DT.columns:
            lp_b = set(np.round(D.Lp[D.Dominio == "BOGOTA"].astype(float), 2)); lp_t = set(np.round(DT.Lp.astype(float), 2))
            out["b_Lp_bogota_ausente_en_test"] = len(lp_b & lp_t) == 0
            out["b_valores_Lp_bogota"] = len(lp_b)
        prohibidas = set(PM.HOG_PROHIBIDAS) | set(PM.OBJETIVO_DERIVADAS) | set(PM.CONSTANTES_EXTERNAS)
        pr = self.conj.get("prohibidas", {})
        for v in (pr.values() if isinstance(pr, dict) else []):
            if isinstance(v, list):
                prohibidas |= {x for x in v if isinstance(x, str)}
            elif isinstance(v, dict):
                prohibidas |= set(v)
        gen = set(self.conj.get("generadas", []) or []) | (set(self.conj.get("S3", [])) - set(self.conj.get("S2", []))
                                                            if self.fuente != "v7" else set())
        prohibidas |= gen
        patrones = list(PM.PATRONES_PROHIBIDOS)
        malas = [c for c in self.todas if c in prohibidas or any(re.search(p, c.lower()) for p in patrones)]
        malas = [c for c in malas if c != "Pobre"]
        out["c_sin_prohibidas"] = not malas
        out["c_prohibidas_encontradas"] = malas
        out["d_mismas_columnas_train_test"] = all(c in DT.columns for c in self.todas)
        if self.fuente == "v7":
            fuera = [c for c in D.columns if c not in DT.columns and c not in ("Pobre", "fold")]
            out["d_columnas_solo_train"] = fuera
            out["d_mismas_columnas_train_test"] = out["d_mismas_columnas_train_test"] and not fuera
        out["ok"] = bool(out["a_dominios_test_en_T"] and out.get("b_Lp_bogota_ausente_en_test", True)
                         and out["c_sin_prohibidas"] and out["d_mismas_columnas_train_test"])
        return out

    # ------------------------------------------------------------------ codificación con la etiqueta en el pliegue
    def _te(self, train_idx) -> dict:
        tr = np.sort(np.asarray(train_idx, int))
        clave = hashlib.md5(tr.tobytes()).hexdigest()
        if clave not in self._te_cache:
            T = self._cod.codificar(tr)
            Tt = self._cod.codificar_test(tr)
            if len(self._te_cache) > 24:
                self._te_cache.pop(next(iter(self._te_cache)))
            self._te_cache[clave] = ({c: T[c].values for c in self.te_cols}, {c: Tt[c].values for c in self.te_cols})
        return self._te_cache[clave]

    def matriz(self, cols, filas, train_idx=None) -> pd.DataFrame:
        """X[filas, cols] con las columnas que dependen de la etiqueta recalculadas con `train_idx` (las filas que
        están en `train_idx` se codifican con una partición interna: ninguna ve su propia etiqueta)."""
        cols = list(cols); filas = np.asarray(filas)
        X = self.X.iloc[filas][cols].copy()
        te = [c for c in cols if c in self.te_cols]
        if te and train_idx is not None:
            vals, _ = self._te(train_idx)
            for c in te:
                X[c] = vals[c][filas].astype("float32")
        return X

    def matriz_test(self, cols, train_idx=None) -> pd.DataFrame:
        cols = list(cols)
        X = self.XT[cols].copy()
        te = [c for c in cols if c in self.te_cols]
        if te:
            _, vt = self._te(np.arange(len(self.y)) if train_idx is None else train_idx)
            for c in te:
                X[c] = vt[c].astype("float32")
        return X

    # ------------------------------------------------------------------ utilidades
    def bloque(self, c) -> str:
        return self.bloques_mapa.get(c, "B")

    def nombre_bloque(self, b) -> str:
        return self.nombres_bloques.get(b, b)          # en v7 los bloques teóricos ya traen su nombre

    def monotonas(self, cols) -> dict:
        return {c: s for c, s in MONOTONAS.items() if c in cols and c not in self.categoricas}

    def prevalencia(self) -> dict:
        from .metricas import prevalencia_esperada_test
        return prevalencia_esperada_test(self.y, self.dominio, self.mT, self.dominio_test)

    def resumen_bogota(self) -> dict:
        return {"hogares_train": int(len(self.y)), "hogares_T": int(self.mT.sum()), "hogares_bogota": int(self.bog.sum()),
                "pi_B": float(self.bog.mean()), "prevalencia_T": float(self.y[self.mT].mean()),
                "prevalencia_bogota": float(self.y[self.bog].mean()) if self.bog.any() else float("nan"),
                "prevalencia_train": float(self.y.mean())}
