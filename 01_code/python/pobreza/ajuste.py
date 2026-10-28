r"""Ajuste a fondo de un modelo, modelo por modelo, con el protocolo común del proyecto.

Lo usan los cuadernos 10 (LightGBM) y 11 (XGBoost), y está pensado para que los siguientes
(Logit, EN, RF, CART, NB, LPM) lo reutilicen: basta con escribir un *motor* nuevo con la
misma interfaz que `MotorLGB` / `MotorXGB` (ajustar, predecir, shap, espacio, nombre_archivo).

Protocolo (no cambia entre modelos):

* Los 5 pliegues de `pliegues.csv` (columna `fold` de `modelo_train`).
* Métrica principal: **F1 anidado sin Bogotá**. Para cada pliegue el umbral se elige con
  las predicciones fuera de pliegue de los otros cuatro, usando sólo hogares fuera de
  Bogotá (el test no tiene Bogotá), y se aplica al pliegue retenido (sin Bogotá).
* Error estándar entre pliegues: EE = sd(F1_k)/√K, con F1_k el F1 del pliegue k.
* Parada temprana con un 10 % **interno** del entrenamiento de cada pliegue (estratificado):
  el pliegue de validación nunca decide el número de árboles.
* Umbral de envío: F1-óptimo OOF sin Bogotá.

Selección de variables sin fuga: todo lo que aprende de los datos (ranking SHAP para el
top-k) se calcula dentro del entrenamiento de cada pliegue externo.
"""
from __future__ import annotations

import json
import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .cv import pliegues as hacer_pliegues
from .modelos_ml import _split_es

# ------------------------------------------------------------------------------ variables
CAT_TODAS = ["Dominio", "Depto", "jefe_oficio", "jefe_posicion", "jefe_nivel_edu", "jefe_regimen_salud",
             "jefe_actividad", "jefe_tam_empresa", "tenencia"]
CAT_ALTA = ["Depto", "jefe_oficio"]          # categóricas de alta cardinalidad (24 y 83 niveles)

# Encuesta corta: ≈ 15 preguntas baratas y verificables (sin variables de modelos).
ENCUESTA_CORTA = ["Nper", "n_menor18", "n_mayor65",                      # tamaño y composición
                  "jefe_mujer", "jefe_edad", "jefe_anios_educ", "educ_max_adultos",  # jefe y educación
                  "n_ocupado", "n_cotiza_pension",                        # ocupados y formalidad
                  "cuartos", "tenencia", "log_arriendo",                  # vivienda y tenencia
                  "n_salud_subsidiado",                                   # régimen de salud
                  "Dominio", "cabecera"]                                  # territorio

_BLOQUE_CAT = {"Dominio": "F", "Depto": "F", "jefe_oficio": "C", "jefe_posicion": "C", "jefe_actividad": "C",
               "jefe_tam_empresa": "C", "jefe_nivel_edu": "B", "jefe_regimen_salud": "D", "tenencia": "E"}
NOMBRE_BLOQUE = {"A": "A. Demografía y escala", "B": "B. Capital humano", "C": "C. Trabajo y formalidad",
                 "D": "D. Protección social e ingresos no laborales", "E": "E. Vivienda",
                 "F": "F. Territorio y diseño", "G": "G. Variable generada (1.ª etapa)",
                 "H": "H. Variables de la búsqueda automática (09)"}


def bloque_variable(c: str, extra: set | None = None, gen: set | None = None) -> str:
    """Letra del bloque teórico (la misma regla del cuaderno 02; H = variables del 09;
    G = cualquier variable declarada como generada en `gen`)."""
    if gen and c in gen:
        return "G"
    if extra and c in extra:
        return "H"
    if c in _BLOQUE_CAT:
        return _BLOQUE_CAT[c]
    if c.startswith(("lz_", "ing_", "log_ing_esp", "brecha_esp", "n_perceptores_esp", "share_principal")):
        return "G"
    if any(k in c for k in ["mincer", "asalariados_formales", "horas_formales", "ocupados_superior", "ocupados_x",
                            "dependientes_por_ocupado", "educ_jefe_x"]):
        return "C"
    if any(k in c for k in ["arriendo", "cuarto", "dormit", "hacinam", "ten_", "amortiz"]):
        return "E"
    if any(k in c for k in ["educ", "anios"]):
        return "B"
    if any(k in c for k in ["salud", "subsid", "recibe", "pension", "transfer", "capital", "ayuda", "remesa"]):
        return "D"
    if any(k in c for k in ["ocup", "formal", "cotiza", "pos_", "asalar", "independ", "horas", "benef", "prima",
                            "empresa", "solo", "subempleo", "segundo", "antig", "sin_rem", "busca", "pago_esp",
                            "extra", "desocup", "inactiv", "cesante", "pet"]):
        return "C"
    if any(k in c for k in ["Lp", "linea", "cabecera", "fex"]):
        return "F"
    return "A"


def preparar(df: pd.DataFrame, columnas, ref: pd.DataFrame) -> pd.DataFrame:
    """Numéricas en float32 y categóricas como `category` con los niveles de train (`ref`)."""
    cols = {}
    for c in columnas:
        if c in CAT_TODAS:
            cats = pd.Index(sorted(ref[c].astype(str).unique()))
            cols[c] = pd.Categorical(df[c].astype(str), categories=cats)
        else:
            cols[c] = df[c].astype("float32").values
    return pd.DataFrame(cols, index=df.index)


@dataclass
class Conjunto:
    """Conjunto de variables. Si `por_pliegue` existe, el pliegue k usa `por_pliegue[k]` (p. ej. un
    top-k cuyo ranking se calculó dentro del entrenamiento de ese pliegue); `final` es la lista que
    usa el modelo reentrenado con todo train."""
    nombre: str
    final: list
    por_pliegue: list | None = None
    grupo: str = ""
    con_gen: bool = True
    # receta de un top-k (para rehacer el ranking dentro de otra partición): pool, k y columnas quitadas
    receta: dict | None = None

    @property
    def n_vars(self) -> int:
        return len(self.final)

    def cols(self, k: int | None = None) -> list:
        return list(self.final if (k is None or self.por_pliegue is None) else self.por_pliegue[k])

    def sin(self, quitar: set, nombre: str | None = None) -> "Conjunto":
        pp = None if self.por_pliegue is None else [[c for c in l if c not in quitar] for l in self.por_pliegue]
        rc = None if self.receta is None else {**self.receta, "quitar": set(self.receta.get("quitar", set())) | set(quitar)}
        return Conjunto(nombre or f"{self.nombre} − {sorted(quitar)}", [c for c in self.final if c not in quitar],
                        pp, self.grupo, self.con_gen, rc)


class Datos:
    """Carga las tablas de modelado, los pliegues y la máscara sin Bogotá."""

    # claves posibles del conjunto ampliado y de la encuesta corta en `conjuntos_variables_v5.json`
    CLAVES_AMPLIADO = ("S3_ampliado", "ampliado", "S3_amp", "S4", "S3plus", "S3_mas", "S5", "S3_v5_ampliado")
    CLAVES_CORTA = ("encuesta_corta", "encuesta_corta_sugerida", "corta", "S0")
    CLAVES_GEN = ("generadas", "GEN", "generadas_todas", "variables_generadas")

    def __init__(self, R, sufijo: str = "", extra: list | None = None, extra_gen: list | None = None,
                 conj_json: str | None = None, especificaciones: list | None = None,
                 submuestra: float | None = None, semilla_sub: int = 1):
        """`sufijo` elige las tablas (`modelo_train{sufijo}`); `conj_json` el archivo de conjuntos
        (por defecto `conjuntos_variables.json`). `especificaciones` (las del 09) permite construir las
        variables de la búsqueda si la tabla no las trae. `submuestra` (fracción) sólo para pruebas:
        submuestra estratificada por pliegue × Pobre de train (test completo)."""
        from . import io as pio
        self.R = R
        self.D = pio.leer_tabla(R, f"modelo_train{sufijo}")
        self.DT = pio.leer_tabla(R, f"modelo_test{sufijo}")
        if submuestra:
            g = self.D.fold.astype(str) + "_" + self.D.Pobre.astype(int).astype(str)
            keep = self.D.groupby(g, group_keys=False).sample(frac=float(submuestra), random_state=semilla_sub).index
            self.D = self.D.loc[np.sort(keep)].reset_index(drop=True)
        self.conj = json.load(open(Path(R.processed) / (conj_json or "conjuntos_variables.json"), encoding="utf-8"))
        self.y = self.D.Pobre.values.astype(int)
        self.folds = [(np.flatnonzero(self.D.fold.values != k), np.flatnonzero(self.D.fold.values == k))
                      for k in sorted(self.D.fold.unique())]
        self.sin_bog = (self.D.Dominio != "BOGOTA").values
        disp = lambda l: [c for c in l if c in self.D.columns and c in self.DT.columns]
        gen_decl = next((self.conj[k] for k in self.CLAVES_GEN if isinstance(self.conj.get(k), list)), None)
        self.GEN = disp([c for c in self.conj["S3"] if c not in self.conj["S2"]] if gen_decl is None else gen_decl)
        # variables de la búsqueda (09): si la tabla no las trae, se construyen con sus especificaciones
        if especificaciones:
            from .busqueda import materializar
            faltan = [s for s in especificaciones if s["nombre"] in (extra or []) and
                      (s["nombre"] not in self.D.columns or s["nombre"] not in self.DT.columns)]
            faltan = [s for s in faltan if all(v in self.D.columns for v in (s.get("a"), s.get("b"), s.get("grupo")) if v)]
            if faltan:
                self.D = pd.concat([self.D, materializar(faltan, self.D, self.D)], axis=1)
                self.DT = pd.concat([self.DT, materializar(faltan, self.D, self.DT)], axis=1)
        self.extra = [c for c in (extra or []) if c in self.D.columns and c in self.DT.columns]
        # variables de la búsqueda (09) construidas con alguna variable generada de 1.ª etapa
        self.extra_gen = [c for c in (extra_gen or []) if c in self.extra]
        self.GEN_TODAS = list(dict.fromkeys(self.GEN + self.extra_gen))
        self.AMPLIADO = disp(next((self.conj[k] for k in self.CLAVES_AMPLIADO if isinstance(self.conj.get(k), list)), []))
        corta = next((self.conj[k] for k in self.CLAVES_CORTA if isinstance(self.conj.get(k), list)), None)
        self.CORTA = disp(corta) if corta else disp(ENCUESTA_CORTA)
        num_s3 = [c for c in self.conj["S3"] if c not in CAT_TODAS]
        self.columnas = list(dict.fromkeys(num_s3 + [c for c in self.extra if c not in num_s3] +
                                           [c for c in self.AMPLIADO if c not in CAT_TODAS] + [c for c in self.GEN] +
                                           [c for c in self.CORTA + ENCUESTA_CORTA + self.conj["S1"] if c not in CAT_TODAS] + CAT_TODAS))
        self.columnas = [c for c in self.columnas if c in self.D.columns and c in self.DT.columns]
        self.X = preparar(self.D, self.columnas, self.D)
        self.XT = preparar(self.DT, self.columnas, self.D)
        self.ids, self.ids_test = self.D.id.values, self.DT.id.values

    def bloques(self) -> dict:
        ex = set(self.extra)
        return {c: bloque_variable(c, ex, set(self.GEN_TODAS)) for c in self.columnas}

    def candidatos_v5(self) -> dict:
        """Candidatos del encargo v5: encuesta corta, S1, S2/S3, S3 + búsqueda (09) y el conjunto ampliado
        v5, cada uno con y sin variables generadas (las generadas son las declaradas en el JSON de conjuntos
        más las variables del 09 construidas con ellas). Las categóricas se añaden a los conjuntos grandes."""
        G = set(self.GEN_TODAS)
        gen_s3 = [c for c in self.conj["S3"] if c in G and c in self.columnas] or [c for c in self.GEN if c in self.columnas]
        # generadas «clave» para los conjuntos chicos: las del hogar y la distribucional de v5 (8) si existen;
        # si no, las generadas de S3 (v3: 7)
        clave = [c for k in ("generadas_v5_hogar", "generadas_v5_distribucion") for c in self.conj.get(k, []) if c in self.columnas]
        gen_chico = clave or gen_s3
        cats = [c for c in CAT_TODAS if c in self.columnas]
        ok = lambda l: list(dict.fromkeys(c for c in l if c in self.columnas))
        sin = lambda l: [c for c in l if c not in G]
        s1 = ok(self.conj["S1"]); corta = ok(self.CORTA)
        s2 = ok([c for c in self.conj["S2"] if c not in CAT_TODAS and c not in G])
        s3 = ok(s2 + gen_s3)
        out, vistos = {}, set()

        def agrega(nombre, cols, grupo, cg):
            clave_c = frozenset(cols)
            if clave_c in vistos or not cols:
                return
            vistos.add(clave_c)
            out[nombre] = Conjunto(nombre, cols, grupo=grupo, con_gen=cg)

        def par(nombre, cols, grupo, con_cat=False):
            cols = ok(cols + (cats if con_cat else []))
            if set(cols) & G:
                agrega(f"{nombre} (con generadas)", cols, grupo, True)
            agrega(f"{nombre} (sin generadas)", sin(cols), grupo, False)

        par("a. Encuesta corta", corta + gen_chico, "a")
        par("b. S1", s1 + gen_chico, "b")
        par("c. S2 / S3 + categóricas", s3, "c", con_cat=True)
        if clave:
            agrega(f"c. S2 + {len(clave)} generadas clave + categóricas (con generadas)", ok(s2 + clave + cats), "c", True)
        if self.extra:
            par("d. S3 + búsqueda 09 + categóricas", s3 + self.extra, "d", con_cat=True)
        if self.AMPLIADO:
            par("e. Ampliado v5 + categóricas", self.AMPLIADO + [c for c in self.GEN if c not in self.AMPLIADO], "e", con_cat=True)
        return out

    def pools_v5(self) -> dict:
        """Pools para el top-k: todas las variables disponibles (con generadas) y las mismas sin generadas."""
        G = set(self.GEN_TODAS)
        todas = list(self.columnas)
        return {"con generadas": Conjunto("pool con generadas", todas, grupo="pool", con_gen=True),
                "sin generadas": Conjunto("pool sin generadas", [c for c in todas if c not in G], grupo="pool", con_gen=False)}

    def candidatos(self) -> dict:
        """Conjuntos candidatos (a)–(e) y la versión con/sin generadas de cada uno."""
        G = self.GEN
        s1 = [c for c in self.conj["S1"] if c in self.columnas]
        s2 = [c for c in self.conj["S2"] if c not in CAT_TODAS]
        bajas = [c for c in CAT_TODAS if c not in CAT_ALTA]
        out = {
            "a. Encuesta corta": Conjunto("a. Encuesta corta", list(ENCUESTA_CORTA), grupo="a", con_gen=False),
            "a. Encuesta corta + generadas": Conjunto("a. Encuesta corta + generadas", ENCUESTA_CORTA + G, grupo="a"),
            "b. S1": Conjunto("b. S1", s1, grupo="b", con_gen=False),
            "b. S1 + generadas": Conjunto("b. S1 + generadas", s1 + G, grupo="b"),
            "c. S2 (+ categóricas)": Conjunto("c. S2 (+ categóricas)", s2 + CAT_TODAS, grupo="c", con_gen=False),
            "d. S3 (S2 + generadas + categóricas)": Conjunto("d. S3 (S2 + generadas + categóricas)", s2 + G + CAT_TODAS, grupo="d"),
            "e. S3 sin categóricas de alta cardinalidad": Conjunto("e. S3 sin categóricas de alta cardinalidad", s2 + G + bajas, grupo="e"),
            "e. S2 sin categóricas de alta cardinalidad": Conjunto("e. S2 sin categóricas de alta cardinalidad", s2 + bajas, grupo="e", con_gen=False),
        }
        if self.extra:
            ex = [c for c in self.extra if c not in s2 + G]
            ex_ng = [c for c in ex if c not in self.extra_gen]
            out["h. S3 + variables de la búsqueda (09)"] = Conjunto("h. S3 + variables de la búsqueda (09)", s2 + G + ex + CAT_TODAS, grupo="h")
            out["h. S2 + variables de la búsqueda (09), sin generadas"] = Conjunto(
                "h. S2 + variables de la búsqueda (09), sin generadas", s2 + ex_ng + CAT_TODAS, grupo="h", con_gen=False)
        return out


# ------------------------------------------------------------------------------ motores
class MotorLGB:
    """LightGBM (Ke et al., 2017): árboles por hoja (leaf-wise), histogramas y categóricas nativas."""
    nombre, corto = "LightGBM", "LGBM"

    def __init__(self, hilos: int = 10):
        import lightgbm as lgb
        self.lgb, self.hilos = lgb, hilos
        self.base = dict(objective="binary", metric="binary_logloss", learning_rate=0.1, num_leaves=15,
                         min_child_samples=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                         lambda_l1=0.0, lambda_l2=5.0, min_gain_to_split=0.0, max_bin=127, max_depth=-1,
                         scale_pos_weight=1.0, cat_smooth=20.0, cat_l2=10.0, max_cat_to_onehot=8, verbose=-1)

    def _p(self, params, semilla):
        p = {**self.base, **params, "num_threads": self.hilos, "seed": semilla, "bagging_seed": semilla,
             "feature_fraction_seed": semilla, "verbose": -1}
        return p

    def ajustar(self, Xtr, ytr, Xes=None, yes=None, params=None, semilla=1, n_arboles=None, paciencia=100, w=None):
        lgb = self.lgb
        p = self._p(params or {}, semilla)
        dtr = lgb.Dataset(Xtr, ytr, weight=w, free_raw_data=True)
        if Xes is not None:
            des = lgb.Dataset(Xes, yes, reference=dtr)
            m = lgb.train(p, dtr, 10000, valid_sets=[des], callbacks=[lgb.early_stopping(paciencia, verbose=False)])
            return m, int(m.best_iteration)
        m = lgb.train(p, dtr, int(n_arboles))
        return m, int(n_arboles)

    def predecir(self, m, X, n_iter):
        return m.predict(X, num_iteration=n_iter)

    def shap(self, m, X, n_iter=None):
        return m.predict(X, num_iteration=n_iter, pred_contrib=True)[:, :-1]

    perdida = "LogLoss_sinBOG"       # objetivo de Optuna (regla de puntuación propia, sin Bogotá)

    def espacio(self, trial) -> dict:
        """Espacio de búsqueda (documentado en el cuaderno 10). El desbalance (scale_pos_weight /
        is_unbalance) NO entra: con log-loss como objetivo siempre perdería (descalibra); se compara aparte."""
        return dict(
            num_leaves=trial.suggest_int("num_leaves", 8, 256, log=True),
            max_depth=trial.suggest_categorical("max_depth", [-1, 6, 8, 10, 12]),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 800, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.2, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
            bagging_freq=trial.suggest_int("bagging_freq", 0, 5),
            lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 100.0, log=True),
            min_gain_to_split=trial.suggest_float("min_gain_to_split", 0.0, 1.0),
            max_bin=trial.suggest_categorical("max_bin", [63, 127, 255]),
            cat_smooth=trial.suggest_float("cat_smooth", 1.0, 100.0, log=True),
        )

    def base_optuna(self) -> dict:
        b = self.base
        return {k: b[k] for k in ["num_leaves", "max_depth", "min_child_samples", "feature_fraction", "bagging_fraction",
                                  "bagging_freq", "min_gain_to_split", "max_bin", "cat_smooth"]} | \
               {"learning_rate": 0.05, "lambda_l1": 1e-3, "lambda_l2": 5.0}

    def desbalance(self, params, razon) -> dict:
        p0 = {k: v for k, v in params.items() if k not in ("scale_pos_weight", "is_unbalance")}
        return {"ninguna (sólo umbral)": {**p0, "scale_pos_weight": 1.0},
                "is_unbalance = True (≡ peso n0/n1)": {**p0, "is_unbalance": True, "scale_pos_weight": 1.0},
                f"scale_pos_weight = √(n0/n1) = {math.sqrt(razon):.2f}": {**p0, "scale_pos_weight": math.sqrt(razon)}}

    def resumen_nombre(self, params) -> str:
        f = lambda v: f"{v:.3g}".replace(".", "_").replace("-", "m")
        return f"leaves_{int(params['num_leaves'])}_lr_{f(params['learning_rate'])}_mcs_{int(params['min_child_samples'])}"

    def nombre_archivo(self, params, n_vars, sufijo="", n_arboles=None) -> str:
        f = lambda v: f"{v:.3g}".replace(".", "_").replace("-", "m")
        return (f"LGBM_leaves_{int(params['num_leaves'])}_lr_{f(params['learning_rate'])}_mcs_{int(params['min_child_samples'])}"
                f"_ff_{f(params['feature_fraction'])}_l2_{f(params['lambda_l2'])}_n_{n_vars}{sufijo}.csv")

    def describir(self, params) -> str:
        return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()
                         if k not in ("objective", "metric", "verbose"))


class MotorXGB:
    """XGBoost (Chen y Guestrin, 2016): árboles por niveles (depth-wise), `hist`, categóricas nativas."""
    nombre, corto = "XGBoost", "XGB"

    def __init__(self, hilos: int = 10):
        import xgboost as xgb
        self.xgb, self.hilos = xgb, hilos
        self.base = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", learning_rate=0.1,
                         max_depth=6, min_child_weight=20.0, subsample=0.8, colsample_bytree=0.7, colsample_bylevel=1.0,
                         gamma=0.0, reg_alpha=0.0, reg_lambda=5.0, scale_pos_weight=1.0, max_bin=256,
                         max_cat_to_onehot=8)

    def _p(self, params, semilla):
        return {**self.base, **params, "nthread": self.hilos, "seed": semilla, "verbosity": 0}

    def _dm(self, X, y=None, w=None):
        return self.xgb.DMatrix(X, y, weight=w, enable_categorical=True, nthread=self.hilos)

    def ajustar(self, Xtr, ytr, Xes=None, yes=None, params=None, semilla=1, n_arboles=None, paciencia=100, w=None):
        xgb = self.xgb
        p = self._p(params or {}, semilla)
        # QuantileDMatrix: cuantiza una sola vez (mucho más rápido y liviano que DMatrix con `hist`)
        mb = int(p.get("max_bin", 256))
        dtr = xgb.QuantileDMatrix(Xtr, ytr, weight=w, enable_categorical=True, nthread=self.hilos, max_bin=mb)
        if Xes is not None:
            des = xgb.QuantileDMatrix(Xes, yes, ref=dtr, enable_categorical=True, nthread=self.hilos, max_bin=mb)
            m = xgb.train(p, dtr, 10000, evals=[(des, "es")], early_stopping_rounds=paciencia, verbose_eval=False)
            return m, int(m.best_iteration) + 1
        m = xgb.train(p, dtr, int(n_arboles))
        return m, int(n_arboles)

    def predecir(self, m, X, n_iter):
        try:
            return m.inplace_predict(X, iteration_range=(0, int(n_iter)))
        except Exception:
            return m.predict(self._dm(X), iteration_range=(0, int(n_iter)))

    def shap(self, m, X, n_iter=None):
        kw = {} if n_iter is None else {"iteration_range": (0, int(n_iter))}
        return m.predict(self._dm(X), pred_contribs=True, **kw)[:, :-1]

    perdida = "LogLoss_sinBOG"

    def espacio(self, trial) -> dict:
        """Espacio de búsqueda (cuaderno 11); el desbalance (scale_pos_weight) se compara aparte."""
        return dict(
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 300.0, log=True),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            colsample_bylevel=trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            gamma=trial.suggest_float("gamma", 1e-3, 5.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
            max_bin=trial.suggest_categorical("max_bin", [64, 128, 256]),
        )

    def base_optuna(self) -> dict:
        return dict(max_depth=6, min_child_weight=20.0, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7,
                    colsample_bylevel=1.0, gamma=1e-3, reg_alpha=1e-3, reg_lambda=5.0, max_bin=256)

    def desbalance(self, params, razon) -> dict:
        p0 = {k: v for k, v in params.items() if k != "scale_pos_weight"}
        return {"ninguna (sólo umbral)": {**p0, "scale_pos_weight": 1.0},
                f"scale_pos_weight = n0/n1 = {razon:.2f}": {**p0, "scale_pos_weight": razon},
                f"scale_pos_weight = √(n0/n1) = {math.sqrt(razon):.2f}": {**p0, "scale_pos_weight": math.sqrt(razon)}}

    def nombre_archivo(self, params, n_vars, sufijo="", n_arboles=None) -> str:
        f = lambda v: f"{v:.3g}".replace(".", "_").replace("-", "m")
        return (f"XGB_depth_{int(params['max_depth'])}_eta_{f(params['learning_rate'])}_mcw_{f(params['min_child_weight'])}"
                f"_sub_{f(params['subsample'])}_col_{f(params['colsample_bytree'])}_n_{n_vars}{sufijo}.csv")

    def describir(self, params) -> str:
        return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()
                         if k not in ("objective", "eval_metric", "tree_method"))


class MotorRF:
    """Random forest (Breiman, 2001) de scikit-learn: *bagging* de árboles con submuestra de variables
    en cada partición; faltantes nativos (sklearn ≥ 1.4) y categóricas como códigos ordinales.
    No hay parada temprana: el número de árboles es un parámetro (300 en la búsqueda, 500 al final)."""
    nombre, corto = "Random forest", "RF"
    usa_shap = False
    parada_temprana = False
    # objetivo de Optuna: Brier sin Bogotá (regla propia y acotada; el log-loss explota cuando un bosque
    # asigna probabilidad 0 a un pobre, algo común con hojas puras)
    perdida = "Brier_sinBOG"

    def __init__(self, hilos: int = 7, n_busqueda: int = 150):
        from sklearn.ensemble import RandomForestClassifier
        self.RF, self.hilos, self.n_busqueda = RandomForestClassifier, hilos, n_busqueda
        self.base = dict(n_estimators=n_busqueda, max_features=0.25, min_samples_leaf=20, max_samples=0.4, max_depth=None,
                         class_weight=None, criterion="gini")

    @staticmethod
    def _num(X):
        X = X.copy()
        for c in X.columns:
            if isinstance(X[c].dtype, pd.CategoricalDtype):
                X[c] = X[c].cat.codes.astype("float32").replace(-1, np.nan)
        return X

    def ajustar(self, Xtr, ytr, Xes=None, yes=None, params=None, semilla=1, n_arboles=None, paciencia=None, w=None):
        p = {**self.base, **(params or {})}
        m = self.RF(**p, n_jobs=self.hilos, random_state=semilla).fit(self._num(Xtr), ytr, sample_weight=w)
        return m, int(p["n_estimators"])

    def n_final(self, iters):
        return int(iters[0])

    def predecir(self, m, X, n_iter=None):
        return m.predict_proba(self._num(X))[:, 1]

    def importancia_ranking(self, m, X, n_iter=None):
        return m.feature_importances_

    def espacio(self, trial) -> dict:
        """Espacio (cuaderno 12); class_weight se compara aparte (estrategias de desbalance)."""
        return dict(
            n_estimators=self.n_busqueda,
            max_features=trial.suggest_float("max_features", 0.05, 0.6, log=True),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 2, 200, log=True),
            max_samples=trial.suggest_float("max_samples", 0.15, 0.6),
            max_depth=trial.suggest_categorical("max_depth", [None, 14, 20, 30]),
            criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
            class_weight=None,
        )

    def base_optuna(self) -> dict:
        return {k: v for k, v in self.base.items() if k not in ("n_estimators", "class_weight")}

    def desbalance(self, params, razon) -> dict:
        p0 = {k: v for k, v in params.items() if k != "class_weight"}
        return {"ninguna (sólo umbral)": {**p0, "class_weight": None},
                "class_weight = balanced": {**p0, "class_weight": "balanced"},
                "class_weight = balanced_subsample": {**p0, "class_weight": "balanced_subsample"}}

    def nombre_archivo(self, params, n_vars, sufijo="", n_arboles=None) -> str:
        f = lambda v: f"{v:.3g}".replace(".", "_")
        d = params.get("max_depth") or "None"
        cw = {None: "none", "balanced": "bal", "balanced_subsample": "balsub"}[params.get("class_weight")]
        return (f"RF_mtry_{f(params['max_features'])}_minleaf_{int(params['min_samples_leaf'])}_ms_{f(params['max_samples'])}"
                f"_depth_{d}_cw_{cw}_trees_{int(n_arboles or params['n_estimators'])}_n_{n_vars}{sufijo}.csv")

    def describir(self, params) -> str:
        return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())


# ------------------------------------------------------------------------------ métricas
def f1_anidado_sb(y, s, folds, m):
    """F1 anidado sin Bogotá: umbral de cada pliegue elegido en los otros cuatro (sin Bogotá).
    Devuelve (F1 global sin Bogotá, F1 por pliegue, umbrales, predicción 0/1 anidada)."""
    y = np.asarray(y).astype(int); s = np.asarray(s, float); m = np.asarray(m, bool)
    yhat = np.zeros_like(y); f1k, umb = [], []
    for tr, va in folds:
        tr_m, va_m = tr[m[tr]], va[m[va]]
        t, _ = M.umbral_optimo_f1(y[tr_m], s[tr_m])
        yhat[va_m] = (s[va_m] >= t).astype(int)
        f1k.append(M.f1(y[va_m], yhat[va_m])); umb.append(t)
    return M.f1(y[m], yhat[m]), np.array(f1k), np.array(umb), yhat


def resumen(y, s, folds, m) -> dict:
    f1, f1k, umb, yhat = f1_anidado_sb(y, s, folds, m)
    tp, fp, fn, _ = M.confusion(y[m], yhat[m])
    return {"F1_anidado_sinBOG": f1, "EE": float(f1k.std(ddof=1) / np.sqrt(len(f1k))),
            "AUC": M.auc_roc(y, s), "AUC_sinBOG": M.auc_roc(y[m], s[m]), "LogLoss": M.log_loss(y, s),
            "LogLoss_sinBOG": M.log_loss(y[m], s[m]), "Brier_sinBOG": M.brier(y[m], s[m]),
            "precision_sinBOG": tp / max(tp + fp, 1), "recall_sinBOG": tp / max(tp + fn, 1),
            **{f"F1_pliegue_{k}": float(v) for k, v in enumerate(f1k)}}


def bootstrap_f1(y, yhat, B=1000, semilla=1):
    """IC95 percentil del F1 con las predicciones (anidadas) fijas; remuestreo de hogares."""
    rng = np.random.default_rng(semilla)
    y = np.asarray(y).astype(int); yhat = np.asarray(yhat).astype(int)
    tp_i = (y == 1) & (yhat == 1); fp_i = (y == 0) & (yhat == 1); fn_i = (y == 1) & (yhat == 0)
    n = len(y); vals = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(0, n, n), minlength=n)
        tp, fp, fn = w @ tp_i, w @ fp_i, w @ fn_i
        vals[b] = 2 * tp / (2 * tp + fp + fn)
    return float(np.quantile(vals, .025)), float(np.quantile(vals, .975))


# ------------------------------------------------------------------------------ validación cruzada
def cv(motor, X, y, folds, params, conjunto: Conjunto, semillas=(1,), X_test=None, al_pliegue=None,
       paciencia=100, verbose=False, bog=None, peso_bogota=1.0, transformar=None, es_sin_bog=False):
    """OOF con parada temprana interna (10 % del entrenamiento de cada pliegue, estratificado) y,
    si se da `X_test`, modelo final con todo train y ⌈1,1·media(árboles)⌉ árboles.
    `al_pliegue(k, va, pred)` se llama tras cada pliegue (poda de Optuna).
    `peso_bogota`: peso de los hogares de Bogotá (máscara `bog`) en el ENTRENAMIENTO: 1 = todos, 0,5 = peso
    reducido, 0 = se excluyen. La evaluación siempre se hace sin Bogotá.
    `es_sin_bog`: el 10 % interno de parada temprana se toma sólo de hogares fuera de Bogotá (la población
    del test), de modo que el número de árboles se elige para el dominio de predicción."""
    t0 = time.time()
    if bog is not None:
        bog = np.asarray(bog, bool)
    if bog is not None and peso_bogota == 0:
        excl = np.asarray(bog, bool)
        folds = [(tr[~excl[tr]], va) for tr, va in folds]
        Xfull_idx = np.flatnonzero(~excl)
    else:
        Xfull_idx = np.arange(len(y))
    wv = None if (bog is None or peso_bogota in (0, 1)) else np.where(np.asarray(bog, bool), float(peso_bogota), 1.0)
    oof = np.zeros(len(y)); iters = []
    X0 = X
    for k, (tr, va) in enumerate(folds):
        X = X0 if transformar is None else transformar(k, tr, X0)   # p. ej. estadísticas de grupo del pliegue
        cols = conjunto.cols(k); pr = np.zeros(len(va))
        for s in semillas:
            fit_m, es_m = _split_es(y[tr], 1000 * s + k)
            if es_sin_bog and bog is not None:
                es_m = es_m & ~bog[tr]; fit_m = ~es_m
            itr, ies = tr[fit_m], tr[es_m]
            if not getattr(motor, "parada_temprana", True):      # p. ej. RF: sin parada, usa todo el pliegue
                mod, n = motor.ajustar(X.iloc[tr][cols], y[tr], params=params, semilla=s, w=None if wv is None else wv[tr])
            else:
                mod, n = motor.ajustar(X.iloc[itr][cols], y[itr], X.iloc[ies][cols], y[ies], params, semilla=s,
                                       paciencia=paciencia, w=None if wv is None else wv[itr])
            iters.append(n)
            pr += motor.predecir(mod, X.iloc[va][cols], n) / len(semillas)
        oof[va] = pr
        if verbose:
            print(f"    pliegue {k}: {iters[-1]} árboles ({time.time() - t0:.0f}s)")
        if al_pliegue is not None:
            al_pliegue(k, va, pr)
    X = X0
    te = None
    n_final = motor.n_final(iters) if hasattr(motor, "n_final") else int(np.ceil(1.1 * np.mean(iters)))
    if X_test is not None:
        cols = conjunto.cols(None); te = np.zeros(len(X_test))
        for s in semillas:
            mod, _ = motor.ajustar(X.iloc[Xfull_idx][cols], y[Xfull_idx], params=params, semilla=s, n_arboles=n_final,
                                   w=None if wv is None else wv[Xfull_idx])
            te += motor.predecir(mod, X_test[cols], n_final) / len(semillas)
    return {"oof": oof, "test": te, "iteraciones": iters, "n_final": n_final, "tiempo_s": time.time() - t0}


# ------------------------------------------------------------------------------ caché
def en_cache(ruta: Path, fn, reusar=True):
    """Calcula `fn()` o lo lee de `ruta` (pickle). Permite reanudar un cuaderno interrumpido."""
    ruta = Path(ruta)
    if reusar and ruta.exists():
        with open(ruta, "rb") as f:
            return pickle.load(f)
    out = fn()
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(out, f)
    return out


# ------------------------------------------------------------------------------ selección de variables
def evaluar_conjunto(motor, dat: Datos, conjunto: Conjunto, params, semillas=(1,), kw: dict | None = None):
    r = cv(motor, dat.X, dat.y, dat.folds, params, conjunto, semillas=semillas, **(kw or {}))
    conjunto.con_gen = any(c in dat.GEN_TODAS for c in conjunto.final)
    fila = {"conjunto": conjunto.nombre, "grupo": conjunto.grupo, "con_generadas": conjunto.con_gen,
            "n_vars": conjunto.n_vars, **resumen(dat.y, r["oof"], dat.folds, dat.sin_bog),
            "arboles_medios": float(np.mean(r["iteraciones"])), "tiempo_s": r["tiempo_s"]}
    return fila, r["oof"]


def ranking_shap(motor, X, y, idx, cols, params, n_shap=20000, semilla=1):
    """Ranking por E|SHAP| de un modelo ajustado sólo con las filas `idx` (parada temprana interna);
    los valores SHAP se calculan sobre una submuestra de esas mismas filas."""
    fit_m, es_m = _split_es(y[idx], 77 + semilla)
    mod, n = motor.ajustar(X.iloc[idx[fit_m]][cols], y[idx[fit_m]], X.iloc[idx[es_m]][cols], y[idx[es_m]], params,
                           semilla=semilla)
    if hasattr(motor, "importancia_ranking"):       # p. ej. RF: importancia por impureza del pliegue
        return pd.Series(motor.importancia_ranking(mod, None, n), index=cols).sort_values(ascending=False)
    rng = np.random.default_rng(semilla)
    sub = rng.choice(idx, min(n_shap, len(idx)), replace=False)
    phi = motor.shap(mod, X.iloc[sub][cols], n)
    return pd.Series(np.abs(phi).mean(0), index=cols).sort_values(ascending=False)


def topk(motor, dat: Datos, pool: Conjunto, params, ks, etiqueta, n_shap=20000, medida="SHAP", kw=None):
    """(f) top-k por importancia SHAP con el ranking calculado DENTRO de cada pliegue de entrenamiento.
    Devuelve las filas de la tabla, los conjuntos y el ranking con todo train (para el modelo final)."""
    rk_pl = [ranking_shap(motor, dat.X, dat.y, tr, pool.cols(), params, n_shap) for tr, _ in dat.folds]
    rk_full = ranking_shap(motor, dat.X, dat.y, np.arange(len(dat.y)), pool.cols(), params, n_shap)
    filas, conjs = [], {}
    for k in ks:
        if k >= len(pool.final):
            continue
        c = Conjunto(f"f. top-{k} {medida} ({etiqueta})", list(rk_full.index[:k]), [list(r.index[:k]) for r in rk_pl],
                     grupo="f", con_gen=pool.con_gen, receta={"pool": pool.cols(), "k": k, "quitar": set()})
        fila, _ = evaluar_conjunto(motor, dat, c, params, kw=kw)
        fila["k"] = k; filas.append(fila); conjs[c.nombre] = c
        print(f"  {c.nombre:55s} F1={fila['F1_anidado_sinBOG']:.4f} ± {fila['EE']:.4f}", flush=True)
    return filas, conjs, rk_full


def conjunto_en_particion(motor, dat: Datos, conjunto: Conjunto, folds, params, n_shap=20000) -> Conjunto:
    """Rehace un top-k dentro de OTRA partición (ranking en el entrenamiento de cada pliegue nuevo), para que
    la segunda partición no herede rankings calculados con los pliegues originales."""
    if conjunto.receta is None:
        return Conjunto(conjunto.nombre, conjunto.final, None, conjunto.grupo, conjunto.con_gen)
    rc = conjunto.receta
    pp = []
    for tr, _ in folds:
        rk = ranking_shap(motor, dat.X, dat.y, tr, rc["pool"], params, n_shap)
        pp.append([c for c in list(rk.index[:rc["k"]]) if c not in rc.get("quitar", set())])
    return Conjunto(conjunto.nombre, conjunto.final, pp, conjunto.grupo, conjunto.con_gen, rc)


def eliminacion_bloques(motor, dat: Datos, inicio: Conjunto, fila_ini: dict, params, kw=None):
    """(g) Una pasada de eliminación hacia atrás por bloques teóricos con la regla de 1 EE: se evalúa
    quitar cada bloque; son admisibles los que dejan F1 ≥ F1_ini − EE_ini. Se prueba quitar todos los
    admisibles a la vez; si eso viola la regla, se quita sólo el admisible de mayor F1 (empates: menos
    variables)."""
    bl = dat.bloques()
    ref, ee = fila_ini["F1_anidado_sinBOG"], fila_ini["EE"]
    presentes = sorted({bl[c] for c in inicio.final})
    filas, adm = [], []
    for b in presentes:
        quitar = {c for c in dat.columnas if bl.get(c) == b}
        c = inicio.sin(quitar, f"g. {inicio.nombre} − bloque {b}")
        if c.n_vars == 0:
            continue
        fila, _ = evaluar_conjunto(motor, dat, c, params, kw=kw)
        fila.update(bloque_quitado=NOMBRE_BLOQUE[b], admisible=fila["F1_anidado_sinBOG"] >= ref - ee)
        filas.append(fila)
        print(f"  sin {NOMBRE_BLOQUE[b]:48s} n={c.n_vars:3d} F1={fila['F1_anidado_sinBOG']:.4f} "
              f"({'admisible' if fila['admisible'] else 'empeora > 1 EE'})", flush=True)
        if fila["admisible"]:
            adm.append((fila["F1_anidado_sinBOG"], -c.n_vars, b, quitar, fila))
    if not adm:
        return inicio, fila_ini, filas
    adm.sort(key=lambda t: (t[1], t[0]), reverse=True)   # regla de 1 EE: entre admisibles, el de MENOS variables
    if len(adm) > 1:
        todos = set().union(*[a[3] for a in adm])
        c = inicio.sin(todos, f"g. {inicio.nombre} − bloques {''.join(sorted(a[2] for a in adm))}")
        if c.n_vars > 0:
            fila, _ = evaluar_conjunto(motor, dat, c, params, kw=kw)
            fila.update(bloque_quitado="+".join(sorted(a[2] for a in adm)), admisible=fila["F1_anidado_sinBOG"] >= ref - ee)
            filas.append(fila)
            print(f"  sin bloques {'+'.join(sorted(a[2] for a in adm))}: n={c.n_vars} F1={fila['F1_anidado_sinBOG']:.4f}")
            if fila["admisible"]:
                return c, fila, filas
    _, _, b, quitar, fila = adm[0]
    return inicio.sin(quitar, f"g. {inicio.nombre} − bloque {b}"), fila, filas


def seleccion(motor, dat: Datos, params, ks=(10, 20, 40, 80), n_shap=20000,
              pools_topk=("S3 + búsqueda 09", "con generadas", "sin generadas"), medida="SHAP"):
    """Selección completa: candidatos (a)–(e) [+ (h) si hay variables del 09], top-k (f) con y sin
    generadas, y eliminación por bloques (g) desde el mejor. Devuelve tabla, conjuntos y elegido."""
    filas, conjs = [], {}
    for nombre, c in dat.candidatos().items():
        fila, _ = evaluar_conjunto(motor, dat, c, params); filas.append(fila); conjs[nombre] = c
        print(f"  {nombre:55s} n={c.n_vars:3d} F1={fila['F1_anidado_sinBOG']:.4f} ± {fila['EE']:.4f}  AUC={fila['AUC']:.4f}  ({fila['tiempo_s']:.0f}s)")
    cand = dat.candidatos()
    pools = [("con generadas", cand["d. S3 (S2 + generadas + categóricas)"]),
             ("sin generadas", cand["c. S2 (+ categóricas)"])]
    if dat.extra:
        pools.insert(0, ("S3 + búsqueda 09", cand["h. S3 + variables de la búsqueda (09)"]))
    rankings = {}
    pools = [(et, p) for et, p in pools if et in pools_topk]
    for et, pool in pools:
        f, cs, rk = topk(motor, dat, pool, params, ks, et, n_shap, medida)
        filas += f; conjs.update(cs); rankings[et] = rk
    tabla = pd.DataFrame(filas)
    mejor = tabla.sort_values("F1_anidado_sinBOG", ascending=False).iloc[0]
    print(f"Mejor conjunto antes de la eliminación: {mejor.conjunto} (F1={mejor.F1_anidado_sinBOG:.4f} ± {mejor.EE:.4f})")
    elegido, fila_el, f_el = eliminacion_bloques(motor, dat, conjs[mejor.conjunto], mejor.to_dict(), params)
    for f in f_el:
        f["grupo"] = "g"
    tabla = pd.concat([tabla, pd.DataFrame(f_el)], ignore_index=True)
    for f in f_el:
        if f["conjunto"] not in conjs:
            pass
    conjs[elegido.nombre] = elegido
    tabla["elegido"] = tabla.conjunto == elegido.nombre
    return {"tabla": tabla, "conjuntos": conjs, "elegido": elegido, "mejor_inicial": mejor.conjunto, "rankings": rankings}


def mejor_parsimonioso(sel: dict, max_vars=20) -> Conjunto:
    t = sel["tabla"]; t = t[(t.n_vars <= max_vars) & t.conjunto.isin(list(sel["conjuntos"]))]
    return sel["conjuntos"][t.sort_values("F1_anidado_sinBOG", ascending=False).iloc[0].conjunto]


def version_sin_generadas(dat: Datos, c: Conjunto) -> Conjunto:
    return c.sin(set(dat.GEN_TODAS), f"{c.nombre} (sin generadas)")


def figura_curva_variables(tabla: pd.DataFrame, titulo: str, ruta: Path):
    import matplotlib.pyplot as plt
    from . import viz
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fijos = tabla[~tabla.grupo.isin(["f", "g"])]
    for cg, mk, col in [(True, "o", viz.AZUL), (False, "s", viz.ROJO)]:
        d = fijos[fijos.con_generadas == cg]
        ax.errorbar(d.n_vars, d.F1_anidado_sinBOG, yerr=d.EE, fmt=mk, color=col, capsize=3, ms=6,
                    label=f"candidatos {'con' if cg else 'sin'} generadas")
        for i, (_, r) in enumerate(d.sort_values("n_vars").iterrows()):
            etq = r.conjunto.split(" (")[0].split(" + ")[0][:22]
            ax.annotate(etq, (r.n_vars, r.F1_anidado_sinBOG), fontsize=7, xytext=(5, 5 if (i % 2 == 0) == cg else -10),
                        textcoords="offset points", color=col)
    tk = tabla[tabla.grupo == "f"].copy()
    tk["pool"] = tk.conjunto.str.extract(r"\((.*)\)")[0]
    for (pool, d), col in zip(tk.groupby("pool"), [viz.VERDE, viz.MORADO, viz.OCRE]):
        d = d.sort_values("n_vars")
        ax.errorbar(d.n_vars, d.F1_anidado_sinBOG, yerr=d.EE, fmt="-^", color=col, capsize=3, label=f"top-k por importancia en el pliegue ({pool})")
    g = tabla[tabla.grupo == "g"]
    if len(g):
        ax.scatter(g.n_vars, g.F1_anidado_sinBOG, marker="x", color="k", s=40, label="eliminación por bloques", zorder=5)
    el = tabla[tabla.elegido]
    if len(el):
        ax.scatter(el.n_vars, el.F1_anidado_sinBOG, s=220, facecolors="none", edgecolors="k", lw=1.8, label="elegido", zorder=6)
    ax.set(xscale="log", xlabel="número de variables (escala log)", ylabel="F1 anidado sin Bogotá (± 1 EE entre pliegues)",
           title=titulo)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(ruta, dpi=150, bbox_inches="tight")
    return fig


# ------------------------------------------------------------------------------ Optuna
def optuna_estudio(motor, dat: Datos, conjunto: Conjunto, n_ensayos=40, timeout=None, semilla=20262,
                   encolar: dict | None = None, nombre="estudio"):
    """Optuna 5 (TPE multivariado con semilla, poda Hyperband por pliegue). Objetivo: log-loss OOF
    (suave y poco ruidoso); se guardan F1 anidado sin Bogotá, AUC y árboles en cada ensayo, y el
    criterio final es el F1 entre los 5 mejores ensayos por log-loss (`elegir_ensayo`)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    y, folds = dat.y, dat.folds
    estudio = optuna.create_study(
        direction="minimize", study_name=nombre,
        sampler=optuna.samplers.TPESampler(seed=semilla, multivariate=True, n_startup_trials=10),
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=len(folds), reduction_factor=3))
    if encolar:
        estudio.enqueue_trial(encolar)

    def objetivo(trial):
        params = motor.espacio(trial)
        acum = []

        def al_pliegue(k, va, pr):
            acum.append(M.log_loss(y[va], pr))
            trial.report(float(np.mean(acum)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        r = cv(motor, dat.X, y, folds, params, conjunto, al_pliegue=al_pliegue)
        res = resumen(y, r["oof"], folds, dat.sin_bog)
        for k in ("F1_anidado_sinBOG", "EE", "AUC", "precision_sinBOG", "recall_sinBOG"):
            trial.set_user_attr(k, float(res[k]))
        trial.set_user_attr("arboles_medios", float(np.mean(r["iteraciones"])))
        trial.set_user_attr("tiempo_s", float(r["tiempo_s"]))
        return res["LogLoss"]

    t0 = time.time()
    estudio.optimize(objetivo, n_trials=n_ensayos, timeout=timeout, gc_after_trial=True)
    c = [t for t in estudio.trials if t.state.name == "COMPLETE"]
    print(f"{len(estudio.trials)} ensayos ({len(c)} completos, "
          f"{sum(t.state.name == 'PRUNED' for t in estudio.trials)} podados) en {(time.time() - t0) / 60:.1f} min")
    return estudio


def elegir_ensayo(estudio, top=5):
    """Entre los `top` ensayos con menor log-loss, el de mayor F1 anidado sin Bogotá."""
    c = sorted([t for t in estudio.trials if t.state.name == "COMPLETE"], key=lambda t: t.value)[:top]
    return max(c, key=lambda t: t.user_attrs["F1_anidado_sinBOG"])


def tabla_ensayos(estudio) -> pd.DataFrame:
    filas = []
    for t in estudio.trials:
        filas.append({"ensayo": t.number, "estado": t.state.name, "LogLoss": t.value,
                      "pliegues_evaluados": len(t.intermediate_values), **t.user_attrs,
                      **{f"param_{k}": v for k, v in t.params.items()}})
    return pd.DataFrame(filas)


def importancia_hiper(estudio) -> pd.Series:
    import optuna
    try:
        imp = optuna.importance.get_param_importances(estudio)
    except Exception as e:  # pragma: no cover
        print("importancia no disponible:", e); return pd.Series(dtype=float)
    return pd.Series(imp).sort_values(ascending=False)


def figura_optuna(estudio, imp: pd.Series, titulo: str, ruta: Path, etiqueta: str = "log-loss"):
    import matplotlib.pyplot as plt
    from . import viz
    t = tabla_ensayos(estudio)
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    c = t[t.estado == "COMPLETE"]; p = t[t.estado == "PRUNED"]
    ax[0].scatter(c.ensayo, c.LogLoss, color=viz.AZUL, s=22, label="completo")
    if len(p):
        ult = [list(tr.intermediate_values.values())[-1] if tr.intermediate_values else np.nan
               for tr in estudio.trials if tr.state.name == "PRUNED"]
        ax[0].scatter(p.ensayo, ult, color=viz.GRIS, marker="x", s=22, label="podado (último valor parcial)")
    mejor = c.sort_values("ensayo").LogLoss.cummin()
    ax[0].plot(c.sort_values("ensayo").ensayo, mejor, color=viz.ROJO, label="mejor hasta el ensayo")
    lo, hi = np.nanquantile(c.LogLoss, [0, .9]) if len(c) > 2 else (c.LogLoss.min(), c.LogLoss.max())
    pad = 0.15 * (hi - lo) + 1e-4
    ax[0].set(title=f"Historia de la búsqueda ({etiqueta} OOF sin Bogotá)", xlabel="ensayo", ylabel=etiqueta,
              ylim=(lo - pad, hi + 2 * pad))
    ax[0].legend(fontsize=8)
    if "F1_anidado_sinBOG" in c:
        ax[1].scatter(c.LogLoss, c.F1_anidado_sinBOG, color=viz.MORADO, s=22)
        ax[1].set(title=f"{etiqueta.capitalize()} vs. F1 anidado sin Bogotá (ensayos completos)", xlabel=f"{etiqueta} OOF sin Bogotá",
                  ylabel="F1 anidado sin Bogotá", xlim=(lo - pad, hi + 2 * pad))
    if len(imp):
        viz.barras_h(ax[2], list(imp.index), list(imp.values), color=viz.VERDE)
        ax[2].set(title="Importancia de hiperparámetros (fANOVA)", xlabel="fracción de la varianza del objetivo")
    fig.suptitle(titulo, fontweight="bold"); fig.tight_layout(); fig.savefig(ruta, dpi=150, bbox_inches="tight")
    return fig


# ------------------------------------------------------------------------------ diagnóstico
def diagnostico(dat: Datos, oof, nombre: str, prefijo: str, FIG: Path, TAB: Path, motor=None, conjunto=None,
                params=None, n_arboles=None, B=1000, figuras=True):
    """Métricas completas y figuras de publicación del modelo final. Devuelve dict de métricas."""
    import matplotlib.pyplot as plt
    from . import viz
    y, m, folds, D = dat.y, dat.sin_bog, dat.folds, dat.D
    s = np.asarray(oof, float)
    f1, f1k, umb, yhat = f1_anidado_sb(y, s, folds, m)
    lo, hi = bootstrap_f1(y[m], yhat[m], B=B)
    t_env, f1_opt = M.umbral_optimo_f1(y[m], s[m])
    tp, fp, fn, tn = M.confusion(y[m], yhat[m])
    met = {"modelo": nombre, "F1_anidado_sinBOG": f1, "IC95_inf": lo, "IC95_sup": hi,
           "EE_pliegues": float(f1k.std(ddof=1) / np.sqrt(len(f1k))), "F1_oof_opt_sinBOG": f1_opt,
           "umbral_envio_sinBOG": t_env, "AUC_ROC": M.auc_roc(y, s), "AUC_ROC_sinBOG": M.auc_roc(y[m], s[m]),
           "AUC_PR": M.pr_auc(y, s), "AUC_PR_sinBOG": M.pr_auc(y[m], s[m]), "Brier": M.brier(y, s),
           "LogLoss": M.log_loss(y, s), "ECE": M.ece(y, s), "precision_sinBOG": tp / max(tp + fp, 1),
           "recall_sinBOG": tp / max(tp + fn, 1), "MCC_sinBOG": M.mcc(y[m], yhat[m]),
           **{f"F1_pliegue_{k}": float(v) for k, v in enumerate(f1k)},
           **{f"umbral_pliegue_{k}": float(v) for k, v in enumerate(umb)}}
    if not figuras:
        return met, None, None
    # --- panel 1: curva F1-umbral, PR, calibración, F1 por pliegue
    cf = M.curva_f1(y[m], s[m], n=2000)
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
    ax[0].plot(cf.umbral, cf.f1, color=viz.ROJO, label="F1"); ax[0].plot(cf.umbral, cf.precision, color=viz.AZUL, lw=1, label="precisión")
    ax[0].plot(cf.umbral, cf.recall, color=viz.VERDE, lw=1, label="exhaustividad")
    ax[0].axvline(t_env, color="k", ls=":", lw=1); ax[0].annotate(f"umbral = {t_env:.3f}\nF1 = {f1_opt:.3f}", (t_env, f1_opt), xytext=(8, -30), textcoords="offset points", fontsize=8)
    ax[0].set(title="F1, precisión y exhaustividad según el umbral", xlabel="umbral de probabilidad", ylabel="valor", xlim=(0, 1)); ax[0].legend(fontsize=8)
    ax[1].plot(cf.recall, cf.precision, color=viz.MORADO)
    ax[1].axhline(y[m].mean(), color=viz.GRIS, ls="--", lw=1, label=f"azar (tasa = {y[m].mean():.3f})")
    ax[1].set(title=f"Curva precisión–exhaustividad (AUC-PR = {met['AUC_PR_sinBOG']:.3f})", xlabel="exhaustividad", ylabel="precisión"); ax[1].legend(fontsize=8)
    q = pd.qcut(s, 20, duplicates="drop"); g = pd.DataFrame({"p": s, "y": y}).groupby(q, observed=True).mean()
    ax[2].plot([0, 1], [0, 1], color=viz.GRIS, ls="--", lw=1); ax[2].plot(g.p, g.y, "o-", color=viz.AZUL, ms=4)
    ax[2].set(title=f"Calibración (20 cuantiles; ECE = {met['ECE']:.4f})", xlabel="probabilidad predicha media", ylabel="frecuencia observada de pobreza")
    ax[3].bar(range(len(f1k)), f1k, color=viz.AZUL); ax[3].axhline(f1, color=viz.ROJO, ls="--", label=f"global = {f1:.4f}")
    for k, v in enumerate(f1k):
        ax[3].text(k, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax[3].set(title="F1 anidado sin Bogotá por pliegue", xlabel="pliegue", ylabel="F1", ylim=(min(f1k) - .02, max(f1k) + .01)); ax[3].legend(fontsize=8)
    fig.suptitle(nombre, fontweight="bold"); fig.tight_layout(); fig.savefig(FIG / f"{prefijo}_diagnostico.png", dpi=150, bbox_inches="tight"); plt.show()
    # --- panel 2: matriz de confusión por 1.000 hogares, F1 por dominio, disyuntiva
    n = m.sum(); cm = np.array([[tn, fp], [fn, tp]]) / n * 1000
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.6), gridspec_kw={"width_ratios": [1, 1.5, 1.3]})
    ax[0].imshow(cm, cmap="Blues"); ax[0].grid(False)
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=15, color="k" if cm[i, j] < cm.max() * .6 else "w")
    ax[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no pobre", "pobre"], yticklabels=["no pobre", "pobre"],
              xlabel="predicción (umbral anidado)", ylabel="realidad", title="Matriz de confusión por 1.000 hogares (sin Bogotá)")
    dd = pd.DataFrame({"Dominio": D.Dominio.values[m], "y": y[m], "yh": yhat[m]})
    fd = []
    for d_, g_ in dd.groupby("Dominio"):
        tp_, fp_, fn_, _ = M.confusion(g_.y, g_.yh)
        fd.append({"Dominio": d_, "hogares": len(g_), "tasa_real": g_.y.mean(), "tasa_predicha": g_.yh.mean(),
                   "F1": M.f1(g_.y, g_.yh), "precision": tp_ / max(tp_ + fp_, 1), "recall": tp_ / max(tp_ + fn_, 1)})
    fd = pd.DataFrame(fd).sort_values("F1"); fd.to_csv(TAB / f"{prefijo}_f1_dominio.csv", index=False)
    ax[1].barh(fd.Dominio, fd.F1, color=viz.AZUL); ax[1].axvline(f1, color=viz.ROJO, ls="--", label=f"global = {f1:.3f}")
    for i, (v, nn) in enumerate(zip(fd.F1, fd.hogares)):
        ax[1].text(v, i, f" {v:.3f} (n={nn:,})".replace(",", "."), va="center", fontsize=7, color=viz.GRIS)
    ax[1].set(title="F1 anidado por dominio (sin Bogotá)", xlabel="F1", xlim=(max(0, fd.F1.min() - .1), 1)); ax[1].tick_params(axis="y", labelsize=8); ax[1].legend(fontsize=8, loc="lower right")
    # disyuntiva filtración / subcobertura
    cf_all = M.curva_f1(y[m], s[m])
    tasa = y[m].mean()
    i_rate = int(np.argmin(np.abs(cf_all.tasa_predicha.values - tasa)))
    i_rec = int(np.flatnonzero(cf_all.recall.values >= 0.90)[0])
    umbrales = {"F1-óptimo (envío)": t_env, "tasa predicha = tasa real": float(cf_all.umbral.values[i_rate]),
                "exhaustividad ≥ 90 %": float(cf_all.umbral.values[i_rec])}
    dis = []
    for nom, t in umbrales.items():
        yy = (s[m] >= t).astype(int); tp_, fp_, fn_, _ = M.confusion(y[m], yy)
        dis.append({"criterio": nom, "umbral": t, "F1": M.f1(y[m], yy), "precision": tp_ / max(tp_ + fp_, 1),
                    "exhaustividad": tp_ / max(tp_ + fn_, 1), "error_inclusion_filtracion": fp_ / max(tp_ + fp_, 1),
                    "error_exclusion_subcobertura": fn_ / max(tp_ + fn_, 1), "tasa_predicha": yy.mean(),
                    "pobres_excluidos_por_1000": fn_ / n * 1000, "no_pobres_incluidos_por_1000": fp_ / n * 1000})
    dis = pd.DataFrame(dis); dis.to_csv(TAB / f"{prefijo}_disyuntiva.csv", index=False)
    ax[2].plot(1 - cf.recall, 1 - cf.precision, color=viz.GRIS)
    for (_, r), col in zip(dis.iterrows(), [viz.ROJO, viz.AZUL, viz.VERDE]):
        ax[2].scatter(r.error_exclusion_subcobertura, r.error_inclusion_filtracion, s=70, color=col, zorder=5,
                      label=f"{r.criterio}: umbral {r.umbral:.3f}, F1 {r.F1:.3f}")
    ax[2].set(title="Disyuntiva filtración / subcobertura", xlabel="error de exclusión (subcobertura) = 1 − exhaustividad",
              ylabel="error de inclusión (filtración) = 1 − precisión", xlim=(0, .6), ylim=(0, .6)); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / f"{prefijo}_confusion_dominio_disyuntiva.png", dpi=150, bbox_inches="tight"); plt.show()
    met.update({f"cm_por_1000_{k}": v for k, v in zip(["TN", "FP", "FN", "TP"], cm.ravel())})
    return met, fd, dis


def shap_final(motor, dat: Datos, conjunto: Conjunto, params, n_arboles, prefijo, FIG, TAB, titulo, n=20000, semilla=1,
               peso_bogota=1.0):
    """Importancia SHAP media (|φ|) y dependencia de las 6 variables principales (modelo con todo train, con el
    mismo tratamiento de Bogotá que el modelo final; SHAP sobre una muestra de hogares fuera de Bogotá)."""
    import matplotlib.pyplot as plt
    from . import viz
    cols = conjunto.cols()
    ifit = np.arange(len(dat.y)) if peso_bogota != 0 else np.flatnonzero(dat.sin_bog)
    w = None if peso_bogota in (0, 1) else np.where(dat.sin_bog, 1.0, float(peso_bogota))[ifit]
    mod, _ = motor.ajustar(dat.X.iloc[ifit][cols], dat.y[ifit], params=params, semilla=semilla, n_arboles=n_arboles, w=w)
    pob = np.flatnonzero(dat.sin_bog)
    idx = np.sort(np.random.default_rng(semilla).choice(pob, min(n, len(pob)), replace=False))
    Xs = dat.X.iloc[idx][cols]
    if not getattr(motor, "usa_shap", True):
        return _importancia_rf(motor, mod, dat, Xs, cols, idx, prefijo, FIG, TAB, titulo)
    phi = motor.shap(mod, Xs, n_arboles)
    imp = pd.Series(np.abs(phi).mean(0), index=cols).sort_values(ascending=False)
    bl = dat.bloques()
    tab = pd.DataFrame({"variable": imp.index, "SHAP_media_abs": imp.values,
                        "bloque": [NOMBRE_BLOQUE[bl.get(c, "A")] for c in imp.index]})
    tab["SHAP_acumulado"] = tab.SHAP_media_abs.cumsum() / tab.SHAP_media_abs.sum()
    tab.to_csv(TAB / f"{prefijo}_importancia_shap.csv", index=False)
    top = tab.head(25).iloc[::-1]
    colores = {b: c for b, c in zip(NOMBRE_BLOQUE.values(), viz.PALETA)}
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.barh(top.variable, top.SHAP_media_abs, color=[colores[b] for b in top.bloque])
    for b in dict.fromkeys(top.bloque[::-1]):
        ax.barh([top.variable.iloc[-1]], [0], color=colores[b], label=b)
    ax.set(title=f"{titulo}: importancia SHAP media (top 25)", xlabel="E|φ_j| (log-odds de pobreza)")
    ax.tick_params(axis="y", labelsize=8); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / f"{prefijo}_shap_importancia.png", dpi=150, bbox_inches="tight"); plt.show()
    top6 = [c for c in tab.variable if c not in CAT_TODAS][:6]
    fig, axs = plt.subplots(2, 3, figsize=(17, 9)); axs = axs.ravel()
    cab = dat.D.cabecera.values[idx]
    for a, c in zip(axs, top6):
        j = cols.index(c); xv = Xs[c].values.astype(float)
        a.scatter(xv, phi[:, j], s=2, alpha=.25, c=np.where(cab == 1, viz.ROJO, viz.AZUL))
        a.axhline(0, color="k", lw=.7); a.set(title=c, xlabel=f"valor de {c}", ylabel="φ (log-odds de pobreza)")
        if np.isfinite(xv).any():
            lo, hi = np.nanquantile(xv, [.005, .995]); a.set_xlim(lo - 1e-9, hi + 1e-9)
    fig.suptitle(f"{titulo}: dependencia SHAP de las 6 variables principales (rojo = cabecera, azul = rural)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / f"{prefijo}_shap_dependencia.png", dpi=150, bbox_inches="tight"); plt.show()
    return tab


def _importancia_rf(motor, mod, dat, Xs, cols, idx, prefijo, FIG, TAB, titulo, n_pd=3000, n_grid=25):
    """RF: importancia por impureza (MDI) y dependencia parcial (Friedman, 2001) de las 6 principales."""
    import matplotlib.pyplot as plt
    from . import viz
    imp = pd.Series(mod.feature_importances_, index=cols).sort_values(ascending=False)
    bl = dat.bloques()
    tab = pd.DataFrame({"variable": imp.index, "importancia_impureza": imp.values,
                        "bloque": [NOMBRE_BLOQUE[bl.get(c, "A")] for c in imp.index]})
    tab["acumulado"] = tab.importancia_impureza.cumsum() / tab.importancia_impureza.sum()
    tab.to_csv(TAB / f"{prefijo}_importancia.csv", index=False)
    top = tab.head(25).iloc[::-1]
    colores = {b: c for b, c in zip(NOMBRE_BLOQUE.values(), viz.PALETA)}
    fig, ax = plt.subplots(figsize=(10, 8.5))
    ax.barh(top.variable, top.importancia_impureza, color=[colores[b] for b in top.bloque])
    for b in dict.fromkeys(top.bloque[::-1]):
        ax.barh([top.variable.iloc[-1]], [0], color=colores[b], label=b)
    ax.set(title=f"{titulo}: importancia por impureza (top 25)", xlabel="reducción media de la impureza (normalizada)")
    ax.tick_params(axis="y", labelsize=8); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / f"{prefijo}_importancia.png", dpi=150, bbox_inches="tight"); plt.show()
    top6 = [c for c in tab.variable if c not in CAT_TODAS][:6]
    Xp = Xs.iloc[:n_pd]
    fig, axs = plt.subplots(2, 3, figsize=(17, 9)); axs = axs.ravel()
    for a, c in zip(axs, top6):
        xv = Xs[c].values.astype(float); xv = xv[np.isfinite(xv)]
        grid = np.unique(np.nanquantile(xv, np.linspace(.01, .99, n_grid)))
        pdp = []
        for g in grid:
            Z = Xp.copy(); Z[c] = np.float32(g); pdp.append(motor.predecir(mod, Z).mean())
        a.plot(grid, pdp, "-o", color=viz.ROJO, ms=3)
        a2 = a.twinx(); a2.hist(xv, bins=40, color=viz.GRIS, alpha=.25); a2.set_yticks([]); a2.grid(False)
        a.set(title=c, xlabel=f"valor de {c}", ylabel="P(pobre) media (dependencia parcial)")
    fig.suptitle(f"{titulo}: dependencia parcial de las 6 variables principales (gris: distribución)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / f"{prefijo}_dependencia_parcial.png", dpi=150, bbox_inches="tight"); plt.show()
    return tab


def version_final(motor, dat: Datos, conjunto: Conjunto, params, semillas, nombre, sufijo, R, notebook, descripcion="",
                  peso_bogota=1.0):
    """Modelo final: OOF y test promediando semillas, registro común y CSV de envío (NO se sube)."""
    from . import registro as REG
    r = cv(motor, dat.X, dat.y, dat.folds, params, conjunto, semillas=semillas, X_test=dat.XT, bog=~dat.sin_bog,
           peso_bogota=peso_bogota)
    if peso_bogota != 1:
        sufijo = f"{sufijo}_bog{str(peso_bogota).replace('.', '_')}"
    m = dat.sin_bog
    t_env, f1o = M.umbral_optimo_f1(dat.y[m], r["oof"][m])
    archivo = motor.nombre_archivo(params, conjunto.n_vars, sufijo)
    desc = (f"{motor.nombre} clasificación; conjunto: {conjunto.nombre}; {conjunto.n_vars} variables; "
            f"{len(semillas)} semilla(s); árboles finales = {r['n_final']}; peso de Bogotá en el entrenamiento = {peso_bogota}; "
            f"{motor.describir(params)}. {descripcion}")
    slug = REG.registrar(R, nombre, r["oof"], r["test"], dat.ids, dat.ids_test, familia=(
        "Random Forest" if motor.corto == "RF" else "Boosting"), descripcion=desc, n_vars=conjunto.n_vars,
        params={**params, "n_final": r["n_final"], "semillas": list(semillas), "peso_bogota": peso_bogota,
                "variables": conjunto.cols()},
        es_probabilidad=True, tiempo_s=r["tiempo_s"], notebook=notebook, archivo_envio=archivo)
    from . import io as pio
    sample = pio.cargar_sample_submission(R)
    ruta = REG.escribir_envio(R, slug, t_env, sample, archivo)
    tasa_test = float((r["test"] >= t_env).mean())
    print(f"→ {nombre}: umbral sin Bogotá = {t_env:.4f}, tasa predicha en test = {tasa_test:.4f}, CSV: {ruta}")
    return {**r, "slug": slug, "umbral": t_env, "archivo": str(ruta), "tasa_test": tasa_test, "conjunto": conjunto,
            "params": params}


def segunda_particion(dat: Datos, semilla: int):
    """Otra partición en 5 pliegues estratificada por Pobre × Dominio (otra semilla)."""
    return hacer_pliegues(dat.y, dat.D.Dominio.values, k=5, semilla=semilla)


# ============================================================================== encargo v5 (cuadernos 10, 11 y 12)
def kw_cv(dat: Datos, peso_bogota=1.0) -> dict:
    """Argumentos comunes de `cv` en los cuadernos 10–12: máscara de Bogotá, su peso en el entrenamiento y
    parada temprana con hogares fuera de Bogotá (la población del test)."""
    return {"bog": ~dat.sin_bog, "peso_bogota": peso_bogota, "es_sin_bog": True}


def seleccion_v5(motor, dat: Datos, params, ks=(10, 20, 40, 80, 120), n_shap=20000, medida="SHAP", kw=None):
    """Selección de variables con el PROPIO modelo y sin fuga: candidatos (encuesta corta, S1, S2/S3, S3 +
    búsqueda 09, ampliado v5; cada uno con y sin generadas), top-k con la importancia del propio algoritmo
    calculada dentro de cada pliegue (pools con y sin generadas) y una pasada de eliminación por bloques
    teóricos con la regla de 1 EE (prefiere menos variables)."""
    filas, conjs = [], {}
    for nombre, c in dat.candidatos_v5().items():
        fila, _ = evaluar_conjunto(motor, dat, c, params, kw=kw); filas.append(fila); conjs[nombre] = c
        print(f"  {nombre:58s} n={c.n_vars:3d} F1={fila['F1_anidado_sinBOG']:.4f} ± {fila['EE']:.4f}  "
              f"AUC={fila['AUC']:.4f}  ({fila['tiempo_s']:.0f}s)", flush=True)
    rankings = {}
    for et, pool in dat.pools_v5().items():
        if pool.n_vars < 12:
            continue
        f, cs, rk = topk(motor, dat, pool, params, ks, et, n_shap, medida, kw=kw)
        filas += f; conjs.update(cs); rankings[et] = rk
    tabla = pd.DataFrame(filas)
    mejor = tabla.sort_values("F1_anidado_sinBOG", ascending=False).iloc[0]
    print(f"Mejor conjunto antes de la eliminación: {mejor.conjunto} (F1={mejor.F1_anidado_sinBOG:.4f} ± {mejor.EE:.4f})", flush=True)
    elegido, fila_el, f_el = eliminacion_bloques(motor, dat, conjs[mejor.conjunto], mejor.to_dict(), params, kw=kw)
    for f in f_el:
        f["grupo"] = "g"
    tabla = pd.concat([tabla, pd.DataFrame(f_el)], ignore_index=True)
    conjs[elegido.nombre] = elegido
    tabla["elegido"] = tabla.conjunto == elegido.nombre
    return {"tabla": tabla, "conjuntos": conjs, "elegido": elegido, "mejor_inicial": mejor.conjunto, "rankings": rankings}


def mejor_sin_generadas(sel: dict) -> Conjunto:
    t = sel["tabla"]; t = t[(~t.con_generadas.astype(bool)) & t.conjunto.isin(list(sel["conjuntos"]))]
    return sel["conjuntos"][t.sort_values("F1_anidado_sinBOG", ascending=False).iloc[0].conjunto]


def params_de_ensayo(motor, trial) -> tuple[dict, float]:
    """Hiperparámetros del modelo (base + los del ensayo) y peso de Bogotá del ensayo."""
    p = dict(trial.params)
    pb = float(p.pop("peso_bogota", 1.0))
    return {**motor.base, **p}, pb


def optuna_v5(motor, dat: Datos, conjunto: Conjunto, n_ensayos=40, timeout=None, semilla=20262, encolar=None,
              nombre="estudio", con_bogota=True, n_inicio=10):
    """Optuna 5: TPE multivariado con semilla y poda Hyperband por pliegue. Objetivo: la pérdida propia del
    motor (log-loss en boosting, Brier en RF) OOF SIN Bogotá, con parada temprana interna. El peso de Bogotá
    en el entrenamiento (1; 0,5; 0) es un hiperparámetro más. Se guardan, por ensayo, el F1 anidado sin Bogotá
    y el OOF (para no repetir la validación cruzada del ensayo elegido)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    y, folds, msk = dat.y, dat.folds, dat.sin_bog
    perd = M.log_loss if motor.perdida.startswith("LogLoss") else M.brier
    estudio = optuna.create_study(
        direction="minimize", study_name=nombre,
        sampler=optuna.samplers.TPESampler(seed=semilla, multivariate=True, n_startup_trials=n_inicio),
        pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=len(folds), reduction_factor=3))
    if encolar:
        estudio.enqueue_trial({**encolar, **({"peso_bogota": 1.0} if con_bogota else {})})
    oofs = {}
    t0 = time.time()

    def objetivo(trial):
        params = motor.espacio(trial)
        pb = trial.suggest_categorical("peso_bogota", [1.0, 0.5, 0.0]) if con_bogota else 1.0
        ys, ps = [], []

        def al_pliegue(k, va, pr):
            vm = msk[va]
            ys.append(y[va][vm]); ps.append(pr[vm])
            trial.report(float(perd(np.concatenate(ys), np.concatenate(ps))), k)
            if trial.should_prune():
                raise optuna.TrialPruned()

        r = cv(motor, dat.X, y, folds, params, conjunto, al_pliegue=al_pliegue, **kw_cv(dat, pb))
        res = resumen(y, r["oof"], folds, msk)
        for k in ("F1_anidado_sinBOG", "EE", "AUC", "AUC_sinBOG", "LogLoss_sinBOG", "Brier_sinBOG", "precision_sinBOG", "recall_sinBOG"):
            trial.set_user_attr(k, float(res[k]))
        trial.set_user_attr("arboles_medios", float(np.mean(r["iteraciones"])))
        trial.set_user_attr("tiempo_s", float(r["tiempo_s"]))
        oofs[trial.number] = r["oof"].astype(np.float32)
        print(f"  ensayo {trial.number:3d}: {motor.perdida}={res[motor.perdida]:.5f} F1={res['F1_anidado_sinBOG']:.4f} "
              f"peso_bog={pb} árboles={np.mean(r['iteraciones']):.0f} ({r['tiempo_s']:.0f}s; {(time.time() - t0) / 60:.1f} min)", flush=True)
        return res[motor.perdida]

    estudio.optimize(objetivo, n_trials=n_ensayos, timeout=timeout, gc_after_trial=True)
    c = [t for t in estudio.trials if t.state.name == "COMPLETE"]
    print(f"{len(estudio.trials)} ensayos ({len(c)} completos, "
          f"{sum(t.state.name == 'PRUNED' for t in estudio.trials)} podados) en {(time.time() - t0) / 60:.1f} min", flush=True)
    return estudio, oofs


def pareado(res_a: dict, res_b: dict, K=5) -> tuple[float, float]:
    """Diferencia media por pliegue F1(a) − F1(b) y su EE pareado."""
    d = np.array([res_a[f"F1_pliegue_{k}"] - res_b[f"F1_pliegue_{k}"] for k in range(K)])
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(K))


def comparar_variantes(motor, dat: Datos, conjunto: Conjunto, variantes: dict, previos: dict | None = None, referencia=None):
    """CV de varias variantes {nombre: (params, peso_bogota)} con el mismo conjunto; `previos` = {nombre: oof}
    reutiliza OOF ya calculados. Añade la diferencia pareada de F1 frente a la `referencia`."""
    filas, oofs, res = [], {}, {}
    for nom, (p, pb) in variantes.items():
        if previos and nom in previos:
            oof, ts = np.asarray(previos[nom], float), np.nan
        else:
            r = cv(motor, dat.X, dat.y, dat.folds, p, conjunto, **kw_cv(dat, pb)); oof, ts = r["oof"], r["tiempo_s"]
        res[nom] = resumen(dat.y, oof, dat.folds, dat.sin_bog); oofs[nom] = oof
        filas.append({"variante": nom, "peso_bogota": pb, **res[nom], "tiempo_s": ts})
        print(f"  {nom:45s} F1={res[nom]['F1_anidado_sinBOG']:.4f} ± {res[nom]['EE']:.4f}  {motor.perdida}={res[nom][motor.perdida]:.5f}", flush=True)
    t = pd.DataFrame(filas)
    ref = referencia or list(variantes)[0]
    dd = [pareado(res[n], res[ref]) for n in t.variante]
    t["dF1_vs_ref"] = [d[0] for d in dd]; t["EE_pareado"] = [d[1] for d in dd]; t["referencia"] = ref
    return t, oofs


def umbrales_dominio(y, s, dom, t0, deltas=None, pasadas=2, min_pobres=30) -> dict:
    """Umbrales por dominio que maximizan el F1 GLOBAL (ascenso por coordenadas sobre una rejilla
    t0 + δ, δ ∈ [−0,12; 0,12]); los dominios con menos de `min_pobres` pobres conservan t0. Empates:
    el δ más cercano a 0."""
    deltas = np.round(np.arange(-0.12, 0.12001, 0.005), 4) if deltas is None else np.asarray(deltas)
    y = np.asarray(y).astype(int); s = np.asarray(s, float); dom = np.asarray(dom).astype(str)
    doms = np.unique(dom); j0 = int(np.argmin(np.abs(deltas))); st = {}
    for d in doms:
        md = dom == d; yd, sd = y[md], s[md]
        cand = np.clip(t0 + deltas, 1e-6, 1 - 1e-6)
        pred = sd[None, :] >= cand[:, None]
        tp = (pred & (yd == 1)[None, :]).sum(1); fp = (pred & (yd == 0)[None, :]).sum(1)
        st[d] = (cand, tp, fp, int(yd.sum()) - tp, int(yd.sum()))
    idx = {d: j0 for d in doms}
    TP = sum(st[d][1][j0] for d in doms); FP = sum(st[d][2][j0] for d in doms); FN = sum(st[d][3][j0] for d in doms)
    for _ in range(pasadas):
        for d in doms:
            cand, tp, fp, fn, npob = st[d]
            if npob < min_pobres:
                continue
            j = idx[d]
            TPo, FPo, FNo = TP - tp[j], FP - fp[j], FN - fn[j]
            f = 2 * (TPo + tp) / np.maximum(2 * (TPo + tp) + FPo + fp + FNo + fn, 1)
            ok = np.flatnonzero(f >= f.max() - 1e-12)
            jn = int(ok[np.argmin(np.abs(deltas[ok]))])
            TP, FP, FN = TPo + tp[jn], FPo + fp[jn], FNo + fn[jn]; idx[d] = jn
    return {d: float(st[d][0][idx[d]]) for d in doms}


def evaluar_umbral_dominio(dat: Datos, s, folds=None):
    """Validación anidada de los umbrales por dominio frente al umbral global (ambos elegidos en los otros
    cuatro pliegues, sin Bogotá). Se adopta sólo si el F1 anidado mejora más de 1 EE (EE entre pliegues del
    umbral global)."""
    y, m, folds = dat.y, dat.sin_bog, folds or dat.folds
    dom = dat.D.Dominio.astype(str).values; s = np.asarray(s, float)
    yg = np.zeros_like(y); yd = np.zeros_like(y); filas = []
    for k, (tr, va) in enumerate(folds):
        tr_m, va_m = tr[m[tr]], va[m[va]]
        t0, _ = M.umbral_optimo_f1(y[tr_m], s[tr_m])
        ud = umbrales_dominio(y[tr_m], s[tr_m], dom[tr_m], t0)
        thr = pd.Series(dom[va_m]).map(ud).fillna(t0).values
        yg[va_m] = (s[va_m] >= t0).astype(int); yd[va_m] = (s[va_m] >= thr).astype(int)
        filas.append({"pliegue": k, "umbral_global": t0, "F1_global": M.f1(y[va_m], yg[va_m]),
                      "F1_por_dominio": M.f1(y[va_m], yd[va_m]),
                      "umbral_dominio_min": min(ud.values()), "umbral_dominio_max": max(ud.values())})
    t = pd.DataFrame(filas)
    d = (t.F1_por_dominio - t.F1_global).values
    f1g, f1d = M.f1(y[m], yg[m]), M.f1(y[m], yd[m])
    ee_g = float(t.F1_global.std(ddof=1) / np.sqrt(len(t)))
    out = {"F1_anidado_global": f1g, "F1_anidado_por_dominio": f1d, "diferencia": f1d - f1g, "EE_global": ee_g,
           "EE_pareado": float(d.std(ddof=1) / np.sqrt(len(d))), "adoptar": bool(f1d - f1g > ee_g)}
    return out, t


def escribir_csv(R, ids_test, score, umbral, archivo, dominio_test=None, umbrales_dom=None):
    """CSV `id,pobre` con los ids y el orden de `sample_submission` (NO se sube a Kaggle desde aquí).
    Devuelve (ruta, tasa de pobres predicha)."""
    from . import io as pio
    sample = pio.cargar_sample_submission(R)
    score = np.asarray(score, float)
    thr = umbral if not umbrales_dom else pd.Series(np.asarray(dominio_test).astype(str)).map(umbrales_dom).fillna(umbral).values
    te = pd.DataFrame({"id": np.asarray(ids_test), "pobre": (score >= thr).astype(int)})
    sub = sample[["id"]].merge(te, on="id", how="left")
    assert len(sub) == len(sample) and sub.pobre.notna().all(), "ids de test incompletos"
    sub["pobre"] = sub.pobre.astype(int)
    p = Path(R.submissions) / archivo
    sub[["id", "pobre"]].to_csv(p, index=False)
    return p, float(sub.pobre.mean())


def version_final_v5(motor, dat: Datos, conjunto: Conjunto, params, semillas, nombre, sufijo, R, notebook, peso_bogota=1.0,
                     descripcion="", umbral_por_dominio=False, familia=None):
    """Modelo final: bolsa de semillas (OOF y test promediados), reentrenado con todo train (con el
    tratamiento de Bogotá elegido), registro común y CSV de envío con el umbral F1-óptimo OOF sin Bogotá."""
    from . import registro as REG
    r = cv(motor, dat.X, dat.y, dat.folds, params, conjunto, semillas=semillas, X_test=dat.XT, **kw_cv(dat, peso_bogota))
    m = dat.sin_bog
    t_env, f1o = M.umbral_optimo_f1(dat.y[m], r["oof"][m])
    ud = umbrales_dominio(dat.y[m], r["oof"][m], dat.D.Dominio.astype(str).values[m], t_env) if umbral_por_dominio else None
    suf = sufijo + (f"_bog{str(peso_bogota).replace('.', '_')}" if peso_bogota != 1 else "")
    if params.get("is_unbalance"):
        suf += "_isunb"
    elif float(params.get("scale_pos_weight", 1.0)) != 1.0:
        suf += f"_spw_{params['scale_pos_weight']:.2f}".replace(".", "_")
    n_arb = r["n_final"] * len(semillas) if motor.corto == "RF" else None
    archivo = motor.nombre_archivo(params, conjunto.n_vars, suf, n_arboles=n_arb)
    desc = (f"{motor.nombre} clasificación; conjunto: {conjunto.nombre}; {conjunto.n_vars} variables; "
            f"bolsa de {len(semillas)} semilla(s); árboles finales = {r['n_final']}{' por bosque' if motor.corto == 'RF' else ''}; "
            f"peso de Bogotá en el entrenamiento = {peso_bogota}; umbral {'por dominio' if ud else 'global'} = {t_env:.4f}; "
            f"{motor.describir(params)}. {descripcion}")
    fam = familia or ("Random Forest" if motor.corto == "RF" else "Boosting")
    slug = REG.registrar(R, nombre, r["oof"], r["test"], dat.ids, dat.ids_test, familia=fam, descripcion=desc,
                         n_vars=conjunto.n_vars,
                         params={**params, "n_final": r["n_final"], "semillas": list(semillas), "peso_bogota": peso_bogota,
                                 "variables": conjunto.cols(), "umbral": t_env, "umbrales_dominio": ud},
                         es_probabilidad=True, tiempo_s=r["tiempo_s"], notebook=notebook, archivo_envio=archivo)
    ruta, tasa_test = escribir_csv(R, dat.ids_test, r["test"], t_env, archivo, dat.DT.Dominio.values, ud)
    print(f"→ {nombre}: umbral sin Bogotá = {t_env:.4f} (F1 OOF {f1o:.4f}), tasa predicha en test = {tasa_test:.4f}, "
          f"CSV: {ruta.name} ({r['tiempo_s'] / 60:.1f} min)", flush=True)
    return {**r, "slug": slug, "umbral": t_env, "umbrales_dominio": ud, "archivo": str(ruta), "tasa_test": tasa_test,
            "conjunto": conjunto, "params": params, "peso_bogota": peso_bogota, "semillas": list(semillas)}
