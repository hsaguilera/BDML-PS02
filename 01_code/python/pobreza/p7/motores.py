r"""Los nueve motores del protocolo v7 (siete algoritmos del enunciado; Boosting en tres implementaciones).

Interfaz común: `ajustar(X, y, w, params, semilla, Xes, yes, n_iter) → Ajuste`, `predecir(aj, X) → probabilidad`,
`espacio()` (distribuciones de Optuna para Sobol y TPE), `base()`, `nombre_archivo(params, n_vars, extra)`.
Todo lo que aprende de los datos (imputación, escalas, discretización, orden de categorías) se ajusta con las filas
de entrenamiento que recibe `ajustar` (dentro del pliegue). Espacios de búsqueda: memo §11.2 (Probst, Boulesteix y
Bischl, 2019; Probst, Wright y Boulesteix, 2019).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import glm as GL


@dataclass
class Ajuste:
    modelo: object
    n_iter: int = 1
    prep: object = None
    cols: list | None = None
    extra: dict = field(default_factory=dict)


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float):
        s = f"{v:.3g}"
    else:
        s = str(v)
    return s.replace(".", "_").replace("-", "m").replace("+", "")


ABREV = {"num_leaves": "hojas", "learning_rate": "lr", "min_child_samples": "mcs", "feature_fraction": "ff", "lambda_l2": "l2",
         "max_depth": "prof", "min_child_weight": "mcw", "subsample": "sub", "colsample_bytree": "col", "reg_lambda": "l2",
         "depth": "prof", "l2_leaf_reg": "l2", "random_strength": "rs", "bagging_temperature": "bt", "max_features": "mtry",
         "min_samples_leaf": "hoja", "max_samples": "ms", "criterion": "crit", "n_estimators": "arb", "min_samples_split": "mss",
         "ccp_alpha": "cp", "l1_ratio": "a", "lam": "lam", "splines": "spl", "tipo": "nb", "n_bins": "bins", "estrategia": "disc",
         "alpha": "alfa"}


def _dist():
    import optuna.distributions as od
    return od


class Motor:
    nombre = corto = familia = ""
    estocastico = False
    parada_temprana = False
    oob = False
    perdida = "logloss"
    usa_pesos = True
    claves_nombre: tuple = ()

    def __init__(self, hilos: int = 4, splines=()):
        self.hilos = int(hilos); self.splines = list(splines)

    def base(self) -> dict:
        return {}

    def espacio(self) -> dict:
        return {}

    def describir(self, params) -> str:
        return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()
                         if not isinstance(v, (list, dict)))

    def importancia_intrinseca(self, aj, cols) -> pd.Series | None:
        """Importancia propia del modelo, sólo para elegir qué variables graficar (no para seleccionar)."""
        try:
            m = aj.modelo
            if hasattr(m, "feature_importance"):                                  # LightGBM (ganancia)
                return pd.Series(m.feature_importance("gain"), index=cols)
            if hasattr(m, "get_score"):                                           # XGBoost (ganancia)
                g = m.get_score(importance_type="gain"); return pd.Series({c: g.get(c, 0.0) for c in cols})
            if hasattr(m, "get_feature_importance"):                              # CatBoost
                return pd.Series(m.get_feature_importance(), index=cols)
            if hasattr(m, "feature_importances_"):                                # árboles de scikit-learn
                return pd.Series(m.feature_importances_, index=aj.prep.cols)
            if isinstance(m, tuple) and hasattr(aj.prep, "origen"):               # lineales: |β| estandarizado por variable
                b = np.abs(np.asarray(m[1])); return pd.Series(b, index=aj.prep.origen).groupby(level=0).sum()
        except Exception:
            return None
        return None

    def nombre_archivo(self, params, n_vars, extra="") -> str:
        hp = "_".join(f"{ABREV.get(k, k)}_{_fmt(params[k])}" for k in self.claves_nombre if k in params)
        return f"{self.corto}_{hp}_n{n_vars}{extra}_v7.csv".replace("__", "_")


# ------------------------------------------------------------------------------ preprocesamiento de árboles
class PrepArbol:
    """Numéricas tal cual (NaN nativo de scikit-learn ≥ 1.4); categóricas → códigos ordenados por la tasa de pobreza
    del entrenamiento del pliegue (orden óptimo de Breiman para particiones binarias con dos clases; Fisher, 1958)."""

    def __init__(self, imputar: bool = False, monotonas=()):
        self.imputar, self.monotonas = imputar, set(monotonas)

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y, float)
        self.cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        self.orden = {}
        for c in self.cat:
            t = pd.Series(y).groupby(X[c].astype(str).values).mean().sort_values()
            self.orden[c] = {k: float(i) for i, k in enumerate(t.index)}
        self.cols = list(X.columns)
        if self.imputar:
            # scikit-learn no predice con faltantes en bosques con restricciones monótonas: centinela bajo el mínimo
            # (el árbol lo separa como una categoría) y mediana en las variables restringidas (no rompe la monotonía)
            A = self.transform(X, imputar=False)
            self.relleno = np.where([c in self.monotonas for c in self.cols], np.nanmedian(A, 0), np.nanmin(A, 0) - 1.0)
            self.relleno = np.where(np.isfinite(self.relleno), self.relleno, -1.0)
        return self

    def transform(self, X: pd.DataFrame, imputar=None) -> np.ndarray:
        A = np.empty((len(X), len(self.cols)), dtype=np.float32)
        for j, c in enumerate(self.cols):
            if c in self.orden:
                A[:, j] = pd.Series(X[c].astype(str).values).map(self.orden[c]).astype(float).values
            else:
                A[:, j] = X[c].to_numpy(dtype=np.float32)
        if (self.imputar if imputar is None else imputar):
            A = np.where(np.isnan(A), self.relleno.astype(np.float32), A)
        return A


# ------------------------------------------------------------------------------ LPM
class MotorLPM(Motor):
    nombre, corto, familia = "Linear Probability Model", "LPM", "Linear Probability Model"
    perdida = "brier"
    claves_nombre = ("splines", "lam")

    def base(self):
        return {"splines": False, "columnas": None}

    def ajustar(self, X, y, w=None, params=None, semilla=1, Xes=None, yes=None, n_iter=None):
        p = {**self.base(), **(params or {})}
        prep = GL.PrepLineal(self.splines if p.get("splines") else (), semilla=semilla).fit(X)
        Z = prep.transform(X)
        est = GL.EstadisticosMCO.de(Z, y, w)
        cols = None if p.get("columnas") is None else [i for i, n in enumerate(prep.nombres) if n in set(p["columnas"])]
        b0, beta = GL.mco(est, cols)
        return Ajuste((b0, beta), 1, prep)

    def predecir(self, aj, X):
        b0, beta = aj.modelo
        return np.clip(b0 + GL.zbeta(aj.prep.transform(X), beta), 1e-4, 1 - 1e-4)


# ------------------------------------------------------------------------------ Logit
class MotorLogit(Motor):
    nombre, corto, familia = "Logit", "LOGIT", "Logit"
    claves_nombre = ("splines",)

    def base(self):
        return {"splines": False, "columnas": None, "cresta": 1e-5}

    def ajustar(self, X, y, w=None, params=None, semilla=1, Xes=None, yes=None, n_iter=None):
        p = {**self.base(), **(params or {})}
        prep = GL.PrepLineal(self.splines if p.get("splines") else (), semilla=semilla).fit(X)
        Z = prep.transform(X)
        cols = None if p.get("columnas") is None else [i for i, n in enumerate(prep.nombres) if n in set(p["columnas"])]
        b0, beta, cols = GL.logit_newton(Z, y, w, cols, cresta=float(p.get("cresta", 1e-5)))
        full = np.zeros(Z.shape[1]); full[cols] = beta
        return Ajuste((b0, full), 1, prep)

    def predecir(self, aj, X):
        b0, beta = aj.modelo
        return GL._sig(b0 + GL.zbeta(aj.prep.transform(X), beta))


# ------------------------------------------------------------------------------ Elastic Net
class MotorEN(Motor):
    nombre, corto, familia = "Elastic Net (logística)", "EN", "Elastic Net"
    claves_nombre = ("l1_ratio", "lam")

    def variables_efectivas(self, aj) -> list:
        b0, beta = aj.modelo
        org = np.asarray(aj.prep.origen)
        return sorted(set(org[beta != 0]))

    def base(self):
        return {"l1_ratio": 0.4, "lam": 1e-3, "splines": False}

    def espacio(self):
        od = _dist()
        return {"l1_ratio": od.FloatDistribution(0.0, 1.0), "lam": od.FloatDistribution(1e-5, 1e-1, log=True)}

    def ajustar(self, X, y, w=None, params=None, semilla=1, Xes=None, yes=None, n_iter=None):
        p = {**self.base(), **(params or {})}
        prep = GL.PrepLineal(self.splines if p.get("splines") else (), semilla=semilla).fit(X)
        Z = prep.transform(X)
        a = float(p["l1_ratio"]); lam = float(p["lam"])
        lmax = GL.lambda_max_logit(Z, y, w, a)
        lams = np.unique(np.r_[GL.grilla_lambda(lmax, 12, max(lam / lmax, 1e-6)), lam])[::-1] if lam < lmax else np.array([lam])
        r = GL.ruta_enet_logit(Z, y, w, lams, a)
        return Ajuste((r["b0"][-1], r["coef"][:, -1]), 1, prep, extra={"df": int(r["df"][-1])})

    def predecir(self, aj, X):
        b0, beta = aj.modelo
        return GL._sig(b0 + GL.zbeta(aj.prep.transform(X), beta))


# ------------------------------------------------------------------------------ CART
def podar_info(tree) -> dict:
    """Secuencia de poda de costo-complejidad mínima (Breiman et al., 1984) sobre el árbol de scikit-learn: para cada
    nodo interno, el α en que colapsa (eslabón más débil, con R(t) = impureza ponderada como en scikit-learn)."""
    t = tree.tree_
    L, Rr = t.children_left, t.children_right
    W = t.weighted_n_node_samples / t.weighted_n_node_samples[0]
    r = W * t.impurity
    n = t.node_count
    padre = np.full(n, -1)
    for i in range(n):
        if L[i] != -1:
            padre[L[i]] = i; padre[Rr[i]] = i
    hoja = L == -1
    RT = np.where(hoja, r, 0.0); nh = hoja.astype(float)
    orden = np.argsort(-np.arange(n))           # hijos antes que padres en el arreglo de sklearn (preorden) → al revés
    for i in range(n - 1, -1, -1):
        if not hoja[i]:
            RT[i] = RT[L[i]] + RT[Rr[i]]; nh[i] = nh[L[i]] + nh[Rr[i]]
    activo = ~hoja.copy()
    colapso = np.full(n, np.inf)
    vivo = np.ones(n, bool)
    import heapq
    g = np.where(activo, (r - RT) / np.maximum(nh - 1, 1e-12), np.inf)
    heap = [(g[i], i) for i in range(n) if activo[i]]
    heapq.heapify(heap)
    alfa_actual = 0.0
    while heap:
        gi, i = heapq.heappop(heap)
        if not activo[i] or not vivo[i] or abs(gi - g[i]) > 1e-15:
            continue
        alfa_actual = max(alfa_actual, gi)
        colapso[i] = alfa_actual
        activo[i] = False
        # descendientes dejan de existir
        pila = [L[i], Rr[i]]
        while pila:
            j = pila.pop()
            if j == -1:
                continue
            vivo[j] = False; activo[j] = False
            pila += [L[j], Rr[j]]
        d_R = r[i] - RT[i]; d_n = 1 - nh[i]
        RT[i] = r[i]; nh[i] = 1
        a = padre[i]
        while a != -1:
            RT[a] += d_R; nh[a] += d_n
            if activo[a]:
                g[a] = (r[a] - RT[a]) / max(nh[a] - 1, 1e-12)
                heapq.heappush(heap, (g[a], a))
            a = padre[a]
    return {"colapso": colapso, "padre": padre, "hoja": hoja}


def probas_hoja(tree, m: float, prior: float) -> np.ndarray:
    """Probabilidad de pobreza por nodo con estimación m (Cestnik, 1990; Zadrozny y Elkan, 2001):
    p = (n₁ + m·π)/(n + m), que evita probabilidades 0/1 en hojas puras."""
    t = tree.tree_
    nW = t.weighted_n_node_samples
    v = t.value[:, 0, :]
    frac1 = v[:, 1] / np.maximum(v.sum(1), 1e-12)
    n1 = frac1 * nW
    return (n1 + m * prior) / (nW + m)


def predecir_podado(tree, X, alfas, info=None, m: float = 2.0, prior: float = 0.2) -> np.ndarray:
    """Probabilidades (n × len(alfas)) del árbol podado en cada α, a partir del árbol completo (sin re-ajustar)."""
    info = info or podar_info(tree)
    P = probas_hoja(tree, m, prior)
    path = tree.decision_path(X); path.sort_indices()
    indptr, idx = path.indptr, path.indices
    n = X.shape[0]; lon = np.diff(indptr); prof = int(lon.max())
    fila = np.repeat(np.arange(n), lon); posi = np.arange(len(idx)) - np.repeat(indptr[:-1], lon)
    M = np.full((n, prof), -1, dtype=np.int64)
    M[fila, posi] = idx                                  # los nodos de un camino tienen índices crecientes (preorden)
    hojas = idx[indptr[1:] - 1]
    M = np.where(M < 0, hojas[:, None], M)
    C = info["colapso"][M]                       # α de colapso de cada nodo del camino (inf para hojas)
    out = np.empty((n, len(alfas)))
    for k, a in enumerate(alfas):
        ok = C <= a
        primero = np.where(ok.any(1), ok.argmax(1), prof - 1)
        out[:, k] = P[M[np.arange(n), primero]]
    return out


class MotorCART(Motor):
    nombre, corto, familia = "CART", "CART", "CART"
    paralelo_pliegues = True
    claves_nombre = ("max_depth", "min_samples_leaf", "min_samples_split", "criterion", "ccp_alpha")

    def base(self):
        return {"max_depth": 21, "min_samples_leaf": 50, "min_samples_split": 100, "criterion": "gini", "ccp_alpha": 0.0,
                "m_estimacion": 2.0}

    def espacio(self):
        od = _dist()
        return {"max_depth": od.IntDistribution(2, 30), "min_samples_leaf": od.IntDistribution(5, 2000, log=True),
                "min_samples_split": od.IntDistribution(10, 4000, log=True),
                "criterion": od.CategoricalDistribution(["gini", "entropy"])}

    def ajustar(self, X, y, w=None, params=None, semilla=1, Xes=None, yes=None, n_iter=None):
        from sklearn.tree import DecisionTreeClassifier
        p = {**self.base(), **(params or {})}
        prep = PrepArbol().fit(X, y)
        A = prep.transform(X)
        m = DecisionTreeClassifier(max_depth=int(p["max_depth"]), min_samples_leaf=int(p["min_samples_leaf"]),
                                   min_samples_split=max(int(p["min_samples_split"]), 2), criterion=p["criterion"],
                                   ccp_alpha=float(p.get("ccp_alpha", 0.0)), random_state=semilla).fit(A, y, sample_weight=w)
        prior = float(np.average(y, weights=w))
        return Ajuste(m, 1, prep, extra={"prior": prior, "m": float(p.get("m_estimacion", 2.0)), "hojas": int(m.get_n_leaves())})

    def predecir(self, aj, X):
        A = aj.prep.transform(X)
        P = probas_hoja(aj.modelo, aj.extra["m"], aj.extra["prior"])
        return P[aj.modelo.apply(A)]

    def variables_usadas(self, aj) -> list:
        f = aj.modelo.tree_.feature
        return sorted({aj.prep.cols[j] for j in f[f >= 0]})

    variables_efectivas = variables_usadas


# ------------------------------------------------------------------------------ Random Forest
class MotorRF(Motor):
    nombre, corto, familia = "Random Forest", "RF", "Random Forest"
    estocastico = True
    oob = True
    claves_nombre = ("max_features", "min_samples_leaf", "max_samples", "criterion", "n_estimators")

    def base(self):
        return {"n_estimators": 300, "max_features": 0.26, "min_samples_leaf": 5, "max_samples": 0.7, "criterion": "gini"}

    def espacio(self):
        od = _dist()
        # Espacio acotado el 20 oct (19:45): con 1.152 variables, max_features > 0,35 y hojas de 1 observación generan
        # árboles enormes que agotan la memoria (un solo ensayo pasó de 5 h paginando). Probst et al. (2019): ajustar
        # min.node.size aporta poco; max_features es el hiperparámetro relevante y su óptimo suele estar en valores bajos.
        return {"max_features": od.FloatDistribution(0.03, 0.35, log=True),
                "min_samples_leaf": od.IntDistribution(5, 200, log=True),
                "max_samples": od.FloatDistribution(0.2, 1.0),
                "criterion": od.CategoricalDistribution(["gini", "log_loss"])}

    def ajustar(self, X, y, w=None, params=None, semilla=1, Xes=None, yes=None, n_iter=None, oob=False, monotonas=None):
        from sklearn.ensemble import RandomForestClassifier
        p = {**self.base(), **(params or {})}
        mon = monotonas if monotonas is not None else p.get("monotonas")
        prep = PrepArbol(imputar=bool(mon), monotonas=(mon or {}).keys()).fit(X, y)
        A = prep.transform(X)
        mono = None
        if mon:
            mono = [int(mon.get(c, 0)) for c in prep.cols]
        ms = float(p["max_samples"]); ms = None if ms >= 0.999 else ms
        m = RandomForestClassifier(n_estimators=int(p["n_estimators"]), max_features=float(p["max_features"]),
                                   min_samples_leaf=int(p["min_samples_leaf"]), max_samples=ms, criterion=p["criterion"],
                                   bootstrap=True, oob_score=bool(oob), n_jobs=self.hilos, random_state=semilla,
                                   monotonic_cst=mono).fit(A, y, sample_weight=w)
        return Ajuste(m, int(p["n_estimators"]), prep)

    def predecir(self, aj, X):
        return aj.modelo.predict_proba(aj.prep.transform(X))[:, 1]

    def oob_proba(self, aj) -> np.ndarray:
        d = aj.modelo.oob_decision_function_[:, 1]
        return d


# ------------------------------------------------------------------------------ Boosting
class MotorBoost(Motor):
    estocastico = True
    parada_temprana = True
    familia = "Boosting"

    def __init__(self, hilos=4, splines=(), lr=0.05, paciencia=200, max_arboles=10000):
        super().__init__(hilos, splines)
        self.lr, self.paciencia, self.max_arboles = float(lr), int(paciencia), int(max_arboles)


class MotorLGB(MotorBoost):
    nombre, corto = "LightGBM", "LGBM"
    claves_nombre = ("num_leaves", "learning_rate", "min_child_samples", "feature_fraction", "lambda_l2")

    def base(self):
        return dict(learning_rate=self.lr, num_leaves=63, min_child_samples=50, feature_fraction=0.5, bagging_fraction=0.8,
                    bagging_freq=1, lambda_l2=1.0, lambda_l1=1e-3, min_gain_to_split=0.0, max_bin=255, cat_smooth=10.0,
                    max_cat_to_onehot=8)

    def espacio(self):
        od = _dist()
        return {"num_leaves": od.IntDistribution(8, 256, log=True), "min_child_samples": od.IntDistribution(10, 1000, log=True),
                "feature_fraction": od.FloatDistribution(0.3, 1.0), "bagging_fraction": od.FloatDistribution(0.5, 1.0),
                "lambda_l2": od.FloatDistribution(1e-3, 100.0, log=True), "lambda_l1": od.FloatDistribution(1e-3, 10.0, log=True),
                "min_gain_to_split": od.FloatDistribution(0.0, 2.0), "max_bin": od.CategoricalDistribution([63, 127, 255]),
                "cat_smooth": od.FloatDistribution(1.0, 100.0, log=True), "max_cat_to_onehot": od.CategoricalDistribution([4, 8, 32])}

    def _p(self, params, semilla, cols):
        p = {**self.base(), **(params or {})}
        mon = p.pop("monotonas", None)
        q = {"objective": "binary", "metric": "binary_logloss", "verbose": -1, "num_threads": self.hilos, "seed": semilla,
             "bagging_seed": semilla, "feature_fraction_seed": semilla, "data_random_seed": semilla, **p}
        if mon:
            q["monotone_constraints"] = [int(mon.get(c, 0)) for c in cols]
            q["monotone_constraints_method"] = "advanced"
        return q

    def ajustar(self, X, y, w=None, params=None, semilla=1, Xes=None, yes=None, n_iter=None, wes=None):
        import lightgbm as lgb
        q = self._p(params, semilla, list(X.columns))
        dtr = lgb.Dataset(X, np.asarray(y), weight=w, free_raw_data=True)
        if Xes is not None and n_iter is None:
            des = lgb.Dataset(Xes, np.asarray(yes), weight=wes, reference=dtr)
            m = lgb.train(q, dtr, self.max_arboles, valid_sets=[des], callbacks=[lgb.early_stopping(self.paciencia, verbose=False)])
            return Ajuste(m, max(int(m.best_iteration), 1))
        m = lgb.train(q, dtr, int(n_iter))
        return Ajuste(m, int(n_iter))

    def predecir(self, aj, X):
        return aj.modelo.predict(X, num_iteration=aj.n_iter)

    def shap(self, aj, X):
        return aj.modelo.predict(X, num_iteration=aj.n_iter, pred_contrib=True)[:, :-1]


class MotorXGB(MotorBoost):
    nombre, corto = "XGBoost", "XGB"
    claves_nombre = ("max_depth", "learning_rate", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda")

    def base(self):
        return dict(learning_rate=self.lr, max_depth=6, min_child_weight=5.0, subsample=0.8, colsample_bytree=0.6,
                    colsample_bylevel=1.0, reg_lambda=1.0, reg_alpha=1e-3, gamma=0.0, max_bin=256)

    def espacio(self):
        od = _dist()
        return {"max_depth": od.IntDistribution(3, 12), "min_child_weight": od.FloatDistribution(1.0, 128.0, log=True),
                "subsample": od.FloatDistribution(0.5, 1.0), "colsample_bytree": od.FloatDistribution(0.3, 1.0),
                "colsample_bylevel": od.FloatDistribution(0.3, 1.0), "reg_lambda": od.FloatDistribution(1e-3, 100.0, log=True),
                "reg_alpha": od.FloatDistribution(1e-3, 10.0, log=True), "gamma": od.FloatDistribution(0.0, 5.0)}

    def _p(self, params, semilla, cols):
        p = {**self.base(), **(params or {})}
        mon = p.pop("monotonas", None)
        q = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "nthread": self.hilos,
             "seed": semilla, "verbosity": 0, **p}
        if mon:
            q["monotone_constraints"] = "(" + ",".join(str(int(mon.get(c, 0))) for c in cols) + ")"
        return q

    def ajustar(self, X, y, w=None, params=None, semilla=1, Xes=None, yes=None, n_iter=None, wes=None):
        import xgboost as xgb
        q = self._p(params, semilla, list(X.columns))
        mb = int(q.get("max_bin", 256))
        dtr = xgb.QuantileDMatrix(X, np.asarray(y), weight=w, enable_categorical=True, nthread=self.hilos, max_bin=mb)
        if Xes is not None and n_iter is None:
            des = xgb.QuantileDMatrix(Xes, np.asarray(yes), weight=wes, ref=dtr, enable_categorical=True, nthread=self.hilos, max_bin=mb)
            m = xgb.train(q, dtr, self.max_arboles, evals=[(des, "es")], early_stopping_rounds=self.paciencia, verbose_eval=False)
            return Ajuste(m, int(m.best_iteration) + 1)
        m = xgb.train(q, dtr, int(n_iter))
        return Ajuste(m, int(n_iter))

    def predecir(self, aj, X):
        import xgboost as xgb
        try:
            return aj.modelo.inplace_predict(X, iteration_range=(0, int(aj.n_iter)))
        except Exception:
            return aj.modelo.predict(xgb.DMatrix(X, enable_categorical=True), iteration_range=(0, int(aj.n_iter)))

    def shap(self, aj, X):
        import xgboost as xgb
        return aj.modelo.predict(xgb.DMatrix(X, enable_categorical=True, nthread=self.hilos), pred_contribs=True,
                                 iteration_range=(0, int(aj.n_iter)))[:, :-1]


class MotorCB(MotorBoost):
    nombre, corto = "CatBoost", "CB"
    claves_nombre = ("depth", "learning_rate", "l2_leaf_reg", "random_strength", "bagging_temperature")

    def base(self):
        return dict(learning_rate=self.lr, depth=6, l2_leaf_reg=3.0, random_strength=1.0, bagging_temperature=1.0,
                    border_count=254, one_hot_max_size=10, max_ctr_complexity=1)

    def espacio(self):
        od = _dist()
        return {"depth": od.IntDistribution(4, 10), "l2_leaf_reg": od.FloatDistribution(1.0, 30.0, log=True),
                "random_strength": od.FloatDistribution(1e-2, 10.0, log=True), "bagging_temperature": od.FloatDistribution(0.0, 1.0),
                "border_count": od.CategoricalDistribution([128, 254]), "one_hot_max_size": od.CategoricalDistribution([2, 10]),
                "max_ctr_complexity": od.CategoricalDistribution([1, 2])}

    @staticmethod
    def _cats(X):
        return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]

    @staticmethod
    def _a_str(X, cats):
        if not cats:
            return X
        X = X.copy()
        for c in cats:
            X[c] = X[c].astype(str)
        return X

    def ajustar(self, X, y, w=None, params=None, semilla=1, Xes=None, yes=None, n_iter=None, wes=None):
        from catboost import CatBoostClassifier, Pool
        p = {**self.base(), **(params or {})}
        mon = p.pop("monotonas", None)
        cats = self._cats(X)
        kw = dict(thread_count=self.hilos, random_seed=semilla, verbose=0, allow_writing_files=False, **p)
        if mon:
            kw["monotone_constraints"] = {c: int(s) for c, s in mon.items() if c in X.columns}
        dtr = Pool(self._a_str(X, cats), np.asarray(y), cat_features=cats, weight=w)
        if Xes is not None and n_iter is None:
            m = CatBoostClassifier(iterations=self.max_arboles, od_type="Iter", od_wait=self.paciencia, **kw)
            m.fit(dtr, eval_set=Pool(self._a_str(Xes, cats), np.asarray(yes), cat_features=cats, weight=wes), use_best_model=True)
            return Ajuste(m, int(m.get_best_iteration()) + 1, extra={"cats": cats})
        m = CatBoostClassifier(iterations=int(n_iter), **kw).fit(dtr)
        return Ajuste(m, int(n_iter), extra={"cats": cats})

    def predecir(self, aj, X):
        return aj.modelo.predict_proba(self._a_str(X, aj.extra["cats"]))[:, 1]

    def shap(self, aj, X):
        from catboost import Pool
        cats = aj.extra["cats"]
        return aj.modelo.get_feature_importance(Pool(self._a_str(X, cats), cat_features=cats), type="ShapValues")[:, :-1]


# ------------------------------------------------------------------------------ Naive Bayes
class PrepNB:
    """Discretización dentro del pliegue: intervalos de igual frecuencia (número como hiperparámetro; Yang y Webb,
    2009) o MDL (Fayyad e Irani, 1993); faltante = categoría propia; categóricas = sus niveles (+ «no visto»)."""

    def __init__(self, n_bins=20, estrategia="cuantiles", n_sub=60000, semilla=0):
        self.n_bins, self.estrategia, self.n_sub, self.semilla = n_bins, estrategia, n_sub, semilla

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        self.cols = list(X.columns)
        self.cat = {c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)}
        rng = np.random.default_rng(self.semilla)
        sub = rng.choice(len(X), min(self.n_sub, len(X)), replace=False)
        self.cortes, self.niveles = {}, {}
        nb = int(self.n_bins) if self.n_bins != "sqrt" else int(np.sqrt(len(X)))
        for c in self.cols:
            if c in self.cat:
                self.niveles[c] = {k: i + 1 for i, k in enumerate(X[c].cat.categories.astype(str))}
                continue
            v = X[c].to_numpy(np.float64)[sub]
            v = v[np.isfinite(v)]
            if len(v) == 0:
                self.cortes[c] = np.array([]); continue
            if self.estrategia == "mdl":
                self.cortes[c] = cortes_mdl(v, y[sub][np.isfinite(X[c].to_numpy(np.float64)[sub])])
            else:
                q = np.unique(np.quantile(v, np.linspace(0, 1, nb + 1)[1:-1]))
                self.cortes[c] = q
        return self

    def transform(self, X) -> np.ndarray:
        A = np.zeros((len(X), len(self.cols)), dtype=np.int16)
        for j, c in enumerate(self.cols):
            if c in self.niveles:
                A[:, j] = pd.Series(X[c].astype(str).values).map(self.niveles[c]).fillna(0).astype(int).values
            else:
                v = X[c].to_numpy(np.float64)
                k = np.searchsorted(self.cortes[c], v, side="right") + 1
                A[:, j] = np.where(np.isfinite(v), k, 0)
        return A

    def n_categorias(self) -> np.ndarray:
        return np.array([len(self.niveles[c]) + 1 if c in self.niveles else len(self.cortes[c]) + 2 for c in self.cols])


def cortes_mdl(v, y, max_cortes: int = 30) -> np.ndarray:
    """Discretización supervisada de Fayyad e Irani (1993): partición recursiva por entropía con el criterio MDL."""
    o = np.argsort(v, kind="mergesort"); v, y = v[o], np.asarray(y)[o].astype(int)
    cortes = []

    def H(c1, n):
        if n == 0:
            return 0.0
        p = c1 / n
        return 0.0 if p <= 0 or p >= 1 else -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    def rec(a, b, prof):
        if len(cortes) >= max_cortes or b - a < 20 or prof > 12:
            return
        yy = y[a:b]; vv = v[a:b]; n = b - a
        c1 = np.cumsum(yy)[:-1]; nl = np.arange(1, n); tot1 = yy.sum()
        ok = vv[1:] != vv[:-1]
        if not ok.any():
            return
        pl = c1 / nl; pr = (tot1 - c1) / (n - nl)
        with np.errstate(all="ignore"):
            hl = -(np.nan_to_num(pl * np.log2(pl)) + np.nan_to_num((1 - pl) * np.log2(1 - pl)))
            hr = -(np.nan_to_num(pr * np.log2(pr)) + np.nan_to_num((1 - pr) * np.log2(1 - pr)))
        e = (nl * hl + (n - nl) * hr) / n
        e = np.where(ok, e, np.inf)
        i = int(np.argmin(e))
        hs = H(tot1, n); gan = hs - e[i]
        k = 2 if 0 < tot1 < n else 1
        k1 = 2 if 0 < c1[i] < nl[i] else 1; k2 = 2 if 0 < tot1 - c1[i] < n - nl[i] else 1
        delta = np.log2(3 ** k - 2) - (k * hs - k1 * H(c1[i], nl[i]) - k2 * H(tot1 - c1[i], n - nl[i]))
        if gan <= (np.log2(n - 1) + delta) / n:
            return
        corte = 0.5 * (vv[i] + vv[i + 1]); cortes.append(corte)
        rec(a, a + i + 1, prof + 1); rec(a + i + 1, b, prof + 1)

    rec(0, len(v), 0)
    return np.unique(np.array(cortes))


class MotorNB(Motor):
    """Naive Bayes categórico sobre variables discretizadas en el pliegue (alternativa: gaussiano sobre puntajes
    normales por cuantiles). El puntaje es log π₁/π₀ + Σ_j log P(x_j|1)/P(x_j|0)."""
    nombre, corto, familia = "Naive Bayes", "NB", "Naive Bayes"
    claves_nombre = ("tipo", "n_bins", "estrategia", "alpha")

    def base(self):
        return {"tipo": "categorico", "n_bins": 20, "estrategia": "cuantiles", "alpha": 1.0, "var_smoothing": 1e-9}

    def espacio(self):
        od = _dist()
        return {"n_bins": od.CategoricalDistribution([5, 10, 20, 50, 100]),
                "estrategia": od.CategoricalDistribution(["cuantiles", "mdl"]),
                "alpha": od.FloatDistribution(0.01, 10.0, log=True)}

    def ajustar(self, X, y, w=None, params=None, semilla=1, Xes=None, yes=None, n_iter=None):
        p = {**self.base(), **(params or {})}
        y = np.asarray(y).astype(int); w = np.ones(len(y)) if w is None else np.asarray(w, float)
        if p["tipo"] == "gaussiano":
            from sklearn.preprocessing import QuantileTransformer
            from sklearn.naive_bayes import GaussianNB
            num = [c for c in X.columns if not isinstance(X[c].dtype, pd.CategoricalDtype)]
            qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=60000, random_state=semilla)
            A = qt.fit_transform(X[num].to_numpy(np.float64))
            med = np.nanmedian(A, axis=0); A = np.where(np.isnan(A), med, A)
            g = GaussianNB(var_smoothing=float(p["var_smoothing"])).fit(A, y, sample_weight=w)
            return Ajuste(g, 1, (qt, num, med))
        prep = PrepNB(p["n_bins"], p["estrategia"], semilla=semilla).fit(X, y)
        A = prep.transform(X); K = prep.n_categorias(); a = float(p["alpha"])
        w1, w0 = w * (y == 1), w * (y == 0)
        tablas = []
        for j in range(A.shape[1]):
            c1 = np.bincount(A[:, j], w1, minlength=K[j]); c0 = np.bincount(A[:, j], w0, minlength=K[j])
            tablas.append(np.log((c1 + a) / (w1.sum() + a * K[j])) - np.log((c0 + a) / (w0.sum() + a * K[j])))
        prior = np.log(w1.sum() / w0.sum())
        return Ajuste((prior, tablas), 1, prep)

    def contribuciones(self, aj, X) -> np.ndarray:
        """log P(x_j|1)/P(x_j|0) por observación y variable (n × p): base del envoltorio hacia adelante."""
        prior, tablas = aj.modelo
        A = aj.prep.transform(X)
        return np.column_stack([tablas[j][np.minimum(A[:, j], len(tablas[j]) - 1)] for j in range(A.shape[1])])

    def predecir(self, aj, X):
        if isinstance(aj.prep, tuple):
            qt, num, med = aj.prep
            A = qt.transform(X[num].to_numpy(np.float64)); A = np.where(np.isnan(A), med, A)
            return aj.modelo.predict_proba(A)[:, 1]
        prior, _ = aj.modelo
        return GL._sig(prior + self.contribuciones(aj, X).sum(1))


MOTORES = {"lpm": MotorLPM, "logit": MotorLogit, "elastic_net": MotorEN, "cart": MotorCART, "random_forest": MotorRF,
           "lightgbm": MotorLGB, "xgboost": MotorXGB, "catboost": MotorCB, "naive_bayes": MotorNB}


def crear(nombre: str, hilos: int = 4, splines=(), **kw) -> Motor:
    cls = MOTORES[nombre]
    if issubclass(cls, MotorBoost):
        return cls(hilos=hilos, splines=splines, **kw)
    return cls(hilos=hilos, splines=splines)
