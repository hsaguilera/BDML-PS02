r"""Protocolo v7 (fase C del plan maestro): implementa el «Protocolo recomendado» del memo metodológico
(`01_code/python/docs/memo_pronostico.md`, §10, con la tabla por algoritmo de §11) para los siete algoritmos del enunciado.

Uso desde un cuaderno 3x (ver `tools/plantilla_v7.py`):

    from pobreza import protocolo_v7 as PV
    pr = PV.Protocolo("35_lightgbm")            # entorno y presupuesto desde variables de entorno
    pr.paso0_datos(); pr.paso1_referencia(); pr.paso2_seleccion(); pr.paso3_busqueda(); pr.paso4_conjuntos()
    pr.paso5_bogota(); pr.paso6_monotonia(); pr.paso7_desbalance(); pr.paso8_sesgo(); pr.paso9_final()
    pr.paso10_diagnostico(); pr.paso11_listo()

Cada paso guarda su resultado en la caché (`P7_CACHE`/`<pref>`), escribe tablas en `02_outputs/tables/<pref>_*.csv` y
figuras en `02_outputs/figures/<pref>_*.png` (de la raíz de **salida**: en las pruebas, el scratchpad). Nada se sube a
Kaggle. Los submódulos están en `pobreza/p7/`.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .p7 import bogota as BG
from .p7 import busqueda as BU
from .p7 import comparar as CP
from .p7 import figuras as FG
from .p7 import final as FI
from .p7 import metricas as MT
from .p7 import motores as MO
from .p7 import seleccion as SE
from .p7.cvmotor import cv, cv_rk, oob_rf, _limpio
from .p7.datos import Datos7
from .p7.entorno import Entorno, presupuesto

# ------------------------------------------------------------------------------ configuración por algoritmo
ALGORITMOS = {
    "30_lpm": dict(motor="lpm", titulo="Linear Probability Model", seleccion="lasso", busqueda="especificacion",
                   flexible=False, monotonia=False, sel_params={}),
    "31_logit": dict(motor="logit", titulo="Logit", seleccion="lasso", busqueda="especificacion", flexible=False,
                     monotonia=False, sel_params={}),
    "32_elastic_net": dict(motor="elastic_net", titulo="Elastic Net (logística)", seleccion="en", busqueda="rejilla_en",
                           flexible=False, monotonia=False, sel_params={}),
    "33_cart": dict(motor="cart", titulo="CART", seleccion="poda", busqueda="cart", flexible=False, monotonia=False,
                    sel_params={}),
    "34_random_forest": dict(motor="random_forest", titulo="Random Forest", seleccion="rfe_perm_oob", busqueda="oob",
                             flexible=True, monotonia=True,
                             sel_params={"n_estimators": 100, "max_features": 0.1, "min_samples_leaf": 20, "max_samples": 0.5}),
    "35_lightgbm": dict(motor="lightgbm", titulo="LightGBM", seleccion="rfe_shap", busqueda="sobol_tpe", flexible=True,
                        monotonia=True, diag_bogota=True, sel_params={"learning_rate": 0.1}),
    "36_xgboost": dict(motor="xgboost", titulo="XGBoost", seleccion="rfe_shap", busqueda="sobol_tpe", flexible=True,
                       monotonia=True, sel_params={"learning_rate": 0.1}),
    "37_catboost": dict(motor="catboost", titulo="CatBoost", seleccion="rfe_shap", busqueda="sobol_tpe", flexible=True,
                        monotonia=True, sel_params={"learning_rate": 0.1}),
    "38_naive_bayes": dict(motor="naive_bayes", titulo="Naive Bayes", seleccion="nb_adelante", busqueda="sobol_tpe",
                           flexible=False, monotonia=False, sel_params={}),
}


def _version():
    import platform, sklearn, scipy, optuna
    v = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
         "scipy": scipy.__version__, "optuna": optuna.__version__}
    for m in ("lightgbm", "xgboost", "catboost"):
        try:
            v[m] = __import__(m).__version__
        except Exception:
            pass
    return v


class Protocolo:
    def __init__(self, clave: str, ent: Entorno | None = None, P=None, datos: Datos7 | None = None,
                 submuestra: float | None = None, verbose: bool = True):
        self.clave = clave; self.cfg = ALGORITMOS[clave]; self.pref = clave[:2]
        self.ent = ent or Entorno.desde_variables(cuaderno=clave)
        self.P = P or presupuesto()
        sub = submuestra if submuestra is not None else float(os.environ.get("P7_SUBMUESTRA", "0") or 0)
        self.datos = datos or Datos7(self.ent, submuestra=sub or None, verbose=verbose)
        d = self.datos
        spl = d.conj.get("splines_logit") or [c for c in ("log_ingreso_requerido", "ocupados_pc", "educ_max_adultos",
                                                          "jefe_edad", "personas_por_cuarto") if c in d.todas]
        self.motor = MO.crear(self.cfg["motor"], hilos=self.ent.hilos, splines=[c for c in spl if c in d.todas][:5],
                              **({"lr": self.P.lr_busqueda, "paciencia": self.P.paciencia, "max_arboles": self.P.max_arboles}
                                 if self.cfg["motor"] in ("lightgbm", "xgboost", "catboost") else {}))
        self.motor_sel = MO.crear(self.cfg["motor"], hilos=self.ent.hilos, splines=self.motor.splines,
                                  **({"lr": self.cfg["sel_params"].get("learning_rate", 0.1), "paciencia": min(self.P.paciencia, 100),
                                      "max_arboles": self.P.max_arboles} if self.cfg["motor"] in ("lightgbm", "xgboost", "catboost") else {}))
        self.pool, self.pool_fuente = None, None
        ruta_pool = os.environ.get("P7_POOL", "")
        if ruta_pool:
            js = json.loads(Path(ruta_pool).read_text(encoding="utf-8"))
            lista = js.get("variables", js.get("cols", [])) if isinstance(js, dict) else js
            self.pool = [c for c in d.todas if c in set(lista)]; self.pool_fuente = str(ruta_pool)
            if verbose:
                print(f"*** Selección restringida a {len(self.pool)} variables candidatas de {ruta_pool} (P7_POOL) ***")
        sufijo_pool = f"_pool{len(self.pool)}" if self.pool else ""
        self.cache = self.ent.cache_de(f"{self.pref}_{d.fuente}{'_sub' if d.submuestra else ''}_{self.P.nombre}_{d.huella}{sufijo_pool}")
        self.tipo = self.motor.perdida
        self.R = tuple(range(1, self.P.R + 1))
        self.S = tuple(range(1, (self.P.S_conf if self.motor.estocastico else 1) + 1))
        self.res, self.tiempos = {}, {}
        self.verbose = verbose
        self.t0 = time.time()
        self.nb = f"{clave}"
        if verbose:
            print(f"=== {clave} · {self.cfg['titulo']} · presupuesto «{self.P.nombre}» · {self.ent.hilos} hilos · salida {self.ent.salida}")

    # ------------------------------------------------------------------ utilidades
    def _paso(self, nombre, fn):
        ruta = self.cache / f"paso_{nombre}.pkl"
        t = time.time()
        if self.ent.reusar and ruta.exists():
            with open(ruta, "rb") as f:
                out = pickle.load(f)
            if self.verbose:
                print(f"[{nombre}] leído de la caché ({ruta.name})")
        else:
            out = fn()
            with open(ruta, "wb") as f:
                pickle.dump(out, f)
        self.tiempos[nombre] = (time.time() - t) / 60
        self.res[nombre] = out
        return out

    def tabla(self, nombre, df: pd.DataFrame):
        p = Path(self.ent.tablas) / f"{self.pref}_{nombre}.csv"
        df.to_csv(p, index=False)
        return p

    def fig(self, nombre) -> Path:
        return Path(self.ent.figuras) / f"{self.pref}_{nombre}.png"

    def _cols_todas(self):
        """Variables candidatas de la selección: todas las de v7 o, si se fija `P7_POOL` (JSON con una lista o una señal
        `*_LISTO.json` con `variables`), ese subconjunto — p. ej. arrancar XGBoost/CatBoost desde las variables de LightGBM
        cuando el plazo no permite su propia selección completa (desviación que queda declarada en la señal)."""
        return list(self.pool) if self.pool else list(self.datos.todas)

    def _params_sel(self):
        return {**self.motor_sel.base(), **{k: v for k, v in self.cfg["sel_params"].items() if k != "learning_rate"}}

    def metricas(self, oof, r=1, completo=True):
        d = self.datos
        return MT.metricas_particion(d.y, oof, d.folds(r), d.mT, B=self.P.B_boot, dominio=d.dominio,
                                     pesos_dominio=d.prevalencia()["pesos_dominio"], completo=completo)

    # ------------------------------------------------------------------ Paso 0: datos, integridad, particiones
    def paso0_datos(self):
        d = self.datos
        ver = d.verificacion
        pi = d.prevalencia()
        out = {"verificacion": ver, "prevalencia": {k: v for k, v in pi.items() if k != "pesos_dominio"},
               "bogota": d.resumen_bogota(), "fuente": d.fuente, "n_variables": len(d.todas),
               "n_preguntas": len(d.MP.unidades(d.todas)), "mapa_fuente": d.mapa_fuente, "versiones": _version(),
               "particiones": {f"r{r}": d.particiones_df[f"fold_r{r}"].value_counts().sort_index().tolist() for r in (1, 2, 3, 4)}}
        self.res["datos"] = out
        self.tabla("mapa_preguntas", d.tabla_mapa)
        assert ver["ok"], f"verificación del Paso 0 fallida: {ver}"
        return out

    # ------------------------------------------------------------------ Paso 1: referencia con todas las variables
    def paso1_referencia(self):
        def f():
            m = self.motor_sel if self.motor.parada_temprana else self.motor
            p = self._params_sel()
            r = cv(m, self.datos, p, self._cols_todas(), 1.0, 1, (1,), cache_dir=self.cache, reusar=self.ent.reusar, verbose=self.verbose,
                   etiqueta="referencia")
            return {"oof": r.oof, "metricas": self.metricas(r.oof), "tiempo_s": r.tiempo_s, "params": p,
                    "iteraciones": r.iteraciones}
        return self._paso("referencia", f)

    # ------------------------------------------------------------------ Paso 2: selección de variables
    def paso2_seleccion(self):
        d, P, cfg = self.datos, self.P, self.cfg
        cols = self._cols_todas()

        def f():
            out = {}
            if cfg["seleccion"] == "lasso":
                s = SE.seleccion_lineal(self.motor, d, cols, "lpm" if cfg["motor"] == "lpm" else "logit", 1,
                                        max_activas=400, verbose=self.verbose)
                out["puntos"] = s["puntos"]; out["rutas"] = s["rutas"]
            elif cfg["seleccion"] == "en":
                cam = SE.caminos_en(self.motor, d, cols, P.en_alfas, 1, P.en_nlambda, getattr(P, "en_razon", 1e-4),
                                    verbose=self.verbose, max_activas=getattr(P, "en_max_activas", None))
                te = SE.tabla_en(d, cam, 1)
                i = te.LogLoss.idxmin(); a_opt = te.loc[i, "l1_ratio"]
                c = cam[a_opt]
                pts = {}
                for l, lam in enumerate(c["lambdas"]):
                    s_ = c["oof"][:, l]
                    if np.isnan(s_[d.mT]).any():
                        continue
                    pts[f"λ = {lam:.2e}"] = {"oof": s_.astype(np.float32), "cols_pliegues": [cl[l] for cl in c["cols"]], "final": None,
                                             "orden": l, "lam": float(lam), "l1_ratio": float(a_opt)}
                out.update({"puntos": pts, "tabla_en": te, "alfa": float(a_opt),
                            "caminos": {a: {"lambdas": c_["lambdas"], "df": c_["df"]} for a, c_ in cam.items()}})
                del cam
            elif cfg["seleccion"] == "poda":
                out["puntos"] = {}                                  # la poda se elige junto con el crecimiento (paso 3)
            elif cfg["seleccion"] in ("rfe_shap", "rfe_perm_oob"):
                s = SE.rfe_preguntas(self.motor_sel, d, cols, self._params_sel(), 1, P.rfe_frac, n_shap=P.n_shap,
                                     n_eval_perm=min(P.n_shap, 10000), verbose=self.verbose)
                out["puntos"] = s["puntos"]; out["ranking_final"] = s["ranking_final"]
            elif cfg["seleccion"] == "nb_adelante":
                s = SE.seleccion_nb(self.motor, d, cols, self.motor.base(), 1, P.nb_M, P.nb_kmax, verbose=self.verbose)
                out["puntos"] = s["puntos"]
            if out["puntos"]:
                tc = SE.tabla_curva(d, out["puntos"], 1, self.tipo)
                out["curva"] = tc
                if P.cvc and len(out["puntos"]) > 1:
                    out["cvc"] = SE.cvc_curva(d, out["puntos"], 1, self.tipo, B=P.B_cvc)
            # stability selection CPSS (Shah y Samworth, 2013) para reportar las variables estables (EN y LPM; memo §11.1)
            if cfg["seleccion"] == "en" and P.cpss_B:
                te = out["tabla_en"]; a_ = out["alfa"]
                lam1 = float(te[(te.l1_ratio == a_) & te.lambda_1se].lam.iloc[0])
                out["cpss"] = SE.cpss(self.motor, d, cols, {"l1_ratio": max(a_, 0.05), "lam": lam1}, B=P.cpss_B, verbose=self.verbose)
            elif cfg["motor"] == "lpm" and P.cpss_B:
                rt = out["rutas"][-1]
                out["cpss"] = SE.cpss(self.motor, d, cols, {"selector": "lpm", "lam": float(rt["lambdas"][rt["i_1se"]])},
                                      B=P.cpss_B, verbose=self.verbose)
            return out
        out = self._paso("seleccion", f)
        if "curva" in out:
            tc = out["curva"]
            self.tabla("curva_parsimonia", tc)
            pts = out["puntos"]
            mejor = tc.loc[tc.perdida.idxmin(), "punto"]; pars = tc.loc[tc.parsimonioso, "punto"].iloc[0]
            self.sel = {"mejor": mejor, "parsimonioso": pars, "puntos": pts}
            for nom in (mejor, pars):
                if pts[nom].get("final") is None and cfg["seleccion"] == "en":
                    pts[nom]["final"] = cols
            out["estabilidad"] = SE.estabilidad(d, pts[pars]["cols_pliegues"])
            FG.parsimonia(tc, f"{cfg['titulo']}: curva de parsimonia (partición 1, selección dentro del pliegue)",
                          self.fig("curva_parsimonia"), self.tipo)
            if "cpss" in out:
                cp_ = out["cpss"]
                self.tabla("cpss", pd.DataFrame({"variable": cp_["frecuencias"].index, "frecuencia": cp_["frecuencias"].values,
                                                 "estable": cp_["frecuencias"].index.isin(cp_["estables"])}))
                if self.verbose:
                    print(f"CPSS: {len(cp_['estables'])} variables estables (τ = {cp_['tau']}), q̄ = {cp_['q_medio']:.1f}, p = {cp_['p']}, "
                          f"cota E(V) ≤ {cp_['cota_EV']:.2f}; preguntas estables: {cp_['preguntas_estables']}")
            FG.estabilidad(pts[pars]["cols_pliegues"] + ([pts[pars]["final"]] if pts[pars].get("final") else []), d.MP,
                           f"{cfg['titulo']}: preguntas del conjunto parsimonioso en cada pliegue y en todo el entrenamiento",
                           self.fig("estabilidad_seleccion"), con_final=bool(pts[pars].get("final")))
        return out

    # ------------------------------------------------------------------ conjuntos de variables
    def conjuntos(self) -> dict:
        d = self.datos; cfg = self.cfg; out = {"todas": list(d.todas), "corta": list(d.corta)}
        sel = self.res.get("seleccion", {})
        if cfg["seleccion"] in ("en", "poda"):
            out["mejor"] = out["pars"] = self._cols_todas()            # la penalización / poda selecciona
        elif "curva" in sel:
            pts = sel["puntos"]; tc = sel["curva"]
            mejor = tc.loc[tc.perdida.idxmin(), "punto"]; pars = tc.loc[tc.parsimonioso, "punto"].iloc[0]
            out["mejor"] = list(pts[mejor]["final"]); out["pars"] = list(pts[pars]["final"])
        else:
            out["mejor"] = out["pars"] = self._cols_todas()
        return out

    # ------------------------------------------------------------------ Paso 3: hiperparámetros
    def paso3_busqueda(self):
        d, P, cfg, m = self.datos, self.P, self.cfg, self.motor
        cj = self.conjuntos()
        cols = cj["pars"]

        def f():
            out = {"cols": cols}
            if cfg["busqueda"] == "sobol_tpe":
                fijos = {"learning_rate": P.lr_busqueda} if m.parada_temprana else {}
                if cfg["motor"] == "naive_bayes":
                    fijos = {"tipo": "categorico"}
                b = BU.Busqueda(m, d, cols, P, f"{self.pref}_busqueda", fijos=fijos, cache_dir=self.cache, verbose=self.verbose,
                                reusar=self.ent.reusar)
                b.ejecutar(P.n_sobol, P.n_tpe, P.timeout_busqueda_min, encolar=[m.base()])
                configs = b.mejores(P.top_confirmar)
                if cfg["motor"] == "naive_bayes":
                    configs.append({**configs[0], "tipo": "gaussiano"})
                out.update({"tabla": b.tabla(), "importancia": b.importancia(), "sesgo_tt": b.sesgo_tt(),
                            "bordes": BU.bordes(configs[0], b.dists)})
            elif cfg["busqueda"] == "oob":
                b = BU.Busqueda(m, d, cols, P, f"{self.pref}_busqueda_oob", fijos={"n_estimators": P.rf_arboles_busqueda},
                                cache_dir=self.cache, oob=True, verbose=self.verbose, reusar=self.ent.reusar)
                b.ejecutar(P.rf_sobol, max(0, P.rf_ensayos - P.rf_sobol), P.timeout_busqueda_min,
                           encolar=[{k: v for k, v in m.base().items() if k in b.dists}])
                configs = b.mejores(P.top_confirmar)
                out.update({"tabla": b.tabla(), "importancia": b.importancia(), "sesgo_tt": {"nota": "búsqueda OOB: sin pliegues"},
                            "bordes": BU.bordes(configs[0], b.dists)})
            elif cfg["busqueda"] == "rejilla_en":
                te = self.res["seleccion"]["tabla_en"].sort_values("LogLoss")
                configs = [{"l1_ratio": float(r.l1_ratio), "lam": float(r.lam)} for r in te.head(P.top_confirmar).itertuples()]
                E = np.array(te.LogLoss_pliegues.tolist())
                out.update({"tabla": te, "sesgo_tt": {"sesgo_perdida": CP.tibshirani_tibshirani(E)["sesgo"], "n_config": len(E)},
                            "importancia": pd.Series(dtype=float)})
            elif cfg["busqueda"] == "cart":
                b = BU.BusquedaCART(m, d, self._cols_todas(), P, f"{self.pref}_busqueda_cart", cache_dir=self.cache,
                                    verbose=self.verbose, reusar=self.ent.reusar, hilos=self.ent.hilos)
                b.ejecutar(P.cart_sobol, P.cart_tpe, P.timeout_busqueda_min, encolar=[{k: v for k, v in m.base().items() if k in b.dists}])
                configs = b.mejores(P.top_confirmar, "min") + b.mejores(1, "1se")
                out.update({"tabla": b.tabla(), "importancia": b.importancia(), "sesgo_tt": b.sesgo_tt(),
                            "bordes": BU.bordes(configs[0], b.dists), "config_1se": configs[-1]})
                out["cols"] = self._cols_todas()
            elif cfg["busqueda"] == "especificacion":
                configs = [{"splines": False}, {"splines": True}]
                out.update({"tabla": pd.DataFrame(configs), "importancia": pd.Series(dtype=float), "sesgo_tt": {"n_config": 2}})
            et = {i: f"config {i}" for i in range(len(configs))}
            if cfg["busqueda"] == "cart":
                et[len(configs) - 1] = "1-EE"
            conf = BU.confirmar(m, d, out["cols"], configs, self.R, self.S, 1.0, cache_dir=self.cache, reusar=self.ent.reusar,
                                verbose=self.verbose, etiquetas=et)
            out.update({"confirmacion": conf["tabla"], "comparaciones": conf["comparaciones"], "elegido": conf["elegido"],
                        "rk_elegido": conf["rk"][conf["etiqueta"]], "configs": configs})
            return out
        out = self._paso("busqueda", f)
        if isinstance(out.get("tabla"), pd.DataFrame):
            self.tabla("busqueda_ensayos", out["tabla"])
        self.tabla("busqueda_confirmacion", out["confirmacion"])
        if cfg["busqueda"] in ("sobol_tpe", "oob", "cart") and "objetivo" in out["tabla"]:
            FG.busqueda(out["tabla"], out["importancia"], f"{cfg['titulo']}: búsqueda Sobol → TPE con poda por la mediana",
                        self.fig("busqueda"), "log-loss" if self.tipo == "logloss" else "Brier")
        self.params = out["elegido"]
        return out

    # ------------------------------------------------------------------ Paso 4: conjuntos (científico) con los hiperparámetros elegidos
    def paso4_conjuntos(self):
        d, P = self.datos, self.P
        cj = self.conjuntos(); p = self.res["busqueda"]["elegido"]

        def f():
            var = {}
            for nom in ("pars", "mejor", "todas", "corta"):
                c = cj[nom]
                if any(set(c) == set(v[1]) for v in var.values()):
                    continue
                var[nom] = (p, c, 1.0)
            return BU.comparar_cientifico(self.motor, d, None, var, self.R, self.S, self.cache, self.ent.reusar, "pars", self.verbose)
        out = self._paso("conjuntos", f)
        t = out["tabla"].copy(); t["n_vars"] = [len(cj[v]) for v in t.variante]; t["n_preguntas"] = [d.MP.costo(cj[v]) for v in t.variante]
        self.tabla("conjuntos", t); self.tabla("conjuntos_comparaciones", out["comparaciones"])
        # principal: menor pérdida media; dentro de la ROPE, el más parsimonioso
        col = f"{self.tipo}_media"
        best = t[col].min(); cand = t[t[col] <= best + MT.rope_de(self.tipo)].sort_values(["n_preguntas", "n_vars"])
        self.principal = cand.iloc[0].variante
        self.cols_principal = cj[self.principal]
        if self.verbose:
            print(f"Conjunto principal: «{self.principal}» ({len(self.cols_principal)} variables, {d.MP.costo(self.cols_principal)} preguntas)")
        return out

    # ------------------------------------------------------------------ Paso 5: Bogotá
    def paso5_bogota(self):
        d, P, cfg = self.datos, self.P, self.cfg
        lams = P.lambdas_flexibles if cfg["flexible"] else P.lambdas_rigidos

        def f():
            var = {f"λ_B = {l:g}": (self.params, self.cols_principal, float(l)) for l in sorted(lams, reverse=True)}
            out = BU.comparar_cientifico(self.motor, d, None, var, self.R, self.S, self.cache, self.ent.reusar, "λ_B = 1", self.verbose)
            r1 = {nom: out["resultados"][nom][1].oof for nom in var}
            out["transferencia"] = BG.transferencia(d, r1["λ_B = 0"], r1["λ_B = 1"]) if "λ_B = 0" in r1 else {}
            if cfg.get("diag_bogota"):
                out["dominio"] = BG.diagnostico(d, self.ent.hilos)
            out = {k: v for k, v in out.items() if k != "resultados"} | {"oof_r1": r1}
            return out
        out = self._paso("bogota", f)
        t = out["tabla"]; col = f"{self.tipo}_media"
        self.tabla("bogota", t); self.tabla("bogota_comparaciones", out["comparaciones"])
        self.lam_B = float(t.loc[t[col].idxmin(), "lam_B"])
        comp = out["comparaciones"]
        if len(comp) and (comp[f"d{self.tipo}_P_rope"] > 0.95).all():
            nota = "equivalentes en la práctica (P(ROPE) > 0,95)"
        else:
            nota = "regla de Bayes: menor pérdida media"
        self.res["bogota"]["decision"] = {"lam_B": self.lam_B, "nota": nota}
        FG.comparaciones(comp, f"{cfg['titulo']}: peso de Bogotá en el entrenamiento (métricas sólo fuera de Bogotá)", self.fig("bogota"), self.tipo)
        if self.verbose:
            print(f"λ_B elegido = {self.lam_B} ({nota})")
        return out

    # ------------------------------------------------------------------ Paso 6: restricciones monótonas
    def paso6_monotonia(self):
        d = self.datos
        mon = d.monotonas(self.cols_principal)
        if not (self.cfg["monotonia"] and self.P.monotonia and mon):
            self.res["monotonia"] = {"aplica": False, "adoptada": False}
            return self.res["monotonia"]

        def f():
            var = {"libre": (self.params, self.cols_principal, self.lam_B),
                   "monótona": ({**self.params, "monotonas": mon}, self.cols_principal, self.lam_B)}
            out = BU.comparar_cientifico(self.motor, d, None, var, self.R, self.S, self.cache, self.ent.reusar, "libre", self.verbose)
            c = out["comparaciones"].iloc[0].to_dict()
            out = {k: v for k, v in out.items() if k != "resultados"}
            out.update({"aplica": True, "restricciones": mon, "adoptada": CP.no_inferior(c, self.tipo), "comparacion": c})
            return out
        out = self._paso("monotonia", f)
        self.tabla("monotonia", pd.DataFrame([out["comparacion"]]))
        if out["adoptada"]:
            self.params = {**self.params, "monotonas": out["restricciones"]}
        if self.verbose:
            print(f"Restricciones monótonas {out['restricciones']}: {'ADOPTADAS' if out['adoptada'] else 'no adoptadas'} (no inferioridad)")
        return out

    # ------------------------------------------------------------------ Paso 7: desbalance (sensibilidad)
    def paso7_desbalance(self):
        d = self.datos
        if not self.P.desbalance:
            return {}

        def f():
            razon = float((d.y[d.mT] == 0).sum() / (d.y[d.mT] == 1).sum())
            base = cv(self.motor, d, self.params, self.cols_principal, self.lam_B, 1, (1,), cache_dir=self.cache, reusar=self.ent.reusar)
            # pesos balanceados w1/w0 = n0/n1: se implementan como pesos de observación
            pes = _cv_pesos_clase(self.motor, d, self.params, self.cols_principal, self.lam_B, razon)
            folds = d.folds(1)
            ma = MT.metricas_rk(d.y, {1: pes}, {1: folds}, d.mT, self.tipo); mb = MT.metricas_rk(d.y, {1: base.oof}, {1: folds}, d.mT, self.tipo)
            c = CP.comparar_rk(ma, mb, self.tipo, 5, "pesos balanceados", "sin pesos")
            m1, m0 = self.metricas(pes, completo=False), self.metricas(base.oof, completo=False)
            return {"razon": razon, "comparacion": c, "dBrier": m1["Brier"] - m0["Brier"], "dACE": m1["ACE"] - m0["ACE"],
                    "F1_pesos": m1["F1_anidado"], "F1_sin": m0["F1_anidado"]}
        out = self._paso("desbalance", f)
        self.tabla("desbalance", pd.DataFrame([{**out["comparacion"], "dBrier": out["dBrier"], "dACE": out["dACE"]}]))
        return out

    # ------------------------------------------------------------------ Paso 8: sesgo de selección y re-evaluación
    def paso8_sesgo(self):
        d, P = self.datos, self.P

        def f():
            out = {"tibshirani": self.res["busqueda"].get("sesgo_tt", {})}
            if P.reevaluar:
                r4 = cv(self.motor, d, self.params, self.cols_principal, self.lam_B, 4, self.S, cache_dir=self.cache, reusar=self.ent.reusar)
                m4 = self.metricas(r4.oof, 4, completo=False)
                rk = self.res["busqueda"]["rk_elegido"]
                out["reevaluacion_r4"] = {"F1": m4["F1_anidado"], "perdida": m4["LogLoss"] if self.tipo == "logloss" else m4["Brier"],
                                          "F1_confirmacion": float(np.mean(list(rk["F1_por_particion"].values()))),
                                          "optimismo_F1": float(np.mean(list(rk["F1_por_particion"].values())) - m4["F1_anidado"])}
            if P.loco and self.cfg["motor"] not in getattr(P, "loco_excluir", ()) and len({d.bloque(c) for c in self.cols_principal}) > 1:
                # en boosting, LOCO con la tasa de la selección (0,1; paciencia 100): es un diagnóstico de importancia por
                # bloque, no una estimación del nivel, y así cuesta la mitad
                if self.motor.parada_temprana:
                    m_loco, p_loco = self.motor_sel, {**self.params, "learning_rate": self.motor_sel.lr}
                elif self.motor.oob:                     # RF: bosques baratos de la selección (100 árboles), mismo diagnóstico
                    m_loco, p_loco = self.motor, {**self.params, **self.cfg["sel_params"]}
                else:
                    m_loco, p_loco = self.motor, self.params
                out["loco"] = SE.loco_bloques(m_loco, d, p_loco, self.cols_principal, 1, (1,), self.lam_B, self.cache,
                                              self.ent.reusar, self.verbose, self.tipo)
            return out
        out = self._paso("sesgo", f)
        if "loco" in out:
            self.tabla("loco_bloques", out["loco"])
        return out

    # ------------------------------------------------------------------ Paso 9: modelos finales y CSV
    def paso9_final(self):
        d, P, cfg = self.datos, self.P, self.cfg
        cj = self.conjuntos()
        sem_cv = self.S

        def f():
            out = {}
            versiones = {"principal": (self.cols_principal, self.params, self.lam_B)}
            pars_cols = cj["pars"]; pars_par = self.params
            if cfg["seleccion"] == "en":
                pp = self.res["seleccion"]["puntos"][self.sel["parsimonioso"]]
                pars_par = {**self.params, "l1_ratio": pp["l1_ratio"], "lam": pp["lam"]}
            if cfg["busqueda"] == "cart":
                pars_par = self.res["busqueda"].get("config_1se", self.params)
            if set(pars_cols) != set(self.cols_principal) or pars_par != self.params:
                versiones["parsimoniosa"] = (pars_cols, pars_par, self.lam_B)
            versiones["encuesta_corta"] = (cj["corta"], self.params, 1.0)
            for v, (c, p, lb) in versiones.items():
                out[v] = FI.modelo_final(self.motor, d, p, c, lb, P, self.ent, f"{cfg['titulo']} v7 {v} ({len(c)} vars)", v,
                                         sem_cv, self.nb, self.cache, reusar=self.ent.reusar, verbose=self.verbose)
                out[v].pop("res_cv", None)
            return out
        out = self._paso("final", f)
        filas = []
        for v, o in out.items():
            m = self.metricas(o["oof"])
            ce = o.get("cols_efectivas", o["cols"])
            filas.append({"version": v, "n_vars": len(ce), "n_preguntas": d.MP.costo(ce), "n_candidatas": len(o["cols"]),
                          "archivo": Path(o["archivo"]).name,
                          **{k: m[k] for k in ("F1_anidado", "EE_psi", "LogLoss", "Brier", "AUC_ROC", "AUC_PR", "ACE", "precision",
                                               "exhaustividad", "MCC")},
                          "IC95_F1": m["IC95_F1"], **{f"diag_{k}": v_ for k, v_ in o["diagnostico"].items() if not isinstance(v_, (list, dict))}})
        self.tabla("finales", pd.DataFrame(filas))
        return out

    # ------------------------------------------------------------------ Paso 10: diagnóstico y figuras
    def paso10_diagnostico(self):
        d, cfg = self.datos, self.cfg
        fin = self.res["final"]["principal"]; oof = fin["oof"]; mT = d.mT; y = d.y
        t = fin["umbral"]["umbral"]
        tit = f"{cfg['titulo']} (v7, principal)"
        FG.umbral_y_pr(y[mT], oof[mT], t, f"{tit}: umbral y curva precisión–exhaustividad", self.fig("umbral_pr"))
        FG.calibracion(y[mT], np.clip(oof[mT], 0, 1), f"{tit}: calibración (OOF, sin Bogotá)", self.fig("calibracion"))
        dis = FG.disyuntiva(y[mT], oof[mT], f"{tit}: disyuntiva filtración / subcobertura", self.fig("disyuntiva"))
        self.tabla("disyuntiva", dis)
        an = MT.f1_anidado(y, oof, d.folds(1), mT)
        fd = MT.f1_por_dominio(y[mT], an["yhat"][mT], d.dominio[mT])
        self.tabla("f1_dominio", fd)
        FG.f1_dominio(fd, an["F1"], f"{tit}: F1 anidado por dominio", self.fig("f1_dominio"))
        cal = MT.calibracion_cruzada(y, oof, d.folds(1), mT)
        tcal = pd.DataFrame(cal).T.reset_index().rename(columns={"index": "calibracion"})
        self.tabla("calibracion", tcal)
        self.res["calibracion"] = {k: {kk: vv for kk, vv in v.items()} for k, v in cal.items()}
        imp = self._importancia_preguntas(fin)
        self.tabla("importancia_preguntas", imp)
        FG.importancia_preguntas(imp, d.MP, f"{tit}: importancia por pregunta (permutación agrupada en validación, Δ log-loss)",
                                 self.fig("importancia_preguntas"))
        aj = self._ajuste_diagnostico(fin)
        top_vars = self._top_variables(imp, fin.get("cols_efectivas", fin["cols"]), aj, cols_modelo=fin["cols"])
        if aj is not None and top_vars:
            tr0, va0 = d.folds(1)[0]
            Xs = d.matriz(fin["cols"], va0[mT[va0]], train_idx=tr0)              # hogares de 𝒯 del pliegue retenido
            FG.dependencia_parcial(self.motor, aj, Xs, top_vars, f"{tit}: dependencia parcial de las variables principales",
                                   self.fig("dependencia_parcial"), n=self.P.pd_n)
        return {"importancia": imp, "f1_dominio": fd, "disyuntiva": dis, "calibracion": tcal}

    def _params_diagnostico(self, fin):
        """Hiperparámetros de los ajustes auxiliares del diagnóstico (importancia por pregunta y dependencia parcial).
        En Random Forest se usan los del modelo final con el número de árboles de la búsqueda (`rf_arboles_busqueda`):
        la importancia por permutación y la dependencia parcial de un bosque se estabilizan con pocos cientos de árboles
        (Probst y Boulesteix, 2018), y con 1.000 árboles × 5 pliegues el paso tardaba más de 5 horas. El modelo final y su
        CSV no cambian; la desviación queda declarada aquí."""
        p = dict(fin["params"])
        if getattr(self.motor, "oob", False) and "n_estimators" in p:
            p["n_estimators"] = int(min(p["n_estimators"], getattr(self.P, "rf_arboles_busqueda", 200)))
        return p

    def _ajuste_diagnostico(self, fin):
        d = self.datos; tr, va = d.folds(1)[0]
        try:
            X = d.matriz(fin["cols"], tr, train_idx=tr)
            if not len(X.columns):
                return None
            p = self._params_diagnostico(fin)
            if self.motor.parada_temprana:
                from .p7.cvmotor import division_es
                a, b = division_es(d.y[tr], d.mT[tr], 1)
                return self.motor.ajustar(X.iloc[a], d.y[tr][a], None, p, 1, Xes=X.iloc[b], yes=d.y[tr][b])
            return self.motor.ajustar(X, d.y[tr], None, p, 1)
        except Exception as e:  # pragma: no cover
            print("dependencia parcial no disponible:", e); return None

    def _importancia_preguntas(self, fin, max_val=8000) -> pd.DataFrame:
        """Permutación agrupada por pregunta en los pliegues de validación (modelos de la partición 1, 1 semilla):
        aumento de la log-loss en 𝒯. Es agnóstica al modelo y honesta (Hooker, Mentch y Zhou, 2021, discuten sus
        límites con variables correlacionadas: por eso se permuta el grupo completo de variables de cada pregunta)."""
        d = self.datos; cols = fin["cols"]; p = self._params_diagnostico(fin)
        grupos = d.MP.grupos(cols)
        filas = []
        for k, (tr, va) in enumerate(d.folds(1)):
            X = d.matriz(cols, tr, train_idx=tr)
            if self.motor.parada_temprana:
                from .p7.cvmotor import division_es
                a, b = division_es(d.y[tr], d.mT[tr], 1000 + k)
                aj = self.motor.ajustar(X.iloc[a], d.y[tr][a], None, p, 1, Xes=X.iloc[b], yes=d.y[tr][b])
            else:
                aj = self.motor.ajustar(X, d.y[tr], None, p, 1)
            vaT = va[d.mT[va]]
            rng = np.random.default_rng(k); ev = np.sort(rng.choice(vaT, min(max_val, len(vaT)), replace=False))
            Xv = d.matriz(cols, ev, train_idx=tr); yv = d.y[ev]
            l0 = MT.logloss(yv, self.motor.predecir(aj, Xv))
            perm = rng.permutation(len(ev))
            for q, cs in grupos.items():
                Z = Xv.copy()
                for c in cs:
                    Z[c] = Xv[c].values[perm]
                filas.append({"pregunta": q, "pliegue": k, "delta": MT.logloss(yv, self.motor.predecir(aj, Z)) - l0})
        t = pd.DataFrame(filas).groupby("pregunta").delta.agg(["mean", "std"]).reset_index()
        t.columns = ["pregunta", "media", "ee"]; t["ee"] = t["ee"] / np.sqrt(len(d.folds(1)))
        t["etiqueta"] = [d.MP.etiqueta(q) for q in t.pregunta]
        return t.sort_values("media", ascending=False).reset_index(drop=True)

    def _top_variables(self, imp, cols, aj=None, n=6, n_q=8, max_cand=40, cols_modelo=None):
        """Variables para la dependencia parcial: entre las de las preguntas más importantes, las de mayor importancia por
        permutación individual en el pliegue de validación 0 (modelo del pliegue)."""
        d = self.datos; g = d.MP.grupos(cols)
        cols_modelo = list(cols_modelo or cols)
        intr = self.motor.importancia_intrinseca(aj, cols_modelo) if aj is not None else None
        cand = []
        for q in list(imp.pregunta[:n_q]) + ["_gratis"]:
            vs = [c for c in (g.get(q, []) if q != "_gratis" else [c for c in cols if not d.MP.mapa.get(c)])
                  if c not in d.categoricas and c not in cand]
            if intr is not None:
                vs = sorted(vs, key=lambda c: -float(intr.get(c, 0.0)))
            cand += vs[:5]                                   # a lo sumo 5 por pregunta: candidatas repartidas
        cand = cand[:max_cand]
        if aj is None or not cand:
            return cand[:n]
        tr, va = d.folds(1)[0]
        vaT = va[d.mT[va]]; ev = np.sort(np.random.default_rng(0).choice(vaT, min(6000, len(vaT)), replace=False))
        Xv = d.matriz(cols_modelo, ev, train_idx=tr); yv = d.y[ev]
        l0 = MT.logloss(yv, self.motor.predecir(aj, Xv)); perm = np.random.default_rng(1).permutation(len(ev))
        delta = {}
        for c in cand:
            Z = Xv.copy(); Z[c] = Xv[c].values[perm]
            delta[c] = MT.logloss(yv, self.motor.predecir(aj, Z)) - l0
        return [c for c, _ in sorted(delta.items(), key=lambda t: -t[1])][:n]

    # ------------------------------------------------------------------ Paso 11: resumen y señal de listo
    def paso11_listo(self) -> dict:
        d, cfg, P = self.datos, self.cfg, self.P
        fin = self.res["final"]; fp = fin["principal"]
        m = self.metricas(fp["oof"])
        rk = self.res["busqueda"]["rk_elegido"]
        f1_part = rk["F1_por_particion"]
        sel = self.res.get("seleccion", {})
        est = sel.get("estabilidad", {})
        ses = self.res.get("sesgo", {})
        hp = {k: v for k, v in fp["params"].items() if not isinstance(v, (list, dict))}
        desc = (f"{cfg['titulo']} (clasificación, {self.motor.corto}); "
                + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in list(hp.items())[:6])
                + f"; {len(fp.get('cols_efectivas', fp['cols']))} variables ({d.MP.costo(fp.get('cols_efectivas', fp['cols']))} preguntas); λ_B {fp['lam_B']:g}; umbral "
                  f"{fp['diagnostico']['regla']} {fp['diagnostico']['umbral_test']:.3f}; F1 CV anidado sin Bogotá {m['F1_anidado']:.4f}; datos v7")
        otras = []
        for v, o in fin.items():
            if v == "principal":
                continue
            mv = self.metricas(o["oof"], completo=False)
            otras.append({"version": v, "ruta_csv": os.path.abspath(o["archivo"]), "archivo": Path(o["archivo"]).name,
                          "n_vars": len(o.get("cols_efectivas", o["cols"])), "n_preguntas": d.MP.costo(o.get("cols_efectivas", o["cols"])),
                          "F1_anidado_sinBOG": mv["F1_anidado"],
                          "EE_psi": mv["EE_psi"], "AUC": mv["AUC_ROC"], "umbral": o["diagnostico"]["umbral_test"],
                          "tasa_predicha_test": o["diagnostico"]["tasa_predicha_test"], "lam_B": o["lam_B"]})
        tt = self.res["busqueda"].get("sesgo_tt", {})
        listo = {
            "modelo": cfg["titulo"], "algoritmo_enunciado": self.motor.familia, "datos": f"v7 ({d.fuente})", "version": "principal",
            "ruta_csv": os.path.abspath(fp["archivo"]), "archivo": Path(fp["archivo"]).name, "descripcion_kaggle": desc,
            "n_vars": len(fp.get("cols_efectivas", fp["cols"])), "n_preguntas": d.MP.costo(fp.get("cols_efectivas", fp["cols"])),
            "conjunto": self.principal, "variables": fp.get("cols_efectivas", fp["cols"]), "n_candidatas": len(fp["cols"]), "hiperparametros": fp["params"], "peso_bogota": fp["lam_B"],
            "desbalance": "ninguno (sólo umbral; sensibilidad con pesos balanceados reportada)",
            "umbral": fp["diagnostico"]["umbral_test"], "regla_umbral": fp["diagnostico"]["regla"], "umbral_por_dominio": False,
            "F1_anidado_sinBOG": m["F1_anidado"], "IC95": m["IC95_F1"], "EE_psi": m["EE_psi"], "EE_bootstrap": m["bootstrap_EE_boot"],
            "EE_pliegues": m["EE_pliegues"], "F1_particiones": f1_part,
            "F1_medio_particiones": float(np.mean(list(f1_part.values()))),
            "EE_entre_particiones": float(np.std(list(f1_part.values()), ddof=1)) if len(f1_part) > 1 else None,
            "F1_segunda_particion": f1_part.get(2), "optimismo": ses.get("reevaluacion_r4", {}).get("optimismo_F1"),
            "F1_reevaluacion_r4": ses.get("reevaluacion_r4", {}).get("F1"), "sesgo_tibshirani": tt,
            "AUC": m["AUC_ROC"], "AUC_PR": m["AUC_PR"], "LogLoss": m["LogLoss"], "Brier": m["Brier"], "ACE": m["ACE"],
            "precision": m["precision"], "exhaustividad": m["exhaustividad"], "MCC": m["MCC"],
            "F1_postestratificado": m.get("F1_postestratificado"), "meseta_umbral": m["meseta"],
            "tasa_predicha_test": fp["diagnostico"]["tasa_predicha_test"], "tasa_predicha_oof_sinBOG": fp["umbral"]["q_hat"],
            "tasa_pobres_train_sinBOG": float(d.y[d.mT].mean()), "prevalencia_esperada_test": d.prevalencia()["prevalencia_test"],
            "media_prob_test": fp["diagnostico"]["media_prob_test"],
            "tasa_coherente": bool(fp["diagnostico"]["tasa_ok"]), "diagnostico_umbral": fp["diagnostico"],
            "estabilidad": {k: v for k, v in est.items() if k != "frecuencia_preguntas"},
            "calibracion_dentro_del_pliegue": self.res.get("calibracion"),
            "bogota": self.res.get("bogota", {}).get("decision"), "monotonia": {k: self.res.get("monotonia", {}).get(k) for k in ("aplica", "adoptada", "restricciones")},
            "otras_versiones": otras, "verificado": "id,pobre; 66.168 filas; ids y orden de sample_submission; valores 0/1",
            "submuestra_prueba": d.submuestra, "presupuesto": P.nombre, "hilos": self.ent.hilos,
            "tiempos_min": self.tiempos, "tiempo_total_min": (time.time() - self.t0) / 60, "versiones_paquetes": _version(),
            "registro_slug": fp.get("slug"), "mapa_preguntas": d.mapa_fuente,
            "pool_candidatas": {"fuente": self.pool_fuente, "n": len(self.pool)} if self.pool else None,
        }
        p = FI.escribir_listo(self.ent, self.pref, listo)
        if self.verbose:
            print(f"Señal escrita: {p}")
        return listo


def _cv_pesos_clase(motor, d, params, cols, lam_B, razon):
    """OOF de la partición 1 con pesos de clase w₁/w₀ = n₀/n₁ (sensibilidad del memo §6.2; el umbral se re-optimiza
    en las OOF, así que sólo cambia lo que los pesos alteran en el ajuste)."""
    from .p7.cvmotor import division_es
    y = d.y; oof = np.full(len(y), np.nan)
    for k, (tr, va) in enumerate(d.folds(1)):
        tr_fit = tr if lam_B > 0 else tr[d.mT[tr]]
        w = np.where(y[tr_fit] == 1, razon, 1.0) * (np.where(d.bog[tr_fit], lam_B, 1.0) if lam_B not in (0, 1) else 1.0)
        X = d.matriz(cols, tr_fit, train_idx=tr); Xv = d.matriz(cols, va, train_idx=tr)
        if motor.parada_temprana:
            a, b = division_es(y[tr_fit], d.mT[tr_fit], 1000 + 10 + k)          # la misma división que `cv` (semilla 1, r = 1)
            aj = motor.ajustar(X.iloc[a], y[tr_fit][a], w[a], params, 1, Xes=X.iloc[b], yes=y[tr_fit][b])
        else:
            aj = motor.ajustar(X, y[tr_fit], w, params, 1)
        oof[va] = motor.predecir(aj, Xv)
    return oof
