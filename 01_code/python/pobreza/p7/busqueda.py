r"""Ajuste de hiperparámetros (memo §4 y Pasos 7–8 del protocolo).

* **Objetivo**: la log-loss OOF en 𝒯 (regla estrictamente propia: su minimizador poblacional es η; por Pinsker,
  $E|\hat\eta-\eta|\le\sqrt{\Delta_{\text{log-loss}}/2}$, y el F1 del *plug-in* converge a $F^*$ cuando
  $E|\hat\eta-\eta|\to0$). El F1 en el umbral óptimo es escalonado y ruidoso entre configuraciones (memo §4.2).
* **Exploración cuasi-aleatoria** (Sobol con *scrambling*; Bergstra y Bengio, 2012; *Deep Learning Tuning Playbook*)
  y luego **TPE** multivariado (Bergstra et al., 2011; Watanabe, 2023) con **poda por la mediana** por pliegue
  (regla de la mediana de Google Vizier; Golovin et al., 2017): tras cada pliegue se informa la pérdida acumulada.
* **Confirmación** de los mejores con R × K × S (particiones 1–3, semillas promediadas dentro de cada pliegue); se elige
  por la pérdida media y, dentro de la ROPE, por el F1 anidado.
* **Random Forest**: la búsqueda usa la pérdida **OOB** del bosque ajustado con todo el entrenamiento (Probst, Wright y
  Boulesteix, 2019) y se confirma con la validación cruzada común.
* **Sesgo de selección**: corrección de Tibshirani y Tibshirani (2009) con las pérdidas por pliegue de los ensayos
  completos; importancia de hiperparámetros por fANOVA (Hutter, Hoos y Leyton-Brown, 2014) y aviso de bordes.
* Cada ensayo se guarda en disco (JSON): una corrida interrumpida se reanuda sin repetirlo.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import comparar as CP
from . import metricas as MT
from .cvmotor import cv, cv_rk, oob_rf, _limpio


# ------------------------------------------------------------------------------ espacios
def sobol_params(dists: dict, n: int, semilla: int) -> list:
    from scipy.stats import qmc
    import optuna.distributions as od
    nombres = sorted(dists)
    if n <= 0 or not nombres:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        U = qmc.Sobol(len(nombres), scramble=True, seed=semilla).random(n)
    out = []
    for u in U:
        p = {}
        for x, nm in zip(u, nombres):
            d = dists[nm]
            if isinstance(d, od.CategoricalDistribution):
                p[nm] = d.choices[min(int(x * len(d.choices)), len(d.choices) - 1)]
            elif isinstance(d, od.IntDistribution):
                v = np.exp(np.log(d.low) + x * (np.log(d.high) - np.log(d.low))) if d.log else d.low + x * (d.high - d.low)
                p[nm] = int(np.clip(round(v), d.low, d.high))
            else:
                v = np.exp(np.log(d.low) + x * (np.log(d.high) - np.log(d.low))) if d.log else d.low + x * (d.high - d.low)
                p[nm] = float(np.clip(v, d.low, d.high))
        out.append(p)
    return out


def sugerir(trial, dists: dict) -> dict:
    import optuna.distributions as od
    p = {}
    for nm, d in dists.items():
        if isinstance(d, od.CategoricalDistribution):
            p[nm] = trial.suggest_categorical(nm, d.choices)
        elif isinstance(d, od.IntDistribution):
            p[nm] = trial.suggest_int(nm, d.low, d.high, log=d.log)
        else:
            p[nm] = trial.suggest_float(nm, d.low, d.high, log=d.log)
    return p


def bordes(params: dict, dists: dict, tol: float = 0.05) -> dict:
    """Posición relativa (escala de la distribución) de cada hiperparámetro numérico; marca los que tocan un borde."""
    import optuna.distributions as od
    out = {}
    for nm, d in dists.items():
        if nm not in params or isinstance(d, od.CategoricalDistribution):
            continue
        v = float(params[nm])
        u = (np.log(v) - np.log(d.low)) / (np.log(d.high) - np.log(d.low)) if d.log else (v - d.low) / max(d.high - d.low, 1e-12)
        out[nm] = {"valor": v, "posicion": float(u), "en_borde": bool(u < tol or u > 1 - tol)}
    return out


# ------------------------------------------------------------------------------ búsqueda Sobol → TPE
class Busqueda:
    def __init__(self, motor, datos, cols, P, nombre: str, fijos: dict | None = None, lam_B: float = 1.0, r: int = 1,
                 semilla: int = 20262, cache_dir: Path | None = None, oob: bool = False, dists: dict | None = None,
                 verbose: bool = True, reusar: bool = True):
        self.motor, self.datos, self.cols, self.P, self.nombre = motor, datos, list(cols), P, nombre
        self.fijos = dict(fijos or {}); self.lam_B, self.r, self.semilla = float(lam_B), int(r), int(semilla)
        self.cache = Path(cache_dir) if cache_dir else None
        self.dir_ensayos = (self.cache / "ensayos" / nombre) if self.cache else None
        if self.dir_ensayos:
            self.dir_ensayos.mkdir(parents=True, exist_ok=True)
        self.oob = oob; self.dists = dists if dists is not None else motor.espacio()
        self.verbose, self.reusar = verbose, reusar
        self.tipo = motor.perdida
        self.t0 = time.time()

    # ---------------------------------------------------------- persistencia de ensayos
    def _guardar(self, trial, estado, valor, inter, attrs):
        if not self.dir_ensayos:
            return
        d = {"numero": trial.number, "estado": estado, "valor": valor, "params": trial.params,
             "intermedios": {int(k): float(v) for k, v in inter.items()}, "attrs": attrs}
        (self.dir_ensayos / f"ensayo_{attrs.get('fase', 'x')}_{attrs.get('indice', trial.number):04d}.json").write_text(
            json.dumps(_limpio(d), ensure_ascii=False, default=str), encoding="utf-8")

    def _cargar(self, estudio) -> dict:
        import optuna
        hechos = {"sobol": set(), "tpe": 0}
        if not (self.dir_ensayos and self.reusar):
            return hechos
        for f in sorted(self.dir_ensayos.glob("ensayo_*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            if any(k not in self.dists for k in d["params"]):
                continue
            estado = optuna.trial.TrialState.COMPLETE if d["estado"] == "COMPLETE" else optuna.trial.TrialState.PRUNED
            try:
                estudio.add_trial(optuna.trial.create_trial(
                    state=estado, value=d["valor"] if estado == optuna.trial.TrialState.COMPLETE else None,
                    params=d["params"], distributions={k: self.dists[k] for k in d["params"]},
                    user_attrs=d["attrs"], intermediate_values={int(k): v for k, v in d["intermedios"].items()}))
            except Exception:
                continue
            if d["attrs"].get("fase") == "sobol":
                hechos["sobol"].add(int(d["attrs"]["indice"]))
            else:
                hechos["tpe"] += 1
        return hechos

    # ---------------------------------------------------------- objetivo
    def _objetivo(self, trial):
        import optuna
        datos, motor, y, mT = self.datos, self.motor, self.datos.y, self.datos.mT
        params = {**self.fijos, **sugerir(trial, self.dists)}
        attrs = {"fase": trial.user_attrs.get("fase", "tpe"), "indice": trial.user_attrs.get("indice", trial.number)}
        inter = {}
        t = time.time()
        if self.oob:
            res = oob_rf(motor, datos, params, self.cols, self.lam_B, semilla=1, cache_dir=self.cache, reusar=self.reusar)
            s = res["oob"]; ok = mT & np.isfinite(s)
            folds = datos.folds(1)
            s2 = np.where(np.isfinite(s), s, np.nanmean(s))
            an = MT.f1_anidado(y, s2, folds, ok)
            valor = MT.perdida(y[ok], s[ok], self.tipo)
            attrs.update({"F1_anidado": an["F1"], "nan_oob": int(res["nan_oob"]), "tiempo_s": time.time() - t,
                          "LogLoss": MT.logloss(y[ok], s[ok]), "Brier": MT.brier(y[ok], s[ok]), "AUC": MT.auc_roc(y[ok], s[ok])})
        else:
            ys, ps = [], []

            def al_pliegue(k, va, pr):
                vm = mT[va]; ys.append(y[va][vm]); ps.append(pr[vm])
                v = MT.perdida(np.concatenate(ys), np.concatenate(ps), self.tipo)
                trial.report(v, k); inter[k] = v
                if trial.should_prune():
                    attrs["tiempo_s"] = time.time() - t
                    self._guardar(trial, "PRUNED", None, inter, attrs)
                    if self.verbose:
                        print(f"    ensayo {trial.number} ({attrs['fase']}) podado en el pliegue {k}: {v:.5f}", flush=True)
                    raise optuna.TrialPruned()

            res = cv(motor, datos, params, self.cols, self.lam_B, self.r, (1,), al_pliegue=al_pliegue,
                     cache_dir=self.cache, reusar=self.reusar)
            folds = datos.folds(self.r)
            m = MT.metricas_particion(y, res.oof, folds, mT, completo=False)
            valor = m["LogLoss"] if self.tipo == "logloss" else m["Brier"]
            attrs.update({"F1_anidado": m["F1_anidado"], "LogLoss": m["LogLoss"], "Brier": m["Brier"], "AUC": m["AUC_ROC"],
                          "perdida_pliegues": m["LogLoss_pliegues"] if self.tipo == "logloss" else m["Brier_pliegues"],
                          "F1_pliegues": m["F1_pliegues"], "iter_mediana": float(np.median(res.iteraciones)),
                          "iteraciones": [int(i) for i in res.iteraciones], "tiempo_s": time.time() - t})
            if not inter:
                inter = {k: v for k, v in enumerate(np.cumsum(attrs["perdida_pliegues"]) / np.arange(1, len(folds) + 1))}
        for k, v in attrs.items():
            trial.set_user_attr(k, v)
        self._guardar(trial, "COMPLETE", valor, inter, attrs)
        if self.verbose:
            print(f"    ensayo {trial.number} ({attrs['fase']}): {self.tipo} = {valor:.5f}, F1 = {attrs['F1_anidado']:.4f} "
                  f"({attrs['tiempo_s']:.0f} s; {(time.time() - self.t0) / 60:.1f} min)", flush=True)
        return valor

    # ---------------------------------------------------------- ejecución
    def ejecutar(self, n_sobol: int, n_tpe: int, timeout_min: float | None = None, encolar: list | None = None):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        estudio = optuna.create_study(
            direction="minimize", study_name=self.nombre,
            sampler=optuna.samplers.TPESampler(seed=self.semilla, multivariate=True, n_startup_trials=0, warn_independent_sampling=False),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=max(3, n_sobol // 2), n_warmup_steps=0))
        hechos = self._cargar(estudio)
        puntos = (encolar or []) + sobol_params(self.dists, n_sobol, self.semilla)
        pend = [i for i in range(len(puntos)) if i not in hechos["sobol"]]
        for i in pend:
            estudio.enqueue_trial({k: v for k, v in puntos[i].items() if k in self.dists}, user_attrs={"fase": "sobol", "indice": i})
        if self.verbose:
            print(f"  {self.nombre}: {len(hechos['sobol'])} Sobol y {hechos['tpe']} TPE en caché; faltan {len(pend)} Sobol y "
                  f"{max(0, n_tpe - hechos['tpe'])} TPE", flush=True)
        if pend:
            estudio.optimize(self._objetivo, n_trials=len(pend), gc_after_trial=True)
        falt = max(0, n_tpe - hechos["tpe"])
        if falt and self.dists:
            to = None if timeout_min is None else max(60.0, timeout_min * 60 - (time.time() - self.t0))
            estudio.optimize(self._objetivo, n_trials=falt, timeout=to, gc_after_trial=True)
        self.estudio = estudio
        return estudio

    # ---------------------------------------------------------- resultados
    def tabla(self) -> pd.DataFrame:
        filas = []
        for t in self.estudio.trials:
            filas.append({"ensayo": t.number, "estado": t.state.name, "fase": t.user_attrs.get("fase"),
                          "objetivo": t.value, **{k: v for k, v in t.user_attrs.items() if not isinstance(v, list)},
                          "pliegues_evaluados": len(t.intermediate_values),
                          **{f"param_{k}": v for k, v in t.params.items()}})
        return pd.DataFrame(filas)

    def mejores(self, n: int) -> list:
        c = sorted([t for t in self.estudio.trials if t.state.name == "COMPLETE" and t.value is not None], key=lambda t: t.value)
        return [{**self.fijos, **t.params} for t in c[:n]]

    def sesgo_tt(self) -> dict:
        c = [t for t in self.estudio.trials if t.state.name == "COMPLETE" and "perdida_pliegues" in t.user_attrs]
        if len(c) < 2:
            return {"sesgo_perdida": 0.0, "sesgo_F1": 0.0, "n_config": len(c)}
        E = np.array([t.user_attrs["perdida_pliegues"] for t in c]); F = np.array([t.user_attrs["F1_pliegues"] for t in c])
        a = CP.tibshirani_tibshirani(E, True); b = CP.tibshirani_tibshirani(F, False)
        return {"sesgo_perdida": a["sesgo"], "sesgo_F1": b["sesgo"], "n_config": a["n_config"]}

    def importancia(self) -> pd.Series:
        import optuna
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imp = optuna.importance.get_param_importances(self.estudio)
            return pd.Series(imp).sort_values(ascending=False)
        except Exception:
            return pd.Series(dtype=float)


# ------------------------------------------------------------------------------ confirmación R × K × S
def confirmar(motor, datos, cols, configs: list, R=(1, 2, 3), semillas=(1, 2, 3), lam_B=1.0, cache_dir=None,
              reusar=True, verbose=True, etiquetas=None) -> dict:
    """Re-evalúa cada configuración con R × K × S y elige por la pérdida media (desempate por F1 dentro de la ROPE)."""
    tipo = motor.perdida
    fpr = datos.folds_por_r(R)
    filas, rks, res_todos = [], {}, {}
    for i, p in enumerate(configs):
        et = (etiquetas or {}).get(i, f"config {i}")
        t = time.time()
        res = cv_rk(motor, datos, p, cols, lam_B, R, semillas, cache_dir=cache_dir, reusar=reusar)
        oofs = {r: res[r].oof for r in R}
        rk = MT.metricas_rk(datos.y, oofs, fpr, datos.mT, tipo)
        rks[et] = rk; res_todos[et] = res
        filas.append({"config": et, "perdida_media": float(np.mean(rk["perdida"])), "perdida_EE": float(np.std(rk["perdida"], ddof=1) / np.sqrt(len(rk["perdida"]))),
                      "F1_medio_particiones": float(np.mean(list(rk["F1_por_particion"].values()))),
                      **{f"F1_r{r}": v for r, v in rk["F1_por_particion"].items()},
                      "iter_mediana": float(np.median([i_ for r in R for i_ in res[r].iteraciones])),
                      "params": json.dumps(_limpio(p), default=str), "tiempo_s": time.time() - t})
        if verbose:
            print(f"  confirmación {et}: {tipo} = {filas[-1]['perdida_media']:.5f}, F1 medio = {filas[-1]['F1_medio_particiones']:.4f} "
                  f"({filas[-1]['tiempo_s']:.0f} s)", flush=True)
    t = pd.DataFrame(filas)
    best = t.perdida_media.min()
    cand = t[t.perdida_media <= best + MT.rope_de(tipo)]
    elegido = cand.sort_values("F1_medio_particiones", ascending=False).iloc[0].config
    comp = CP.tabla_comparaciones(rks, elegido, tipo) if len(rks) > 1 else pd.DataFrame()
    t["elegido"] = t.config == elegido
    i_el = int(t.index[t.elegido][0])
    return {"tabla": t, "comparaciones": comp, "elegido": configs[i_el], "etiqueta": elegido, "rk": rks, "resultados": res_todos}


def comparar_cientifico(motor, datos, cols, variantes: dict, R=(1, 2, 3), semillas=(1,), cache_dir=None, reusar=True,
                        referencia=None, verbose=True) -> dict:
    """Hiperparámetro «científico» (λ_B, monotonía, pesos de clase, conjunto de variables): cada variante
    {nombre: (params, cols, lam_B)} con R × K × S; comparación pareada con la referencia (t de Nadeau–Bengio, prueba
    bayesiana con ROPE, bootstrap pareado de las OOF de la partición 1)."""
    tipo = motor.perdida
    fpr = datos.folds_por_r(R)
    rks, filas, res = {}, [], {}
    for nom, (p, c, lb) in variantes.items():
        t = time.time()
        rr = cv_rk(motor, datos, p, c if c is not None else cols, lb, R, semillas, cache_dir=cache_dir, reusar=reusar)
        res[nom] = rr
        rks[nom] = MT.metricas_rk(datos.y, {r: rr[r].oof for r in R}, fpr, datos.mT, tipo)
        filas.append({"variante": nom, "lam_B": lb, f"{tipo}_media": float(np.mean(rks[nom]["perdida"])),
                      "F1_medio": float(np.mean(list(rks[nom]["F1_por_particion"].values()))), "tiempo_s": time.time() - t})
        if verbose:
            print(f"  {nom}: {tipo} = {filas[-1][f'{tipo}_media']:.5f}, F1 = {filas[-1]['F1_medio']:.4f} ({filas[-1]['tiempo_s']:.0f} s)", flush=True)
    ref = referencia or list(variantes)[0]
    comp = CP.tabla_comparaciones(rks, ref, tipo)
    f1 = datos.folds(R[0])
    if len(comp):
        yb = MT.f1_anidado(datos.y, res[ref][R[0]].oof, f1, datos.mT)["yhat"][datos.mT]
        boots = []
        for nom in comp.A:
            ya = MT.f1_anidado(datos.y, res[nom][R[0]].oof, f1, datos.mT)["yhat"][datos.mT]
            boots.append(CP.bootstrap_pareado(datos.y[datos.mT], ya, yb, B=1000))
        comp["dF1_boot_IC95_r1"] = [b["IC95_boot"] for b in boots]
        comp["P_A_mejor_boot_r1"] = [b["P_A_mejor"] for b in boots]
    return {"tabla": pd.DataFrame(filas), "comparaciones": comp, "rk": rks, "resultados": res, "referencia": ref}


# ------------------------------------------------------------------------------ CART: crecimiento + poda con 1-EE
def cv_cart_poda(motor, datos, params, cols, r=1, alfas=None, lam_B=1.0, hilos=4):
    """Para una configuración de crecimiento: árbol completo en cada pliegue (en paralelo) y predicciones del árbol
    podado en una grilla común de α (sin re-ajustar). Devuelve OOF (n × A), hojas y variables usadas por α."""
    from joblib import Parallel, delayed
    from .motores import podar_info, predecir_podado
    y = datos.y; n = len(y); folds = datos.folds(r)
    alfas = np.r_[0.0, np.logspace(-6, -1.5, 30)] if alfas is None else np.asarray(alfas)
    m_est = float(params.get("m_estimacion", 2.0))

    def uno(k, tr, va):
        tr_fit = tr if lam_B > 0 else tr[datos.mT[tr]]
        w = None if lam_B in (0, 1) else np.where(datos.bog[tr_fit], lam_B, 1.0)
        aj = motor.ajustar(datos.matriz(cols, tr_fit, train_idx=tr), y[tr_fit], w, {**params, "ccp_alpha": 0.0}, 1)
        A = aj.prep.transform(datos.matriz(cols, va, train_idx=tr))
        info = podar_info(aj.modelo)
        P = predecir_podado(aj.modelo, A, alfas, info, m_est, aj.extra["prior"])
        hojas, usadas = [], []
        f = aj.modelo.tree_.feature
        for a in alfas:
            vivo = _vivos(aj.modelo, info, a)
            interno = vivo & (aj.modelo.tree_.children_left != -1) & (info["colapso"] > a)
            hojas.append(int((vivo & ~interno).sum()))
            usadas.append(sorted({aj.prep.cols[j] for j in f[interno] if j >= 0}))
        return k, va, P, hojas, usadas

    outs = Parallel(n_jobs=min(hilos, len(folds)), prefer="threads")(delayed(uno)(k, tr, va) for k, (tr, va) in enumerate(folds))
    OOF = np.full((n, len(alfas)), np.nan); H, U = [], []
    for k, va, P, h, u in sorted(outs, key=lambda t: t[0]):
        OOF[va] = P; H.append(h); U.append(u)
    return {"alfas": alfas, "oof": OOF, "hojas": np.array(H), "usadas": U}


def _vivos(tree, info, a) -> np.ndarray:
    """Nodos que existen en el árbol podado con α = a (ningún ancestro colapsado)."""
    t = tree.tree_; n = t.node_count
    vivo = np.ones(n, bool)
    for i in range(n):                                    # preorden: el padre se procesa antes que los hijos
        p = info["padre"][i]
        if p != -1 and (not vivo[p] or info["colapso"][p] <= a):
            vivo[i] = False
    return vivo


def evaluar_poda(datos, poda: dict, r=1, tipo="logloss") -> pd.DataFrame:
    y, mT, folds = datos.y, datos.mT, datos.folds(r)
    filas = []
    for j, a in enumerate(poda["alfas"]):
        s = poda["oof"][:, j]
        Lk = MT.por_pliegue(y, s, folds, mT, tipo)
        an = MT.f1_anidado(y, s, folds, mT)
        filas.append({"alfa": float(a), "perdida": MT.perdida(y[mT], s[mT], tipo), "perdida_EE": float(Lk.std(ddof=1) / np.sqrt(len(Lk))),
                      "perdida_pliegues": Lk.tolist(), "F1_anidado": an["F1"], "hojas_media": float(poda["hojas"][:, j].mean()),
                      "n_vars_media": float(np.mean([len(u[j]) for u in poda["usadas"]])),
                      "n_preguntas_media": float(np.mean([datos.MP.costo(u[j]) for u in poda["usadas"]]))})
    t = pd.DataFrame(filas)
    i = int(t.perdida.idxmin())
    ok = t[t.perdida <= t.loc[i, "perdida"] + t.loc[i, "perdida_EE"]]
    t["alfa_min"] = t.index == i
    t["alfa_1se"] = t.index == ok.alfa.idxmax()
    return t


class BusquedaCART(Busqueda):
    """Sobol + TPE sobre los hiperparámetros de crecimiento; para cada configuración, el α de poda se elige por la
    pérdida CV mínima (y se registra el de la regla 1-EE)."""

    def __init__(self, *a, hilos=4, **kw):
        super().__init__(*a, **kw); self.hilos = hilos

    def _objetivo(self, trial):
        datos = self.datos
        params = {**self.fijos, **sugerir(trial, self.dists)}
        attrs = {"fase": trial.user_attrs.get("fase", "tpe"), "indice": trial.user_attrs.get("indice", trial.number)}
        t = time.time()
        poda = cv_cart_poda(self.motor, datos, params, self.cols, self.r, lam_B=self.lam_B, hilos=self.hilos)
        tb = evaluar_poda(datos, poda, self.r, self.tipo)
        i, j = int(tb.index[tb.alfa_min][0]), int(tb.index[tb.alfa_1se][0])
        valor = float(tb.loc[i, "perdida"])
        attrs.update({"F1_anidado": float(tb.loc[i, "F1_anidado"]), "alfa_min": float(tb.loc[i, "alfa"]),
                      "alfa_1se": float(tb.loc[j, "alfa"]), "hojas_min": float(tb.loc[i, "hojas_media"]),
                      "hojas_1se": float(tb.loc[j, "hojas_media"]), "perdida_1se": float(tb.loc[j, "perdida"]),
                      "F1_1se": float(tb.loc[j, "F1_anidado"]), "perdida_pliegues": tb.loc[i, "perdida_pliegues"],
                      "F1_pliegues": MT.f1_anidado(datos.y, poda["oof"][:, i], datos.folds(self.r), datos.mT)["F1_pliegues"].tolist(),
                      "tiempo_s": time.time() - t})
        for k, v in attrs.items():
            trial.set_user_attr(k, v)
        self._guardar(trial, "COMPLETE", valor, {}, attrs)
        if self.verbose:
            print(f"    CART ensayo {trial.number} ({attrs['fase']}): {self.tipo} = {valor:.5f} (α = {attrs['alfa_min']:.2g}, "
                  f"{attrs['hojas_min']:.0f} hojas), F1 = {attrs['F1_anidado']:.4f}; 1-EE: {attrs['hojas_1se']:.0f} hojas "
                  f"({attrs['tiempo_s']:.0f} s)", flush=True)
        return valor

    def mejores(self, n: int, regla: str = "min") -> list:
        c = sorted([t for t in self.estudio.trials if t.state.name == "COMPLETE" and t.value is not None], key=lambda t: t.value)
        return [{**self.fijos, **t.params, "ccp_alpha": t.user_attrs["alfa_min" if regla == "min" else "alfa_1se"]} for t in c[:n]]
