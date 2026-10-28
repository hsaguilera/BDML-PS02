r"""Motor de validación cruzada común (Paso 2 del protocolo) y error fuera de bolsa de Random Forest (Paso 8).

Para cada pliegue k de la partición r:
* entrenamiento = (pliegues ≠ k) ∩ (𝒯 ∪ ℬ si λ_B > 0), con peso λ_B para Bogotá y 1 para 𝒯; si λ_B = 0, Bogotá se
  elimina **físicamente** (con pesos nulos ocuparía lugares del *bootstrap* y de los histogramas);
* las codificaciones con la etiqueta se recalculan con el entrenamiento del pliegue (`Datos7.matriz`);
* parada temprana (boosting) con un 15 % interno, estratificado por la etiqueta, **sólo de hogares de 𝒯**;
* las semillas se promedian **dentro** del pliegue antes de cualquier métrica.
Devuelve las predicciones OOF de todas las filas (las de Bogotá existen pero nunca entran a una métrica), las de test
promediadas sobre los K × S modelos de pliegue (contraste del modelo final), iteraciones y tiempos. Los resultados se
guardan en disco por clave (motor, hiperparámetros, variables, λ_B, partición, semillas, datos), de modo que una
corrida interrumpida continúa sin repetir ajustes.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _limpio(o):
    if isinstance(o, dict):
        return {str(k): _limpio(v) for k, v in sorted(o.items(), key=lambda t: str(t[0]))}
    if isinstance(o, (list, tuple)):
        return [_limpio(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, float):
        return float(f"{o:.10g}")
    return o


def clave(*partes) -> str:
    return hashlib.md5(json.dumps(_limpio(list(partes)), sort_keys=True, default=str).encode()).hexdigest()[:20]


@dataclass
class ResultadoCV:
    oof: np.ndarray
    test: np.ndarray | None
    iteraciones: list
    tiempo_s: float
    r: int
    semillas: tuple
    lam_B: float
    params: dict
    extra: dict = field(default_factory=dict)

    def guardar(self, ruta: Path):
        np.savez(ruta, oof=self.oof.astype(np.float32), test=(self.test if self.test is not None else np.zeros(0)).astype(np.float32),
                 iteraciones=np.asarray(self.iteraciones, float), meta=json.dumps(_limpio({
                     "tiempo_s": self.tiempo_s, "r": self.r, "semillas": list(self.semillas), "lam_B": self.lam_B,
                     "params": self.params, "extra": self.extra}), default=str))

    @classmethod
    def cargar(cls, ruta: Path) -> "ResultadoCV":
        z = np.load(ruta, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        te = z["test"]
        return cls(z["oof"].astype(np.float64), te.astype(np.float64) if te.size else None, list(z["iteraciones"]),
                   m["tiempo_s"], m["r"], tuple(m["semillas"]), m["lam_B"], m["params"], m.get("extra", {}))


def division_es(y, mascara_T, semilla, frac=0.15):
    """(filas de ajuste, filas de parada temprana) dentro del entrenamiento: 15 % de 𝒯 estratificado por la etiqueta."""
    y = np.asarray(y).astype(int); rng = np.random.default_rng(semilla)
    es = np.zeros(len(y), bool)
    for v in (0, 1):
        ii = np.flatnonzero((y == v) & mascara_T); rng.shuffle(ii)
        es[ii[:max(1, int(round(frac * len(ii))))]] = True
    return ~es, es


def _cols_k(cols, k):
    return list(cols[k]) if (cols and isinstance(cols[0], (list, tuple))) else list(cols)


def canonico(datos, cols):
    """Mismo orden de columnas que `datos.todas` (la clave de caché y el muestreo de columnas no dependen del orden
    en que se pasan)."""
    if cols and isinstance(cols[0], (list, tuple)):
        return [canonico(datos, c) for c in cols]
    s = set(cols)
    return [c for c in datos.todas if c in s]


def cv(motor, datos, params: dict, cols, lam_B: float = 1.0, r: int = 1, semillas=(1,), con_test: bool = False,
       al_pliegue=None, cache_dir: Path | None = None, etiqueta: str = "", frac_es: float = 0.15, reusar: bool = True,
       verbose: bool = False, extra_clave=None) -> ResultadoCV:
    semillas = tuple(int(s) for s in semillas)
    cols = canonico(datos, cols)
    k_ = None
    if cache_dir is not None:
        k_ = clave(motor.corto, getattr(motor, "lr", None), getattr(motor, "paciencia", None), getattr(motor, "max_arboles", None),
                   motor.splines, params, cols, float(lam_B), int(r),
                   semillas, bool(con_test), datos.huella, frac_es, extra_clave)
        ruta = Path(cache_dir) / f"cv_{k_}.npz"
        if reusar and ruta.exists():
            try:
                return ResultadoCV.cargar(ruta)
            except Exception:
                pass
    t0 = time.time()
    y = datos.y; n = len(y)
    folds = datos.folds(r)
    oof = np.full(n, np.nan); te = np.zeros(len(datos.ids_test)) if con_test else None
    iters, tiempos = [], []

    def pliegue(k, tr, va):
        tk = time.time()
        ck = _cols_k(cols, k)
        tr_fit = tr if lam_B > 0 else tr[datos.mT[tr]]
        w_fit = None if lam_B in (0, 1) else np.where(datos.bog[tr_fit], float(lam_B), 1.0)
        Xtr = datos.matriz(ck, tr_fit, train_idx=tr)
        Xva = datos.matriz(ck, va, train_idx=tr)
        Xte = datos.matriz_test(ck, train_idx=tr) if con_test else None
        pva = np.zeros(len(va)); pte = np.zeros(len(datos.ids_test)) if con_test else None; it = []
        for s in semillas:
            if motor.parada_temprana:
                fit_m, es_m = division_es(y[tr_fit], datos.mT[tr_fit], 1000 * s + 10 * r + k, frac_es)
                aj = motor.ajustar(Xtr.iloc[fit_m], y[tr_fit][fit_m], None if w_fit is None else w_fit[fit_m], params, s,
                                   Xes=Xtr.iloc[es_m], yes=y[tr_fit][es_m])
            else:
                aj = motor.ajustar(Xtr, y[tr_fit], w_fit, params, s)
            it.append(aj.n_iter)
            pva += motor.predecir(aj, Xva) / len(semillas)
            if con_test:
                pte += motor.predecir(aj, Xte) / (len(semillas) * len(folds))
        return k, va, pva, pte, it, time.time() - tk, len(ck)

    # motores de un solo hilo (CART): los pliegues se ajustan en paralelo con hilos (scikit-learn libera el GIL)
    if getattr(motor, "paralelo_pliegues", False) and al_pliegue is None and getattr(motor, "hilos", 1) > 1:
        from joblib import Parallel, delayed
        salidas = Parallel(n_jobs=min(motor.hilos, len(folds)), prefer="threads")(
            delayed(pliegue)(k, tr, va) for k, (tr, va) in enumerate(folds))
    else:
        salidas = []
        for k, (tr, va) in enumerate(folds):
            o = pliegue(k, tr, va); salidas.append(o)
            if verbose:
                print(f"    {etiqueta} r{r} pliegue {k}: {o[6]} variables, iter {o[4]}, {o[5]:.0f} s", flush=True)
            if al_pliegue is not None:
                al_pliegue(k, o[1], o[2])
    for k, va, pva, pte, it, tk, _ in sorted(salidas, key=lambda t: t[0]):
        oof[va] = pva; iters += it; tiempos.append(tk)
        if con_test:
            te += pte
    res = ResultadoCV(oof, te, iters, time.time() - t0, r, semillas, float(lam_B), dict(params),
                      {"tiempos_pliegue": tiempos, "n_vars": len(_cols_k(cols, 0))})
    if k_ is not None:
        res.guardar(Path(cache_dir) / f"cv_{k_}.npz")
    return res


def cv_rk(motor, datos, params, cols, lam_B=1.0, R=(1, 2, 3), semillas=(1,), cache_dir=None, reusar=True, verbose=False,
          etiqueta="", con_test_r1=False, extra_clave=None) -> dict:
    """R × K × S: un ResultadoCV por partición (semillas promediadas dentro de cada pliegue)."""
    return {r: cv(motor, datos, params, cols, lam_B, r, semillas, con_test=(con_test_r1 and r == 1), cache_dir=cache_dir,
                  reusar=reusar, verbose=verbose, etiqueta=etiqueta, extra_clave=extra_clave) for r in R}


# ------------------------------------------------------------------------------ OOB de Random Forest (Paso 8)
def oob_rf(motor, datos, params, cols, lam_B: float = 1.0, semilla: int = 1, cache_dir=None, reusar=True,
           con_test=False, devolver_modelo=False) -> dict:
    """Bosque ajustado con TODO el entrenamiento (λ_B; Bogotá eliminada si λ_B = 0) y probabilidades OOB honestas
    (`oob_decision_function_`): cada hogar se predice sólo con los árboles que no lo usaron. Las métricas y el umbral
    se calculan sólo en 𝒯."""
    k_ = None
    if cache_dir is not None and not devolver_modelo:
        cols = canonico(datos, cols)
        k_ = clave("OOB", motor.corto, params, cols, float(lam_B), int(semilla), datos.huella, bool(con_test))
        ruta = Path(cache_dir) / f"oob_{k_}.npz"
        if reusar and ruta.exists():
            z = np.load(ruta)
            return {"oob": z["oob"].astype(float), "test": z["test"].astype(float) if z["test"].size else None,
                    "tiempo_s": float(z["tiempo_s"]), "nan_oob": int(z["nan_oob"])}
    t0 = time.time()
    fit = np.arange(len(datos.y)) if lam_B > 0 else np.flatnonzero(datos.mT)
    w = None if lam_B in (0, 1) else np.where(datos.bog[fit], float(lam_B), 1.0)
    X = datos.matriz(cols, fit, train_idx=np.arange(len(datos.y)))
    aj = motor.ajustar(X, datos.y[fit], w, params, semilla, oob=True)
    oob = np.full(len(datos.y), np.nan); oob[fit] = motor.oob_proba(aj)
    nan_oob = int(np.isnan(oob[fit]).sum())
    te = motor.predecir(aj, datos.matriz_test(cols)) if con_test else None
    out = {"oob": oob, "test": te, "tiempo_s": time.time() - t0, "nan_oob": nan_oob}
    if k_ is not None:
        np.savez(Path(cache_dir) / f"oob_{k_}.npz", oob=oob.astype(np.float32), test=(te if te is not None else np.zeros(0)),
                 tiempo_s=out["tiempo_s"], nan_oob=nan_oob)
    if devolver_modelo:
        out["ajuste"] = aj
    return out
