r"""Comparaciones pareadas entre procedimientos (memo §2.3–2.5 y Paso 4 del protocolo).

* **t corregido de Nadeau y Bengio (2003)**: con diferencias por (partición, pliegue) $d_j$, $J=R\cdot K$,
  $t=\bar d/\sqrt{(1/J+n_2/n_1)\hat\sigma_d^2}$ con $n_2/n_1=1/(K-1)$ y $J-1$ g.l. (las diferencias no son
  independientes porque los conjuntos de entrenamiento se solapan; Bengio y Grandvalet, 2004, prueban que no hay un
  estimador insesgado de esa varianza, así que la corrección es heurística y conservadora; Bouckaert y Frank, 2004).
* **Prueba bayesiana correlacionada** (Corani y Benavoli, 2015; Benavoli et al., 2017): posterior
  $\mu\mid d\sim t_{J-1}\bigl(\bar d,(1/J+\rho/(1-\rho))\hat\sigma^2\bigr)$, $\rho=1/K$; se reportan $P(\mu<-r)$,
  $P(|\mu|\le r)$ y $P(\mu>r)$ con la ROPE $r$ (0,002 en F1; 0,0005 en log-loss).
* **Bootstrap pareado** de las predicciones OOF (incertidumbre de evaluación condicional a los modelos ajustados:
  «¿le ganaría A a B en el tablero privado?») y EE por función de influencia de la diferencia.
* **Holm** para comparaciones múltiples; **Tibshirani y Tibshirani (2009)** para el sesgo de selección de una búsqueda;
  **CVC** (Lei, 2020) para el conjunto de confianza de modelos.
Convención de signos: `d = A − B` en la métrica tal como se reporta; para pérdidas, «mejor» es menor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import metricas as MT


def nadeau_bengio(d, K: int = 5) -> dict:
    d = np.asarray(d, float); J = len(d)
    m = float(d.mean()); s2 = float(d.var(ddof=1)) if J > 1 else 0.0
    fac = 1.0 / J + 1.0 / (K - 1)
    ee = float(np.sqrt(fac * s2)); ee_ing = float(np.sqrt(s2 / J))
    if ee == 0:
        t, p = (np.inf * np.sign(m) if m != 0 else 0.0), (0.0 if m != 0 else 1.0)
    else:
        t = m / ee; p = float(2 * stats.t.sf(abs(t), J - 1))
    q = stats.t.ppf(0.975, max(J - 1, 1))
    return {"media": m, "EE_corregido": ee, "EE_ingenuo": ee_ing, "t": float(t), "gl": J - 1, "p": p,
            "IC95": [m - q * ee, m + q * ee], "J": J}


def bayes_correlacionado(d, rope: float, K: int = 5) -> dict:
    d = np.asarray(d, float); J = len(d)
    rho = 1.0 / K
    m = float(d.mean()); s2 = float(d.var(ddof=1)) if J > 1 else 0.0
    esc = np.sqrt((1.0 / J + rho / (1 - rho)) * s2)
    if esc == 0:
        izq = float(m < -rope); der = float(m > rope)
        return {"P_izq": izq, "P_rope": 1 - izq - der, "P_der": der}
    T = stats.t(df=J - 1, loc=m, scale=esc)
    izq = float(T.cdf(-rope)); der = float(T.sf(rope))
    return {"P_izq": izq, "P_rope": float(1 - izq - der), "P_der": der}


def bootstrap_pareado(y, yhat_a, yhat_b, B: int = 2000, semilla: int = 20262) -> dict:
    """ΔF1 = F1(A) − F1(B) en los mismos hogares (predicciones 0/1 fijas): IC 95 % bootstrap y EE por influencia."""
    y = np.asarray(y).astype(int); a = np.asarray(yhat_a).astype(int); b = np.asarray(yhat_b).astype(int); n = len(y)
    Fa, _, pa = MT.influencia_f1(y, a); Fb, _, pb = MT.influencia_f1(y, b)
    T = np.stack([(y == 1) & (a == 1), (y == 0) & (a == 1), (y == 1) & (a == 0),
                  (y == 1) & (b == 1), (y == 0) & (b == 1), (y == 1) & (b == 0)], 1).astype(float)
    rng = np.random.default_rng(semilla); vals = np.empty(B)
    for i in range(B):
        c = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        v = c @ T
        vals[i] = 2 * v[0] / max(2 * v[0] + v[1] + v[2], 1e-12) - 2 * v[3] / max(2 * v[3] + v[4] + v[5], 1e-12)
    return {"dF1": Fa - Fb, "IC95_boot": [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))],
            "EE_boot": float(vals.std(ddof=1)), "EE_psi": float((pa - pb).std(ddof=1) / np.sqrt(n)),
            "P_A_mejor": float((vals > 0).mean())}


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float); m = len(p)
    o = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj


def tibshirani_tibshirani(E, menor_mejor: bool = True) -> dict:
    """Sesgo de selección (Tibshirani y Tibshirani, 2009). E: matriz (configuraciones × pliegues) del error por pliegue.
    θ̂ minimiza la media; θ̂_k el error del pliegue k. sesgo = (1/K) Σ_k [e_k(θ̂) − e_k(θ̂_k)] ≥ 0 (en la escala del error;
    para métricas «mayor es mejor» se invierte el signo antes)."""
    E = np.asarray(E, float)
    E = E[np.all(np.isfinite(E), axis=1)]
    if len(E) < 2:
        return {"sesgo": 0.0, "n_config": int(len(E))}
    X = E if menor_mejor else -E
    j = int(np.argmin(X.mean(1)))
    sesgo = float(np.mean(X[j] - X.min(0)))
    return {"sesgo": sesgo, "n_config": int(len(E)), "indice_elegido": j}


def comparar_rk(A: dict, B: dict, tipo: str = "logloss", K: int = 5, nombre_a="A", nombre_b="B") -> dict:
    """A y B: salidas de `metricas.metricas_rk` en las MISMAS particiones y pliegues (semillas promediadas dentro de
    cada par). Devuelve ΔF1 = F1(A) − F1(B) y Δpérdida = pérdida(A) − pérdida(B) con t corregido y prueba bayesiana."""
    assert A["etiquetas"] == B["etiquetas"], "particiones/pliegues distintos: la comparación no sería pareada"
    dF = np.asarray(A["F1"]) - np.asarray(B["F1"]); dL = np.asarray(A["perdida"]) - np.asarray(B["perdida"])
    nbF, nbL = nadeau_bengio(dF, K), nadeau_bengio(dL, K)
    bF, bL = bayes_correlacionado(dF, MT.ROPE_F1, K), bayes_correlacionado(dL, MT.rope_de(tipo), K)
    return {"A": nombre_a, "B": nombre_b, "J": len(dF),
            "dF1": nbF["media"], "dF1_EE": nbF["EE_corregido"], "dF1_IC95": nbF["IC95"], "dF1_p": nbF["p"],
            "dF1_P_A_peor": bF["P_izq"], "dF1_P_rope": bF["P_rope"], "dF1_P_A_mejor": bF["P_der"],
            f"d{tipo}": nbL["media"], f"d{tipo}_EE": nbL["EE_corregido"], f"d{tipo}_IC95": nbL["IC95"], f"d{tipo}_p": nbL["p"],
            f"d{tipo}_P_A_mejor": bL["P_izq"], f"d{tipo}_P_rope": bL["P_rope"], f"d{tipo}_P_A_peor": bL["P_der"],
            "pliegues_A_mejor_F1": int((dF > 0).sum()), f"pliegues_A_mejor_{tipo}": int((dL < 0).sum())}


def no_inferior(comp: dict, tipo: str = "logloss") -> bool:
    """Regla de adopción por no inferioridad (memo §4.4): ΔF1 ≥ −ROPE y Δpérdida ≤ +ROPE, ambas con el t corregido
    (se usa el extremo desfavorable del IC 95 %)."""
    return bool(comp["dF1_IC95"][0] >= -MT.ROPE_F1 and comp[f"d{tipo}_IC95"][1] <= MT.rope_de(tipo))


def tabla_comparaciones(res: dict, referencia: str, tipo: str = "logloss", K: int = 5) -> pd.DataFrame:
    """res: {nombre: metricas_rk}. Compara cada uno con la referencia; añade Holm sobre los p del ΔF1."""
    filas = [comparar_rk(res[n], res[referencia], tipo, K, n, referencia) for n in res if n != referencia]
    t = pd.DataFrame(filas)
    if len(t):
        t["dF1_p_Holm"] = holm(t.dF1_p.values)
    return t


def cvc(L: np.ndarray, alfa: float = 0.10, B: int = 500, semilla: int = 20262, lote: int = 50) -> dict:
    """Conjunto de confianza de modelos por validación cruzada (Lei, 2020), versión con bootstrap gaussiano de
    multiplicadores sobre las pérdidas OOF por observación. L: (n × M). Para cada candidato m se contrasta
    H0: «m no es peor que ningún otro» con T_m = max_j √n·μ̂_{mj}/σ̂_{mj}, μ̂_{mj} = media(L_m − L_j); se conserva m si
    el p-valor bootstrap supera α."""
    L = np.asarray(L, float); n, M = L.shape
    rng = np.random.default_rng(semilla); pv = np.ones(M)
    for m in range(M):
        otros = [j for j in range(M) if j != m]
        if not otros:
            break
        D = L[:, [m]] - L[:, otros]
        mu = D.mean(0); sd = D.std(0, ddof=1); sd[sd == 0] = np.inf
        Tm = float(np.max(np.sqrt(n) * mu / sd))
        Z = (D - mu) / sd
        Ts = []
        for b0 in range(0, B, lote):
            k = min(lote, B - b0)
            xi = rng.standard_normal((k, n))
            Ts.append(np.max(xi @ Z / np.sqrt(n), axis=1))
        Ts = np.concatenate(Ts)
        pv[m] = float((Ts >= Tm).mean())
    return {"p_valores": pv.tolist(), "conjunto": [int(m) for m in range(M) if pv[m] > alfa], "alfa": alfa}
