r"""Modelos lineales del protocolo v7: preprocesamiento en el pliegue, caminos lasso / red elástica y reajustes.

**Preprocesamiento (`PrepLineal`)**, ajustado sólo con el entrenamiento del pliegue (memo §11.1): recorte al
0,5 %–99,5 %, imputación por la mediana con indicador de faltante, *splines* cúbicos (4 g.l., nodos en cuantiles)
opcionales en continuas clave, *one-hot* de las categóricas (niveles con ≥ 30 hogares; el resto agrupado) y
estandarización de todas las columnas (glmnet estandariza para que la penalización sea equitativa).

**LPM (MCO con ℓ1)**: con $Q=Z_c^\top WZ_c/s_w$ y $q=Z_c^\top Wy_c/s_w$ el problema
$\min_\beta \tfrac12\beta^\top Q\beta-q^\top\beta+\lambda\|\beta\|_1$ sólo depende de la matriz de Gram; el camino se
calcula por descenso por coordenadas sobre $Q$ (Friedman, Hastie y Tibshirani, 2010) y la validación cruzada interna
para $\lambda_{1se}$ se obtiene de Grams por bloque (el Gram del entrenamiento interno es el total menos el del bloque).

**Logit con red elástica**: $\min_{\beta_0,\beta}\ \frac1{s_w}\sum_iw_i\ell(y_i,\beta_0+z_i^\top\beta)+
\lambda[\alpha\|\beta\|_1+\tfrac{1-\alpha}2\|\beta\|_2^2]$ por IRLS (aproximación cuadrática con pesos
$W_i=w_ip_i(1-p_i)$ y respuesta de trabajo $\eta_i+(y_i-p_i)/\{p_i(1-p_i)\}$) y descenso por coordenadas sobre el Gram
ponderado del **conjunto fuerte**: $S_k=\{j:|g_j|\ge\alpha(2\lambda_k-\lambda_{k-1})\}\cup\{\beta_j\ne0\}$ (reglas
fuertes secuenciales de Tibshirani et al., 2012), con verificación de las condiciones KKT en todas las columnas
($|g_j|\le\alpha\lambda$ para las inactivas); si alguna falla, se agrega y se repite. Es el algoritmo de glmnet.

**Reajustes sin penalizar**: post-lasso por MCO (Belloni y Chernozhukov, 2013) y logit por Newton sobre el soporte
(*relaxed lasso* con γ = 0; Meinshausen, 2007), con una cresta mínima (10⁻⁶) sólo por estabilidad numérica.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model._cd_fast import enet_coordinate_descent_gram as _CDG
except Exception:  # pragma: no cover
    _CDG = None

BLOQUE = 20000


# ------------------------------------------------------------------------------ preprocesamiento
class PrepLineal:
    def __init__(self, splines=(), n_nodos: int = 3, grado: int = 3, min_frec: int = 30, n_sub: int = 50000,
                 semilla: int = 0, recorte=(0.005, 0.995), indicadores: bool = True):
        self.splines, self.n_nodos, self.grado, self.min_frec = list(splines), n_nodos, grado, min_frec
        self.n_sub, self.semilla, self.recorte, self.indicadores = n_sub, semilla, recorte, indicadores

    def fit(self, X: pd.DataFrame):
        from sklearn.preprocessing import SplineTransformer
        self.cat = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        num = [c for c in X.columns if c not in self.cat]
        A = X[num].to_numpy(dtype=np.float32) if num else np.zeros((len(X), 0), np.float32)
        n = len(A)
        rng = np.random.default_rng(self.semilla)
        sub = A[rng.choice(n, min(self.n_sub, n), replace=False)].astype(np.float64)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lo, hi = (np.nanquantile(sub, list(self.recorte), axis=0) if len(num) else (np.zeros(0), np.zeros(0)))
            med = np.nanmedian(sub, axis=0) if len(num) else np.zeros(0)
        ok = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
        lo, hi = np.where(ok, lo, -np.inf), np.where(ok, hi, np.inf)
        med = np.where(np.isfinite(med), med, 0.0)
        s1 = np.zeros(len(num)); s2 = np.zeros(len(num)); nna = np.zeros(len(num))
        for i in range(0, n, BLOQUE):
            a = A[i:i + BLOQUE].astype(np.float64)
            na = np.isnan(a); nna += na.sum(0)
            b = np.where(na, med, np.clip(a, lo, hi))
            s1 += b.sum(0); s2 += (b ** 2).sum(0)
        mu = s1 / max(n, 1); sd = np.sqrt(np.maximum(s2 / max(n, 1) - mu ** 2, 0))
        keep = sd > 1e-9
        self.num = [c for c, k in zip(num, keep) if k]
        self.lo, self.hi, self.med, self.mu, self.sd = lo[keep], hi[keep], med[keep], mu[keep], sd[keep]
        fna = nna / max(n, 1)
        self.na_cols = [c for c, f in zip(num, fna) if 0.001 < f < 0.999] if self.indicadores else []
        if self.na_cols:
            Zn = X[self.na_cols].isna().to_numpy(np.float64)
            self.na_mu, self.na_sd = Zn.mean(0), np.maximum(Zn.std(0), 1e-6)
        self.spl = {}
        for c in self.splines:
            if c in self.num:
                j = self.num.index(c)
                col = sub[:, num.index(c)]
                col = np.where(np.isnan(col), self.med[j], np.clip(col, self.lo[j], self.hi[j]))
                if np.unique(col).size < 6:
                    continue
                st = SplineTransformer(n_knots=self.n_nodos, degree=self.grado, knots="quantile", extrapolation="linear",
                                       include_bias=False)
                Zs = st.fit_transform(col[:, None])
                self.spl[c] = (j, st, Zs.mean(0), np.maximum(Zs.std(0), 1e-6))
        self.niveles = {}
        for c in self.cat:
            vc = X[c].astype(str).value_counts()
            self.niveles[c] = list(vc.index[vc >= self.min_frec])
        self.nombres = list(self.num) + [f"NA_{c}" for c in self.na_cols]
        self.origen = list(self.num) + list(self.na_cols)
        for c, (j, st, m, s) in self.spl.items():
            self.nombres += [f"spl{i}_{c}" for i in range(len(m))]; self.origen += [c] * len(m)
        for c in self.cat:
            self.nombres += [f"{c}={l}" for l in self.niveles[c]] + [f"{c}=otros"]
            self.origen += [c] * (len(self.niveles[c]) + 1)
        self.p = len(self.nombres)
        # medias y desviaciones de los dummies (para estandarizar)
        self.cat_mu, self.cat_sd = {}, {}
        for c in self.cat:
            O = self._onehot(X[c])
            self.cat_mu[c] = O.mean(0); self.cat_sd[c] = np.maximum(O.std(0), 1e-6)
        return self

    def _onehot(self, s):
        v = s.astype(str).values; niv = self.niveles[s.name]
        return np.column_stack([(v == l) for l in niv] + [~np.isin(v, niv)]).astype(np.float64)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X); Z = np.empty((n, self.p), dtype=np.float32)
        k = len(self.num)
        if k:
            A = X[self.num].to_numpy(dtype=np.float32)
            for i in range(0, n, BLOQUE):
                a = A[i:i + BLOQUE].astype(np.float64)
                Z[i:i + BLOQUE, :k] = (np.where(np.isnan(a), self.med, np.clip(a, self.lo, self.hi)) - self.mu) / self.sd
            del A
        if self.na_cols:
            Z[:, k:k + len(self.na_cols)] = (X[self.na_cols].isna().to_numpy(np.float64) - self.na_mu) / self.na_sd
            k += len(self.na_cols)
        for c, (j, st, m, s) in self.spl.items():
            b = Z[:, j].astype(np.float64) * self.sd[j] + self.mu[j]
            S = (st.transform(b[:, None]) - m) / s
            Z[:, k:k + S.shape[1]] = S; k += S.shape[1]
        for c in self.cat:
            O = (self._onehot(X[c]) - self.cat_mu[c]) / self.cat_sd[c]
            Z[:, k:k + O.shape[1]] = O; k += O.shape[1]
        assert k == self.p
        return Z


# ------------------------------------------------------------------------------ álgebra
def gram(Z, w=None, cols=None, v=None):
    """(Σ w z zᵀ, Σ w z, Σ w z v) en float64 por bloques de filas (sin centrar)."""
    n = Z.shape[0]; idx = slice(None) if cols is None else cols
    p = Z.shape[1] if cols is None else len(cols)
    G = np.zeros((p, p)); s = np.zeros(p); b = np.zeros(p) if v is not None else None
    for i in range(0, n, BLOQUE):
        a = np.asarray(Z[i:i + BLOQUE][:, idx], dtype=np.float64)
        ww = np.ones(len(a)) if w is None else np.asarray(w[i:i + BLOQUE], float)
        aw = a * ww[:, None]
        G += aw.T @ a; s += aw.sum(0)
        if v is not None:
            b += aw.T @ np.asarray(v[i:i + BLOQUE], float)
    return G, s, b


def xtv(Z, v, cols=None) -> np.ndarray:
    n = Z.shape[0]; idx = slice(None) if cols is None else cols
    out = None
    for i in range(0, n, BLOQUE):
        a = np.asarray(Z[i:i + BLOQUE][:, idx], dtype=np.float64)
        r = a.T @ np.asarray(v[i:i + BLOQUE], float)
        out = r if out is None else out + r
    return out


def zbeta(Z, beta, cols=None) -> np.ndarray:
    idx = slice(None) if cols is None else cols
    out = np.empty(Z.shape[0])
    for i in range(0, Z.shape[0], BLOQUE):
        out[i:i + BLOQUE] = np.asarray(Z[i:i + BLOQUE][:, idx], dtype=np.float64) @ beta
    return out


CD_TOL = 1e-4          # tolerancia del hueco de dualidad relativa a ‖y‖² (la de scikit-learn por defecto)


def cd_gram(Q, q, yy, alpha, beta, w0=None, max_iter=1000, tol=None):
    """min ½ wᵀQw − qᵀw + α‖w‖₁ + (β/2)‖w‖² (descenso por coordenadas de scikit-learn sobre el Gram)."""
    p = len(q)
    if p == 0:
        return np.zeros(0)
    if alpha <= 0:
        return np.linalg.solve(Q + (beta + 1e-10) * np.eye(p), q)
    tol = CD_TOL if tol is None else tol
    w = np.zeros(p) if w0 is None else np.array(w0, dtype=np.float64, copy=True)
    Qc = np.ascontiguousarray(Q, dtype=np.float64); qc = np.ascontiguousarray(q, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w, _, _, _ = _CDG(w, float(alpha), float(beta), Qc, qc, np.array([np.sqrt(max(yy, 1e-300))]), int(max_iter),
                          float(tol), np.random.RandomState(0), 0, 0)
    return np.asarray(w)


def grilla_lambda(lmax, n=60, razon=1e-4) -> np.ndarray:
    return float(lmax) * np.logspace(0, np.log10(razon), int(n))


# ------------------------------------------------------------------------------ LPM: lasso-MCO con Gram
class EstadisticosMCO:
    """Estadísticos suficientes (sin centrar) de un conjunto de filas: G = Σwzzᵀ, s = Σwz, b = Σwzy, sw, sy, syy."""

    def __init__(self, G, s, b, sw, sy, syy):
        self.G, self.s, self.b, self.sw, self.sy, self.syy = G, s, b, float(sw), float(sy), float(syy)

    @classmethod
    def de(cls, Z, y, w=None):
        y = np.asarray(y, float); ww = np.ones(len(y)) if w is None else np.asarray(w, float)
        G, s, b = gram(Z, ww, v=y)
        return cls(G, s, b, ww.sum(), (ww * y).sum(), (ww * y * y).sum())

    def __sub__(self, o):
        return EstadisticosMCO(self.G - o.G, self.s - o.s, self.b - o.b, self.sw - o.sw, self.sy - o.sy, self.syy - o.syy)

    def centrado(self):
        mz = self.s / self.sw; my = self.sy / self.sw
        Q = (self.G - self.sw * np.outer(mz, mz)) / self.sw
        q = (self.b - self.sw * mz * my) / self.sw
        yy = (self.syy - self.sw * my * my) / self.sw
        return Q, q, yy, mz, my

    def sse(self, b0, beta):
        """Σ w (y − b0 − zᵀβ)² sobre las filas de estos estadísticos."""
        return float(self.syy - 2 * b0 * self.sy - 2 * beta @ self.b + b0 * b0 * self.sw + 2 * b0 * beta @ self.s
                     + beta @ self.G @ beta)


def ruta_lasso_mco(est: EstadisticosMCO, lambdas, l1_ratio: float = 1.0, max_activas: int | None = None, kkt_tol: float = 1e-3):
    """Camino lasso/red elástica MCO sobre el Gram con reglas fuertes secuenciales y verificación KKT: el descenso por
    coordenadas sólo recorre el bloque del conjunto fuerte (Tibshirani et al., 2012)."""
    Q, q, yy, mz, my = est.centrado()
    p = len(q); a = float(l1_ratio)
    B = np.zeros((p, len(lambdas))); b0 = np.zeros(len(lambdas)); w = np.zeros(p)
    g = q.copy(); lam_prev = lambdas[0]
    L = len(lambdas)
    for k, lam in enumerate(lambdas):
        S = np.flatnonzero((np.abs(g) >= a * (2 * lam - lam_prev)) | (w != 0)) if a > 0 else np.arange(p)
        while True:
            wS = cd_gram(Q[np.ix_(S, S)], q[S], yy, lam * a, lam * (1 - a), w[S])
            w = np.zeros(p); w[S] = wS
            g = q - Q[:, S] @ wS - lam * (1 - a) * w
            fuera = np.setdiff1d(np.arange(p), S)
            viol = fuera[np.abs(g[fuera]) > a * lam * (1 + kkt_tol)] if a > 0 else np.array([], int)
            if len(viol) == 0:
                break
            S = np.union1d(S, viol)
        B[:, k] = w; b0[k] = my - mz @ w; lam_prev = lam
        if max_activas and (w != 0).sum() >= max_activas:
            L = k + 1
            break
    return B[:, :L], b0[:L]


def lambda_max_mco(est: EstadisticosMCO, l1_ratio=1.0) -> float:
    Q, q, yy, mz, my = est.centrado()
    return float(np.max(np.abs(q)) / max(l1_ratio, 1e-3))


def cv_lasso_mco(Z, y, w, lambdas, k_interno=5, semilla=0, max_activas=None):
    """Validación cruzada interna del lasso-MCO con Grams por bloque. Devuelve (ECM medio por λ, EE, índices λ_min y
    λ_1se)."""
    n = len(y); rng = np.random.default_rng(semilla)
    bloque = rng.permutation(n) % k_interno
    ests = [EstadisticosMCO.de(Z[bloque == b], np.asarray(y)[bloque == b], None if w is None else np.asarray(w)[bloque == b])
            for b in range(k_interno)]
    tot = ests[0]
    for e in ests[1:]:
        tot = EstadisticosMCO(tot.G + e.G, tot.s + e.s, tot.b + e.b, tot.sw + e.sw, tot.sy + e.sy, tot.syy + e.syy)
    E = np.full((k_interno, len(lambdas)), np.nan)
    for b in range(k_interno):
        B, b0 = ruta_lasso_mco(tot - ests[b], lambdas, max_activas=max_activas)
        E[b, :B.shape[1]] = [ests[b].sse(b0[k], B[:, k]) / ests[b].sw for k in range(B.shape[1])]
    m = np.nanmean(E, 0); ee = np.nanstd(E, 0, ddof=1) / np.sqrt(k_interno)
    i_min = int(np.nanargmin(m)); cand = np.flatnonzero(m <= m[i_min] + ee[i_min])
    return m, ee, i_min, int(cand.min())


def mco(est: EstadisticosMCO, cols=None, cresta=1e-6):
    """MCO (post-lasso) sobre las columnas `cols` con una cresta mínima por estabilidad."""
    Q, q, yy, mz, my = est.centrado()
    beta = np.zeros(len(q))
    if cols is None:
        cols = np.arange(len(q))
    cols = np.asarray(cols, int)
    if len(cols):
        Qs = Q[np.ix_(cols, cols)]
        beta[cols] = np.linalg.solve(Qs + cresta * np.eye(len(cols)) * max(np.mean(np.diag(Qs)), 1e-12), q[cols])
    return my - mz @ beta, beta


# ------------------------------------------------------------------------------ logit con red elástica (IRLS + CD)
def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def _nll(y, p, w, sw):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / sw)


def _qcuad(Z, S, W, zr, sw):
    """Q, q, yy y medias del problema cuadrático ponderado (IRLS) restringido a las columnas S."""
    G, s, b = gram(Z, W, cols=S, v=zr)
    sW = W.sum(); mu = s / sW; zbar = (W * zr).sum() / sW
    Q = (G - sW * np.outer(mu, mu)) / sw
    q = (b - sW * mu * zbar) / sw
    yy = ((W * zr * zr).sum() - sW * zbar * zbar) / sw
    return Q, q, yy, mu, zbar


def lambda_max_logit(Z, y, w=None, l1_ratio=1.0) -> float:
    y = np.asarray(y, float); w = np.ones(len(y)) if w is None else np.asarray(w, float)
    sw = w.sum(); pbar = (w * y).sum() / sw
    g = xtv(Z, w * (y - pbar)) / sw
    return float(np.max(np.abs(g)) / max(l1_ratio, 1e-3))


def ruta_enet_logit(Z, y, w=None, lambdas=None, l1_ratio: float = 1.0, max_irls: int = 30, tol: float = 1e-6,
                    kkt_tol: float = 1e-3, max_activas: int | None = None, verbose: bool = False) -> dict:
    """Camino del logit con red elástica (glmnet). Devuelve coeficientes (p × L), interceptos, log-verosimilitud media
    negativa, grados de libertad (no nulos) e iteraciones IRLS por λ."""
    y = np.asarray(y, float); n, p = Z.shape
    w = np.ones(n) if w is None else np.asarray(w, float); sw = w.sum()
    a = float(l1_ratio)
    if lambdas is None:
        lambdas = grilla_lambda(lambda_max_logit(Z, y, w, a), 60)
    L = len(lambdas)
    B = np.zeros((p, L)); b0s = np.zeros(L); nll = np.zeros(L); df = np.zeros(L, int); its = np.zeros(L, int)
    beta = np.zeros(p); b0 = np.log((w * y).sum() / (sw - (w * y).sum()))
    eta = np.full(n, b0); pr = _sig(eta)
    g = xtv(Z, w * (y - pr)) / sw
    lam_prev = lambdas[0] if a == 0 else max(lambdas[0], np.max(np.abs(g)) / max(a, 1e-3))
    for k, lam in enumerate(lambdas):
        if a == 0:
            S = np.arange(p)
        else:
            S = np.flatnonzero((np.abs(g) >= a * (2 * lam - lam_prev)) | (beta != 0))
        tot_it = 0
        while True:
            bS = beta[S].copy(); dev_prev = np.inf
            for it in range(max_irls):
                eta = b0 + (zbeta(Z, bS, S) if len(S) else 0.0)
                pr = _sig(eta); v = np.maximum(pr * (1 - pr), 1e-6)
                W = w * v; zr = eta + (y - pr) / v
                if len(S):
                    Q, q, yy, mu, zbar = _qcuad(Z, S, W, zr, sw)
                    bS = cd_gram(Q, q, yy, lam * a, lam * (1 - a), bS)
                    b0 = zbar - mu @ bS
                else:
                    b0 = (W * zr).sum() / W.sum()
                eta = b0 + (zbeta(Z, bS, S) if len(S) else 0.0); pr = _sig(eta)
                dev = _nll(y, pr, w, sw) + lam * (a * np.abs(bS).sum() + 0.5 * (1 - a) * (bS ** 2).sum())
                tot_it += 1
                if abs(dev_prev - dev) < tol * (abs(dev) + 0.1):
                    break
                dev_prev = dev
            beta = np.zeros(p); beta[S] = bS
            g = xtv(Z, w * (y - pr)) / sw
            if a == 0:
                break
            fuera = np.setdiff1d(np.arange(p), S)
            viol = fuera[np.abs(g[fuera]) > a * lam * (1 + kkt_tol)]
            if len(viol) == 0:
                break
            S = np.union1d(S, viol)
        B[:, k] = beta; b0s[k] = b0; nll[k] = _nll(y, pr, w, sw); df[k] = int((beta != 0).sum()); its[k] = tot_it
        lam_prev = lam
        if verbose:
            print(f"    λ[{k}] = {lam:.3g}: activas {df[k]}, nll {nll[k]:.5f}, IRLS {tot_it}")
        if max_activas and df[k] >= max_activas:
            B, b0s, nll, df, its, lambdas = B[:, :k + 1], b0s[:k + 1], nll[:k + 1], df[:k + 1], its[:k + 1], lambdas[:k + 1]
            break
    return {"coef": B, "b0": b0s, "nll": nll, "df": df, "irls": its, "lambdas": np.asarray(lambdas), "l1_ratio": a}


def predecir_ruta(Z, ruta: dict, cols=None) -> np.ndarray:
    """Probabilidades (n × L) para todos los λ del camino."""
    B = ruta["coef"] if cols is None else ruta["coef"]
    out = np.empty((Z.shape[0], B.shape[1]))
    for i in range(0, Z.shape[0], BLOQUE):
        out[i:i + BLOQUE] = _sig(np.asarray(Z[i:i + BLOQUE], np.float64) @ B + ruta["b0"])
    return out


def logit_newton(Z, y, w=None, cols=None, cresta: float = 1e-5, max_iter: int = 30, tol: float = 1e-8):
    """Logit sin penalizar (salvo una cresta mínima) por Newton–Raphson sobre las columnas `cols`."""
    y = np.asarray(y, float); n = len(y)
    w = np.ones(n) if w is None else np.asarray(w, float); sw = w.sum()
    cols = np.arange(Z.shape[1]) if cols is None else np.asarray(cols, int)
    k = len(cols); beta = np.zeros(k); b0 = np.log((w * y).sum() / (sw - (w * y).sum()))
    prev = np.inf
    for it in range(max_iter):
        eta = b0 + (zbeta(Z, beta, cols) if k else 0.0); pr = _sig(eta)
        W = w * np.maximum(pr * (1 - pr), 1e-9)
        nll = _nll(y, pr, w, sw) + 0.5 * cresta * (beta ** 2).sum()
        if abs(prev - nll) < tol * (abs(nll) + 0.1):
            break
        prev = nll
        r = w * (y - pr)
        if k:
            G, s, _ = gram(Z, W, cols=cols)
            H = np.zeros((k + 1, k + 1)); H[0, 0] = W.sum(); H[0, 1:] = H[1:, 0] = s; H[1:, 1:] = G + cresta * sw * np.eye(k)
            gr = np.r_[r.sum(), xtv(Z, r, cols) - cresta * sw * beta]
        else:
            H = np.array([[W.sum()]]); gr = np.array([r.sum()])
        try:
            d = np.linalg.solve(H, gr)
        except np.linalg.LinAlgError:
            d = np.linalg.lstsq(H, gr, rcond=None)[0]
        paso = 1.0
        for _ in range(20):                     # búsqueda lineal (amortiguamiento) para garantizar descenso
            b0n, bn = b0 + paso * d[0], beta + paso * d[1:]
            pn = _sig(b0n + (zbeta(Z, bn, cols) if k else 0.0))
            if _nll(y, pn, w, sw) + 0.5 * cresta * (bn ** 2).sum() <= nll + 1e-12:
                break
            paso *= 0.5
        b0, beta = b0n, bn
    return b0, beta, cols


def modelos_por_costo(coef: np.ndarray, origen, MP, grilla_c) -> dict:
    """Para cada costo máximo c: el modelo más denso del camino (menor λ) cuyo soporte usa ≤ c preguntas.
    Devuelve {c: (índice λ, columnas activas, variables fuente, costo)}."""
    origen = np.asarray(origen)
    info = []
    for l in range(coef.shape[1]):
        act = np.flatnonzero(coef[:, l] != 0)
        fuentes = sorted(set(origen[act]))
        info.append((l, act, fuentes, MP.costo(fuentes)))
    out = {}
    for c in grilla_c:
        ok = [t for t in info if t[3] <= c and len(t[1]) > 0]
        if ok:
            out[c] = ok[-1]
    return out
