r"""Modelos probit implementados desde cero (NumPy/SciPy), con inferencia completa.

Modelo de utilidad latente (Bliss, 1934; Finney, 1947; McFadden, 1974):

    y*_i = x_i'β + ε_i,  ε_i ~ N(0,1),   pobre_i = 1{y*_i > 0}
    ⇒  P(pobre_i = 1 | x_i) = Φ(x_i'β).

Log-verosimilitud con q_i = 2y_i − 1 (forma simétrica, numéricamente estable):

    ℓ(β) = Σ_i w_i log Φ(q_i x_i'β)
    ∇ℓ   = Σ_i w_i λ_i q_i x_i,                     λ_i = φ(q_i η_i) / Φ(q_i η_i)  (razón inversa de Mills)
    ∇²ℓ  = −Σ_i w_i λ_i (λ_i + q_i η_i) x_i x_i'     (definida negativa: ℓ es estrictamente cóncava)

Clases
------
Probit              MV / ridge por Newton–Raphson; EE de modelo y sándwich (Huber–White);
                    efectos marginales promedio (AME) con método delta.
ProbitElasticNet    Probit penalizado (L1/elastic net) por IRLS + FISTA sobre la matriz de
                    Gram (Friedman, Hastie y Tibshirani, 2010; Beck y Teboulle, 2009);
                    camino completo de λ con arranques en caliente.
ProbitLatente       "Probit con variable latente observada": en train se observa el
                    log-ingreso del hogar y pobre = 1{log I < log R} con R observable. Si
                    log I | x ~ N(μ(x), σ(x)²) con μ = x'β y log σ = x'γ (heterocedasticidad
                    multiplicativa, Harvey 1976), entonces P(pobre | x) = Φ((log R − μ(x))/σ(x)):
                    un probit heterocedástico cuyos parámetros se estiman con TODA la
                    información continua del ingreso, más eficiente que el probit binario
                    (cf. Tobin, 1958; Brown, Ravallion y van de Walle, 2018).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import log_ndtr, ndtr

_LOG_SQRT_2PI = 0.5 * np.log(2 * np.pi)


def mills(t):
    """φ(t)/Φ(t) estable para t muy negativo."""
    return np.exp(-0.5 * t * t - _LOG_SQRT_2PI - log_ndtr(t))


class Estandarizador:
    """Centra y escala columnas (sd poblacional); columnas constantes quedan con sd=1."""

    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd < 1e-12] = 1.0
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd


def _con_intercepto(X):
    return np.column_stack([np.ones(len(X)), X])


# ---------------------------------------------------------------------------
# 1. Probit por máxima verosimilitud (con ridge opcional)
# ---------------------------------------------------------------------------
class Probit:
    def __init__(self, l2: float = 0.0, max_iter: int = 100, tol: float = 1e-9, estandarizar=True):
        self.l2, self.max_iter, self.tol, self.estandarizar = l2, max_iter, tol, estandarizar

    def _prep(self, X):
        X = np.asarray(X, float)
        Xs = self.est.transform(X) if self.estandarizar else X
        return _con_intercepto(Xs)

    def fit(self, X, y, w=None, nombres=None):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.nombres = list(nombres) if nombres is not None else [f"x{j}" for j in range(X.shape[1])]
        self.est = Estandarizador().fit(X) if self.estandarizar else None
        Z = self._prep(X)
        n, p = Z.shape
        w = np.ones(n) if w is None else np.asarray(w, float) * n / np.sum(w)
        q = 2 * y - 1
        pen = np.full(p, self.l2); pen[0] = 0.0
        # arranque: intercepto en Φ^{-1}(ȳ)
        b = np.zeros(p); b[0] = stats.norm.ppf(np.clip(np.average(y, weights=w), 1e-3, 1 - 1e-3))
        ll_old = -np.inf
        for it in range(self.max_iter):
            eta = Z @ b
            lam = mills(q * eta)
            g = Z.T @ (w * lam * q) - pen * b
            Wd = w * lam * (lam + q * eta)
            H = (Z * Wd[:, None]).T @ Z + np.diag(pen)
            step = np.linalg.solve(H, g)
            # búsqueda lineal (Armijo) por seguridad
            t = 1.0
            ll0 = np.sum(w * log_ndtr(q * eta)) - 0.5 * np.sum(pen * b * b)
            while True:
                bn = b + t * step
                ll = np.sum(w * log_ndtr(q * (Z @ bn))) - 0.5 * np.sum(pen * bn * bn)
                if ll >= ll0 - 1e-10 or t < 1e-6:
                    break
                t *= 0.5
            b = bn
            if abs(ll - ll_old) < self.tol * (1 + abs(ll)):
                break
            ll_old = ll
        self.coef_ = b; self.n_iter_ = it + 1; self.loglik_ = float(np.sum(w * log_ndtr(q * (Z @ b))))
        # Inferencia (escala estandarizada)
        eta = Z @ b; lam = mills(q * eta)
        Wd = w * lam * (lam + q * eta)
        H = (Z * Wd[:, None]).T @ Z + np.diag(pen)
        Hinv = np.linalg.inv(H)
        S = Z * (w * lam * q)[:, None]
        meat = S.T @ S
        self.vcov_modelo_ = Hinv
        self.vcov_robusta_ = Hinv @ meat @ Hinv
        # log-verosimilitud nula y pseudo-R2
        ybar = np.average(y, weights=w)
        self.loglik_nulo_ = float(np.sum(w * (y * np.log(ybar) + (1 - y) * np.log(1 - ybar))))
        self.pseudo_r2_mcfadden_ = 1 - self.loglik_ / self.loglik_nulo_
        self.n_, self.k_ = n, p
        self.aic_ = -2 * self.loglik_ + 2 * p
        self.bic_ = -2 * self.loglik_ + np.log(n) * p
        self._calcular_ame(Z, w)
        return self

    def decision_function(self, X):
        return self._prep(X) @ self.coef_

    def predict_proba(self, X):
        return ndtr(self.decision_function(X))

    # --- coeficientes en escala original -------------------------------------------------
    def coef_original(self):
        b = self.coef_.copy()
        if self.est is None:
            return b
        beta = b[1:] / self.est.sd
        b0 = b[0] - np.sum(b[1:] * self.est.mu / self.est.sd)
        return np.r_[b0, beta]

    def _calcular_ame(self, Z, w):
        """AME_j = mean_i φ(η_i) β_j (en unidades de 1 d.e. de x_j si se estandarizó).
        EE por método delta: ∂AME_j/∂β = mean_i φ(η_i)[e_j − η_i β_j z_i]."""
        b = self.coef_
        eta = Z @ b
        ph = stats.norm.pdf(eta)
        wn = w / w.sum()
        p = len(b)
        ame = np.array([np.sum(wn * ph) * b[j] for j in range(p)])
        m_phz = (Z * (wn * ph * (-eta))[:, None]).sum(0)  # mean φ(η)(−η) z
        mph = np.sum(wn * ph)
        G = np.outer(b, m_phz) + np.eye(p) * mph
        V = self.vcov_robusta_
        se = np.sqrt(np.einsum("ij,jk,ik->i", G, V, G))
        self.ame_, self.ame_se_ = ame, se

    def resumen(self, robusta=True) -> pd.DataFrame:
        V = self.vcov_robusta_ if robusta else self.vcov_modelo_
        se = np.sqrt(np.diag(V))
        z = self.coef_ / se
        df = pd.DataFrame({
            "variable": ["(Intercepto)"] + self.nombres,
            "coef_std": self.coef_, "ee": se, "z": z, "p_valor": 2 * stats.norm.sf(np.abs(z)),
            "coef_original": self.coef_original(),
            "AME_1de": self.ame_, "AME_ee": self.ame_se_,
        })
        df["IC95_inf"] = df.coef_std - 1.96 * df.ee; df["IC95_sup"] = df.coef_std + 1.96 * df.ee
        return df

    def wald_grupo(self, indices) -> tuple[float, float, int]:
        """Test de Wald conjunto H0: β_g = 0 (índices sobre self.nombres)."""
        idx = np.asarray(indices) + 1
        b = self.coef_[idx]; V = self.vcov_robusta_[np.ix_(idx, idx)]
        W = float(b @ np.linalg.solve(V, b))
        return W, float(stats.chi2.sf(W, len(idx))), len(idx)


# ---------------------------------------------------------------------------
# 2. Probit elastic net (camino de λ)
# ---------------------------------------------------------------------------
def _soft(x, t):
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


class ProbitElasticNet:
    r"""min_β  −(1/n)ℓ(β) + λ[α‖β‖₁ + (1−α)/2‖β‖²₂]  (intercepto sin penalizar).

    Algoritmo: IRLS externo (aproximación cuadrática de Newton de la log-verosimilitud
    probit, pesos W_i = λ_i(λ_i + q_iη_i) > 0) + FISTA interno sobre la matriz de Gram
    G = Z'WZ/n (subproblema cuadrático penalizado, vectorizado). Camino decreciente de λ
    con arranques en caliente."""

    def __init__(self, alpha=1.0, n_lambdas=30, ratio_min=1e-3, lambdas=None,
                 max_outer=25, max_inner=3000, tol=1e-7):
        self.alpha, self.n_lambdas, self.ratio_min, self.lambdas = alpha, n_lambdas, ratio_min, lambdas
        self.max_outer, self.max_inner, self.tol = max_outer, max_inner, tol

    def _cd(self, G, bvec, beta, lam):
        """Resuelve el subproblema cuadrático penalizado
            min_β ½β'Gβ − b'β + λ[α‖β_{-0}‖₁ + (1−α)/2‖β_{-0}‖²]
        con FISTA (Beck y Teboulle, 2009) totalmente vectorizado y reinicio adaptativo
        (O'Donoghue y Candès, 2015). Paso 1/L con L = λ_max(G) + λ(1−α)."""
        a = self.alpha
        p = len(beta)
        pen_l1 = np.full(p, lam * a); pen_l1[0] = 0.0
        pen_l2 = np.full(p, lam * (1 - a)); pen_l2[0] = 0.0
        L = self._L + lam * (1 - a)
        x, yk, t = beta.copy(), beta.copy(), 1.0
        for _ in range(self.max_inner):
            grad = G @ yk - bvec + pen_l2 * yk
            xn = _soft(yk - grad / L, pen_l1 / L)
            tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            if np.dot(yk - xn, xn - x) > 0:   # reinicio adaptativo
                tn, yn = 1.0, xn.copy()
            else:
                yn = xn + ((t - 1) / tn) * (xn - x)
            if np.max(np.abs(xn - x)) < self.tol:
                x = xn; break
            x, yk, t = xn, yn, tn
        return x

    def fit(self, X, y, nombres=None):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.nombres = list(nombres) if nombres is not None else [f"x{j}" for j in range(X.shape[1])]
        self.est = Estandarizador().fit(X)
        Z = _con_intercepto(self.est.transform(X))
        n, p = Z.shape
        q = 2 * y - 1
        beta = np.zeros(p); beta[0] = stats.norm.ppf(np.clip(y.mean(), 1e-3, 1 - 1e-3))
        # λ_max: gradiente en el modelo nulo
        lam0 = mills(q * beta[0])
        grad0 = np.abs(Z[:, 1:].T @ (lam0 * q)) / n
        lmax = grad0.max() / max(self.alpha, 1e-3)
        lambdas = self.lambdas if self.lambdas is not None else np.geomspace(lmax, lmax * self.ratio_min, self.n_lambdas)
        self.lambdas_ = np.asarray(lambdas)
        coefs = []
        for lam in self.lambdas_:
            for _ in range(self.max_outer):
                eta = Z @ beta
                l = mills(q * eta)
                W = l * (l + q * eta)
                zt = eta + q * l / W
                G = (Z * W[:, None]).T @ Z / n
                bv = Z.T @ (W * zt) / n
                self._L = float(np.linalg.eigvalsh(G)[-1])
                old = beta.copy()
                beta = self._cd(G, bv, beta.copy(), lam)
                if np.max(np.abs(beta - old)) < 1e-5:
                    break
            coefs.append(beta.copy())
        self.coefs_ = np.array(coefs)  # (n_lambdas, p) escala estandarizada
        return self

    def predict_proba(self, X, i=None):
        Z = _con_intercepto(self.est.transform(np.asarray(X, float)))
        if i is None:
            return ndtr(Z @ self.coefs_.T)  # (n, n_lambdas)
        return ndtr(Z @ self.coefs_[i])

    def soporte(self, i):
        return [self.nombres[j] for j in np.flatnonzero(self.coefs_[i, 1:] != 0)]

    def tabla_camino(self) -> pd.DataFrame:
        df = pd.DataFrame(self.coefs_[:, 1:], columns=self.nombres)
        df.insert(0, "lambda", self.lambdas_)
        df.insert(1, "n_activas", (self.coefs_[:, 1:] != 0).sum(1))
        return df


# ---------------------------------------------------------------------------
# 3. Probit latente-gaussiano heterocedástico
# ---------------------------------------------------------------------------
class ProbitLatente:
    r"""z_i | x_i ~ N(x_i'β, exp(2 x_i'γ)) por MV (L-BFGS, gradiente analítico), con
    penalización ridge suave. P(pobre_i) = Φ((c_i − x_i'β) / exp(x_i'γ)) con corte c_i
    (0 si z es log(ingreso/línea); log R_i si z es log del ingreso del hogar).

    `cols_var` permite restringir las variables de la ecuación de varianza."""

    def __init__(self, l2=1e-4, heterocedastico=True, cols_var=None, max_iter=500):
        self.l2, self.het, self.cols_var, self.max_iter = l2, heterocedastico, cols_var, max_iter

    def fit(self, X, z, nombres=None):
        X = np.asarray(X, float); z = np.asarray(z, float)
        self.nombres = list(nombres) if nombres is not None else [f"x{j}" for j in range(X.shape[1])]
        self.est = Estandarizador().fit(X)
        Zm = _con_intercepto(self.est.transform(X))
        self.idx_var = (np.arange(X.shape[1]) if self.cols_var is None
                        else np.array([self.nombres.index(c) for c in self.cols_var]))
        Zv = _con_intercepto(self.est.transform(X)[:, self.idx_var]) if self.het else np.ones((len(X), 1))
        n, p = Zm.shape; k = Zv.shape[1]
        lam = self.l2
        # Máxima verosimilitud por MCG factible iterado con *scoring* en γ (Harvey, 1976):
        #   β ← (Z'WZ + nλI)⁻¹ Z'Wz,  W = diag(exp(−2 Zv γ))       (MCP dado γ)
        #   γ ← γ + (Zv'Zv + nλI)⁻¹ Zv'(e² − 1)/2,  e = (z − Zβ)·exp(−Zv γ)   (scoring exacto)
        # Es estable con diseños casi colineales (la versión L-BFGS conjunta divergía con
        # los datos reales); log σ se acota en [−6, 4] por seguridad numérica.
        pen_b = n * lam * np.r_[0.0, np.ones(p - 1)]
        pen_g = n * lam * np.r_[0.0, np.ones(k - 1)]
        def wls(w):
            A = (Zm * w[:, None]).T @ Zm + np.diag(pen_b) + 1e-9 * np.eye(p)
            return np.linalg.solve(A, Zm.T @ (w * z))
        b = wls(np.ones(n))
        r = z - Zm @ b
        g = np.zeros(k); g[0] = np.log(r.std())
        G = Zv.T @ Zv + np.diag(pen_g) + 1e-9 * np.eye(k)
        nll_old = np.inf
        for it in range(self.max_iter):
            ls = np.clip(Zv @ g, -6, 4)
            b = wls(np.exp(-2 * ls))
            r = z - Zm @ b
            for _ in range(3 if self.het else 1):
                ls = np.clip(Zv @ g, -6, 4)
                e2 = (r * np.exp(-ls)) ** 2
                g = g + np.linalg.solve(G, Zv.T @ (e2 - 1) / 2 - np.r_[0.0, pen_g[1:] * g[1:]])
            ls = np.clip(Zv @ g, -6, 4)
            nll = np.mean(ls + 0.5 * (r * np.exp(-ls)) ** 2)
            if abs(nll_old - nll) < 1e-9 * (1 + abs(nll)):
                break
            nll_old = nll
        self.res_ = optimize.OptimizeResult(x=np.r_[b, g], fun=nll, nit=it + 1, success=True,
                                            message="MCG factible iterado (Harvey, 1976)")
        self.beta_, self.gamma_ = b, g
        self._Zv_fn = lambda Xs: (_con_intercepto(Xs[:, self.idx_var]) if self.het else np.ones((len(Xs), 1)))
        return self

    def mu_sigma(self, X):
        Xs = self.est.transform(np.asarray(X, float))
        mu = _con_intercepto(Xs) @ self.beta_
        sg = np.exp(np.clip(self._Zv_fn(Xs) @ self.gamma_, -6, 4))
        return mu, sg

    def predict_proba(self, X, corte=0.0):
        mu, sg = self.mu_sigma(X)
        return ndtr((corte - mu) / sg)

    def resumen(self) -> pd.DataFrame:
        filas = [{"ecuacion": "media", "variable": v, "coef_std": c}
                 for v, c in zip(["(Intercepto)"] + self.nombres, self.beta_)]
        nv = ["(Intercepto)"] + [self.nombres[j] for j in self.idx_var] if self.het else ["(Intercepto)"]
        filas += [{"ecuacion": "log_sigma", "variable": v, "coef_std": c} for v, c in zip(nv, self.gamma_)]
        return pd.DataFrame(filas)
