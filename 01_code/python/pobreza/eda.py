"""Herramientas de análisis exploratorio (EDA)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def perfil_columnas(tablas: dict[str, pd.DataFrame], top: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perfil por columna (tipo, faltantes, únicos, momentos) y valores más frecuentes.

    Devuelve (perfil, frecuentes). `frecuentes` está en formato largo
    (tabla, columna, valor, n, pct)."""
    filas, freq = [], []
    for nom, df in tablas.items():
        n = len(df)
        for c in df.columns:
            s = df[c]
            num = pd.api.types.is_numeric_dtype(s)
            d = {
                "tabla": nom, "columna": c, "dtype": str(s.dtype), "n": n,
                "n_na": int(s.isna().sum()), "pct_na": float(s.isna().mean()),
                "n_unicos": int(s.nunique(dropna=True)),
            }
            if num and s.notna().any():
                q = s.quantile([0, .01, .25, .5, .75, .99, 1]).values
                d.update(media=float(s.mean()), de=float(s.std()), min=q[0], p01=q[1], p25=q[2],
                         p50=q[3], p75=q[4], p99=q[5], max=q[6])
            filas.append(d)
            if d["n_unicos"] <= 60 or not num:
                vc = s.value_counts(dropna=False).head(top)
                for v, k in vc.items():
                    freq.append({"tabla": nom, "columna": c, "valor": v, "n": int(k), "pct": k / n})
    return pd.DataFrame(filas), pd.DataFrame(freq)


def comparar_train_test(perfil: pd.DataFrame, base: str) -> pd.DataFrame:
    """Columnas presentes en train y ausentes en test (y viceversa) para hogares o personas."""
    tr = set(perfil.loc[perfil.tabla == f"train_{base}", "columna"])
    te = set(perfil.loc[perfil.tabla == f"test_{base}", "columna"])
    filas = [{"columna": c, "en_train": c in tr, "en_test": c in te} for c in sorted(tr | te)]
    return pd.DataFrame(filas)


def chequeos_integridad(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Unicidad de id, cobertura persona-hogar y consistencia Nper vs. conteo de personas."""
    out = []
    for split in ("train", "test"):
        h, p = t[f"{split}_hogares"], t[f"{split}_personas"]
        cnt = p.groupby("id").size()
        nper = h.set_index("id")["Nper"]
        out.append({
            "split": split,
            "hogares": len(h), "personas": len(p),
            "id_hogar_unico": h["id"].is_unique,
            "hogares_sin_personas": int((~h["id"].isin(p["id"])).sum()),
            "personas_sin_hogar": int((~p["id"].isin(h["id"])).sum()),
            "Nper==conteo_personas": float((cnt.reindex(nper.index) == nper).mean()),
            "orden_1_por_hogar": float((p.loc[p.Orden == 1].groupby("id").size() == 1).mean()),
            "jefe_unico (P6050==1)": float((p.loc[p.P6050 == 1].groupby("id").size().reindex(h.id).fillna(0) == 1).mean()),
        })
    return pd.DataFrame(out)


def tasa_ponderada(y: pd.Series, w: pd.Series | None = None) -> float:
    if w is None:
        return float(y.mean())
    return float(np.average(y, weights=w))


def tabla_pobreza_por(df: pd.DataFrame, por: str, y: str = "Pobre", w: str = "Fex_c") -> pd.DataFrame:
    g = df.groupby(por, observed=True)
    t = pd.DataFrame({
        "hogares": g.size(),
        "tasa_muestral": g[y].mean(),
        "tasa_ponderada": g.apply(lambda d: np.average(d[y], weights=d[w]), include_groups=False),
    })
    t["pct_muestra"] = t.hogares / t.hogares.sum()
    return t.sort_values("tasa_ponderada", ascending=False)


def auc_univariada(x: pd.Series, y: pd.Series) -> float:
    """AUC de una variable (orientada: max(AUC, 1-AUC)), ignorando faltantes.
    Implementación por rangos (Mann–Whitney)."""
    m = x.notna() & y.notna()
    x, y = x[m].astype(float).values, y[m].astype(int).values
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0 or np.unique(x).size < 2:
        return np.nan
    r = pd.Series(x).rank().values
    auc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(auc, 1 - auc))


def informacion_mutua_discreta(x: pd.Series, y: pd.Series, bins: int = 10) -> float:
    """Información mutua I(X;Y) en nats con X discretizada por cuantiles (NA = categoría)."""
    xs = x.copy()
    if pd.api.types.is_numeric_dtype(xs) and xs.nunique() > bins:
        xs = pd.qcut(xs, bins, duplicates="drop")
    xs = xs.astype(str)
    ct = pd.crosstab(xs, y).values.astype(float)
    p = ct / ct.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (px @ py)[nz])).sum())


def v_cramer(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x.astype(str), y.astype(str)).values.astype(float)
    n = ct.sum()
    e = ct.sum(1, keepdims=True) @ ct.sum(0, keepdims=True) / n
    chi2 = ((ct - e) ** 2 / np.where(e > 0, e, 1)).sum()
    k = min(ct.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else np.nan
