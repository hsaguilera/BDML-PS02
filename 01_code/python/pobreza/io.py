"""Lectura de datos crudos de la competencia y almacenamiento intermedio.

Los CSV de Kaggle traen montos como texto con 'NA' y notación científica
(p. ej. '3e+05'); `pd.read_csv` los interpreta correctamente como float.
Los códigos 98/99 ("no sabe / no informa") se tratan en `features.py`.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARCHIVOS, Rutas

TABLAS = ["train_hogares", "train_personas", "test_hogares", "test_personas"]


def _zip_en(raw: Path) -> Path | None:
    zs = sorted(raw.glob("*.zip"))
    return zs[0] if zs else None


def descomprimir(rutas: Rutas, forzar: bool = False) -> list[str]:
    """Extrae el zip de Kaggle en 00_data/raw (si los CSV aún no existen)."""
    faltan = [f for f in ARCHIVOS.values() if not (rutas.raw / f).exists()]
    if not faltan and not forzar:
        return sorted(p.name for p in rutas.raw.glob("*.csv"))
    z = _zip_en(rutas.raw)
    if z is None:
        raise FileNotFoundError(f"No hay CSV ni zip en {rutas.raw}")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(rutas.raw)
    return sorted(p.name for p in rutas.raw.glob("*.csv"))


def _guardar(df: pd.DataFrame, ruta: Path) -> Path:
    try:
        p = ruta.with_suffix(".parquet")
        df.to_parquet(p, index=False)
    except Exception:  # sin pyarrow -> pickle comprimido
        p = ruta.with_suffix(".pkl.gz")
        df.to_pickle(p)
    return p


def _leer(ruta_base: Path) -> pd.DataFrame | None:
    for suf in (".parquet", ".pkl.gz"):
        p = ruta_base.with_suffix(suf)
        if p.exists():
            return pd.read_parquet(p) if suf == ".parquet" else pd.read_pickle(p)
    return None


def guardar_tabla(df: pd.DataFrame, rutas: Rutas, nombre: str) -> Path:
    return _guardar(df, rutas.processed / nombre)


def leer_tabla(rutas: Rutas, nombre: str) -> pd.DataFrame:
    df = _leer(rutas.processed / nombre)
    if df is None:
        raise FileNotFoundError(f"{nombre} no está en {rutas.processed}")
    return df


def _tipar(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte a float32 todo lo numérico salvo identificadores/categorías texto."""
    for c in df.columns:
        if c in ("id", "Dominio", "Depto"):
            continue
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        if pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype("float64")
    if "Depto" in df:
        df["Depto"] = df["Depto"].astype(str).str.zfill(2)
    if "Dominio" in df:
        df["Dominio"] = df["Dominio"].astype(str).str.strip()
    return df


def cargar_crudos(rutas: Rutas, usar_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Devuelve las 4 tablas crudas tipadas (con caché en 00_data/processed)."""
    out = {}
    for t in TABLAS:
        cache = _leer(rutas.processed / f"raw_{t}") if usar_cache else None
        if cache is None:
            descomprimir(rutas)
            df = pd.read_csv(rutas.raw / ARCHIVOS[t], low_memory=False)
            df = _tipar(df)
            _guardar(df, rutas.processed / f"raw_{t}")
            cache = df
        out[t] = cache
    return out


def cargar_sample_submission(rutas: Rutas) -> pd.DataFrame:
    descomprimir(rutas)
    return pd.read_csv(rutas.raw / ARCHIVOS["sample_submission"])


def memoria_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / 2**20)
