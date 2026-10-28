"""Cadena de datos de Jhon: construye, si faltan, sus bases en 00_data/processed a partir de 00_data/raw.

Uso (lo llaman los scripts 26–37 de 01_code/ antes de cada modelo; también se puede correr solo):
    python 01_code/python/jhon_datos.py v5      # base, variables y primera etapa (modelos 26–28)
    python 01_code/python/jhon_datos.py v7      # lo anterior y las variables v7 sin primera etapa (modelos 29–37)

Pasos, en orden (scripts en 01_code/python/jhon_datos/; cada uno se salta si ya existen sus salidas):
    00_ingesta_y_perfil     integridad de los CSV, perfil de columnas y comparación train-test (Etapa 1)
    01_EDA_completo         análisis exploratorio: desbalance, dominios, techo del F1, faltantes (Etapa 1)
    02_teoria_y_variables   variables de hogar desde la teoría, particiones `pliegues.csv` (5 pliegues por Pobre ×
                            Dominio, semilla 20262) y conjuntos de variables
    09_busqueda_variables   búsqueda de variables adicionales (base v4)
    02b_primera_etapa_v5    primera etapa: ingreso esperado fuera de pliegue como variable (v5) y listas S1, S2 y
                            encuesta corta que reutiliza v7
    02d_variables_v7        variables v7 (1.152 candidatas, 64 preguntas) sin nada derivado de modelos de ingreso
Tablas y figuras: 02_outputs/tables/jhon_datos/ y 02_outputs/figures/jhon_datos/.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = Path(os.environ.get("TALLER2_ROOT") or AQUI.parents[1])
PROC = RAIZ / "00_data" / "processed"
TAB = RAIZ / "02_outputs" / "tables" / "jhon_datos"


def _existe(rel: str) -> bool:
    """Una tabla guardada con `io.guardar_tabla` puede quedar en parquet o, si falta pyarrow, en pkl.gz."""
    p = RAIZ / rel
    return p.exists() or (p.suffix == ".parquet" and p.with_suffix(".pkl.gz").exists())


PASOS = [
    ("00_ingesta_y_perfil", ["02_outputs/tables/jhon_datos/00_resumen.json"]),
    ("01_EDA_completo", ["02_outputs/tables/jhon_datos/01_techo_f1.csv"]),
    ("02_teoria_y_variables", ["00_data/processed/pliegues.csv", "00_data/processed/conjuntos_variables.json",
                               "00_data/processed/modelo_train.parquet", "00_data/processed/modelo_test.parquet"]),
    ("09_busqueda_variables", ["00_data/processed/variables_busqueda.json", "00_data/processed/modelo_train_v4.parquet",
                               "00_data/processed/modelo_test_v4.parquet"]),
    ("02b_primera_etapa_v5", ["00_data/processed/conjuntos_variables_v5.json", "00_data/processed/modelo_train_v5.parquet",
                              "00_data/processed/modelo_test_v5.parquet"]),
    ("02d_variables_v7", ["00_data/processed/conjuntos_variables_v7.json", "00_data/processed/modelo_train_v7.parquet",
                          "00_data/processed/modelo_test_v7.parquet"]),
]
ETAPAS = {"v5": 5, "v7": 6}


def main(etapa: str) -> None:
    if etapa not in ETAPAS:
        raise SystemExit(f"etapa desconocida: {etapa!r} (use v5 o v7)")
    faltan = [n for n in ("train_hogares.csv", "train_personas.csv", "test_hogares.csv", "test_personas.csv",
                          "sample_submission.csv") if not (RAIZ / "00_data" / "raw" / n).exists()]
    zips = sorted((RAIZ / "00_data" / "raw").glob("*.zip"))
    if faltan and not zips:
        raise SystemExit(f"Faltan datos en 00_data/raw: {faltan}. Descárguelos de Kaggle (ver 00_data/README.md).")
    entorno = {**os.environ, "TALLER2_ROOT": str(RAIZ), "JHON_CARPETA": "jhon_datos", "PYTHONUTF8": "1",
               "MPLBACKEND": "Agg"}
    for nombre, salidas in PASOS[:ETAPAS[etapa]]:
        if all(_existe(s) for s in salidas):
            print(f"[jhon_datos] {nombre}: ya existe, se omite", flush=True)
            continue
        print(f"[jhon_datos] {nombre}: ejecutando…", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, "-X", "utf8", str(AQUI / "jhon_datos" / f"{nombre}.py")], cwd=RAIZ,
                       env=entorno, check=True)
        faltantes = [s for s in salidas if not _existe(s)]
        if faltantes:
            raise SystemExit(f"{nombre} terminó sin producir {faltantes}")
        print(f"[jhon_datos] {nombre}: listo en {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v7")
