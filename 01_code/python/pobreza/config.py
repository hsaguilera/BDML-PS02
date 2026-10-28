"""Configuración: rutas, semillas y constantes del proyecto."""
from __future__ import annotations

import os
from pathlib import Path

SEED = 20262  # semilla global (semestre 2026-20)

# Nombre de la competencia y archivos esperados
COMPETENCIA = "uniandes-bdml-2026-20-ps-2"
ARCHIVOS = {
    "train_hogares": "train_hogares.csv",
    "train_personas": "train_personas.csv",
    "test_hogares": "test_hogares.csv",
    "test_personas": "test_personas.csv",
    "sample_submission": "sample_submission.csv",
}


def _candidatos_raiz() -> list[Path]:
    env = os.environ.get("TALLER2_ROOT")
    cands = []
    if env:
        cands.append(Path(env))
    cands += [
        Path("/content/drive/MyDrive/Curso BigData/Taller2"),
        Path("/content/drive/My Drive/Curso BigData/Taller2"),
        Path.cwd(),
        Path.cwd().parent,
    ]
    return cands


def es_repositorio(raiz: Path) -> bool:
    """¿`raiz` tiene la plantilla del repositorio del curso (00_data, 01_code/python/pobreza, 02_outputs)?"""
    raiz = Path(raiz)
    return (raiz / "00_data").is_dir() and (raiz / "01_code" / "python" / "pobreza").is_dir()


def raiz_proyecto() -> Path:
    """Devuelve la raíz del proyecto: el repositorio (plantilla del curso) o la carpeta Taller2 original."""
    env = os.environ.get("TALLER2_ROOT")
    if env and es_repositorio(Path(env)):
        return Path(env)
    for p in Path(__file__).resolve().parents:       # 01_code/python/pobreza/config.py → raíz del repositorio
        if es_repositorio(p):
            return p
    for c in _candidatos_raiz():
        if (c / "src" / "pobreza").exists() and (c / "data").exists():
            return c
    # búsqueda amplia en Drive montado
    base = Path("/content/drive")
    if base.exists():
        for p in base.glob("**/Taller2"):
            if (p / "src").exists():
                return p
    raise FileNotFoundError(
        "No encontré la carpeta Taller2. Define la variable de entorno TALLER2_ROOT."
    )


class Rutas:
    """Rutas estándar del proyecto, creadas si no existen."""

    def __init__(self, raiz: Path | None = None):
        self.raiz = Path(raiz) if raiz else raiz_proyecto()
        if es_repositorio(self.raiz):                 # plantilla del repositorio BDML-PS02
            self.raw = self.raiz / "00_data" / "raw"
            self.processed = self.raiz / "00_data" / "processed"
            self.figuras_base = self.raiz / "02_outputs" / "figures"
            self.tablas_base = self.raiz / "02_outputs" / "tables"
            self.submissions = self.raiz / "02_outputs" / "predictions"
            self.modelos = self.processed / "modelos_ajustados"
        else:                                         # carpeta Taller2 original
            self.raw = self.raiz / "data" / "raw"
            self.processed = self.raiz / "data" / "processed"
            self.figuras_base = self.raiz / "reports" / "figures"
            self.tablas_base = self.raiz / "reports" / "tables"
            self.submissions = self.raiz / "submissions"
            self.modelos = self.raiz / "models"
        # SALIDAS_CARPETA: subcarpeta de tablas y figuras del script que corre (la fija su script .R)
        carpeta = os.environ.get("SALIDAS_CARPETA", "")
        self.figuras = self.figuras_base / carpeta if carpeta else self.figuras_base
        self.tablas = self.tablas_base / carpeta if carpeta else self.tablas_base
        for d in [self.raw, self.processed, self.figuras, self.tablas,
                  self.submissions, self.modelos]:
            d.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Rutas(raiz='{self.raiz}')"
