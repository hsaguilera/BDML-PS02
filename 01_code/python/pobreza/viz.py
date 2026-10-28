"""Estilo gráfico común y utilidades para guardar figuras."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

AZUL = "#2563A6"      # no pobre / modelo base
ROJO = "#C2410C"      # pobre / énfasis
VERDE = "#2E7D5B"
GRIS = "#6B7280"
MORADO = "#6D28D9"
OCRE = "#B7791F"
PALETA = [AZUL, ROJO, VERDE, MORADO, OCRE, "#0E7490", "#9D174D", GRIS]


def estilo():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.figsize": (8, 4.5),
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linestyle": "-", "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelsize": 10.5, "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=PALETA), "font.size": 10,
    })


def guardar(fig, carpeta: Path, nombre: str):
    carpeta = Path(carpeta); carpeta.mkdir(parents=True, exist_ok=True)
    fig.savefig(carpeta / f"{nombre}.png", bbox_inches="tight")
    return carpeta / f"{nombre}.png"


def barras_h(ax, etiquetas, valores, color=AZUL, fmt="{:.2f}", err=None):
    y = np.arange(len(valores))
    ax.barh(y, valores, color=color, xerr=err, alpha=.9, error_kw={"elinewidth": .8, "capsize": 2})
    ax.set_yticks(y); ax.set_yticklabels(etiquetas)
    for i, v in enumerate(valores):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8, color=GRIS)
    ax.invert_yaxis()
    return ax
