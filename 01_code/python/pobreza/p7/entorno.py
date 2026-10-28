r"""Entorno de ejecución del protocolo v7: rutas de lectura y de escritura separadas, hilos y presupuestos.

* **Lectura** (`datos`, `raw`): tablas de modelado, conjuntos de variables, `pliegues.csv` y `sample_submission.csv`.
* **Escritura** (`salida`): `00_data/processed/pliegues_v7.csv`, el registro común (`00_data/processed/modelos/`),
  `02_outputs/tables`, `02_outputs/figures` y `submissions`. En las corridas reales `salida` es la raíz del proyecto; en las
  pruebas es una carpeta del scratchpad, de modo que nada se escribe en `data/`, `reports/` ni `02_outputs/predictions/` reales.
* **Caché** reanudable (`cache`): resultados de validación cruzada y ensayos de la búsqueda, por clave.

Variables de entorno (las fija el lanzador el script `.R` que lo ejecuta (`01_code/NN_*.R`) o la prueba):
TALLER2_ROOT (raíz del proyecto), P7_SALIDA (raíz de escritura), P7_CACHE, P7_HILOS, P7_PRESUPUESTO
(`memo` | `reducido` | `express` | `prueba`), P7_SUBMUESTRA (fracción de train, sólo pruebas), P7_FUENTE (`auto` | `v7` |
`respaldo`), P7_DATOS (carpeta alternativa de lectura de las tablas, sólo pruebas), P7_REUSAR (1/0).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

def _raiz_defecto() -> Path:
    from ..config import raiz_proyecto
    try:
        return raiz_proyecto()
    except FileNotFoundError:
        return Path.cwd()


RAIZ_DEFECTO = _raiz_defecto()


def _sub(raiz: Path, clave: str) -> tuple:
    """Subcarpetas de lectura y escritura según la disposición: plantilla del repositorio o carpeta original."""
    from ..config import es_repositorio
    repo = es_repositorio(Path(raiz))
    return {"procesados": ("00_data", "processed") if repo else ("data", "processed"),
            "raw": ("00_data", "raw") if repo else ("data", "raw"),
            "tablas": ("02_outputs", "tables") if repo else ("reports", "tables"),
            "figuras": ("02_outputs", "figures") if repo else ("reports", "figures"),
            "envios": ("02_outputs", "predictions") if repo else ("submissions",)}[clave]


def _carpeta() -> tuple:
    """SALIDAS_CARPETA: subcarpeta de tablas y figuras del script que corre (la fija su script .R)."""
    c = os.environ.get("SALIDAS_CARPETA", "")
    return (c,) if c else ()

# ------------------------------------------------------------------------------ presupuestos
# «memo»: valores por defecto del memo (§4.5, §8.3, §9.3, §11.2). «reducido»: el mismo protocolo con menos ensayos y
# semillas. «express»: los mismos pasos con el presupuesto mínimo defendible (Sobol 8 + TPE 16, confirmación 2 × 5 × 1,
# RFE con pasos del 35 %; modelo final de boosting con tasa 0,05 y 5 semillas en vez de 0,02 y 10 — desviación declarada
# del memo §8.3), para los algoritmos caros (XGBoost, CatBoost, Random Forest) cuando hay plazo.
# «prueba»: de punta a punta en una submuestra.
PRESUPUESTOS = {
    "memo": dict(
        n_sobol=64, n_tpe=150, top_confirmar=5, R=3, S_conf=3, S_final_boost=10, rf_arboles_busqueda=300,
        rf_arboles_final=1000, rf_ensayos=60, rf_sobol=10, n_shap=20000, B_boot=2000, B_umbral=500, cpss_B=50,
        en_alfas=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0), en_nlambda=60, en_razon=1e-4, en_max_activas=None, cart_sobol=64, cart_tpe=36, nb_M=60, nb_kmax=25,
        rfe_frac=0.2, loco=True, lambdas_flexibles=(0.0, 0.5, 1.0), lambdas_rigidos=(0.0, 1.0), monotonia=True,
        desbalance=True, reevaluar=True, paciencia=200, lr_busqueda=0.05, lr_final=0.02, max_arboles=10000,
        seleccion_R=1, seleccion_S=1, timeout_busqueda_min=None, cvc=True, B_cvc=500, pd_n=3000,
    ),
    "reducido": dict(
        n_sobol=16, n_tpe=40, top_confirmar=3, R=3, S_conf=2, S_final_boost=10, rf_arboles_busqueda=200,
        rf_arboles_final=1000, rf_ensayos=30, rf_sobol=8, n_shap=10000, B_boot=2000, B_umbral=500, cpss_B=25,
        en_alfas=(0.0, 0.25, 0.5, 0.75, 1.0), en_nlambda=40, en_razon=1e-3, en_max_activas=800, cart_sobol=32, cart_tpe=16, nb_M=50, nb_kmax=25,
        rfe_frac=0.25, loco=True, lambdas_flexibles=(0.0, 0.5, 1.0), lambdas_rigidos=(0.0, 1.0), monotonia=True,
        desbalance=True, reevaluar=True, paciencia=200, lr_busqueda=0.05, lr_final=0.02, max_arboles=10000,
        seleccion_R=1, seleccion_S=1, timeout_busqueda_min=240, cvc=True, B_cvc=300, pd_n=2000,
    ),
    "express": dict(
        n_sobol=8, n_tpe=16, top_confirmar=2, R=2, S_conf=1, S_final_boost=5, rf_arboles_busqueda=150,
        rf_arboles_final=1000, rf_ensayos=12, rf_sobol=4, n_shap=5000, B_boot=1000, B_umbral=500, cpss_B=10,
        en_alfas=(0.0, 0.5, 1.0), en_nlambda=30, en_razon=1e-3, en_max_activas=600, cart_sobol=16, cart_tpe=8,
        nb_M=40, nb_kmax=20, rfe_frac=0.35, loco=True, loco_excluir=("elastic_net",), lambdas_flexibles=(0.0, 0.5, 1.0), lambdas_rigidos=(0.0, 1.0),
        monotonia=True, desbalance=True, reevaluar=True, paciencia=200, lr_busqueda=0.05, lr_final=0.05,
        max_arboles=10000, seleccion_R=1, seleccion_S=1, timeout_busqueda_min=120, cvc=True, B_cvc=200, pd_n=1500,
    ),
    # «minimo» (23 oct, por plazo, aprobado por el usuario para XGBoost): los pasos de «express» que cambian el modelo
    # elegido (selección, búsqueda acotada por tiempo, confirmación, Bogotá λ_B ∈ {0, 1}, re-evaluación en r4, finales,
    # diagnóstico) con una sola partición en las comparaciones; sin monotonía, sin pesos de desbalance y sin LOCO
    # (diagnósticos que en LightGBM no cambiaron la decisión). Desviación declarada del memo §4.5 y §8.3.
    "minimo": dict(
        n_sobol=3, n_tpe=4, top_confirmar=1, R=1, S_conf=1, S_final_boost=3, rf_arboles_busqueda=150,
        rf_arboles_final=1000, rf_ensayos=8, rf_sobol=4, n_shap=5000, B_boot=1000, B_umbral=500, cpss_B=10,
        en_alfas=(0.0, 0.5, 1.0), en_nlambda=30, en_razon=1e-3, en_max_activas=600, cart_sobol=16, cart_tpe=8,
        nb_M=40, nb_kmax=20, rfe_frac=0.35, loco=False, loco_excluir=("elastic_net",), lambdas_flexibles=(0.0, 1.0), lambdas_rigidos=(0.0, 1.0),
        monotonia=False, desbalance=False, reevaluar=True, paciencia=200, lr_busqueda=0.05, lr_final=0.05,
        max_arboles=10000, seleccion_R=1, seleccion_S=1, timeout_busqueda_min=150, cvc=True, B_cvc=200, pd_n=1500,
    ),
    "prueba": dict(
        n_sobol=3, n_tpe=3, top_confirmar=2, R=2, S_conf=1, S_final_boost=2, rf_arboles_busqueda=40,
        rf_arboles_final=80, rf_ensayos=4, rf_sobol=2, n_shap=1500, B_boot=200, B_umbral=100, cpss_B=4,
        en_alfas=(0.5, 1.0), en_nlambda=12, en_razon=1e-2, en_max_activas=400, cart_sobol=4, cart_tpe=2, nb_M=15, nb_kmax=6,
        rfe_frac=0.35, loco=True, lambdas_flexibles=(0.0, 1.0), lambdas_rigidos=(0.0, 1.0), monotonia=True,
        desbalance=True, reevaluar=True, paciencia=30, lr_busqueda=0.1, lr_final=0.1, max_arboles=600,
        seleccion_R=1, seleccion_S=1, timeout_busqueda_min=None, cvc=True, B_cvc=100, pd_n=400,
    ),
}


def presupuesto(nombre: str | None = None, **cambios) -> SimpleNamespace:
    nombre = nombre or os.environ.get("P7_PRESUPUESTO", "reducido")
    if nombre not in PRESUPUESTOS:
        raise ValueError(f"presupuesto desconocido: {nombre} (opciones: {list(PRESUPUESTOS)})")
    d = dict(PRESUPUESTOS[nombre]); d.update(cambios); d["nombre"] = nombre
    return SimpleNamespace(**d)


@dataclass
class Entorno:
    raiz: Path = RAIZ_DEFECTO
    salida: Path | None = None
    datos: Path | None = None
    raw: Path | None = None
    cache: Path | None = None
    hilos: int = 4
    reusar: bool = True
    cuaderno: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.raiz = Path(self.raiz)
        self.salida = Path(self.salida) if self.salida else self.raiz
        self.datos = Path(self.datos) if self.datos else self.raiz.joinpath(*_sub(self.raiz, "procesados"))
        self.raw = Path(self.raw) if self.raw else self.raiz.joinpath(*_sub(self.raiz, "raw"))
        self.cache = Path(self.cache) if self.cache else self.procesados / "cache_v7"
        for d in (self.procesados, self.tablas, self.figuras, self.envios, self.cache):
            d.mkdir(parents=True, exist_ok=True)

    # rutas de escritura
    @property
    def procesados(self) -> Path:
        return self.salida.joinpath(*_sub(self.raiz, "procesados"))

    @property
    def tablas(self) -> Path:
        return self.salida.joinpath(*_sub(self.raiz, "tablas"), *_carpeta())

    @property
    def figuras(self) -> Path:
        return self.salida.joinpath(*_sub(self.raiz, "figuras"), *_carpeta())

    @property
    def envios(self) -> Path:
        return self.salida.joinpath(*_sub(self.raiz, "envios"))

    @property
    def es_prueba(self) -> bool:
        return Path(self.salida).resolve() != Path(self.raiz).resolve()

    def cache_de(self, *partes) -> Path:
        d = self.cache.joinpath(*[str(p) for p in partes])
        d.mkdir(parents=True, exist_ok=True)
        return d

    def rutas_registro(self) -> SimpleNamespace:
        """Objeto con la interfaz de `config.Rutas` que usa `registro.py` (escribe en `salida`)."""
        return SimpleNamespace(raiz=self.salida, processed=self.procesados, submissions=self.envios, raw=self.raw,
                               tablas=self.tablas, figuras=self.figuras)

    @classmethod
    def desde_variables(cls, cuaderno: str = "", **kw) -> "Entorno":
        raiz = Path(os.environ.get("TALLER2_ROOT", RAIZ_DEFECTO))
        args = dict(raiz=raiz, salida=os.environ.get("P7_SALIDA") or None, datos=os.environ.get("P7_DATOS") or None,
                    cache=os.environ.get("P7_CACHE") or None, hilos=int(os.environ.get("P7_HILOS", "4")),
                    reusar=os.environ.get("P7_REUSAR", "1") == "1", cuaderno=cuaderno)
        args.update(kw)
        return cls(**args)

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Entorno(raiz='{self.raiz}', salida='{self.salida}', datos='{self.datos}', cache='{self.cache}', "
                f"hilos={self.hilos}, prueba={self.es_prueba})")
