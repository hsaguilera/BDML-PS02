# %% [markdown]
# # 00 · Ingesta y perfil de los datos
# **Problem Set 2 — Predicción de pobreza en Colombia (BDML, Uniandes 2026-20)**
#
# Este notebook es el punto de entrada del proyecto:
#
# 1. Monta Google Drive y localiza la carpeta `Curso BigData/Taller2`.
# 2. Descomprime el zip de Kaggle (`00_data/raw/`) y guarda las cuatro tablas tipadas en `00_data/processed/` (parquet) para que los demás notebooks carguen en segundos.
# 3. Produce un **perfil exhaustivo de las 239 columnas** (tipo, faltantes, cardinalidad, cuantiles, valores más frecuentes) y los chequeos de integridad del vínculo hogar–persona.
#
# Todas las salidas quedan en `02_outputs/tables/00_*.csv` para trazabilidad.
#
# > Recomendado: *Entorno de ejecución → Cambiar tipo de entorno → RAM alta*.

# %%
# ---- Configuración del entorno (común a todos los notebooks) ----
import os, sys, warnings
warnings.filterwarnings("ignore")
EN_COLAB = "google.colab" in sys.modules
if EN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    os.environ.setdefault("TALLER2_ROOT", "/content/drive/MyDrive/Curso BigData/Taller2")
def _raiz_repo():
    """Raíz del repositorio: TALLER2_ROOT o la carpeta con el archivo `.here` (subiendo desde este script)."""
    if os.environ.get("TALLER2_ROOT"):
        return os.environ["TALLER2_ROOT"]
    d = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    while not os.path.exists(os.path.join(d, ".here")):
        if os.path.dirname(d) == d:
            raise FileNotFoundError("No encuentro la raíz del repositorio (archivo .here); defina TALLER2_ROOT.")
        d = os.path.dirname(d)
    return d


RAIZ = _raiz_repo()
sys.path.insert(0, os.path.join(RAIZ, "01_code", "python"))

import numpy as np, pandas as pd
import pobreza
from pobreza.config import Rutas, SEED
from pobreza import io as pio, eda
R = Rutas(RAIZ)
pd.set_option("display.max_columns", 60); pd.set_option("display.width", 200)
print("Raíz del proyecto:", R.raiz, "| pobreza v", pobreza.__version__)
print("Contenido de 00_data/raw:", sorted(p.name for p in R.raw.iterdir()))

# %% [markdown]
# ## 1. Descompresión y carga tipada
# Los montos de vivienda (`P5100`, `P5130`, `P5140`) vienen como texto con `NA` y notación científica; se convierten a numérico. `Depto` se trata como código de dos dígitos y `Dominio` como texto.

# %%
import time
t0 = time.time()
print("Archivos extraídos:", pio.descomprimir(R))
T = pio.cargar_crudos(R, usar_cache=False)
for k, df in T.items():
    print(f"{k:16s} {df.shape[0]:>9,d} filas × {df.shape[1]:>3d} columnas  ({pio.memoria_mb(df):,.0f} MB)")
print(f"Tiempo: {time.time()-t0:.1f} s")

# %% [markdown]
# ## 2. Integridad del vínculo hogar–persona
# Se verifica: unicidad del `id` de hogar, que cada persona pertenezca a un hogar, que `Nper` coincida con el número de registros de persona y que exista exactamente un jefe (`P6050 == 1`).

# %%
integ = eda.chequeos_integridad(T)
integ.to_csv(R.tablas / "00_integridad.csv", index=False)
display(integ.T)

# %% [markdown]
# ## 3. Columnas disponibles en train vs. test
# La competencia elimina deliberadamente variables del test (todos los ingresos y el estrato). **Sólo las variables presentes en ambos conjuntos pueden usarse como predictores**; las demás sirven únicamente para entender el fenómeno o como objetivos auxiliares.

# %%
perfil, frecuentes = eda.perfil_columnas(T)
perfil.to_csv(R.tablas / "00_perfil_columnas.csv", index=False)
frecuentes.to_csv(R.tablas / "00_valores_frecuentes.csv", index=False)
cmp_h = eda.comparar_train_test(perfil, "hogares"); cmp_p = eda.comparar_train_test(perfil, "personas")
cmp = pd.concat([cmp_h.assign(nivel="hogar"), cmp_p.assign(nivel="persona")])
cmp.to_csv(R.tablas / "00_train_vs_test_columnas.csv", index=False)
print("Hogar — sólo en train:", cmp_h.query("en_train and not en_test").columna.tolist())
print("Persona — sólo en train (", cmp_p.query("en_train and not en_test").shape[0], "):",
      cmp_p.query("en_train and not en_test").columna.tolist())
print("Persona — comunes:", cmp_p.query("en_train and en_test").shape[0])

# %% [markdown]
# ## 4. Perfil de columnas
# Tabla completa en `02_outputs/tables/00_perfil_columnas.csv`. Abajo, el resumen de faltantes y la cardinalidad de las columnas comunes.

# %%
comunes_p = cmp_p.query("en_train and en_test").columna.tolist()
vista = (perfil[perfil.tabla.isin(["train_personas", "test_personas"]) & perfil.columna.isin(comunes_p)]
         .pivot_table(index="columna", columns="tabla", values=["pct_na", "n_unicos"]))
display(vista.round(3))
vista_h = perfil[perfil.tabla.str.endswith("hogares")].pivot_table(index="columna", columns="tabla", values=["pct_na", "n_unicos", "media"])
display(vista_h.round(3))

# %% [markdown]
# ## 5. Resumen compacto para auditoría
# Se exporta un archivo JSON con los hechos clave (tamaños, tasa de pobreza, dominios, códigos de las variables categóricas) que se usan para diseñar las variables del notebook 02.

# %%
import json
th = T["train_hogares"]
resumen = {
    "dimensiones": {k: list(v.shape) for k, v in T.items()},
    "tasa_pobreza_muestral": float(th.Pobre.mean()),
    "tasa_pobreza_ponderada": float(np.average(th.Pobre, weights=th.Fex_c)),
    "tasa_indigencia_muestral": float(th.Indigente.mean()),
    "dominios_train": th.Dominio.value_counts().to_dict(),
    "dominios_test": T["test_hogares"].Dominio.value_counts().to_dict(),
    "deptos_train": th.Depto.value_counts().to_dict(),
    "pobre_igual_regla": float(((th.Ingpcug < th.Lp).astype(int) == th.Pobre).mean()),
    "ingpcug_igual_ingtotugarr_sobre_npersug": float(np.isclose(th.Ingpcug, th.Ingtotugarr / th.Npersug, rtol=1e-4).mean()),
    "codigos": {c: frecuentes[(frecuentes.tabla == "train_personas") & (frecuentes.columna == c)]
                .set_index("valor").n.astype(int).to_dict()
                for c in comunes_p if perfil.loc[(perfil.tabla == "train_personas") & (perfil.columna == c), "n_unicos"].iloc[0] <= 20},
}
resumen["codigos"] = {k: {str(a): b for a, b in v.items()} for k, v in resumen["codigos"].items()}
for c in ["P5090", "Clase", "Npersug"]:
    resumen[f"hog_{c}"] = {str(a): int(b) for a, b in th[c].value_counts(dropna=False).head(25).items()}
for c in ["P5100", "P5130", "P5140"]:
    s = th[c]
    resumen[f"hog_{c}"] = {"pct_na": float(s.isna().mean()), "n_98": int((s == 98).sum()), "n_99": int((s == 99).sum()),
                           "p50": float(s[s > 99].median()) if (s > 99).any() else None,
                           "por_tenencia_pct_no_na": th.groupby("P5090")[c].apply(lambda x: float(x.notna().mean())).to_dict()}
with open(R.tablas / "00_resumen.json", "w", encoding="utf-8") as f:
    json.dump(resumen, f, ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: resumen[k] for k in ["dimensiones", "tasa_pobreza_muestral", "tasa_pobreza_ponderada", "pobre_igual_regla", "ingpcug_igual_ingtotugarr_sobre_npersug"]}, indent=1))
