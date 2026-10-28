# %% [markdown]
# # 09 · Búsqueda automática de variables
# **Problem Set 2 — Predicción de pobreza (MESE/DANE) · BDML Uniandes 2026-20**
#
# Antes de ajustar cada modelo (cuadernos 10 LightGBM y 11 XGBoost) se busca, de forma automática y con el mismo
# protocolo de validación cruzada, si hay **combinaciones de variables** que el *boosting* no descubre por sí solo.
#
# **Por qué puede valer la pena.** Un árbol aproxima interacciones y cocientes con muchas particiones escalonadas;
# cuando la relación verdadera es un cociente (p. ej. ocupados por miembro, ingreso potencial frente al requerido),
# darla explícita ahorra particiones y reduce varianza (Friedman, 2001; Kuhn y Johnson, 2019, cap. 7). En cambio,
# las **transformaciones monótonas de una sola variable** (log, raíz, cuadrado) **no cambian** las particiones de un
# árbol: se generan y reportan, pero no se evalúan.
#
# **Diseño.**
# 1. LightGBM base sobre S3 → importancia SHAP (Lundberg y Lee, 2017). Como LightGBM no calcula **interacciones**
#    SHAP, éstas se calculan con XGBoost (`pred_interactions`, Lundberg et al., 2020) sobre las 30 variables
#    principales, en una submuestra: 25 pares con mayor $E|\phi_{ij}|$.
# 2. Candidatos: productos y cocientes de esos pares, per cápita, transformaciones, rango y desviación frente a la
#    mediana del Dominio y del Depto (con estadísticas de **X de train**, nunca con `Pobre`), indicadores de umbral y
#    un conteo de carencias.
# 3. Filtro en dos etapas: (i) correlación de Spearman con el **residuo OOF** del modelo base (información que el modelo
#    aún no usa); (ii) selección **hacia adelante por lotes** con CV de 5 pliegues en un LightGBM rápido (lr 0,1,
#    parada temprana interna): un lote entra si mejora el F1 anidado sin Bogotá o el log-loss en más de **1 EE pareado**
#    entre pliegues; dentro de un lote aceptado se poda lo que no aporta. Las ganadoras se confirman con XGBoost.
# 4. Se guardan `variables_busqueda.json` y las tablas `modelo_train_v4.parquet` / `modelo_test_v4.parquet`.
#
# *Nota de honestidad.* La lista de candidatos y el filtro (i) usan todo train; el filtro (ii) es por CV. Esa
# selección introduce un optimismo de segundo orden, que los cuadernos 10 y 11 miden con una segunda partición.

# %%
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
HILOS = int(os.environ.get("AJUSTE_HILOS", "20"))
os.environ["OMP_NUM_THREADS"] = str(HILOS)
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
import matplotlib.pyplot as plt
from IPython.display import display
from pobreza.config import Rutas, SEED
from pobreza import io as pio, viz, ajuste as AJ, busqueda as BU
R = Rutas(RAIZ); viz.estilo(); FIG, TAB = R.figuras, R.tablas
RAPIDO = os.environ.get("AJUSTE_RAPIDO", "0") == "1"      # prueba sintética corta
T0 = time.time()
dat = AJ.Datos(R)
motor = AJ.MotorLGB(hilos=HILOS)
P_RAPIDO = {**motor.base, "learning_rate": 0.1}
S3 = dat.candidatos()["d. S3 (S2 + generadas + categóricas)"].final
print(dat.D.shape, dat.DT.shape, f"S3 = {len(S3)} columnas | hilos = {HILOS}")

# %% [markdown]
# ## 1. Modelo base, importancia SHAP e interacciones SHAP

# %%
imp, pares, pares_todos, A, top30 = BU.importancia_e_interacciones(dat, S3, P_RAPIDO, n_shap=5000 if RAPIDO else 20000, hilos=HILOS)
imp.rename("SHAP_media_abs").to_frame().to_csv(TAB / "09_importancia_shap_base.csv")
pares_todos.to_csv(TAB / "09_interacciones_shap.csv", index=False)
display(pares.round(4))

# %%
fig, ax = plt.subplots(1, 2, figsize=(19, 7.5), gridspec_kw={"width_ratios": [1, 1.25]})
t = pares.iloc[::-1]
ax[0].barh([f"{a} × {b}" for a, b in zip(t.a, t.b)], t.interaccion_media_abs, color=viz.MORADO)
ax[0].set(title="25 pares con mayor interacción SHAP (XGBoost, top 30 variables)", xlabel="2·E|φ_ij| (log-odds)")
ax[0].tick_params(axis="y", labelsize=7)
k = 15; Ak = A[:k, :k].copy(); np.fill_diagonal(Ak, np.nan)
im = ax[1].imshow(Ak, cmap="magma_r"); ax[1].grid(False)
ax[1].set_xticks(range(k)); ax[1].set_yticks(range(k))
ax[1].set_xticklabels(top30[:k], rotation=75, fontsize=7); ax[1].set_yticklabels(top30[:k], fontsize=7)
ax[1].set_title("Mapa de interacciones SHAP E|φ_ij| (15 variables principales)")
fig.colorbar(im, ax=ax[1], shrink=.8); fig.tight_layout()
viz.guardar(fig, FIG, "09_interacciones_shap"); plt.show()

# %% [markdown]
# ## 2. Generación de candidatos

# %%
specs = BU.generar_candidatos(dat, imp, pares)
C_tr = BU.materializar(specs, dat.D, dat.D); C_te = BU.materializar(specs, dat.D, dat.DT)
print(f"{len(specs)} candidatos:", pd.Series([s['origen'] for s in specs]).value_counts().to_dict())

# %% [markdown]
# ## 3. Filtro 1: correlación con el residuo OOF del modelo base

# %%
base = AJ.cv(motor, dat.X, dat.y, dat.folds, P_RAPIDO, AJ.Conjunto("S3", S3))
res_base = AJ.resumen(dat.y, base["oof"], dat.folds, dat.sin_bog)
print(f"Modelo base S3 (lr 0,1): F1 anidado sin Bogotá = {res_base['F1_anidado_sinBOG']:.4f} ± {res_base['EE']:.4f} | log-loss = {res_base['LogLoss']:.5f}")
f1t = BU.filtro_residuo(C_tr, dat.y - base["oof"], specs)
f1t.to_csv(TAB / "09_candidatos_filtro1.csv", index=False)
N1 = 15 if RAPIDO else 40
elegibles = f1t[~f1t.monotona_univariada & (f1t.pct_na < .5)].head(N1)
display(elegibles.round(4))

# %% [markdown]
# ## 4. Filtro 2: selección hacia adelante por lotes (CV de 5 pliegues, LightGBM rápido)

# %%
dat.X = pd.concat([dat.X, C_tr[elegibles.variable.tolist()]], axis=1)
dat.XT = pd.concat([dat.XT, C_te[elegibles.variable.tolist()]], axis=1)
ganadoras, camino, ini, fin = BU.adelante_por_lotes(motor, dat, S3, elegibles.variable.tolist(), P_RAPIDO, lote=5,
                                                    limite_s=None if RAPIDO else 25 * 60)
camino.to_csv(TAB / "09_seleccion_adelante.csv", index=False)
display(camino.drop(columns="variables").round(5))
print("Ganadoras:", ganadoras)
print(f"F1 anidado sin Bogotá: {ini['F1_anidado_sinBOG']:.4f} → {fin['F1_anidado_sinBOG']:.4f} | log-loss {ini['LogLoss']:.5f} → {fin['LogLoss']:.5f}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(16, 4.6))
cl = camino[camino.accion != "base S3"]
col = np.where(cl.aceptado, viz.VERDE, viz.GRIS)
ax[0].bar(range(len(cl)), cl.dF1, yerr=cl.EE_dF1, color=col, capsize=2); ax[0].axhline(0, color="k", lw=.7)
ax[0].set(title="Cambio pareado en F1 anidado sin Bogotá (± 1 EE)", xlabel="paso (lotes y podas)", ylabel="ΔF1")
ax[1].bar(range(len(cl)), cl.dLogLoss, yerr=cl.EE_dLL, color=col, capsize=2); ax[1].axhline(0, color="k", lw=.7)
ax[1].set(title="Mejora pareada en log-loss (± 1 EE; positivo = mejor)", xlabel="paso (lotes y podas)", ylabel="−Δ log-loss")
fig.tight_layout(); viz.guardar(fig, FIG, "09_seleccion_adelante"); plt.show()

# %% [markdown]
# ## 5. Confirmación con XGBoost

# %%
conf = []
if ganadoras:
    mx = AJ.MotorXGB(hilos=HILOS); px = {**mx.base, "learning_rate": 0.1}
    for nom, cols in [("S3", S3), ("S3 + ganadoras", S3 + ganadoras)]:
        r = AJ.cv(mx, dat.X, dat.y, dat.folds, px, AJ.Conjunto(nom, cols))
        conf.append({"modelo": "XGBoost (lr 0,1)", "conjunto": nom, **AJ.resumen(dat.y, r["oof"], dat.folds, dat.sin_bog), "tiempo_s": r["tiempo_s"]})
    conf.append({"modelo": "LightGBM (lr 0,1)", "conjunto": "S3", **ini}); conf.append({"modelo": "LightGBM (lr 0,1)", "conjunto": "S3 + ganadoras", **fin})
    conf = pd.DataFrame(conf); conf.to_csv(TAB / "09_confirmacion_xgb.csv", index=False)
    display(conf[["modelo", "conjunto", "F1_anidado_sinBOG", "EE", "AUC", "LogLoss"]].round(5))

# %% [markdown]
# ## 6. Guardar la lista ganadora y las tablas v4

# %%
spec_g = [s for s in specs if s["nombre"] in ganadoras]
salida = {"ganadoras": ganadoras, "usan_generadas": [s["nombre"] for s in spec_g if s["usa_generadas"]],
          "especificaciones": spec_g, "F1_base": ini["F1_anidado_sinBOG"], "F1_con_ganadoras": fin["F1_anidado_sinBOG"],
          "LogLoss_base": ini["LogLoss"], "LogLoss_con_ganadoras": fin["LogLoss"], "n_candidatos": len(specs),
          "n_filtro1": len(elegibles), "tiempo_min": (time.time() - T0) / 60}
json.dump(salida, open(R.processed / "variables_busqueda.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
pio.guardar_tabla(pd.concat([dat.D, C_tr[ganadoras]], axis=1), R, "modelo_train_v4")
pio.guardar_tabla(pd.concat([dat.DT, C_te[ganadoras]], axis=1), R, "modelo_test_v4")
print(f"{len(ganadoras)} variables nuevas guardadas; tiempo total {(time.time() - T0) / 60:.1f} min")
