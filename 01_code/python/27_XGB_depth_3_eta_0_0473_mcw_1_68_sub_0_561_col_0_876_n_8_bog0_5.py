# %% [markdown]
# > **Repositorio BDML-PS02 · script 27.** Lo ejecuta `01_code/27_XGB_depth_3_eta_0_0473_mcw_1_68_sub_0_561_col_0_876_n_8_bog0_5.R`. Es el cuaderno interno `11_xgboost`:
# > sus tablas y figuras conservan el prefijo `11_` y quedan en `02_outputs/tables/27_XGB_v5/` y
# > `02_outputs/figures/27_XGB_v5/`; los CSV de envío, con el nombre con que se generaron (y, si se subieron,
# > con el que se subieron a Kaggle), en `02_outputs/predictions/`.

# %% [markdown]
# # 11 · XGBoost ajustado a fondo
# **Problem Set 2 — Predicción de pobreza (MESE/DANE) · BDML Uniandes 2026-20**
#
# Un cuaderno por modelo: se ajusta **XGBoost** «a la perfección» con el protocolo común del proyecto y después se
# compara con los demás en el registro (`00_data/processed/modelos/`). Ruta de **clasificación** (objetivo binario
# `Pobre`), algoritmo de la lista del enunciado: «Boosting». **No se sube nada a Kaggle desde aquí**: el CSV queda en
# `02_outputs/predictions/` y la señal `02_outputs/tables/11_LISTO.json` resume lo necesario para el envío.
#
# **Por qué XGBoost.** XGBoost (Chen y Guestrin, 2016) es *gradient boosting* de segundo orden (usa gradiente y
# hessiano de la pérdida) con regularización explícita de la complejidad del árbol (γ por hoja, L1/L2 sobre los
# pesos) y crecimiento **por niveles** (profundidad máxima), más conservador que el crecimiento por hojas de LightGBM.
# Se usa `tree_method = hist` con `QuantileDMatrix` y categóricas nativas; la importancia de la selección es SHAP
# exacta de los árboles (`pred_contribs`).
#
# **Protocolo común (no cambia entre modelos).**
# * Los 5 pliegues de `pliegues.csv` (estratificados por Pobre × Dominio).
# * Métrica principal: **F1 anidado sin Bogotá**: el umbral de cada pliegue se elige con las predicciones fuera de
#   pliegue de los otros cuatro, sólo con hogares fuera de Bogotá (el test no tiene Bogotá; validación adversarial
#   AUC = 0,50 sin Bogotá), y se aplica al pliegue retenido. EE entre pliegues = sd(F1_k)/√5.
# * Umbral de envío: F1-óptimo OOF sin Bogotá (Lipton, Elkan y Naryanaswamy, 2014).
# * Parada temprana con un 10 % **interno** del entrenamiento de cada pliegue (estratificado y sólo con hogares
#   fuera de Bogotá): el pliegue de validación nunca decide el número de árboles; el modelo final usa
#   ⌈1,1·media(árboles)⌉ con todo train.
#
# **Plan.** (1) Selección de variables con el propio modelo; (2) hiperparámetros con Optuna (el peso de Bogotá es un
# hiperparámetro más); (3) Bogotá; (4) desbalance; (5) umbral global frente a umbrales por dominio; (6) segunda
# partición (optimismo); (7) modelos finales (bolsa de 5 semillas), registro y CSV; (8) métricas y figuras;
# (9) resumen y señal de listo.
#
# La versión de los datos se lee de la variable de entorno `TALLER2_DATOS` (`v5` = primera etapa rediseñada;
# `v4` = tablas del cuaderno 09).

# %%
import os, sys, json, time, warnings
warnings.filterwarnings("ignore")
HILOS = int(os.environ.get("AJUSTE_HILOS", "6"))            # 3 cuadernos en paralelo (≈ 6 hilos cada uno)
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
from IPython.display import display, Markdown
from pobreza.config import Rutas, SEED
from pobreza import io as pio, viz, metrics as M, ajuste as AJ
R = Rutas(RAIZ); viz.estilo(); FIG, TAB = R.figuras, R.tablas
PREF, NB = "11", "11_xgboost"
VERSION = os.environ.get("TALLER2_DATOS") or ("v5" if (R.processed / "v5_LISTO.json").exists() else "v4")
RAPIDO = os.environ.get("AJUSTE_RAPIDO", "0") == "1"         # prueba corta de punta a punta
SUB = float(os.environ.get("AJUSTE_SUBMUESTRA", "0") or 0)   # submuestra de train (sólo pruebas)
REUSAR = os.environ.get("AJUSTE_REUSAR", "1") == "1"         # reanudar desde la caché si el cuaderno se interrumpe
CACHE = R.processed / "ajuste" / f"{PREF}_{VERSION}{'_sub' if SUB else ''}{'_rapido' if RAPIDO else ''}"
T0 = time.time(); TIEMPOS = {}


class _Tee:
    """Copia la salida impresa a un archivo de progreso (para vigilar la corrida desde fuera de Jupyter)."""
    def __init__(self, a, ruta):
        self.a, self.f = a, open(ruta, "a", encoding="utf-8", buffering=1)
    def write(self, s):
        self.f.write(s); return self.a.write(s)
    def flush(self):
        self.f.flush(); self.a.flush()
    def __getattr__(self, k):
        return getattr(self.a, k)


PROGRESO = R.processed / f"progreso_{PREF}.log"
if not isinstance(sys.stdout, _Tee):
    sys.stdout = _Tee(sys.stdout, PROGRESO)
print(f"\n=== {NB} · inicio {time.strftime('%Y-%m-%d %H:%M:%S')} ===")


def SEC(t):
    print(f"--- {t} · {time.strftime('%H:%M:%S')} · {(time.time() - T0) / 60:.1f} min ---", flush=True)

# Variables de la búsqueda automática (cuaderno 09) y conjuntos de la versión elegida
vb_ruta = R.processed / "variables_busqueda.json"
vb = json.load(open(vb_ruta, encoding="utf-8")) if vb_ruta.exists() else {"ganadoras": [], "usan_generadas": []}
conj_json = f"conjuntos_variables_{VERSION}.json"
conj_json = conj_json if (R.processed / conj_json).exists() else "conjuntos_variables.json"
dat = AJ.Datos(R, sufijo=f"_{VERSION}", extra=vb["ganadoras"], extra_gen=vb["usan_generadas"], conj_json=conj_json,
               especificaciones=vb.get("especificaciones"), submuestra=SUB or None)
motor = AJ.MotorXGB(hilos=HILOS)
KW = {**AJ.kw_cv(dat, 1.0), "paciencia": 50}   # selección: peso 1 para todos, parada temprana fuera de Bogotá (paciencia 50)
print(f"datos {VERSION} ({conj_json}) | train {dat.D.shape} | test {dat.DT.shape} | {len(dat.columnas)} columnas candidatas | "
      f"generadas: {len(dat.GEN_TODAS)} | búsqueda 09: {len(dat.extra)} | ampliado: {len(dat.AMPLIADO)} | hilos = {HILOS}")
if SUB:
    print(f"*** PRUEBA con submuestra del {SUB:.0%} de train ***")

# Presupuesto (meta: ≤ 75–90 minutos por cuaderno con 6 hilos)
KS = (10, 20) if RAPIDO else (10, 20, 40, 80, 120)
N_OPT, T_OPT = (4, 3 * 60) if RAPIDO else (40, 28 * 60)
SEMILLAS = (1, 2) if RAPIDO else (1, 2, 3, 4, 5)               # bolsa del modelo principal
SEMILLAS_V = (1,) if RAPIDO else (1, 2, 3)                     # versiones parsimoniosa y sin generadas
N_SHAP = 5000 if RAPIDO else 20000
PARAMS_SEL = {**motor.base, "learning_rate": 0.1}
print("Hiperparámetros base para la selección:", motor.describir(PARAMS_SEL))

# %% [markdown]
# ## 1. Selección de variables con el propio modelo (validación cruzada sin fuga)
#
# Todos los candidatos se evalúan con los **mismos pliegues** y los hiperparámetros base (XGBoost con profundidad 6, min_child_weight = 20, lr = 0,1 y parada temprana interna):
#
# | id | conjunto | idea |
# |---|---|---|
# | a | **Encuesta corta** | preguntas baratas y verificables: tamaño y composición, jefe y educación, ocupados y cotizantes, vivienda y tenencia, régimen de salud, territorio |
# | b | S1 | PMT parsimonioso del 02: un representante por mecanismo |
# | c | S2 / S3 + categóricas | todas las construidas (S3 = S2 + generadas de 1.ª etapa) |
# | d | S3 + búsqueda 09 | más las combinaciones aceptadas por CV en el 09 |
# | e | ampliado v5 | el conjunto ampliado de la versión 5 (si existe) |
# | f | top-k por importancia del propio modelo (SHAP), k ∈ {10, 20, 40, 80, 120} | el ranking se calcula **dentro** del entrenamiento de cada pliegue externo y se evalúa en el pliegue retenido; un pool con generadas y otro sin ellas |
# | g | eliminación por bloques A–H | desde el mejor conjunto, una pasada: son admisibles los bloques cuya salida no empeora el F1 más de 1 EE; se intenta quitar todos los admisibles a la vez y, si viola la regla, el admisible que deja **menos** variables |
#
# Cada candidato aparece **con y sin** las variables generadas de la primera etapa (modelos auxiliares del ingreso),
# por si se consideran ruta de regresión.

# %%
SEC("1. Selección de variables"); t = time.time()
sel = AJ.en_cache(CACHE / "seleccion.pkl", lambda: AJ.seleccion_v5(motor, dat, PARAMS_SEL, ks=KS, n_shap=N_SHAP,
                                                                   medida="SHAP", kw=KW), REUSAR)
TIEMPOS["seleccion_min"] = (time.time() - t) / 60
tabla_sel = sel["tabla"].sort_values("F1_anidado_sinBOG", ascending=False)
tabla_sel.to_csv(TAB / f"{PREF}_seleccion_variables.csv", index=False)
display(tabla_sel[["conjunto", "n_vars", "con_generadas", "F1_anidado_sinBOG", "EE", "AUC", "LogLoss_sinBOG", "tiempo_s"]].round(4))
elegido = sel["elegido"]; pars = AJ.mejor_parsimonioso(sel, 20); singen = AJ.mejor_sin_generadas(sel)
pd.DataFrame({"variable": elegido.cols(), "bloque": [AJ.NOMBRE_BLOQUE[dat.bloques().get(c, "A")] for c in elegido.cols()]}).to_csv(
    TAB / f"{PREF}_variables_elegidas.csv", index=False)
print(f"Mejor antes de la eliminación: {sel['mejor_inicial']}\nElegido: {elegido.nombre} ({elegido.n_vars} variables)\n"
      f"Mejor parsimonioso (≤ 20 variables): {pars.nombre} ({pars.n_vars})\nMejor sin generadas: {singen.nombre} ({singen.n_vars})")
print(f"Selección: {TIEMPOS['seleccion_min']:.1f} min")

# %%
fig = AJ.figura_curva_variables(sel["tabla"], f"XGBoost ({VERSION}): F1 anidado sin Bogotá frente al número de variables",
                                FIG / f"{PREF}_curva_variables.png")
plt.show()

# %% [markdown]
# ## 2. Hiperparámetros con Optuna
#
# **Espacio de búsqueda.** max_depth ∈ {3,…,10}, min_child_weight ∈ [1, 300] (log), eta (learning_rate) ∈ [0,02; 0,12] (log),
# subsample ∈ [0,5; 1], colsample_bytree ∈ [0,3; 1], colsample_bylevel ∈ [0,5; 1], gamma ∈ [10⁻³, 5] (log),
# reg_alpha ∈ [10⁻³, 10] (log), reg_lambda ∈ [10⁻², 100] (log), max_bin ∈ {64, 128, 256}. Además, **peso de Bogotá** en el entrenamiento ∈ {1; 0,5; 0}.
#
# **Objetivo.** Se minimiza el **log-loss OOF sin Bogotá** (parada temprana interna) y el F1 anidado sin Bogotá es el
# **criterio final** entre los 5 mejores ensayos. Razón: el F1 depende de un umbral y es escalonado, así que entre
# configuraciones parecidas su diferencia es casi ruido; el log-loss es una regla de puntuación
# propia, suave y continua, que premia ordenar y calibrar bien a **todos** los hogares y es mucho menos ruidosa.
# Elegir por F1 entre los mejores por log-loss evita perseguir el ruido del F1. Las estrategias de desbalance no entran en la
# búsqueda (una pérdida propia siempre castiga reponderar porque descalibra): se comparan aparte, por F1 (sección 4).
#
# **Muestreador y poda.** TPE multivariado (Bergstra et al., 2011; Watanabe, 2023) con semilla 20262 y 10 ensayos
# iniciales aleatorios; poda **Hyperband** (Li et al., 2018) por pliegue: tras cada pliegue se informa la pérdida
# acumulada y los ensayos malos se detienen. El primer ensayo es la configuración base (mide cuánto aporta el ajuste).

# %%
SEC("2. Optuna"); t = time.time()
def _optuna():
    est, oofs = AJ.optuna_v5(motor, dat, elegido, n_ensayos=N_OPT, timeout=T_OPT, semilla=SEED,
                             encolar=motor.base_optuna(), nombre=f"{PREF}_{VERSION}")
    mt = AJ.elegir_ensayo(est)
    return {"estudio": est, "oof_mejor": oofs[mt.number], "tabla": AJ.tabla_ensayos(est), "importancia": AJ.importancia_hiper(est)}
op = AJ.en_cache(CACHE / "optuna.pkl", _optuna, REUSAR)
TIEMPOS["optuna_min"] = (time.time() - t) / 60
est = op["estudio"]; mejor_t = AJ.elegir_ensayo(est)
params, PB_OPT = AJ.params_de_ensayo(motor, mejor_t)
base_t = est.trials[0]
op["tabla"].rename(columns={"LogLoss": "objetivo"}).to_csv(TAB / f"{PREF}_optuna_ensayos.csv", index=False)
op["importancia"].rename("importancia").to_frame().to_csv(TAB / f"{PREF}_optuna_importancia.csv")
display(op["tabla"].rename(columns={"LogLoss": "objetivo"}).sort_values("objetivo").head(10).round(5))
print(f"Ensayo elegido #{mejor_t.number}: {motor.perdida} = {mejor_t.value:.5f}, F1 anidado sin Bogotá = "
      f"{mejor_t.user_attrs['F1_anidado_sinBOG']:.4f}, peso de Bogotá = {PB_OPT}")
print(f"Ensayo base #0: {motor.perdida} = {base_t.value}, F1 = {base_t.user_attrs.get('F1_anidado_sinBOG')}")
print("Hiperparámetros elegidos:", motor.describir(params))
print(f"Optuna: {TIEMPOS['optuna_min']:.1f} min")

# %%
fig = AJ.figura_optuna(est, op["importancia"], f"XGBoost ({VERSION}) — Optuna sobre «{elegido.nombre}»",
                       FIG / f"{PREF}_optuna_historia.png", etiqueta="log-loss"); plt.show()

# %% [markdown]
# ## 3. Bogotá: ¿entrenar con todos, con peso 0,5 o sin Bogotá?
# El test no tiene Bogotá, pero sus 10.567 hogares aportan información sobre la relación entre variables y pobreza.
# Con el resto de hiperparámetros fijos en los de Optuna se comparan las tres opciones de **entrenamiento** (la
# evaluación y el umbral siempre son sin Bogotá, y la parada temprana usa sólo hogares fuera de Bogotá). Se elige la de
# mayor F1 anidado sin Bogotá (es un hiperparámetro más, elegido por validación cruzada con los mismos pliegues).

# %%
SEC("3. Bogotá"); t = time.time()
var_bog = {f"peso Bogotá = {pb}": (dict(params), pb) for pb in (1.0, 0.5, 0.0)}
bogt, oofs_bog = AJ.en_cache(CACHE / "bogota.pkl", lambda: AJ.comparar_variantes(
    motor, dat, elegido, var_bog, previos={f"peso Bogotá = {PB_OPT}": op["oof_mejor"]}, referencia="peso Bogotá = 1.0"), REUSAR)
TIEMPOS["bogota_min"] = (time.time() - t) / 60
bogt.to_csv(TAB / f"{PREF}_bogota.csv", index=False)
display(bogt[["variante", "F1_anidado_sinBOG", "EE", "dF1_vs_ref", "EE_pareado", "AUC_sinBOG", "LogLoss_sinBOG", "Brier_sinBOG"]].round(4))
PESO_BOG = float(bogt.sort_values("F1_anidado_sinBOG", ascending=False).iloc[0].peso_bogota)
print(f"Peso de Bogotá elegido para el entrenamiento: {PESO_BOG}")

# %% [markdown]
# ## 4. Estrategias de desbalance (≈ 20 % de pobres)
# Con F1 y un umbral elegido por validación, reponderar la clase minoritaria no debería ser necesario: mover el umbral
# ya corrige el desbalance (Lipton et al., 2014) y reponderar sólo descalibra las probabilidades. Se comparan, con el
# resto fijo, ninguna (sólo umbral), `scale_pos_weight` = n0/n1 y `scale_pos_weight` = √(n0/n1). Se adopta una reponderación sólo si mejora el F1 anidado sin Bogotá en más de 1 EE pareado.

# %%
SEC("4. Desbalance"); t = time.time()
razon = float((dat.y == 0).sum() / (dat.y == 1).sum())
var_desb = {nom: (p, PESO_BOG) for nom, p in motor.desbalance(params, razon).items()}
ref_desb = list(var_desb)[0]
desb, oofs_desb = AJ.en_cache(CACHE / "desbalance.pkl", lambda: AJ.comparar_variantes(
    motor, dat, elegido, var_desb, previos={ref_desb: oofs_bog[f"peso Bogotá = {PESO_BOG}"]}, referencia=ref_desb), REUSAR)
TIEMPOS["desbalance_min"] = (time.time() - t) / 60
desb.to_csv(TAB / f"{PREF}_desbalance.csv", index=False)
display(desb[["variante", "F1_anidado_sinBOG", "EE", "dF1_vs_ref", "EE_pareado", "AUC_sinBOG", "precision_sinBOG", "recall_sinBOG", "Brier_sinBOG"]].round(4))
alt = desb[desb.variante != ref_desb].sort_values("F1_anidado_sinBOG", ascending=False).iloc[0]
DESB = alt.variante if alt.dF1_vs_ref > alt.EE_pareado else ref_desb
params_fin = var_desb[DESB][0]; oof_1 = oofs_desb[DESB]
print(f"Estrategia adoptada: {DESB}")

# %% [markdown]
# ## 5. Umbral: global frente a umbrales por dominio
# El umbral de envío es el F1-óptimo OOF sin Bogotá. Si las probabilidades están bien calibradas en cada dominio, un
# umbral único es óptimo (Lipton et al., 2014: p ≥ F1*/2). Se evalúan además, con **validación anidada**, umbrales por
# dominio que maximizan el F1 global (ascenso por coordenadas, δ ∈ [−0,12; 0,12] alrededor del global; dominios con
# menos de 30 pobres conservan el global): se adoptan sólo si mejoran el F1 anidado en más de 1 EE.

# %%
SEC("5. Umbral"); um, um_t = AJ.evaluar_umbral_dominio(dat, oof_1)
UMBRAL_DOM = um["adoptar"]
um_t.to_csv(TAB / f"{PREF}_umbral_dominio_pliegues.csv", index=False)
pd.DataFrame([um]).to_csv(TAB / f"{PREF}_umbral_dominio.csv", index=False)
display(um_t.round(4)); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in um.items()})
m_ = dat.sin_bog
t_glob, f1_glob = M.umbral_optimo_f1(dat.y[m_], oof_1[m_])
print(f"Umbral global F1-óptimo OOF sin Bogotá = {t_glob:.4f} (F1 OOF = {f1_glob:.4f}); regla de Lipton F1*/2 = {f1_glob / 2:.4f}")
print(f"Umbrales por dominio {'ADOPTADOS' if UMBRAL_DOM else 'no adoptados'} "
      f"(ΔF1 anidado = {um['diferencia']:+.4f}, 1 EE = {um['EE_global']:.4f})")

# %% [markdown]
# ## 6. Estimación honesta: segunda partición
# La selección de variables y el ajuste usaron los mismos 5 pliegues, así que el mejor F1 está sesgado hacia arriba
# (el máximo de muchas estimaciones ruidosas; Cawley y Talbot, 2010). Se repite la configuración final con una
# **segunda partición** en 5 pliegues (otra semilla, estratificada por Pobre × Dominio). Si el conjunto es un top-k, el
# ranking se rehace dentro de cada pliegue nuevo. Optimismo = F1(original) − F1(segunda), ambos con una semilla.
# *Advertencia:* las variables generadas son OOF respecto de la partición original; en la segunda partición queda una
# fuga de segundo orden (débil) por esa vía.

# %%
SEC("6. Segunda partición"); t = time.time()
folds2 = AJ.segunda_particion(dat, SEED + 101)
def _segunda():
    c2 = AJ.conjunto_en_particion(motor, dat, elegido, folds2, PARAMS_SEL, N_SHAP)
    r2 = AJ.cv(motor, dat.X, dat.y, folds2, params_fin, c2, **AJ.kw_cv(dat, PESO_BOG))
    return AJ.resumen(dat.y, r2["oof"], folds2, dat.sin_bog)
res2 = AJ.en_cache(CACHE / "segunda_particion.pkl", _segunda, REUSAR)
TIEMPOS["segunda_particion_min"] = (time.time() - t) / 60
res1 = AJ.resumen(dat.y, oof_1, dat.folds, dat.sin_bog)
optim = pd.DataFrame([{"particion": "original (pliegues.csv)", **{k: res1[k] for k in ("F1_anidado_sinBOG", "EE", "AUC", "LogLoss_sinBOG")}},
                      {"particion": f"segunda (semilla {SEED + 101})", **{k: res2[k] for k in ("F1_anidado_sinBOG", "EE", "AUC", "LogLoss_sinBOG")}}])
optim["optimismo_F1"] = [np.nan, res1["F1_anidado_sinBOG"] - res2["F1_anidado_sinBOG"]]
optim.to_csv(TAB / f"{PREF}_optimismo.csv", index=False); display(optim.round(4))

# %% [markdown]
# ## 7. Modelos finales, registro y CSV de envío
# Tres versiones, cada una con OOF y test promediando semillas y reentrenada con todo train (con el tratamiento de
# Bogotá elegido):
# 1. **principal**: conjunto elegido + hiperparámetros de Optuna, bolsa de 5 semillas;
# 2. **parsimoniosa**: el mejor conjunto con ≤ 20 variables (mismos hiperparámetros, 3 semillas);
# 3. **sin generadas**: el mejor conjunto sin variables generadas de 1.ª etapa (3 semillas).

# %%
SEC("7. Modelos finales"); t = time.time()
PF = params_fin
versiones = {"principal": (elegido, SEMILLAS, ""), "parsimoniosa": (pars, SEMILLAS_V, "_pars"), "sin generadas": (singen, SEMILLAS_V, "_singen")}
fin = {}
for nom, (cj, sem, suf) in versiones.items():
    if nom != "principal" and cj.nombre == elegido.nombre:      # el elegido ya cumple: no se repite el ajuste
        fin[nom] = fin["principal"]; print(f"{nom}: coincide con la principal ({cj.n_vars} variables)"); continue
    fin[nom] = AJ.en_cache(CACHE / f"final_{suf or 'principal'}.pkl", lambda: AJ.version_final_v5(
        motor, dat, cj, PF, sem, f"XGB {VERSION} {nom} ({cj.n_vars} vars)", suf, R, NB, peso_bogota=PESO_BOG,
        descripcion=f"Cuaderno {NB}; datos {VERSION}; selección con el propio modelo y Optuna; desbalance: {DESB}.",
        umbral_por_dominio=(UMBRAL_DOM and nom == "principal")), REUSAR)
TIEMPOS["finales_min"] = (time.time() - t) / 60
print(f"Modelos finales: {TIEMPOS['finales_min']:.1f} min")

# %% [markdown]
# ## 8. Métricas y figuras
# F1 anidado sin Bogotá con IC95 bootstrap (remuestreo de hogares con las predicciones anidadas fijas), F1 por pliegue,
# AUC-ROC, AUC-PR, Brier, log-loss, ECE, precisión y exhaustividad; curva F1-umbral, curva PR, calibración, matriz de
# confusión por 1.000 hogares, F1 por dominio y la disyuntiva filtración/subcobertura en 3 umbrales.

# %%
SEC("8. Métricas y figuras"); filas = []
for nom, f in fin.items():
    met, fd, dis = AJ.diagnostico(dat, f["oof"], f"XGBoost ({VERSION}) — versión {nom} ({f['conjunto'].n_vars} variables)",
                                  PREF if nom == "principal" else f"{PREF}_{nom.replace(' ', '_')}", FIG, TAB, figuras=(nom == "principal"))
    met.update({"Brier_sinBOG": M.brier(dat.y[m_], f["oof"][m_]), "LogLoss_sinBOG": M.log_loss(dat.y[m_], f["oof"][m_])})
    filas.append({"version": nom, "conjunto": f["conjunto"].nombre, "n_vars": f["conjunto"].n_vars, **met,
                  "umbral_csv": f["umbral"], "tasa_predicha_test": f["tasa_test"], "archivo_envio": os.path.basename(f["archivo"]),
                  "semillas": len(f["semillas"]), "peso_bogota": f["peso_bogota"], "params": motor.describir(f["params"])})
    if nom == "principal":
        f1_dom, disy = fd, dis
metricas = pd.DataFrame(filas)
metricas.loc[metricas.version == "principal", "F1_segunda_particion"] = res2["F1_anidado_sinBOG"]
metricas.to_csv(TAB / f"{PREF}_metricas.csv", index=False)
p0 = metricas.set_index("version")
display(metricas[["version", "n_vars", "F1_anidado_sinBOG", "IC95_inf", "IC95_sup", "AUC_ROC", "AUC_PR", "Brier", "LogLoss", "ECE",
                  "precision_sinBOG", "recall_sinBOG", "umbral_csv", "tasa_predicha_test"]].round(4))
display(disy.round(4))

# %%
imp_final = AJ.shap_final(motor, dat, elegido, PF, fin["principal"]["n_final"], PREF, FIG, TAB, f"XGBoost ({VERSION}, principal)",
                          n=N_SHAP, peso_bogota=PESO_BOG)
display(imp_final.head(20).round(4))

# %% [markdown]
# ## 9. Resumen

# %%
TIEMPOS["total_min"] = (time.time() - T0) / 60
fp = fin["principal"]; mp = p0.loc["principal"]
tasa_tr = float(dat.y[dat.sin_bog].mean())
tasa_oof = float((fp["oof"][m_] >= fp["umbral"]).mean())         # tasa predicha OOF sin Bogotá con el umbral de envío
coherente = abs(fp["tasa_test"] - tasa_oof) < 0.02 and 0.15 <= fp["tasa_test"] <= 0.28
f1_ini = tabla_sel[tabla_sel.conjunto == sel["mejor_inicial"]].iloc[0]
txt = f"""
**Datos {VERSION}.** {len(tabla_sel)} conjuntos evaluados; el mejor por F1 anidado sin Bogotá fue «{sel['mejor_inicial']}»
(F1 = {f1_ini.F1_anidado_sinBOG:.4f} ± {f1_ini.EE:.4f}). Tras la eliminación por bloques (regla de 1 EE) queda **«{elegido.nombre}»
con {elegido.n_vars} variables**.

**Hiperparámetros (Optuna, ensayo #{mejor_t.number} de {len(est.trials)}).** {motor.describir(params_fin)}.
Ensayo base #0: F1 = {base_t.user_attrs.get('F1_anidado_sinBOG', float('nan')):.4f}; ensayo elegido: F1 = {mejor_t.user_attrs['F1_anidado_sinBOG']:.4f}.

**Bogotá.** Peso en el entrenamiento = {PESO_BOG} ({"; ".join(f"{r.variante}: {r.F1_anidado_sinBOG:.4f}" for r in bogt.itertuples())}).
**Desbalance.** Adoptado: {DESB} ({"; ".join(f"{r.variante}: {r.F1_anidado_sinBOG:.4f}" for r in desb.itertuples())}).
**Umbral.** Global F1-óptimo OOF sin Bogotá = {fp['umbral']:.4f}; umbrales por dominio {'adoptados' if UMBRAL_DOM else 'no adoptados'}
(ΔF1 anidado = {um['diferencia']:+.4f} frente a 1 EE = {um['EE_global']:.4f}).

**Modelo final ({len(SEMILLAS)} semillas).** **F1 anidado sin Bogotá = {mp.F1_anidado_sinBOG:.4f}** (IC95 {mp.IC95_inf:.4f}–{mp.IC95_sup:.4f}),
AUC = {mp.AUC_ROC:.4f}, AUC-PR = {mp.AUC_PR:.4f}, precisión = {mp.precision_sinBOG:.4f}, exhaustividad = {mp.recall_sinBOG:.4f}.
Segunda partición (1 semilla): F1 = {res2['F1_anidado_sinBOG']:.4f} frente a {res1['F1_anidado_sinBOG']:.4f} en la original
(optimismo = {res1['F1_anidado_sinBOG'] - res2['F1_anidado_sinBOG']:+.4f}).

**Parsimonia.** «{pars.nombre}» ({pars.n_vars} variables): F1 = {p0.loc['parsimoniosa', 'F1_anidado_sinBOG']:.4f};
sin generadas «{singen.nombre}» ({singen.n_vars} variables): F1 = {p0.loc['sin generadas', 'F1_anidado_sinBOG']:.4f}.

**Envío (no subido).** `{os.path.basename(fp['archivo'])}`; tasa de pobres predicha en test = {fp['tasa_test']:.4f}
(OOF sin Bogotá con el mismo umbral: {tasa_oof:.4f}; tasa real en train sin Bogotá: {tasa_tr:.4f}; {'coherente' if coherente else 'REVISAR'}).
Tiempo total: {TIEMPOS['total_min']:.0f} min.
"""
display(Markdown(txt))
json.dump({"modelo": "XGBoost", "datos": VERSION, "conjunto": elegido.nombre, "n_vars": elegido.n_vars, "variables": elegido.cols(),
           "params": params_fin, "peso_bogota": PESO_BOG, "desbalance": DESB, "umbral_por_dominio": UMBRAL_DOM,
           "metricas": filas, "F1_segunda_particion": res2["F1_anidado_sinBOG"],
           "optimismo": float(res1["F1_anidado_sinBOG"] - res2["F1_anidado_sinBOG"]),
           "F1_base_ensayo0": base_t.user_attrs.get("F1_anidado_sinBOG"), "tiempos_min": TIEMPOS, "resumen": txt},
          open(TAB / f"{PREF}_resumen.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)

# %% [markdown]
# ## 10. Verificación del CSV y señal de listo para el envío

# %%
sample = pio.cargar_sample_submission(R)
def verificar(ruta):
    csv = pd.read_csv(ruta)
    assert list(csv.columns) == ["id", "pobre"], csv.columns
    assert len(csv) == len(sample) == 66168, len(csv)
    assert (csv.id.values == sample.id.values).all()
    assert set(csv.pobre.unique()) <= {0, 1}
    return float(csv.pobre.mean())
for v in fin:
    verificar(fin[v]["archivo"])
CLAVES = ["max_depth", "learning_rate", "min_child_weight", "subsample", "colsample_bytree", "reg_lambda"]
hp = ", ".join(f"{k}={params_fin[k]:.4g}" if isinstance(params_fin.get(k), float) else f"{k}={params_fin.get(k)}" for k in CLAVES)
listo = {"modelo": "XGBoost", "algoritmo_enunciado": "Boosting", "datos": VERSION, "version": "principal",
         "ruta_csv": os.path.abspath(fp["archivo"]), "archivo": os.path.basename(fp["archivo"]),
         "descripcion_kaggle": (f"XGBoost (clasificación, {motor.corto}); {hp}; {elegido.n_vars} variables ({elegido.nombre}); "
                                f"peso Bogotá {PESO_BOG}; desbalance: {DESB}; umbral {fp['umbral']:.3f} (F1-óptimo OOF sin Bogotá"
                                f"{', por dominio' if UMBRAL_DOM else ''}); F1 CV anidado sin Bogotá {mp.F1_anidado_sinBOG:.4f}; "
                                f"bolsa de {len(SEMILLAS)} semillas; datos {VERSION}"),
         "n_vars": elegido.n_vars, "conjunto": elegido.nombre, "hiperparametros": params_fin, "peso_bogota": PESO_BOG,
         "desbalance": DESB, "umbral": fp["umbral"], "umbral_por_dominio": UMBRAL_DOM,
         "F1_anidado_sinBOG": float(mp.F1_anidado_sinBOG), "IC95": [float(mp.IC95_inf), float(mp.IC95_sup)],
         "F1_segunda_particion": float(res2["F1_anidado_sinBOG"]), "optimismo": float(res1["F1_anidado_sinBOG"] - res2["F1_anidado_sinBOG"]),
         "AUC": float(mp.AUC_ROC), "precision": float(mp.precision_sinBOG), "exhaustividad": float(mp.recall_sinBOG),
         "tasa_predicha_test": float(fp["tasa_test"]), "tasa_predicha_oof_sinBOG": tasa_oof, "tasa_pobres_train_sinBOG": tasa_tr,
         "tasa_coherente": bool(coherente),
         "otras_versiones": [{"version": v, "conjunto": fin[v]["conjunto"].nombre, "ruta_csv": os.path.abspath(fin[v]["archivo"]),
                              "archivo": os.path.basename(fin[v]["archivo"]), "n_vars": fin[v]["conjunto"].n_vars,
                              "F1_anidado_sinBOG": float(p0.loc[v, "F1_anidado_sinBOG"]), "AUC": float(p0.loc[v, "AUC_ROC"]),
                              "umbral": fin[v]["umbral"], "tasa_predicha_test": fin[v]["tasa_test"]} for v in ("parsimoniosa", "sin generadas")],
         "verificado": "id,pobre; 66.168 filas; ids y orden de sample_submission; valores 0/1",
         "submuestra_prueba": SUB or None, "tiempos_min": TIEMPOS, "tiempo_total_min": TIEMPOS["total_min"]}
json.dump(listo, open(TAB / f"{PREF}_LISTO.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(listo, ensure_ascii=False, indent=1, default=str))
print(f"=== {NB} · fin {time.strftime('%Y-%m-%d %H:%M:%S')} ({TIEMPOS['total_min']:.0f} min) ===")
