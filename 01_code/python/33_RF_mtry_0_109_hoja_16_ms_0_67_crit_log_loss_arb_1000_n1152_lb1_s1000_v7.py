# %% [markdown]
# > **Repositorio BDML-PS02 · script 33.** Lo ejecuta `01_code/33_RF_mtry_0_109_hoja_16_ms_0_67_crit_log_loss_arb_1000_n1152_lb1_s1000_v7.R`. Es el cuaderno interno `34_random_forest`:
# > sus tablas y figuras conservan el prefijo `34_` y quedan en `02_outputs/tables/33_RF_v7/` y
# > `02_outputs/figures/33_RF_v7/`; los CSV de envío, con el nombre con que se generaron (y, si se subieron,
# > con el que se subieron a Kaggle), en `02_outputs/predictions/`.

# %% [markdown]
# # 34 · Random Forest con el protocolo v7
# **Problem Set 2 — Predicción de pobreza (MESE/DANE) · BDML Uniandes 2026-20 · ruta de clasificación**
#
# Algoritmo del enunciado: **«Random Forest»**. Este cuaderno aplica, sin cambios, el **protocolo recomendado** del memo
# metodológico (`01_code/python/docs/memo_pronostico.md`, §10 y tabla §11) programado en `01_code/python/pobreza/protocolo_v7.py`
# (submódulos en `01_code/python/pobreza/p7/`). **No se sube nada a Kaggle**: los CSV quedan en `02_outputs/predictions/` y la señal
# `02_outputs/tables/34_LISTO.json` resume lo necesario para que el coordinador decida.
#
# **Por qué y cómo.** *Bagging* de árboles con submuestreo de variables en cada partición (Breiman, 2001). Es el algoritmo **menos ajustable** (Probst, Boulesteix y Bischl, 2019: sus defaults son casi óptimos y `mtry` es lo más relevante) y el número de árboles no se ajusta: la log-loss esperada decrece monótonamente con él (Probst y Boulesteix, 2018). Cada predicción OOB usa sólo los árboles que no vieron al hogar: es tan honesta como una OOF.
#
# ## El problema estadístico en una página
#
# * **Estimando.** El test es una muestra aleatoria (estratificada por dominio) de los 24 dominios distintos de Bogotá
#   ($\mathcal T$): validación adversarial $\mathcal T$ vs test con AUC 0,499. El riesgo que importa es
#   $R_{\mathcal T}(f)=E_{P_S}[w(X)\ell(f(X),Y)]$ con $w=\mathbb 1\{d\neq B\}/(1-\pi_B)$ **exactos**, de modo que la
#   validación cruzada ponderada por importancia (Sugiyama, Krauledat y Müller, 2007) es promediar **sólo sobre los
#   hogares de $\mathcal T$** de los pliegues de validación. Umbral, prevalencia, métricas, comparaciones y selección:
#   siempre sin Bogotá (memo §1.2).
# * **Métrica principal: F1 agregado anidado en $\mathcal T$.** Se suman TP, FP y FN de todos los pliegues y se calcula
#   un solo cociente (consistente; Forman y Scholz, 2010). El umbral del pliegue $k$ es el argmax del F1 en las OOF de
#   $\mathcal T$ de los otros pliegues (*plug-in* en dos pasos, consistente: Koyejo et al., 2014; Narasimhan, Vaish y
#   Agarwal, 2014).
# * **Incertidumbre.** Linealizando $F=N/D$ con $N=2E[t]$, $D=E[2t+f+g]$:
#   $\psi_i=\{2t_i-F(2t_i+f_i+g_i)\}/D$ y $\mathrm{EE}(\hat F)\approx\mathrm{sd}(\psi)/\sqrt n$; además IC bootstrap
#   percentil de hogares. Las diferencias entre procedimientos usan las $J=R\cdot K$ diferencias por (partición, pliegue)
#   con el **t corregido de Nadeau y Bengio (2003)**, $t=\bar d/\sqrt{(1/J+1/(K-1))\hat\sigma^2_d}$, y la **prueba bayesiana
#   correlacionada** (Benavoli et al., 2017) con ROPE de 0,002 en F1 y 0,0005 en log-loss.
# * **Particiones.** $R=3$ particiones × $K=5$ pliegues estratificados por Pobre × Dominio (la 1 es `pliegues.csv`;
#   semillas 20263 y 20264), más una cuarta (20265) reservada a la **re-evaluación** de la configuración elegida.
#   Mismas particiones y semillas para todos los algoritmos (comparaciones pareadas). Con el presupuesto `express` las
#   comparaciones usan $R=2$ (el número efectivo queda en la señal de listo).
#
# **Ejecución.** Las variables de entorno (las fija el script `.R` que lo ejecuta (`01_code/NN_*.R`)): `P7_PRESUPUESTO` (`memo`, `reducido`,
# `express`, `prueba`), `P7_HILOS`, `P7_SUBMUESTRA` (sólo pruebas), `P7_SALIDA` (raíz de escritura), `P7_CACHE`, `P7_REUSAR`. Cada
# paso guarda su resultado en la caché: si la corrida se interrumpe, al relanzarla continúa donde iba.

# %%
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
HILOS = int(os.environ.get("P7_HILOS", "4"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):   # BLAS acotado: varios cuadernos en paralelo
    os.environ[_v] = str(HILOS)
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
from pobreza import protocolo_v7 as PV
from pobreza.p7.entorno import Entorno, presupuesto
ENT = Entorno.desde_variables(cuaderno="34_random_forest")
P = presupuesto()


class _Tee:
    """Copia la salida a un archivo de progreso (para vigilar la corrida desde fuera de Jupyter)."""
    def __init__(self, a, ruta):
        self.a, self.f = a, open(ruta, "a", encoding="utf-8", buffering=1)
    def write(self, s):
        self.f.write(s); return self.a.write(s)
    def flush(self):
        self.f.flush(); self.a.flush()
    def __getattr__(self, k):
        return getattr(self.a, k)


PROGRESO = os.path.join(str(ENT.procesados), "progreso_34.log")
if not isinstance(sys.stdout, _Tee):
    sys.stdout = _Tee(sys.stdout, PROGRESO)
print(f"\n=== 34_random_forest · inicio {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
pr = PV.Protocolo("34_random_forest", ENT, P)
d = pr.datos


def SEC(t):
    print(f"--- {t} · {time.strftime('%H:%M:%S')} · {(time.time() - pr.t0) / 60:.1f} min ---", flush=True)

# %% [markdown]
# ## 0. Datos, verificaciones de integridad y particiones (Pasos 0–1)
# Se afirma que (a) los dominios del test están contenidos en $\mathcal T$; (b) ningún valor de $L_p$ de Bogotá aparece en
# el test (soporte disjunto: por eso Bogotá nunca puede evaluarse como si fuera test); (c) ninguna variable prohibida
# (sólo-train, derivadas de modelos de ingreso, constantes externas) está en el conjunto; (d) train y test tienen las
# mismas columnas (mismo código de construcción; Zinkevich, reglas #29/#32). La prevalencia esperada del test es la
# de $\mathcal T$ post-estratificada por la composición de dominios del test.

# %%
SEC("0. Datos")
info0 = pr.paso0_datos()
display(pd.Series({k: v for k, v in info0["verificacion"].items() if not isinstance(v, list)}, name="verificación").to_frame())
display(pd.Series(info0["prevalencia"], name="prevalencia").to_frame())
display(pd.Series(info0["bogota"], name="Bogotá").to_frame())
print(f"Fuente {info0['fuente']}: {info0['n_variables']} variables, {info0['n_preguntas']} preguntas (mapa {info0['mapa_fuente']}); "
      f"particiones: {info0['particiones']}")

# %% [markdown]
# ## 1. Referencia: todas las variables con hiperparámetros por defecto
# Punto de partida para medir lo que aportan la selección y el ajuste (partición 1, λ_B = 1, una semilla).

# %%
SEC("1. Referencia")
ref = pr.paso1_referencia()
mr = ref["metricas"]
display(pd.Series({k: mr[k] for k in ("F1_anidado", "EE_psi", "IC95_F1", "LogLoss", "Brier", "AUC_ROC", "AUC_PR", "ACE",
                                       "precision", "exhaustividad", "MCC", "F1_oraculo", "brecha_oraculo")}, name="referencia").to_frame())

# %% [markdown]
# ## 2. Selección de variables dentro del pliegue, con la pregunta como unidad (Paso 6; memo §3)
#
# **La selección es parte del procedimiento.** Si las variables se eligieran mirando la etiqueta de toda la muestra, la
# validación cruzada sería optimista (Ambroise y McLachlan, 2002; Varma y Simon, 2006): cada paso se re-ejecuta en el
# entrenamiento de cada pliegue externo y se evalúa en el retenido; el conjunto final se recalcula con todo el
# entrenamiento con la misma receta.
#
# **Unidad de parsimonia.** El costo de un conjunto es el número de **preguntas** del cuestionario,
# $c(S)=|\bigcup_{j\in S}Q(j)|$; dominio, departamento, clase y las líneas son de costo cero. Una importancia aditiva se
# asigna a las preguntas con el reparto de Shapley de un juego de unanimidad ($\phi_j/|Q(j)|$ a cada pregunta de $Q(j)$).
#
# **Método de esta familia.** **Eliminación recursiva por bloques de pregunta** con importancia por **permutación agrupada sobre las observaciones fuera de bolsa** del bosque de cada pliegue (se permutan juntas todas las variables que dependen de la pregunta). Nunca la importancia por impureza, sesgada hacia variables continuas o de muchas categorías (Strobl et al., 2007); la permutación marginal se sesga con correlación (Strobl et al., 2008; Hooker, Mentch y Zhou, 2021), por eso se agrupa y se confirma con LOCO (§8). Para la eliminación se usan bosques baratos (100 árboles, mtry 0,1, hoja 20, muestra 0,5): hiperparámetros de molestia.
#
# **Regla de elección.** Conjunto parsimonioso = el de menor $c(S)$ cuyo F1 anidado no es peor que el del mejor en más de
# 0,002 **y** cuya pérdida propia no es peor en más de su ROPE. Se reporta además el conjunto de confianza CVC al 90 %
# (Lei, 2020) y la **estabilidad**: Jaccard medio entre los conjuntos de los pliegues (un conjunto con $\bar J<0{,}5$ no
# debe presentarse como «las variables que importan»; Meinshausen y Bühlmann, 2010; Yu, 2013).

# %%
SEC("2. Selección")
sel = pr.paso2_seleccion()
if "curva" in sel:
    display(sel["curva"][["punto", "n_vars", "n_preguntas", "n_preguntas_final", "F1_anidado", "EE_psi", "perdida", "AUC_ROC",
                          "dF1_vs_mejor", "dF1_EE_psi", "dperdida_vs_mejor", "en_ROPE", "parsimonioso"]].round(5))
    print("Estabilidad del conjunto parsimonioso:", {k: v for k, v in sel["estabilidad"].items() if k != "frecuencia_preguntas"})
    if "cvc" in sel:
        print("Conjunto de confianza CVC (90 %):", sel["cvc"]["conjunto"])
else:
    print("En esta familia la selección se hace junto con el ajuste (sección 3).")

# %% [markdown]
# ## 3. Hiperparámetros (Pasos 7–8; memo §4)
#
# **Objetivo: log-loss OOF en $\mathcal T$, no F1.** La log-loss es estrictamente propia (Gneiting y Raftery, 2007): su
# minimizador poblacional es $\eta(x)=P(Y=1\mid x)$; el exceso de log-loss es $E_X[\mathrm{KL}(\eta\|\hat\eta)]$ y, por
# Pinsker, $E|\hat\eta-\eta|\le\sqrt{\Delta/2}$; como el F1 del *plug-in* converge a $F^*$ cuando
# $E|\hat\eta-\eta|\to0$, minimizar la log-loss persigue la señal correcta, mientras que el F1 en el umbral óptimo es
# escalonado y su rango entre configuraciones (0,0012–0,0015 en las búsquedas previas) es del orden del ruido entre
# semillas (≈ 0,0007).
#
# **Estrategia.** Sobol (puntos iniciales) + TPE sobre `max_features` ∈ [0,03; 0,7] (log), `min_samples_leaf` ∈ [1, 200] (log), `max_samples` ∈ [0,2; 1] y `criterion` ∈ {gini, log_loss}, minimizando la **log-loss OOB** en $\mathcal T$ de un bosque ajustado con todo el entrenamiento (un ajuste por configuración; Probst, Wright y Boulesteix, 2019). Con n = 165 mil los sesgos conocidos del OOB (Mitchell, 2011; Janitza y Hornung, 2018) son despreciables. Los mejores se confirman con la **validación cruzada común**, que es la que entra a las comparaciones pareadas.
#
# **Confirmación y elección.** Los mejores se re-evalúan con $R\times K\times S$ y se elige por la pérdida media; dentro
# de la ROPE, por el F1 anidado. El sesgo de seleccionar el mínimo de muchas estimaciones ruidosas se estima con la
# corrección de Tibshirani y Tibshirani (2009),
# $\widehat{\text{sesgo}}=\frac1K\sum_k[e_k(\hat\theta)-e_k(\hat\theta_k)]$, y se verifica en la partición nueva (§8).

# %%
SEC("3. Búsqueda")
bus = pr.paso3_busqueda()
display(bus["confirmacion"].drop(columns=["params"]).round(5))
if len(bus.get("comparaciones", [])):
    display(bus["comparaciones"].round(5))
print("Configuración elegida:", pr.motor.describir(bus["elegido"]))
print("Sesgo de selección (Tibshirani–Tibshirani):", bus.get("sesgo_tt"))
if bus.get("bordes"):
    print("Posición en el espacio (0–1; en_borde = ampliar):", bus["bordes"])

# %% [markdown]
# ## 4. El conjunto de variables como hiperparámetro científico
# Con los hiperparámetros elegidos se comparan, con $R\times K\times S$ y pruebas pareadas, el conjunto parsimonioso,
# el mejor de la curva, **todas las variables v7** y la **encuesta corta** (los dos últimos son los conjuntos comunes de
# la comparación «en igualdad de condiciones» de la fase E). Principal = menor pérdida media; dentro de la ROPE, el de
# menos preguntas (*Deep Learning Tuning Playbook*: hiperparámetros científicos frente a los de molestia).

# %%
SEC("4. Conjuntos")
cj = pr.paso4_conjuntos()
display(cj["tabla"].round(5)); display(cj["comparaciones"].round(5))

# %% [markdown]
# ## 5. Bogotá: peso en el entrenamiento como hiperparámetro (Paso 5; memo §1.3–1.9)
# La evaluación es fija (sólo $\mathcal T$). El peso relativo $\lambda_B$ de los hogares de Bogotá en el entrenamiento
# equivale a minimizar $\alpha\hat\varepsilon_{\mathcal T}+(1-\alpha)\hat\varepsilon_B$ con
# $\alpha=n_{\mathcal T}/(n_{\mathcal T}+\lambda_Bn_B)$. La cota de Ben-David et al. (2010) es de peor caso y no puede
# certificar ganancia (soporte disjunto en Dominio y $L_p$); con el dominio como variable la condicional es compartida
# por construcción y $dP_{\mathcal T}/dP_S\le1{,}068$, el régimen en que el estimador agrupado es minimax-óptimo (Ma,
# Pathak y Wainwright, 2023). La decisión es empírica: $\hat\lambda_B=\arg\min$ de la log-loss media en $\mathcal T$
# (regla de Bayes con previa plana; sólo 2–3 valores, sesgo de selección despreciable), con $\lambda_B=1$ por defecto.
# Con $\lambda_B=0$ Bogotá se elimina **físicamente** del ajuste. Para este modelo flexible se evalúan $\lambda_B\in\{0;0{,}5;1\}$: Bogotá sólo puede ayudar a modelos de complejidad efectiva alta y si la función de etiquetado es compartida (en LightGBM se midió −0,0005 de log-loss con $\lambda_B=1$, n. s.).

# %%
SEC("5. Bogotá")
bog = pr.paso5_bogota()
display(bog["tabla"].round(5)); display(bog["comparaciones"].round(5))
print("Transferencia (OOF de Bogotá):", json.dumps(bog.get("transferencia", {}), indent=1, default=float))
if "dominio" in bog:
    print("Clasificador de dominio Bogotá vs otras cabeceras:", json.dumps(bog["dominio"], indent=1, default=float))
print("Decisión:", bog.get("decision"))

# %% [markdown]
# ## 6. Restricciones monótonas por teoría económica (Paso 9; memo §4.4)
# Si $Y=\mathbb 1\{I<R\}$ e $I\mid x$ tiene distribución $F(\cdot\mid x)$, entonces $\eta(x,R)=F(R\mid x)$ es creciente en
# el ingreso requerido $R$; las variables que desplazan $I$ en dominancia estocástica (ocupados y formales por miembro,
# educación) dan $\eta$ decreciente, y los marcadores de privación (hacinamiento, dependencia, régimen subsidiado),
# creciente. Son supuestos *ceteris paribus* que se **contrastan**: se adoptan por **no inferioridad** (extremo
# desfavorable del IC 95 % corregido: ΔF1 ≥ −0,002 y Δlog-loss ≤ +0,0005). Valen aunque no mejoren: reducen varianza
# y dan legitimidad a una regla de focalización (Gupta et al., 2016; Wang y Gupta, 2020). En Random Forest se usa `monotonic_cst` de scikit-learn ≥ 1.4.

# %%
SEC("6. Monotonía")
mon = pr.paso6_monotonia()
print({k: v for k, v in mon.items() if k in ("aplica", "adoptada", "restricciones")})
if mon.get("aplica"):
    display(pd.DataFrame([mon["comparacion"]]).round(5))

# %% [markdown]
# ## 7. Desbalance (20 % de pobres): sensibilidad (Paso 10; memo §6)
# Con pesos de clase $k$, el minimizador poblacional de una pérdida propia ponderada es
# $\eta_k=\eta/\{\eta+k(1-\eta)\}$, función creciente de $\eta$: ponderar sólo **mueve el umbral** (Elkan, 2001, teorema
# 1). Como el umbral se elige explícitamente en las OOF, los pesos no pueden mejorar el F1 poblacional y sí descalibran.
# Especificación principal: sin pesos ni remuestreo; aquí, la sensibilidad con pesos balanceados ($w_1/w_0=n_0/n_1$) con
# el umbral re-optimizado (lo pide la etapa 3 del enunciado). Predicción teórica: ΔF1 dentro de la ROPE, Brier y ACE peores.

# %%
SEC("7. Desbalance")
des = pr.paso7_desbalance()
if des:
    print({k: v for k, v in des.items() if k != "comparacion"})
    display(pd.DataFrame([des["comparacion"]]).round(5))

# %% [markdown]
# ## 8. Sesgo de selección, partición nueva y LOCO por bloque teórico (memo §2.3, §3.4)
# La configuración elegida se re-evalúa en la **partición 4**, no usada en la búsqueda ni en la confirmación: la diferencia
# con la confirmación estima directamente el optimismo. LOCO (*leave one covariate out*; Lei et al., 2018): se reajusta
# sin cada bloque teórico y se mide la caída de F1 y el aumento de la pérdida con IC pareados (permutar una variable
# correlacionada obliga a extrapolar; reajustar sin ella no: Hooker, Mentch y Zhou, 2021).

# %%
SEC("8. Sesgo y LOCO")
ses = pr.paso8_sesgo()
print("Re-evaluación en la partición 4:", ses.get("reevaluacion_r4")); print("Tibshirani–Tibshirani:", ses.get("tibshirani"))
if "loco" in ses:
    display(ses["loco"].round(5))

# %% [markdown]
# ## 9. Modelos finales, umbral y CSV (Paso 11; memo §5 y §8)
# **El umbral F1-óptimo es $\mathbb 1\{\eta>F^*/2\}$** (argumento de Dinkelbach: $\theta^*$ maximiza
# $E[(2\eta-F^*)\theta]$; Lipton, Elkan y Naryanaswamy, 2014): un solo umbral global, estimado con predicciones honestas
# de $\mathcal T$ y suavizado por la **mediana bootstrap** del argmax (B = 500). El F1 está en una meseta (error de umbral
# de segundo orden). Random Forest: el umbral se elige en las probabilidades **OOB del propio bosque final** (misma escala que las de test; memo §9.3). Se reportan ambas reglas (umbral fijo y tasa igualada), la proporción de hogares de test en
# que difieren, el contraste con el promedio de los modelos de pliegue y los diagnósticos: $\hat t$ frente a $\hat F/2$,
# tasa predicha en test frente a $\hat q$ (±0,003) y media de las probabilidades frente a la prevalencia esperada.
#
# Bosque de 1.000 árboles con todo el entrenamiento (en RF, «más semillas» es lo mismo que más árboles). Versiones: **principal**, **parsimoniosa** (si difiere) y **encuesta corta**; el nombre del CSV lleva el
# algoritmo, los hiperparámetros, el número de variables, $\lambda_B$, las semillas/árboles y `_v7`.

# %%
SEC("9. Finales")
fin = pr.paso9_final()
display(pd.read_csv(pr.ent.tablas / f"{pr.pref}_finales.csv").round(5).T)

# %% [markdown]
# ## 10. Diagnóstico y figuras (sólo $\mathcal T$)
# Curva F1-umbral con la meseta y curva PR con iso-F1; calibración (bins de igual masa; Nixon et al., 2019) — la
# calibración se **exige** y la nitidez es lo que premia el F1 (Gneiting, Balabdaoui y Raftery, 2007); disyuntiva
# filtración/subcobertura con los umbrales de F1, F2 y un costo explícito (memo §5.6); F1 por dominio con IC bootstrap;
# importancia por pregunta (permutación **agrupada** en validación) y dependencia parcial (Friedman, 2001) de las
# variables principales. **Calibración dentro del pliegue** (memo §5.3): Platt e isotónica ajustadas en las OOF de los otros
# pliegues y evaluadas en el retenido; se adopta sólo si mejora el F1 anidado más allá de la ROPE o la log-loss más allá de
# la suya (para el F1 basta el orden; Niculescu-Mizil y Caruana, 2005, esperan ganancia de log-loss en NB y RF).

# %%
SEC("10. Diagnóstico")
dg = pr.paso10_diagnostico()
display(dg["calibracion"].round(5))
display(dg["importancia"].head(15).round(5)); display(dg["disyuntiva"].round(4)); display(dg["f1_dominio"].round(4))

# %% [markdown]
# ## 11. Resumen y señal de listo
# `02_outputs/tables/34_LISTO.json`: el formato de `1x_LISTO.json` más la incertidumbre (EE por influencia y bootstrap,
# IC, F1 por partición, re-evaluación en la partición 4, sesgo de Tibshirani–Tibshirani, estabilidad de la selección) y los
# diagnósticos del umbral. Los modelos quedan en el registro común (`00_data/processed/modelos/`) para la fase E.

# %%
SEC("11. Listo")
listo = pr.paso11_listo()
claves = ["modelo", "archivo", "n_vars", "n_preguntas", "F1_anidado_sinBOG", "IC95", "EE_psi", "F1_particiones", "F1_reevaluacion_r4",
          "optimismo", "sesgo_tibshirani", "AUC", "LogLoss", "peso_bogota", "umbral", "regla_umbral", "tasa_predicha_test",
          "tasa_predicha_oof_sinBOG", "prevalencia_esperada_test", "media_prob_test", "estabilidad", "tiempo_total_min"]
display(Markdown("\n".join(f"* **{k}**: {listo.get(k)}" for k in claves)))
print(f"=== 34_random_forest · fin {time.strftime('%Y-%m-%d %H:%M:%S')} ({listo['tiempo_total_min']:.0f} min) ===")
