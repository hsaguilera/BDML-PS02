# %% [markdown]
# # 02b · Primera etapa v5: ingreso por componentes, distribución del ingreso del hogar y etapa intermedia
# **Problem Set 2 — Predicción de pobreza (MESE/DANE) · BDML Uniandes 2026-20**
#
# **Por qué rehacer la primera etapa.** La búsqueda automática de combinaciones (cuaderno 09) apenas movió el F1
# (0,7725 → 0,7736 con LightGBM rápido). El techo lo fija la precisión con que se predice el ingreso del hogar: si
# $\hat z=z+\varepsilon$, $\varepsilon\sim N(0,\sigma^2)$, el F1 máximo es 0,81 con $\sigma=0{,}30$ y 0,78 con
# $\sigma\approx0{,}38$ (TRASPASO §3.7). Como $\text{Pobre}_h=\mathbb 1\{\text{Ingtotug}_h<R_h\}$ con
# $R_h=\text{Npersug}_h\cdot\text{Lp}_h-A_h$ **observable en test**, toda mejora en la predicción del ingreso
# monetario del hogar se traduce casi uno a uno en F1.
#
# **Qué se hace (en orden de impacto esperado).**
# 1. **Primera etapa por componentes con esquema de valla** (Cragg, 1971; Duan et al., 1983; Mullahy, 1986). En train
#    se cumple exactamente $\text{Ingtot}_i=\sum_c (I_{ic}+I^{es}_{ic})$ con nueve componentes (laboral, segunda
#    actividad, especie, desocupados, intereses, pensiones, ayudas de hogares, ayudas institucionales, arriendos). Cada
#    componente sólo puede ser positivo si una pregunta de «¿recibió…?» presente en test lo permite: se usa como
#    **compuerta exacta**. Dentro de la compuerta, un clasificador estima $\pi_c(x)=P(y_{ic}>0)$ y dos regresores la
#    media y la varianza del log-monto entre los positivos.
# 2. **Primera etapa distribucional.** Con $\pi_c,\mu_c,s^2_c$ se obtienen la media y la varianza del ingreso de cada
#    persona y, sumando en la unidad de gasto, los dos primeros momentos del ingreso del hogar; una lognormal con
#    esos momentos (Fenton, 1960) y masa en cero da la **probabilidad estructural de pobreza**
#    $P(\text{Ingtotug}_h<R_h)$, la variable que resume el problema en la literatura de mapas de pobreza (Elbers,
#    Lanjouw y Lanjouw, 2003; Molina y Rao, 2010).
# 3. **Etapa intermedia del hogar.** Un LightGBM de $z_h=\log(\text{Ingtotugarr}/(\text{Npersug}\cdot\text{Lp}))$ con
#    las variables del hogar y los agregados de la primera etapa produce $\hat z_h$ y, con un segundo modelo del error
#    absoluto, $\hat\sigma_h$ y $P_h=\Phi(-\hat z_h/\hat\sigma_h)$ (la versión flexible del probit latente P4).
# 4. **Ajuste de hiperparámetros** del LightGBM de persona (el submodelo laboral, ≈ 75 % del ingreso).
# 5. **Variables guiadas por la literatura** (escalas de equivalencia, dependencia económica, capital humano, calidad del
#    empleo, vivienda, composición, protección social y posición relativa en el dominio dentro del pliegue).
#
# **Protocolo sin fuga.** Todo es fuera de pliegue con los **mismos cinco pliegues de hogares** de `pliegues.csv`
# (apilamiento con pliegues comunes en todos los niveles; Wolpert, 1992; van der Laan, Polley y Hubbard, 2007); test con
# modelos reentrenados en todo train. Sólo se usan como predictores columnas comunes a train y test; los montos de
# ingreso sólo como objetivo. Cada mejora se mide con el protocolo rápido del cuaderno 09: LightGBM (lr 0,1, parada
# temprana interna), **F1 anidado sin Bogotá** (umbral elegido en los otros pliegues), log-loss y cambios pareados por
# pliegue con su error estándar. Las variables nuevas que salen de modelos del ingreso se etiquetan como **generadas**
# (Pagan, 1984) para poder entregar también modelos sin ellas.

# %%
import os, sys, json, time, math, warnings
warnings.filterwarnings("ignore")
HILOS = int(os.environ.get("AJUSTE_HILOS", "14"))
os.environ["OMP_NUM_THREADS"] = str(HILOS)
SUB = float(os.environ.get("V5_SUBMUESTRA", "1"))          # < 1 sólo para pruebas rápidas
RAPIDO = SUB < 1
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
from scipy.stats import norm
from IPython.display import display
from pobreza.config import Rutas, SEED
from pobreza import io as pio, viz, metrics as M, ajuste as AJ, busqueda as BU, primera_etapa_v5 as PE
R = Rutas(RAIZ); viz.estilo(); FIG, TAB = R.figuras, R.tablas
CACHE = R.processed / ("cache_v5" if not RAPIDO else "cache_v5_prueba"); CACHE.mkdir(exist_ok=True)
T0 = time.time()
T = pio.cargar_crudos(R)
th, tp, teh, tep = T["train_hogares"], T["train_personas"], T["test_hogares"], T["test_personas"]
D = pio.leer_tabla(R, "modelo_train_v4"); DT = pio.leer_tabla(R, "modelo_test_v4")
conj4 = json.load(open(R.processed / "conjuntos_variables.json", encoding="utf-8"))
vb = json.load(open(R.processed / "variables_busqueda.json", encoding="utf-8"))
if RAPIDO:   # submuestra de hogares (sólo para probar el código)
    rng = np.random.default_rng(1)
    ids = set(D.id[rng.random(len(D)) < SUB]); ids_t = set(DT.id[rng.random(len(DT)) < SUB])
    D = D[D.id.isin(ids)].reset_index(drop=True); DT = DT[DT.id.isin(ids_t)].reset_index(drop=True)
    th = th[th.id.isin(ids)].reset_index(drop=True); tp = tp[tp.id.isin(ids)].reset_index(drop=True)
    teh = teh[teh.id.isin(ids_t)].reset_index(drop=True); tep = tep[tep.id.isin(ids_t)].reset_index(drop=True)
assert (D.id.values == th.id.values).all()
y = D.Pobre.values.astype(int)
folds = [(np.flatnonzero(D.fold.values != k), np.flatnonzero(D.fold.values == k)) for k in sorted(D.fold.unique())]
sin_bog = (D.Dominio != "BOGOTA").values
fold_hogar = D.set_index("id").fold
GEN4 = [c for c in conj4["S3"] if c not in conj4["S2"]]
S2 = [c for c in conj4["S2"] if c not in AJ.CAT_TODAS]
CAT = list(AJ.CAT_TODAS)
print(f"train {D.shape} | test {DT.shape} | personas {tp.shape[0]} / {tep.shape[0]} | hilos {HILOS} | submuestra {SUB}")

# %% [markdown]
# ## 1. Identidad contable del ingreso y compuertas exactas
# El DANE construye el ingreso total de cada persona como la suma de nueve componentes observados más sus montos
# imputados por no respuesta (`*es`): $\text{Ingtot}=\text{Ingtotob}+\text{Ingtotes}$. La tabla verifica la identidad
# y, para cada componente, su participación en el ingreso, qué parte imputa el DANE, la tasa de positivos dentro de la
# compuerta y el porcentaje del monto que cae **fuera** de la compuerta (debe ser 0: la compuerta es exacta).

# %%
T_id, chk = PE.verificar_identidad(tp, th)
T_id.to_csv(TAB / "09b_identidad_componentes.csv", index=False)
print(chk)
display(T_id.drop(columns="descripcion").round(4))

# %% [markdown]
# **Lectura.** El ingreso laboral de la primera actividad es ≈ 75 % del ingreso total (con su imputación), las pensiones
# ≈ 12 % y las ayudas de otros hogares ≈ 5 %. Las compuertas son exactas (0 % del monto fuera), salvo los arriendos:
# su pregunta exacta (`P7500s1`) no está en test y la compuerta usa `P7495` («¿recibió arriendos y/o pensiones?»), así
# que el clasificador de participación hace el resto. Una persona sin ninguna compuerta abierta tiene ingreso cero
# **con certeza**: esa información ya no hay que aprenderla.

# %% [markdown]
# ## 2. Matriz de la persona y ajuste de hiperparámetros del submodelo laboral
# Predictores de la persona (comunes a train y test): edad, educación (nivel, grado, años y experiencia potencial de
# Mincer), sexo, parentesco, salud, actividad, oficio, posición ocupacional, tamaño de empresa, cotización, horas,
# antigüedad, prestaciones (que revelan tramos salariales por ley), segunda actividad, búsqueda de empleo, todas las
# preguntas de «¿recibió…?», dominio y zona; y el **contexto observable del hogar** (vivienda, perceptores, formalidad,
# educación, régimen de salud, ingreso requerido). Se ajusta el LightGBM de la media del log-ingreso laboral con una
# búsqueda acotada (hojas, mínimo por hoja, regularización L2, fracción de variables; lr 0,1 con parada temprana
# interna) evaluada en el pliegue 0 por R² y log-verosimilitud gaussiana; la configuración ganadora se usa con lr 0,05.

# %%
Xtr, Xte = PE.preparar_matrices(tp, tep, D, DT)
Ycomp = PE.objetivos_componentes(tp); Gtr = PE.compuertas(tp)
f_per = tp.id.map(fold_hogar).values
print(f"Matriz de persona: {Xtr.shape[1]} columnas")
CONFIGS = [dict(num_leaves=127, min_child_samples=100, lambda_l2=5.0, learning_rate=0.1),
           dict(num_leaves=255, min_child_samples=100, lambda_l2=5.0, learning_rate=0.1),
           dict(num_leaves=255, min_child_samples=50, lambda_l2=10.0, learning_rate=0.1),
           dict(num_leaves=511, min_child_samples=100, lambda_l2=10.0, learning_rate=0.1),
           dict(num_leaves=63, min_child_samples=200, lambda_l2=5.0, learning_rate=0.1),
           dict(num_leaves=255, min_child_samples=200, lambda_l2=1.0, feature_fraction=0.6, learning_rate=0.1),
           dict(num_leaves=127, min_child_samples=30, lambda_l2=20.0, learning_rate=0.1),
           dict(num_leaves=255, min_child_samples=100, lambda_l2=5.0, feature_fraction=0.6, learning_rate=0.1)]
if RAPIDO:
    CONFIGS = CONFIGS[:2]
hip = AJ.en_cache(CACHE / "hiper_laboral.pkl", lambda: PE.busqueda_hiper(
    Xtr, Ycomp["laboral"].values, Gtr["laboral"].values, f_per, CONFIGS, hilos=HILOS))
hip.to_csv(TAB / "09b_hiper_primera_etapa.csv", index=False)
display(hip.round(4))
mejor = hip.sort_values("R2_val", ascending=False).iloc[0]
PM_LAB = {k: (int(mejor[k]) if k in ("num_leaves", "min_child_samples") else float(mejor[k]))
          for k in ("num_leaves", "min_child_samples", "lambda_l2") }
if "feature_fraction" in hip.columns and pd.notna(mejor.get("feature_fraction")):
    PM_LAB["feature_fraction"] = float(mejor["feature_fraction"])
PM_LAB["learning_rate"] = 0.05
ganancia_hiper = float(mejor.R2_val - hip.loc[0, "R2_val"])
print(f"Configuración elegida (con lr 0,05): {PM_LAB} | ganancia de R² frente a la de la versión 3: {ganancia_hiper:+.4f}")

# %% [markdown]
# ## 3. Primera etapa por componentes (valla), fuera de pliegue
# Para cada componente $c$ y cada pliegue $k$ se entrena con las personas de los hogares de los otros cuatro pliegues:
# (1) $\pi_c(x)$ con LightGBM binario dentro de la compuerta (si casi todos los de la compuerta son positivos se usa la
# tasa); (2) $\mu_c(x)=E[\log y\mid y>0,x]$ con LightGBM L2; (3) $s^2_c(x)$ con LightGBM gamma sobre los residuos al
# cuadrado, calibrado con los residuos de la submuestra de parada temprana (fuera de muestra). La complejidad se adapta
# al tamaño de cada submuestra (más regularización en los componentes pequeños). Los montos se acotan en el percentil
# 99,9 de los positivos del pliegue y con un piso de 1.000 pesos.

# %%
t0 = time.time()
P_tr, P_te, info1 = AJ.en_cache(CACHE / "etapa1_componentes.pkl", lambda: PE.primera_etapa_oof(
    tp, tep, Xtr, Xte, fold_hogar, hilos=HILOS, semilla=SEED, pm={"laboral": PM_LAB}))
print(f"Primera etapa por componentes: {time.time() - t0:.0f}s")
diag = PE.diagnostico_componentes(tp, P_tr, info1)
diag.to_csv(TAB / "09b_componentes.csv", index=False)
display(diag.round(4))

# %% [markdown]
# **Calidad a nivel de persona.** Se compara $\hat E[\text{Ingtot}_i]$ con el ingreso observado (acotado en el
# percentil 99,9, como en la versión 3) en niveles y en log(1 + y). La versión 3 (un solo LightGBM Poisson) tenía
# R² OOF de 0,682 en niveles y 0,622 en log(1 + y).

# %%
Ei_tr = PE.ingreso_persona(P_tr); Ei_te = PE.ingreso_persona(P_te)
yi = tp.Ingtot.fillna(0).clip(lower=0).values; tope_i = np.quantile(yi[yi > 0], 0.999); yi = np.minimum(yi, tope_i)
r2_niv = 1 - ((yi - Ei_tr) ** 2).sum() / ((yi - yi.mean()) ** 2).sum()
r2_log = 1 - ((np.log1p(yi) - np.log1p(Ei_tr)) ** 2).sum() / ((np.log1p(yi) - np.log1p(yi).mean()) ** 2).sum()
pe_v3 = conj4.get("primera_etapa", {})
tabla_persona = pd.DataFrame([
    {"version": "v3 (LightGBM Poisson, un modelo)", "R2_niveles_oof": pe_v3.get("R2_niveles_oof"), "R2_log1p_oof": pe_v3.get("R2_log1p_oof")},
    {"version": "v5 (valla por componentes)", "R2_niveles_oof": r2_niv, "R2_log1p_oof": r2_log}])
display(tabla_persona.round(4))
imp1 = info1["laboral"].get("importancia_mu")
if imp1 is not None:
    display((imp1 / imp1.sum()).head(15).round(4).to_frame("importancia (ganancia), media del log-ingreso laboral"))

# %% [markdown]
# ## 4. Agregación al hogar y probabilidad estructural de pobreza
# Con $E_{ic}=\pi e^{\mu+s^2/2}$ y $V_{ic}=\pi e^{2\mu+2s^2}-E_{ic}^2$ se suman media y varianza en la unidad de gasto.
# La varianza del hogar admite una correlación intrahogar $\rho$ entre las desviaciones de las personas
# ($V_\rho=(1-\rho)\sum V_i+\rho(\sum\sqrt{V_i})^2$): los choques de ingreso de un hogar no son independientes
# (mercado laboral local, emparejamiento selectivo). $\rho$ se elige por AUC **dentro del entrenamiento de cada
# pliegue**. Con masa en cero $P_0=\prod(1-\pi_{ic})$ y la lognormal de Fenton–Wilkinson para la parte positiva:
# $$P(\text{Ingtotug}_h<R_h)=P_0+(1-P_0)\,\Phi\!\left(\frac{\log R_h-\mu_h}{\sigma_h}\right),\qquad
# \sigma_h^2=\log\!\Big(1+\frac{V^+_h}{(E^+_h)^2}\Big),\ \mu_h=\log E^+_h-\tfrac12\sigma_h^2 .$$
# También se guardan $\mu_h$ y $\sigma_h$ (media y desviación del log-ingreso del hogar predicho), el ingreso esperado
# por grupos de fuentes (laboral, pensiones, transferencias, capital), los perceptores esperados y el principal perceptor.

# %%
RHOS = (0.0, 0.25, 0.5, 1.0)
A_tr = PE.agregar_hogar_v5(tp, P_tr, D, RHOS); A_te = PE.agregar_hogar_v5(tep, P_te, DT, RHOS)
prob_fw, rho_k = PE.elegir_rho_en_pliegues(y, A_tr, folds, RHOS)
auc_rho = {r: M.auc_roc(y, A_tr[f"v5_prob_pobre_fw_rho{r:g}"].values) for r in RHOS}
rho_fin = max(auc_rho, key=auc_rho.get)
print("AUC por ρ (todo train):", {k: round(v, 4) for k, v in auc_rho.items()}, "| ρ por pliegue:", rho_k, "| ρ test:", rho_fin)
V5_E1 = [c for c in A_tr.columns if "_rho" not in c]                # agregados que no dependen de ρ
for nom in ("v5_mu_log", "v5_sd_log", "v5_gap_fw", "v5_prob_pobre_fw"):
    oof = np.zeros(len(D))
    for (tr, va), r in zip(folds, rho_k):
        oof[va] = A_tr[f"{nom}_rho{r:g}"].values[va]
    D[nom] = oof; DT[nom] = A_te[f"{nom}_rho{rho_fin:g}"].values
for c in V5_E1:
    D[c] = A_tr[c].values; DT[c] = A_te[c].values
V5_FW = ["v5_mu_log", "v5_sd_log", "v5_gap_fw", "v5_prob_pobre_fw"]
V5_ETAPA1 = V5_E1                                                    # ingreso por componentes (sin FW)
print(f"{len(V5_ETAPA1)} agregados de la 1.ª etapa + {len(V5_FW)} de la distribución (FW)")

# %%
z_nb = D.z.values[sin_bog]
tabla_sigma = PE.f1_max_ruido(z_nb, np.round(np.arange(0.05, 1.01, 0.025), 3))
cal = [PE.calidad_prediccion("v3: Σ E[Ingtot] (Poisson, un modelo)", D, D.log_ing_esperado_ug.values, mask=sin_bog),
       PE.calidad_prediccion("v5: Σ E[Ingtot] (valla por componentes)", D, D.v5_log_ing_esp.values, mask=sin_bog),
       PE.calidad_prediccion("v5: probabilidad estructural FW", D, score_pobre=D.v5_prob_pobre_fw.values, mask=sin_bog)]
display(pd.DataFrame(cal).round(4))

# %% [markdown]
# ## 5. Etapa intermedia del hogar: $\hat z_h$, su error esperado y $P_h$
# LightGBM de $z_h$ (pérdida cuadrática) con las variables del hogar (S2 y categóricas), las generadas de la versión 3
# y los agregados v5; luego un LightGBM del error absoluto fuera de pliegue $|z-\hat z|$ da
# $\hat\sigma_h=\sqrt{\pi/2}\,E|z-\hat z|$ (bajo normalidad) y $P_h=\Phi(-\hat z_h/\hat\sigma_h)$. Ambos se estiman
# fuera de pliegue con los mismos pliegues: cuando se predice el pliegue $k$, ni el modelo del hogar ni los modelos de
# persona que generaron sus variables vieron ese pliegue.

# %%
COLS_H = list(dict.fromkeys(S2 + GEN4 + V5_ETAPA1 + V5_FW + CAT))
Xh = AJ.preparar(D, COLS_H, D); Xh_te = AJ.preparar(DT, COLS_H, D)
zh, zh_te, sig, sig_te, info3 = AJ.en_cache(CACHE / "etapa_hogar.pkl", lambda: PE.etapa_hogar(
    Xh, D.z.values, folds, Xh_te, hilos=HILOS, semilla=SEED))
for dd, a, b in ((D, zh, sig), (DT, zh_te, sig_te)):
    dd["v5_zhat_hogar"] = a; dd["v5_sigma_hogar"] = b
    dd["v5_gap_hogar"] = a / b
    dd["v5_prob_pobre_hogar"] = norm.cdf(-a / b)
V5_HOGAR = ["v5_zhat_hogar", "v5_sigma_hogar", "v5_gap_hogar", "v5_prob_pobre_hogar"]
print(f"R² OOF de z: {info3['R2_z_oof']:.4f} | árboles {info3['arboles_z']} | {info3['tiempo_s']:.0f}s")
cal.append(PE.calidad_prediccion("v5: etapa del hogar ẑ_h", D, zhat=D.v5_zhat_hogar.values,
                                 score_pobre=D.v5_prob_pobre_hogar.values, mask=sin_bog))
# F1 de la «ruta de regresión»: clasificar por −ẑ con umbral anidado (sin Bogotá)
f1_reg, f1k_reg, _, _ = AJ.f1_anidado_sb(y, -D.v5_zhat_hogar.values, folds, sin_bog)
f1_prob, _, _, _ = AJ.f1_anidado_sb(y, D.v5_prob_pobre_hogar.values, folds, sin_bog)
f1_fw, _, _, _ = AJ.f1_anidado_sb(y, D.v5_prob_pobre_fw.values, folds, sin_bog)
f1_v3, _, _, _ = AJ.f1_anidado_sb(y, -D.brecha_esperada.values, folds, sin_bog)
f1_v5b, _, _, _ = AJ.f1_anidado_sb(y, -D.v5_brecha.values, folds, sin_bog)
tab_cal = pd.DataFrame(cal)
tab_cal["F1_umbral_directo"] = [f1_v3, f1_v5b, f1_fw, f1_reg]
tab_cal["sigma_implicita_F1"] = [PE.sigma_implicito(v, tabla_sigma) if pd.notna(v) else np.nan for v in tab_cal.F1_umbral_directo]
tab_cal.to_csv(TAB / "09b_calidad_ingreso.csv", index=False)
display(tab_cal.round(4))
display(info3["importancia"].head(15).round(4).to_frame("importancia (ganancia), modelo de z del hogar"))

# %%
fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
ax[0].hexbin(D.v5_zhat_hogar.clip(-3, 4), D.z.clip(-3, 4), gridsize=70, cmap="magma", mincnt=2, bins="log")
ax[0].plot([-3, 4], [-3, 4], "c--", lw=1); ax[0].axhline(0, color="w", lw=.8); ax[0].axvline(0, color="w", lw=.8)
ax[0].set(title="Etapa del hogar (OOF): ẑ frente a z", xlabel="ẑ_h", ylabel="z_h observado")
b = pd.qcut(D.v5_prob_pobre_fw, 30, duplicates="drop")
g = pd.DataFrame({"p": D.v5_prob_pobre_fw, "y": y}).groupby(b, observed=True).mean()
ax[1].plot(g.p, g.y, "o-", color=viz.ROJO, ms=4, label="P estructural (FW)")
b2 = pd.qcut(D.v5_prob_pobre_hogar, 30, duplicates="drop")
g2 = pd.DataFrame({"p": D.v5_prob_pobre_hogar, "y": y}).groupby(b2, observed=True).mean()
ax[1].plot(g2.p, g2.y, "s-", color=viz.AZUL, ms=4, label="P del hogar Φ(−ẑ/σ̂)")
ax[1].plot([0, 1], [0, 1], "k:", lw=1); ax[1].legend()
ax[1].set(title="Calibración de las probabilidades estructurales (30 bins)", xlabel="probabilidad predicha", ylabel="tasa observada de pobres")
ax[2].plot(tabla_sigma.sigma, tabla_sigma.F1_max, color=viz.MORADO)
for _, r in tab_cal.dropna(subset=["F1_umbral_directo"]).iterrows():
    ax[2].plot(r.sigma_implicita_F1, r.F1_umbral_directo, "o", ms=6)
    ax[2].annotate(r.prediccion.split(":")[0] + ": " + r.prediccion.split(":")[1][:22], (r.sigma_implicita_F1, r.F1_umbral_directo), fontsize=7)
ax[2].set(title="Techo de F1 según el error σ de ẑ (sin Bogotá)", xlabel="σ del error en z", ylabel="F1 máximo")
fig.tight_layout(); viz.guardar(fig, FIG, "09b_primera_etapa_v5"); plt.show()

# %% [markdown]
# ## 6. Medición de cada mejora con el protocolo rápido
# LightGBM lr 0,1 con parada temprana interna (el del cuaderno 09), los cinco pliegues, F1 anidado sin Bogotá y
# log-loss; los cambios se miden **pareados por pliegue** frente a la referencia de la versión 4 (S3) y frente al paso
# anterior. La «σ implícita» es la que, con error gaussiano en $z$, daría el mismo F1.

# %%
motor = AJ.MotorLGB(hilos=HILOS); P_RAPIDO = {**motor.base, "learning_rate": 0.1}
EXTRA09 = [c for c in vb["ganadoras"] if c in D.columns]
EXTRA09_GEN = [c for c in vb["usan_generadas"] if c in EXTRA09]
GEN5 = V5_ETAPA1 + V5_FW + V5_HOGAR


class Rapido:
    """Contenedor mínimo con la interfaz que usan `busqueda._cv_eval` y `candidatos_literatura`."""
    def __init__(self, D, DT, cols):
        self.D, self.DT = D, DT
        self.X = AJ.preparar(D, cols, D); self.XT = AJ.preparar(DT, cols, D)
        self.y, self.folds, self.sin_bog = y, folds, sin_bog
        self.GEN = GEN4 + EXTRA09_GEN + GEN5


dat = Rapido(D, DT, list(dict.fromkeys(S2 + CAT + GEN4 + EXTRA09 + GEN5)))
REF = {}; filas = []


def medir(paso, cols, idea, comparar=("v4: S3",), transformar=None):
    res, f1k, llk, ts = BU._cv_eval(motor, dat, cols, P_RAPIDO, transformar)
    fila = {"paso": paso, "idea": idea, "n_vars": len(cols), "F1_anidado_sinBOG": res["F1_anidado_sinBOG"], "EE": res["EE"],
            "LogLoss": res["LogLoss"], "AUC_sinBOG": res["AUC_sinBOG"],
            "sigma_implicita": PE.sigma_implicito(res["F1_anidado_sinBOG"], tabla_sigma), "tiempo_s": ts}
    for ref in comparar:
        if ref in REF and ref != paso:
            d1, e1, d2, e2, ok = BU._mejora(f1k, llk, *REF[ref])
            tag = "v4" if ref == "v4: S3" else "previo"
            fila.update({f"dF1_vs_{tag}": d1, f"EE_dF1_vs_{tag}": e1, f"dLogLoss_vs_{tag}": d2, f"EE_dLL_vs_{tag}": e2, f"mejora_1EE_vs_{tag}": ok})
    REF[paso] = (f1k, llk); filas.append(fila)
    print(f"{paso:62s} F1 = {fila['F1_anidado_sinBOG']:.4f} ± {fila['EE']:.4f} | LL = {fila['LogLoss']:.5f} | σ* = {fila['sigma_implicita']:.3f} ({ts:.0f}s)")
    return fila


S3_v4 = S2 + GEN4 + CAT
medir("v4: S3", S3_v4, "Referencia: S2 + generadas v3 + categóricas")
medir("v4: S3 + 10 variables del 09", S3_v4 + EXTRA09, "Referencia: combinaciones automáticas (09)")
medir("1. S2 + 1.ª etapa por componentes (sin generadas v3)", S2 + V5_ETAPA1 + CAT, "1. Valla por componentes", ("v4: S3",))
medir("1b. S3 v4 + 1.ª etapa por componentes", S3_v4 + V5_ETAPA1, "1. Valla por componentes", ("v4: S3",))
medir("2. 1b + probabilidad estructural (FW)", S3_v4 + V5_ETAPA1 + V5_FW, "2. Primera etapa distribucional",
      ("v4: S3", "1b. S3 v4 + 1.ª etapa por componentes"))
medir("3. 2 + etapa intermedia del hogar (ẑ, σ̂, P)", S3_v4 + V5_ETAPA1 + V5_FW + V5_HOGAR, "3. Etapa del hogar",
      ("v4: S3", "2. 1b + probabilidad estructural (FW)"))
medir("3b. S2 + sólo FW y etapa del hogar", S2 + V5_FW + V5_HOGAR + CAT, "3. Etapa del hogar (generadas compactas)", ("v4: S3",))
medir("3c. 3 + 10 variables del 09", S3_v4 + V5_ETAPA1 + V5_FW + V5_HOGAR + EXTRA09, "Combinaciones 09 sobre v5",
      ("v4: S3", "3. 2 + etapa intermedia del hogar (ẑ, σ̂, P)"))
tab = pd.DataFrame(filas)
display(tab.drop(columns=["idea"]).round(5))

# %% [markdown]
# **Elección de la base v5.** Se toma como base el paso con mayor F1 anidado entre «3», «3b» y «3c»; si las
# generadas de la versión 3 o las combinaciones del 09 no aportan más de 1 EE, se conserva la especificación más corta.

# %%
cand_base = {"3. 2 + etapa intermedia del hogar (ẑ, σ̂, P)": S3_v4 + V5_ETAPA1 + V5_FW + V5_HOGAR,
             "3b. S2 + sólo FW y etapa del hogar": S2 + V5_FW + V5_HOGAR + CAT,
             "3c. 3 + 10 variables del 09": S3_v4 + V5_ETAPA1 + V5_FW + V5_HOGAR + EXTRA09}
f1s = {k: tab.set_index("paso").loc[k, "F1_anidado_sinBOG"] for k in cand_base}
mejor_paso = max(f1s, key=f1s.get)
# regla de 1 EE: preferir la más corta si su F1 está a menos de 1 EE (pareado) de la mejor
for k in sorted(cand_base, key=lambda k: len(cand_base[k])):
    if k == mejor_paso:
        break
    d1, e1, _, _, _ = BU._mejora(REF[k][0], REF[k][1], *REF[mejor_paso])
    if d1 > -e1:
        mejor_paso = k; break
BASE_V5 = cand_base[mejor_paso]
print(f"Base v5: «{mejor_paso}» ({len(BASE_V5)} variables)")

# %% [markdown]
# ## 7. Variables guiadas por la literatura
# Candidatas de `busqueda.candidatos_literatura` (escalas de equivalencia OCDE y raíz, dependencia demográfica y
# económica, capital humano y Mincer, calidad del empleo, vivienda y hacinamiento, composición, protección social,
# posición relativa en el dominio y el departamento) más algunas nuevas sobre la primera etapa v5. Las posiciones
# relativas (rango y desviación frente a la mediana del grupo) se recalculan **dentro de cada pliegue** con las X de
# los hogares de entrenamiento. Selección hacia adelante **por familia** con la regla de 1 EE pareado y poda dentro de
# la familia aceptada. Las transformaciones monótonas de una sola variable no se evalúan (no cambian un árbol).

# %%
LIT_V5 = [
    ("x_v5_lz_ocde", "v5_log_ing_esp - np.log(x_escala_ocde) - np.log(Lp)", "a. Escalas de equivalencia",
     "Ingreso esperado (1.ª etapa v5) por adulto equivalente OCDE frente a la línea (Deaton, 1997)"),
    ("x_dependencia_economica", "(Npersug - np.minimum(n_ocupado + n_pensionado, Npersug))/np.maximum(n_ocupado + n_pensionado, 1)",
     "b. Dependencia", "Dependencia económica: miembros sin ingreso laboral ni pensión por perceptor"),
    ("x_v5_perceptores_por_ae", "v5_n_perceptores_esp/x_escala_ocde", "b. Dependencia",
     "Perceptores esperados (1.ª etapa v5) por adulto equivalente"),
    ("x_superior_por_adulto", "n_educ_superior/np.maximum(n_edad18_64 + n_mayor65, 1)", "c. Capital humano",
     "Proporción de adultos con educación superior (prima de la educación terciaria en Colombia)"),
    ("x_pension_por_mayor", "n_recibe_pension_jubilacion/np.maximum(n_mayor65, 1)", "g. Protección social",
     "Cobertura pensional de los adultos mayores"),
    ("x_hacinamiento_x_menores", "personas_por_dormitorio*share_menores", "e. Vivienda", "Hacinamiento con niños (IPM)"),
    ("x_v5_brecha_x_cabecera", "v5_brecha*cabecera", "h. Territorio", "Brecha esperada v5 × cabecera"),
]
LIT_GRUPO_V5 = [("v5_zhat_hogar", "rango_grupo", "Dominio", "Posición de ẑ_h en la distribución de su dominio (Elbers, Lanjouw y Lanjouw, 2003)"),
                ("v5_zhat_hogar", "desv_mediana_grupo", "Dominio", "ẑ_h frente a la mediana del dominio"),
                ("v5_brecha", "desv_mediana_grupo", "Depto", "Brecha v5 frente a la mediana del departamento")]
specs = BU.candidatos_literatura(dat)
gen_set = set(dat.GEN)
nombres = {s["nombre"] for s in specs}
for nom, e, fam, mot in LIT_V5:
    usa = any(g in e for g in gen_set) or any(p in e for p in nombres if p.startswith("x_") and
                                               next((s for s in specs if s["nombre"] == p), {}).get("usa_generadas"))
    specs.append(dict(nombre=nom, tipo="expr", expr=e, familia=fam, motivacion=mot, origen="literatura (v5)",
                      usa_generadas=bool(usa), monotona_univariada=False)); nombres.add(nom)
for a, tp_, gr, mot in LIT_GRUPO_V5:
    specs.append(dict(nombre=f"x_{a}__{'rango' if tp_ == 'rango_grupo' else 'desv'}_{gr}", tipo=tp_, a=a, grupo=gr,
                      familia="h. Territorio", motivacion=mot, origen="literatura (v5)", usa_generadas=True,
                      monotona_univariada=False))
specs = [s for s in specs if not s["monotona_univariada"] and s["nombre"] not in D.columns]
C_tr = BU.materializar(specs, D, D); C_te = BU.materializar(specs, D, DT)
dat.X = pd.concat([dat.X, C_tr], axis=1); dat.XT = pd.concat([dat.XT, C_te], axis=1)
transf = BU.transformador_grupal(specs, D, folds)
familias = {}
for s in specs:
    familias.setdefault(s["familia"], []).append(s["nombre"])
print({k: len(v) for k, v in familias.items()})

# %%
def adelante_por_familia(base, familias, transformar, limite_s=15 * 60):
    """Añade cada familia completa; si mejora F1 o log-loss en más de 1 EE pareado (sin empeorar la otra más de
    1 EE), se poda dentro de la familia quitando las variables cuya salida no empeora."""
    t0 = time.time(); actual = list(base)
    res, f1k, llk, ts = BU._cv_eval(motor, dat, actual, P_RAPIDO, transformar)
    camino = [{"paso": "base v5", "familia": "", "variables": "", "n_vars": len(actual), "F1": res["F1_anidado_sinBOG"],
               "LogLoss": res["LogLoss"], "aceptado": True}]
    ganadoras = []
    for fam, L in familias.items():
        if time.time() - t0 > limite_s:
            print("  límite de tiempo"); break
        r2, f2, l2, ts = BU._cv_eval(motor, dat, actual + L, P_RAPIDO, transformar)
        d1, e1, d2, e2, ok = BU._mejora(f2, l2, f1k, llk)
        camino.append({"paso": "añadir familia", "familia": fam, "variables": ", ".join(L), "n_vars": len(actual) + len(L),
                       "F1": r2["F1_anidado_sinBOG"], "LogLoss": r2["LogLoss"], "dF1": d1, "EE_dF1": e1, "dLogLoss": d2,
                       "EE_dLL": e2, "aceptado": ok})
        print(f"  {fam:32s} ΔF1={d1:+.4f} (EE {e1:.4f}) Δll={d2:+.5f} (EE {e2:.5f}) → {'ACEPTA' if ok else 'rechaza'}")
        if not ok:
            continue
        quedan = list(L)
        if len(L) > 1:
            for v in L:
                prueba = [c for c in actual + quedan if c != v]
                r3, f3, l3, _ = BU._cv_eval(motor, dat, prueba, P_RAPIDO, transformar)
                dd1, ee1, dd2, ee2, _ = BU._mejora(f2, l2, f3, l3)
                aporta = (dd1 > 0 and dd2 > -ee2) or (dd2 > 0 and dd1 > -ee1)
                camino.append({"paso": f"poda: quitar {v}", "familia": fam, "variables": v, "n_vars": len(prueba),
                               "F1": r3["F1_anidado_sinBOG"], "LogLoss": r3["LogLoss"], "dF1": dd1, "EE_dF1": ee1,
                               "dLogLoss": dd2, "EE_dLL": ee2, "aceptado": aporta})
                if not aporta:
                    quedan.remove(v); r2, f2, l2 = r3, f3, l3
        actual += quedan; ganadoras += quedan; f1k, llk = f2, l2
        print(f"    quedan {quedan}; F1 = {r2['F1_anidado_sinBOG']:.4f}")
    return ganadoras, pd.DataFrame(camino)


LIT_GAN, camino_lit = AJ.en_cache(CACHE / "literatura.pkl", lambda: adelante_por_familia(BASE_V5, familias, transf,
                                                                                         limite_s=3 * 60 if RAPIDO else 15 * 60))
camino_lit.to_csv(TAB / "09b_literatura_seleccion.csv", index=False)
display(camino_lit.drop(columns="variables").round(5))
print("Variables de la literatura aceptadas:", LIT_GAN)
spec_lit = [s for s in specs if s["nombre"] in LIT_GAN]
LIT_GAN_GEN = [s["nombre"] for s in spec_lit if s["usa_generadas"]]
pd.DataFrame([{k: s.get(k) for k in ("nombre", "familia", "motivacion", "tipo", "expr", "a", "grupo", "usa_generadas")}
              for s in specs]).assign(aceptada=lambda d: d.nombre.isin(LIT_GAN)).to_csv(TAB / "09b_literatura_candidatas.csv", index=False)

# %%
# Para la tabla final, las posiciones relativas usan como referencia las X de todo train (sin la respuesta)
for c in LIT_GAN:
    D[c] = C_tr[c].values; DT[c] = C_te[c].values
medir("4. Base v5 + variables de la literatura", BASE_V5 + LIT_GAN, "5. Literatura", ("v4: S3",))
S3_V5_AMPL = list(dict.fromkeys(BASE_V5 + LIT_GAN))

# %% [markdown]
# ## 8. Conjuntos sin generadas y encuesta corta
# Se mide también la versión **sin ninguna variable generada** (para poder entregar modelos que no dependan de modelos
# auxiliares del ingreso) y una **encuesta corta**: la de `ajuste.ENCUESTA_CORTA` (15 preguntas baratas y
# verificables) y una sugerida que añade las compuertas de ingreso no laboral, que se responden con «sí/no» y son
# verificables (pensión, ayudas institucionales, cotización a pensión).

# %%
GEN_TODAS = list(dict.fromkeys(GEN4 + EXTRA09_GEN + GEN5 + LIT_GAN_GEN))
S2_AMPL = [c for c in S3_V5_AMPL if c not in GEN_TODAS]
medir("5. S2 ampliado (sin generadas)", S2_AMPL, "Sin generadas", ("v4: S3",))
EC = list(AJ.ENCUESTA_CORTA)
medir("6a. Encuesta corta (15 preguntas)", EC, "Encuesta corta", ())
EC_SUG = EC + ["n_recibe_pension_jubilacion", "n_recibe_ayuda_instituciones", "n_asalariado_formal"]
medir("6b. Encuesta corta sugerida (18)", EC_SUG, "Encuesta corta", ("6a. Encuesta corta (15 preguntas)",))
medir("6c. Encuesta corta sugerida + P estructural del hogar", EC_SUG + ["v5_prob_pobre_hogar"], "Encuesta corta",
      ("6b. Encuesta corta sugerida (18)",))
tab = pd.DataFrame(filas)
tab.to_csv(TAB / "09b_mejoras_v5.csv", index=False)
display(tab.drop(columns=["idea"]).round(5))

# %%
fig, ax = plt.subplots(figsize=(11, 6))
t = tab.iloc[::-1]
col = [viz.GRIS if p.startswith("v4") else (viz.VERDE if "mejora_1EE_vs_v4" in t and bool(m) else viz.AZUL)
       for p, m in zip(t.paso, t.get("mejora_1EE_vs_v4", pd.Series([False] * len(t))).fillna(False))]
ax.barh(t.paso, t.F1_anidado_sinBOG, xerr=t.EE, color=col, capsize=2)
ax.set_xlim(max(0.6, t.F1_anidado_sinBOG.min() - 0.02), t.F1_anidado_sinBOG.max() + 0.01)
ax.axvline(tab.set_index("paso").loc["v4: S3", "F1_anidado_sinBOG"], color="k", ls=":", lw=1, label="v4: S3")
ax.set(title="F1 anidado sin Bogotá (LightGBM rápido) por mejora de la versión 5 (± 1 EE)", xlabel="F1 anidado sin Bogotá")
ax.tick_params(axis="y", labelsize=8); ax.legend()
fig.tight_layout(); viz.guardar(fig, FIG, "09b_mejoras_v5"); plt.show()

# %% [markdown]
# ## 9. Salidas
# * `modelo_train_v5.parquet` / `modelo_test_v5.parquet`: las tablas v4 más las variables v5 (primera etapa por
#   componentes, distribución FW, etapa del hogar) y las de la literatura aceptadas.
# * `conjuntos_variables_v5.json`: S1, S2, S3 (S2 + generadas), sus ampliaciones, la lista **completa** de generadas,
#   la encuesta corta sugerida y, para `ajuste.Datos`, `extra` / `extra_gen`.
# * `v5_LISTO.json`: bandera con las cifras clave (la espera el agente de los modelos finales).

# %%
S3_V5 = [c for c in BASE_V5 if c not in EXTRA09]
S2_V5 = [c for c in S3_V5 if c not in GEN_TODAS]
extra = [c for c in S3_V5_AMPL if c not in S3_v4]
conj5 = {
    "version": "v5", "descripcion": "Primera etapa por componentes (valla), distribución FW, etapa del hogar y variables de la literatura (cuaderno 02b)",
    "S1": conj4["S1"], "S2": S2_V5, "S3": S3_V5, "S2_ampliado": S2_AMPL, "S3_ampliado": S3_V5_AMPL,
    "categoricas": CAT, "generadas": GEN_TODAS,
    "generadas_v3": GEN4, "generadas_v5_etapa1": V5_ETAPA1, "generadas_v5_distribucion": V5_FW, "generadas_v5_hogar": V5_HOGAR,
    "busqueda_09": EXTRA09, "busqueda_09_generadas": EXTRA09_GEN, "literatura": LIT_GAN, "literatura_generadas": LIT_GAN_GEN,
    "encuesta_corta": EC_SUG, "encuesta_corta_v4": EC,
    "extra": extra, "extra_gen": [c for c in extra if c in GEN_TODAS],
    "base_elegida": mejor_paso,
    "uso": "dat = ajuste.Datos(R, sufijo='_v5', extra=c['extra'], extra_gen=c['extra_gen']) con c = este json; "
           "o bien columnas = c['S3_ampliado'] (con generadas) / c['S2_ampliado'] (sin generadas).",
    "primera_etapa_v5": {"R2_persona_niveles_oof": float(r2_niv), "R2_persona_log1p_oof": float(r2_log),
                         "hiperparametros_laboral": PM_LAB, "rho_pliegues": list(map(float, rho_k)), "rho_test": float(rho_fin),
                         "R2_z_hogar_oof": info3["R2_z_oof"], "componentes": diag.to_dict(orient="records")},
}
for nom, cols in [("S2", S2_V5), ("S3", S3_V5), ("S3_ampliado", S3_V5_AMPL), ("S2_ampliado", S2_AMPL), ("encuesta_corta", EC_SUG)]:
    falt = [c for c in cols if c not in D.columns or c not in DT.columns]
    assert not falt, (nom, falt)
assert not [c for c in D.columns if c not in DT.columns and c not in
            ("Pobre", "z", "log_ingtotug", "brecha", "Indigente", "fold")], "columnas de train que no están en test"
if not RAPIDO:
    pio.guardar_tabla(D, R, "modelo_train_v5"); pio.guardar_tabla(DT, R, "modelo_test_v5")
json.dump(conj5, open(R.processed / "conjuntos_variables_v5.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
tb = tab.set_index("paso")
listo = {
    "listo": not RAPIDO, "fecha": time.strftime("%Y-%m-%d %H:%M"), "cuaderno": "02b_primera_etapa_v5",
    "archivos": {"train": str(R.processed / "modelo_train_v5.parquet"), "test": str(R.processed / "modelo_test_v5.parquet"),
                 "conjuntos": str(R.processed / "conjuntos_variables_v5.json"), "tabla_mejoras": str(TAB / "09b_mejoras_v5.csv")},
    "F1_rapido_v4_S3": float(tb.loc["v4: S3", "F1_anidado_sinBOG"]),
    "F1_rapido_v4_S3_mas_09": float(tb.loc["v4: S3 + 10 variables del 09", "F1_anidado_sinBOG"]),
    "F1_rapido_v5_base": float(tb.loc[mejor_paso, "F1_anidado_sinBOG"]),
    "F1_rapido_v5_ampliado": float(tb.loc["4. Base v5 + variables de la literatura", "F1_anidado_sinBOG"]),
    "F1_rapido_v5_sin_generadas": float(tb.loc["5. S2 ampliado (sin generadas)", "F1_anidado_sinBOG"]),
    "EE_v5_ampliado": float(tb.loc["4. Base v5 + variables de la literatura", "EE"]),
    "R2_persona_niveles": {"v3": pe_v3.get("R2_niveles_oof"), "v5": float(r2_niv)},
    "R2_log_ingtotug_sinBOG": {r["prediccion"]: r.get("R2_log_ingtotug") for r in cal},
    "R2_z_sinBOG": {r["prediccion"]: r.get("R2_z") for r in cal},
    "sigma_error_z_sinBOG": {r["prediccion"]: r.get("sigma_error_z") for r in cal},
    "sigma_implicita_F1": {"v4_S3": float(tb.loc["v4: S3", "sigma_implicita"]),
                           "v5_ampliado": float(tb.loc["4. Base v5 + variables de la literatura", "sigma_implicita"])},
    "n_vars": {"S3_v5": len(S3_V5), "S3_ampliado": len(S3_V5_AMPL), "S2_ampliado": len(S2_AMPL), "generadas": len(GEN_TODAS)},
    "uso": conj5["uso"], "tiempo_min": (time.time() - T0) / 60,
}
json.dump(listo, open(R.processed / ("v5_LISTO.json" if not RAPIDO else "v5_PRUEBA.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: v for k, v in listo.items() if k.startswith("F1") or k in ("tiempo_min", "n_vars")}, ensure_ascii=False, indent=1, default=float))
