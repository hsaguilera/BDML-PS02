# %% [markdown]
# # 01 · Análisis exploratorio completo
# **Problem Set 2 — Predicción de pobreza (MESE/DANE) · BDML Uniandes 2026-20**
#
# Objetivo: entender a fondo el fenómeno que se va a predecir y la estructura de los datos antes de modelar. El análisis se organiza en torno a cinco preguntas:
#
# 1. **¿Qué es exactamente `Pobre`?** Verificamos la regla de construcción y la distribución del ingreso relativo a la línea de pobreza (de ella depende el techo de F1 alcanzable).
# 2. **¿Cómo se distribuye la pobreza en el territorio?** Dominio, clase urbano/rural y costo de vida (líneas de pobreza).
# 3. **¿Son comparables train y test?** Composición, faltantes y una *validación adversarial*.
# 4. **¿Qué patrones de faltantes hay?** Distinguir faltantes estructurales (flujo del cuestionario) de faltantes informativos.
# 5. **¿Qué variables observables en test se asocian con la pobreza?** Composición del hogar, educación, trabajo, formalidad, salud y vivienda.
#
# Todas las figuras se guardan en `02_outputs/figures/01_*.png` y las tablas en `02_outputs/tables/01_*.csv`.

# %%
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

import json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pobreza.config import Rutas, SEED
from pobreza import io as pio, eda, viz
from pobreza.features import variables_persona, anios_educacion
R = Rutas(RAIZ); viz.estilo()
FIG, TAB = R.figuras, R.tablas
def guardar(fig, nombre):
    viz.guardar(fig, FIG, nombre); plt.show()

T = pio.cargar_crudos(R)
th, tp = T["train_hogares"].copy(), T["train_personas"]
teh, tep = T["test_hogares"].copy(), T["test_personas"]
print({k: v.shape for k, v in T.items()})

# %% [markdown]
# ## 1. La variable objetivo
# ### 1.1 Regla de construcción
# En la metodología MESE (DANE, 2012), un hogar es pobre si su ingreso per cápita de la unidad de gasto (con arriendo imputado a propietarios) es inferior a la línea de pobreza de su dominio:
#
# $$\text{Pobre}_h = \mathbb{1}\{\text{Ingpcug}_h < \text{Lp}_h\},\qquad \text{Ingpcug}_h=\frac{\text{Ingtotugarr}_h}{\text{Npersug}_h}.$$
#
# Si esta identidad se cumple exactamente, el problema de clasificación es equivalente a predecir el **signo** de $z_h=\log(\text{Ingpcug}_h/\text{Lp}_h)$. Esto tiene dos implicaciones de primer orden para el modelado:
#
# * Un modelo de **regresión para $z$** usa estrictamente más información que un clasificador sobre `Pobre` (el clasificador descarta la distancia a la línea).
# * La **densidad de $z$ alrededor de 0** determina cuán difícil es el problema: hogares con $|z|$ pequeño son prácticamente imposibles de clasificar con proxies.

# %%
regla = ((th.Ingpcug < th.Lp).astype(int) == th.Pobre).mean()
ident = np.isclose(th.Ingpcug, th.Ingtotugarr / th.Npersug, rtol=1e-6).mean()
indig = ((th.Ingpcug < th.Li).astype(int) == th.Indigente).mean()
th["z"] = np.log(np.maximum(th.Ingpcug, 1) / th.Lp)
print(f"Pobre == 1{{Ingpcug < Lp}}: {regla:.4%}  |  Ingpcug == Ingtotugarr/Npersug: {ident:.4%}  |  Indigente == 1{{Ingpcug < Li}}: {indig:.4%}")
print(f"Tasa de pobreza (hogares, muestral): {th.Pobre.mean():.4f}   ponderada por Fex_c: {np.average(th.Pobre, weights=th.Fex_c):.4f}")
print(f"Tasa de indigencia: {th.Indigente.mean():.4f}  |  Hogares con ingreso cero: {(th.Ingpcug==0).mean():.4%}")
# Pobreza de personas (Npobres / Npersug) ponderada
print(f"Incidencia en personas (ponderada): {np.sum(th.Npobres*th.Fex_c)/np.sum(th.Npersug*th.Fex_c):.4f}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
zz = th.z.clip(-4, 4)
ax[0].hist(zz[th.Pobre == 0], bins=160, color=viz.AZUL, alpha=.75, label="No pobre")
ax[0].hist(zz[th.Pobre == 1], bins=160, color=viz.ROJO, alpha=.75, label="Pobre")
ax[0].axvline(0, color="k", lw=1); ax[0].axvline(np.log(th.Li/th.Lp).median(), color="k", lw=1, ls="--")
ax[0].set(title="Distribución de z = log(Ingpcug / Lp)", xlabel="z (0 = línea de pobreza; -- = línea de indigencia)", ylabel="Hogares")
ax[0].legend()
# Masa alrededor de la línea
bandas = [0.05, 0.1, 0.2, 0.3, 0.5]
masa = [(th.z.abs() < b).mean() for b in bandas]
ax[1].bar([f"|z|<{b}" for b in bandas], masa, color=viz.MORADO)
for i, v in enumerate(masa):
    ax[1].text(i, v, f"{v:.1%}", ha="center", va="bottom")
ax[1].set(title="Hogares cerca de la línea (difíciles de clasificar)", ylabel="Proporción de hogares")
guardar(fig, "01_distribucion_z")

# %% [markdown]
# ### 1.2 Techo teórico de F1 bajo ruido de medición
# Supongamos que el mejor modelo posible predice $\hat z = z + e$ con $e\sim N(0,\sigma_e^2)$. Entonces el F1 que se alcanza clasificando con $\hat z<c$ (umbral $c$ óptimo) depende sólo de la distribución empírica de $z$ y de $\sigma_e$. Simulando con la distribución observada de $z$ obtenemos una curva *F1 alcanzable vs. error de predicción del log-ingreso*. Sirve como brújula: un PMT típico tiene $R^2\approx 0.5$–$0.6$ en log-consumo (Brown, Ravallion y van de Walle, 2018), es decir $\sigma_e \approx 0.5$–$0.6$.

# %%
from pobreza import metrics as M
rng = np.random.default_rng(SEED)
z = th.z.clip(-4, 4).values; y = th.Pobre.values
sig = np.linspace(0.05, 1.2, 24); techo = []
for s in sig:
    zh = z + rng.normal(0, s, len(z))
    _, f = M.umbral_optimo_f1(y, -zh)
    techo.append(f)
r2 = 1 - sig**2 / z.var()
fig, ax = plt.subplots(figsize=(8, 4.3))
ax.plot(sig, techo, "-o", ms=3, color=viz.ROJO)
for s_ref in (0.4, 0.5, 0.6):
    i = np.argmin(abs(sig - s_ref)); ax.annotate(f"σe={s_ref}: F1≈{techo[i]:.3f}\n(R²≈{r2[i]:.2f})", (sig[i], techo[i]), textcoords="offset points", xytext=(10, 10), fontsize=8)
ax.set(title="F1 máximo alcanzable según el error de predicción del log-ingreso relativo", xlabel="σ del error de predicción de z", ylabel="F1 (umbral óptimo)")
guardar(fig, "01_techo_f1")
pd.DataFrame({"sigma_e": sig, "R2_implicito": r2, "F1_max": techo}).to_csv(TAB / "01_techo_f1.csv", index=False)

# %% [markdown]
# ## 2. Pobreza en el territorio
# El DANE estima con 24 dominios (13 áreas metropolitanas + 11 ciudades), *resto urbano* y *rural*. La línea de pobreza (`Lp`) varía por dominio y refleja el costo de la canasta. **Hallazgo clave:** Bogotá está en train pero **no** aparece en test (ver tabla), lo que implica un desplazamiento de covariables que se debe tener en cuenta en la validación.

# %%
dom = eda.tabla_pobreza_por(th, "Dominio")
dom["Lp_mediana"] = th.groupby("Dominio").Lp.median()
dom["z_mediana"] = th.groupby("Dominio").z.median()
cnt_te = teh.Dominio.value_counts()
dom["hogares_test"] = cnt_te.reindex(dom.index).fillna(0).astype(int)
dom["pct_test"] = dom.hogares_test / dom.hogares_test.sum()
dom.to_csv(TAB / "01_pobreza_por_dominio.csv")
display(dom.round(3))
fig, ax = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.3, 1]})
d = dom.sort_values("tasa_ponderada")
se = np.sqrt(d.tasa_muestral * (1 - d.tasa_muestral) / d.hogares)
ax[0].barh(d.index, d.tasa_ponderada, xerr=1.96 * se, color=[viz.ROJO if i in ("RURAL", "RESTO URBANO") else viz.AZUL for i in d.index], alpha=.85)
ax[0].axvline(np.average(th.Pobre, weights=th.Fex_c), color="k", ls="--", lw=1)
ax[0].set(title="Incidencia de pobreza por dominio (ponderada, IC 95%)", xlabel="Proporción de hogares pobres")
ax[1].scatter(dom.Lp_mediana / 1e3, dom.tasa_ponderada, s=dom.hogares / 60, color=viz.MORADO, alpha=.6)
for i, r in dom.iterrows():
    ax[1].annotate(i.title()[:12], (r.Lp_mediana / 1e3, r.tasa_ponderada), fontsize=7)
ax[1].set(title="Línea de pobreza vs. incidencia", xlabel="Lp mediana (miles de pesos/mes)", ylabel="Incidencia")
guardar(fig, "01_pobreza_dominio")

# %%
fig, ax = plt.subplots(figsize=(13, 5))
orden = dom.sort_values("z_mediana").index
sns.boxplot(data=th.assign(zc=th.z.clip(-3, 4)), x="Dominio", y="zc", order=orden, ax=ax, fliersize=0, color=viz.AZUL, linewidth=.8)
ax.axhline(0, color=viz.ROJO, lw=1.2); ax.set(title="z = log(Ingpcug/Lp) por dominio", xlabel="", ylabel="z")
plt.setp(ax.get_xticklabels(), rotation=60, ha="right")
guardar(fig, "01_z_por_dominio")

# %% [markdown]
# ## 3. ¿Son comparables train y test?
# ### 3.1 Composición
# Comparamos las distribuciones de las variables de hogar comunes. Además de Bogotá, verificamos tamaños del hogar, tenencia y montos de vivienda.

# %%
comunes_h = [c for c in teh.columns if c not in ("id", "Dominio", "Depto")]
filas = []
for c in comunes_h:
    a, b = th[c].dropna(), teh[c].dropna()
    ks = stats.ks_2samp(a, b)
    filas.append({"variable": c, "media_train": a.mean(), "media_test": b.mean(), "pct_na_train": th[c].isna().mean(),
                  "pct_na_test": teh[c].isna().mean(), "KS": ks.statistic, "p_valor_KS": ks.pvalue})
comp = pd.DataFrame(filas); comp.to_csv(TAB / "01_train_vs_test_hogar.csv", index=False)
display(comp.round(4))
th_nb = th[th.Dominio != "BOGOTA"]
print("Tasa de pobreza train sin Bogotá:", round(th_nb.Pobre.mean(), 4), "| con Bogotá:", round(th.Pobre.mean(), 4))

# %% [markdown]
# ### 3.2 Validación adversarial
# Entrenamos un clasificador para distinguir hogares de train vs. test con las variables comunes de hogar. Un AUC cercano a 0.5 indica que los conjuntos son intercambiables (mismo proceso generador); un AUC alto revela desplazamiento de covariables. Lo hacemos con y sin Bogotá.

# %%
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
def adversarial(a, b, cols):
    X = pd.concat([a[cols], b[cols]], ignore_index=True)
    X["Dominio"] = pd.Categorical(X["Dominio"]).codes
    yy = np.r_[np.zeros(len(a)), np.ones(len(b))]
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=.1, categorical_features=[cols.index("Dominio")], random_state=SEED)
    p = cross_val_predict(clf, X, yy, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    return M.auc_roc(yy, p)
cols_adv = ["Clase", "Dominio", "P5000", "P5010", "P5090", "P5100", "P5130", "P5140", "Nper", "Npersug", "Lp", "Fex_c"]
auc_con = adversarial(th, teh, cols_adv)
auc_sin = adversarial(th_nb, teh, cols_adv)
auc_sin_fex = adversarial(th_nb, teh, [c for c in cols_adv if c != "Fex_c"])
adv = pd.DataFrame({"escenario": ["train completo vs test", "train sin Bogotá vs test", "sin Bogotá y sin Fex_c"], "AUC": [auc_con, auc_sin, auc_sin_fex]})
adv.to_csv(TAB / "01_validacion_adversarial.csv", index=False); display(adv.round(4))

# %% [markdown]
# ## 4. Faltantes: estructurales vs. informativos
# En la GEIH casi todos los faltantes de persona son **estructurales** (saltos del cuestionario): las preguntas laborales sólo se hacen a la población en edad de trabajar (`Pet`), las de prestaciones sólo a asalariados, etc. Por eso, **imputar no tiene sentido**; la ausencia codifica pertenencia a una rama del cuestionario. La estrategia correcta es construir indicadores (`ocupado`, `asalariado`, `cotiza`, ...) y agregarlos al hogar, donde no hay faltantes.

# %%
comunes_p = [c for c in tep.columns if c in tp.columns and c not in ("id",)]
na = pd.DataFrame({"train": tp[comunes_p].isna().mean(), "test": tep[comunes_p].isna().mean()})
na["universo_si_estructural"] = ""
na.loc[[c for c in comunes_p if c.startswith(("P65", "P66"))], "universo_si_estructural"] = "asalariados (P6430∈{1,2,3,...})"
na.loc[["P6426", "P6430", "P6800", "P6870", "P6920", "P7040", "P7090", "Oficio"], "universo_si_estructural"] = "ocupados (Oc=1)"
na.to_csv(TAB / "01_faltantes_personas.csv")
fig, ax = plt.subplots(figsize=(14, 4))
na_s = na.sort_values("train")
ax.bar(range(len(na_s)), na_s.train, color=viz.AZUL, label="train", alpha=.8)
ax.plot(range(len(na_s)), na_s.test, "o", ms=3, color=viz.ROJO, label="test")
ax.set_xticks(range(len(na_s))); ax.set_xticklabels(na_s.index, rotation=90, fontsize=7)
ax.set(title="Proporción de faltantes por variable de persona (comunes)", ylabel="% faltante"); ax.legend()
guardar(fig, "01_faltantes_personas")
# Verificación del carácter estructural: faltantes de P6920 vs condición de ocupación
tab_na = pd.crosstab(tp.Oc.fillna(0), tp.P6920.isna(), normalize="index")
tab_na.index = ["No ocupado", "Ocupado"]; tab_na.columns = ["P6920 observado", "P6920 faltante"]
display(tab_na.round(4))

# %% [markdown]
# ## 5. Correlatos de la pobreza observables en test
# Construimos indicadores individuales con el módulo `features` y los resumimos a nivel de hogar para describir los gradientes de pobreza. (La construcción definitiva de variables está en el notebook 02.)

# %%
# Verificación de la codificación educativa (nivel P6210 × grado P6210s1) para construir años de educación
ct_edu = pd.crosstab(tp.P6210, tp.P6210s1); ct_edu.to_csv(TAB / "01_nivel_x_grado.csv"); display(ct_edu)
vp = variables_persona(tp)
vp = vp.merge(th[["id", "Pobre", "z", "Fex_c"]], on="id", how="left")
jefe = vp[vp.jefe == 1].copy()
def tasa_por(df, col, etiquetas=None, nmin=50):
    t = df.groupby(col, observed=True).agg(hogares=("Pobre", "size"), tasa=("Pobre", "mean"))
    t = t[t.hogares >= nmin]
    if etiquetas:
        t.index = [etiquetas.get(i, i) for i in t.index]
    t["ee"] = np.sqrt(t.tasa * (1 - t.tasa) / t.hogares)
    return t
from pobreza.features import NIVEL_EDU, POSICION, TENENCIA, PARENTESCO
paneles = {
    "Nivel educativo del jefe": tasa_por(jefe, "nivel_edu", NIVEL_EDU),
    "Posición ocupacional del jefe": tasa_por(jefe, "posicion", POSICION),
    "Régimen de salud del jefe": tasa_por(jefe, "regimen_salud", {1: "contributivo", 2: "especial", 3: "subsidiado", 9: "ns"}),
    "Actividad principal del jefe": tasa_por(jefe, "actividad", {1: "trabajando", 2: "buscando", 3: "estudiando", 4: "oficios hogar", 5: "incapacitado", 6: "otra"}),
    "Tamaño de empresa del jefe": tasa_por(jefe, "tam_empresa", {1: "solo", 2: "2-3", 3: "4-5", 4: "6-10", 5: "11-19", 6: "20-30", 7: "31-50", 8: "51-100", 9: "101+"}),
    "Jefe cotiza a pensión": tasa_por(jefe, "cotiza_pension", {0: "no/no aplica", 1: "sí"}),
}
fig, axs = plt.subplots(2, 3, figsize=(16, 9)); axs = axs.ravel()
media = th.Pobre.mean()
for a, (tit, t) in zip(axs, paneles.items()):
    a.barh([str(i) for i in t.index], t.tasa, xerr=1.96 * t.ee, color=viz.AZUL, alpha=.85)
    a.axvline(media, color=viz.ROJO, ls="--", lw=1); a.set_title(tit, fontsize=10.5); a.invert_yaxis()
    for i, (v, n) in enumerate(zip(t.tasa, t.hogares)):
        a.text(v, i, f" {v:.0%} (n={n:,})", va="center", fontsize=7, color=viz.GRIS)
fig.suptitle("Incidencia de pobreza según características del jefe de hogar (línea roja = promedio)", fontweight="bold")
fig.tight_layout(); guardar(fig, "01_pobreza_jefe")
pd.concat({k: v for k, v in paneles.items()}).to_csv(TAB / "01_pobreza_por_jefe.csv")

# %%
# Agregados del hogar
agg = vp.groupby("id").agg(n_ocupados=("ocupado", "sum"), n_cotizan=("cotiza_pension", "sum"), n_menores=("menor18", "sum"),
                            n_mayores=("mayor65", "sum"), educ_max=("anios_educ", "max"), n_subsidiado=("salud_subsidiado", "sum"),
                            n_benef=("n_beneficios", "sum"), n_dom=("empleado_domestico_hogar", "sum"))
H = th.merge(agg, left_on="id", right_index=True)
H["ocupados_pc"] = H.n_ocupados / H.Npersug
H["personas_dorm"] = H.Nper / H.P5010
H["arriendo"] = H.P5140.where(~H.P5140.isin([98, 99])).fillna(H.P5130.where(~H.P5130.isin([98, 99])))
H["log_arr_linea"] = np.log(H.arriendo / (H.Lp * H.Npersug))
fig, axs = plt.subplots(2, 3, figsize=(16, 8.5)); axs = axs.ravel()
def curva(ax, x, tit, xl, q=20):
    d = H[[x, "Pobre"]].dropna()
    if d[x].nunique() > q:
        d["b"] = pd.qcut(d[x], q, duplicates="drop"); g = d.groupby("b", observed=True).agg(x=(x, "mean"), t=("Pobre", "mean"), n=("Pobre", "size"))
    else:
        g = d.groupby(x).agg(x=(x, "mean"), t=("Pobre", "mean"), n=("Pobre", "size")); g = g[g.n >= 30]
    se = np.sqrt(g.t * (1 - g.t) / g.n)
    ax.plot(g.x, g.t, "-o", ms=3, color=viz.ROJO); ax.fill_between(g.x, g.t - 1.96 * se, g.t + 1.96 * se, color=viz.ROJO, alpha=.15)
    ax.set(title=tit, xlabel=xl, ylabel="P(pobre)")
curva(axs[0], "Npersug", "Tamaño de la unidad de gasto", "personas")
curva(axs[1], "ocupados_pc", "Ocupados por miembro", "ocupados / Npersug")
curva(axs[2], "educ_max", "Máxima educación de los adultos", "años")
curva(axs[3], "personas_dorm", "Personas por dormitorio", "personas/dormitorio")
curva(axs[4], "log_arr_linea", "Arriendo (pagado o estimado) relativo a la línea del hogar", "log(arriendo / (Lp·Npersug))")
curva(axs[5], "n_cotizan", "Miembros que cotizan a pensión", "número")
fig.suptitle("Gradientes de pobreza (bins por cuantiles, IC 95%)", fontweight="bold"); fig.tight_layout()
guardar(fig, "01_gradientes_hogar")

# %%
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
d = H[["log_arr_linea", "z", "P5090"]].dropna()
hb = ax[0].hexbin(d.log_arr_linea.clip(-3, 2), d.z.clip(-3, 4), gridsize=60, cmap="viridis", mincnt=3, bins="log")
ax[0].axhline(0, color="w", lw=1); ax[0].set(title="Vivienda como proxy del ingreso permanente", xlabel="log(arriendo / línea del hogar)", ylabel="z = log(Ingpcug/Lp)")
plt.colorbar(hb, ax=ax[0], label="log10(n)")
rho = d.groupby("P5090").apply(lambda g: stats.spearmanr(g.log_arr_linea, g.z).statistic, include_groups=False)
ax[1].bar([TENENCIA.get(int(i), i) for i in rho.index], rho.values, color=viz.VERDE)
ax[1].set(title="Correlación de Spearman (arriendo vs z) por tenencia", ylabel="ρ"); plt.setp(ax[1].get_xticklabels(), rotation=30, ha="right")
guardar(fig, "01_vivienda_ingreso")
t_ten = tasa_por(H, "P5090", TENENCIA); t_ten.to_csv(TAB / "01_pobreza_por_tenencia.csv"); display(t_ten.round(3))

# %% [markdown]
# ### 5.1 Arriendo imputado en el ingreso (κ)
# MESE suma al ingreso de los propietarios el valor de uso de la vivienda. Estimamos qué fracción del arriendo estimado (`P5130`) aparece en `Ingtotugarr − Ingtotug`.

# %%
d = th[["Ingtotugarr", "Ingtotug", "P5130", "P5090"]].copy()
d = d[(d.P5130 > 1000) & ~d.P5130.isin([98, 99])]
d["dif"] = d.Ingtotugarr - d.Ingtotug
kap = d.groupby("P5090").apply(lambda g: pd.Series({"n": len(g), "kappa_MCO": (g.dif * g.P5130).sum() / (g.P5130 ** 2).sum(),
                                                   "mediana_ratio": (g.dif / g.P5130).median(), "pct_dif_cero": (g.dif.abs() < 1).mean()}), include_groups=False)
kap.index = [TENENCIA.get(int(i), i) for i in kap.index]; kap.to_csv(TAB / "01_kappa_arriendo_imputado.csv"); display(kap.round(3))
print("Hogares arrendatarios con diferencia ≠ 0:", ((th.P5090 == 3) & ((th.Ingtotugarr - th.Ingtotug).abs() > 1)).mean().round(4))

# %% [markdown]
# ## 6. Lo que se pierde en test: estrato e ingresos
# El estrato socioeconómico (`Estrato1`) y todos los ingresos sólo están en train. El estrato es un predictor poderoso pero no utilizable; lo mostramos para dimensionar la información que hay que reconstruir con *proxies* (vivienda, formalidad, territorio).

# %%
jefe_est = tp.loc[tp.P6050 == 1, ["id", "Estrato1"]].merge(th[["id", "Pobre"]], on="id")
t_est = tasa_por(jefe_est, "Estrato1"); t_est.to_csv(TAB / "01_pobreza_por_estrato.csv"); display(t_est.round(3))
print("AUC univariada del estrato (no disponible en test):", round(eda.auc_univariada(jefe_est.Estrato1, jefe_est.Pobre), 4))

# %% [markdown]
# ## 7. Pesos muestrales
# `Fex_c` es el factor de expansión. La métrica de Kaggle es F1 **no ponderado** sobre los hogares de test, que provienen del mismo diseño muestral; por tanto, para predecir no se requiere ponderar (Solon, Haider y Wooldridge, 2015: ponderar se justifica para estimar efectos poblacionales con heterogeneidad, no para pronosticar unidades muestreadas igual). Aun así, el factor contiene información del diseño (estrato muestral, tamaño del segmento) y se evalúa como predictor candidato.

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].hist(np.log(th.Fex_c), bins=80, color=viz.AZUL, alpha=.7, density=True, label="train")
ax[0].hist(np.log(teh.Fex_c), bins=80, color=viz.ROJO, alpha=.5, density=True, label="test")
ax[0].set(title="log(Fex_c): train vs test", xlabel="log factor de expansión"); ax[0].legend()
q = pd.qcut(np.log(th.Fex_c), 20, duplicates="drop")
g = th.groupby(q, observed=True).Pobre.mean()
ax[1].plot(range(len(g)), g.values, "-o", color=viz.ROJO, ms=3); ax[1].set(title="P(pobre) por veintil de log(Fex_c)", xlabel="veintil", ylabel="P(pobre)")
guardar(fig, "01_pesos")

# %% [markdown]
# ## 8. Asociación univariada de todas las variables comunes con la pobreza
# Para cada variable de persona común (resumida en el jefe de hogar) y cada variable de hogar, calculamos el **AUC univariado** (orientado) y la **información mutua** con `Pobre`. Es una primera criba; el notebook 02 hace el análisis definitivo con las variables construidas.

# %%
filas = []
jefe_raw = tp[tp.P6050 == 1].merge(th[["id", "Pobre"]], on="id")
for c in [c for c in comunes_p if c not in ("Orden", "Dominio", "Depto", "Fex_dpto", "Fex_c")]:
    filas.append({"nivel": "jefe", "variable": c, "AUC": eda.auc_univariada(jefe_raw[c], jefe_raw.Pobre),
                  "IM": eda.informacion_mutua_discreta(jefe_raw[c].fillna(-1), jefe_raw.Pobre), "pct_na": jefe_raw[c].isna().mean()})
for c in comunes_h + ["Dominio", "Depto"]:
    x = th[c] if c not in ("Dominio", "Depto") else th[c].astype("category").cat.codes
    filas.append({"nivel": "hogar", "variable": c, "AUC": eda.auc_univariada(x, th.Pobre) if c not in ("Dominio", "Depto") else np.nan,
                  "IM": eda.informacion_mutua_discreta(th[c].fillna(-1) if c not in ("Dominio", "Depto") else th[c], th.Pobre), "pct_na": th[c].isna().mean()})
uni = pd.DataFrame(filas).sort_values("IM", ascending=False); uni.to_csv(TAB / "01_asociacion_univariada.csv", index=False)
fig, ax = plt.subplots(figsize=(9, 10))
top = uni.head(35)
viz.barras_h(ax, [f"{v} ({n})" for v, n in zip(top.variable, top.nivel)], top.IM.values, color=viz.MORADO, fmt="{:.3f}")
ax.set(title="Información mutua con Pobre (top 35, nats)", xlabel="I(X; Pobre)")
guardar(fig, "01_informacion_mutua")
display(uni.head(25).round(4))

# %% [markdown]
# ## 9. Conclusiones del EDA (guía para el modelado)
# Las cifras exactas quedan en las tablas exportadas; las conclusiones operativas son:
#
# 1. `Pobre` es **exactamente** $\mathbb 1\{z<0\}$ con $z=\log(\text{Ingpcug}/\text{Lp})$. ⇒ Modelar $z$ (regresión/probit latente) aprovecha más información que un clasificador binario.
# 2. Una fracción no trivial de hogares está a menos de ±10% de la línea: el **techo de F1** con proxies es claramente inferior a 1 (sección 1.2). Ganancias de F1 de 0.01 son económicamente relevantes.
# 3. La pobreza tiene un fuerte componente **territorial** (dominio, rural/urbano) ⇒ efectos fijos o **aleatorios** por territorio (motivación del GLMM).
# 4. **Bogotá no está en test**: la validación debe reportar el F1 también excluyendo Bogotá de los pliegues de validación, y los efectos de dominio deben estimarse sin que Bogotá distorsione el intercepto común.
# 5. Los faltantes de persona son **estructurales** ⇒ se codifican como indicadores y se agregan al hogar, no se imputan.
# 6. Los correlatos más fuertes son: vivienda (arriendo relativo a la línea, hacinamiento, tenencia), número de perceptores por miembro, formalidad (cotización, prestaciones), educación de los adultos, régimen subsidiado de salud y composición demográfica.
# 7. Los ingresos individuales de train permiten construir una **variable generada** (ingreso esperado del hogar) con predicción fuera de pliegue.
