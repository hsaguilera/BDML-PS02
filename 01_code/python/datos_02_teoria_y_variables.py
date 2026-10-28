# %% [markdown]
# # 02 · Marco teórico, literatura y construcción de variables
# **Problem Set 2 — Predicción de pobreza (MESE/DANE) · BDML Uniandes 2026-20**
#
# Este notebook traduce la teoría económica y social de la pobreza en **variables observables en test**, construye la tabla de modelado (una fila por hogar), estima la **primera etapa** (ingreso individual esperado → variable generada del hogar) con predicción fuera de pliegue y diagnostica el poder predictivo de cada variable. Salidas: `00_data/processed/modelo_{train,test}.parquet`, `00_data/processed/pliegues.csv`, `00_data/processed/conjuntos_variables.json`.
#
# ---
# ## 1. Marco teórico
#
# ### 1.1 El objeto a predecir
# La pobreza monetaria oficial (DANE–MESE) compara el ingreso per cápita de la unidad de gasto con una línea que valora una canasta básica de bienes alimentarios y no alimentarios (enfoque de costo de necesidades básicas; Ravallion, 1998). Formalmente $\text{Pobre}_h=\mathbb 1\{y_h/n_h<L_{d(h)}\}$, donde $y_h$ es el ingreso total del hogar con arriendo imputado, $n_h$ el tamaño de la unidad de gasto y $L_d$ la línea del dominio $d$. Tomando logaritmos,
#
# $$\text{Pobre}_h=\mathbb 1\{z_h<0\},\qquad z_h=\log y_h-\log n_h-\log L_{d(h)}.$$
#
# Esta descomposición organiza todo el modelado: **(i)** $\log n_h$ y $\log L_d$ son *observables* en test (entran con coeficiente teórico −1); **(ii)** lo que hay que predecir es $\log y_h$, el ingreso del hogar, a partir de sus determinantes; **(iii)** la clasificación es un problema de *umbral sobre un índice latente*, exactamente la estructura del modelo probit.
#
# ### 1.2 Determinantes del ingreso del hogar
# El ingreso del hogar es la suma de ingresos laborales y no laborales de sus miembros:
# $$y_h=\sum_{i\in h}\big[\underbrace{e_i\cdot w_i\cdot h_i}_{\text{laboral}}+\underbrace{T_i+K_i+P_i}_{\text{transferencias, capital, pensiones}}\big].$$
#
# * **Participación y empleo ($e_i$).** Número de perceptores por miembro: la razón de dependencia económica es el predictor más robusto de pobreza en América Latina (CEPAL, 2018; Banco Mundial, 2015).
# * **Salario por hora ($w_i$).** Ecuación de Mincer (1974): $\log w_i=\alpha+\rho\,S_i+\beta_1X_i+\beta_2X_i^2$, con $S_i$ escolaridad y $X_i$ experiencia potencial. La teoría del capital humano (Becker, 1964) y la evidencia de retornos convexos en Colombia (alta prima de la educación superior) motivan años de educación, su máximo en el hogar y la educación superior como variables.
# * **Segmentación y formalidad.** El mercado laboral colombiano es dual (≈ 60% de informalidad): la cotización a pensión, el tamaño de la empresa y las prestaciones legales (prima de servicios, cesantías) identifican empleos formales con salario ≥ 1 SMMLV (Perry et al., 2007; Maloney, 2004; Fields, 2011). Ciertas prestaciones revelan **tramos salariales por ley**: el auxilio de transporte sólo se paga si el salario ≤ 2 SMMLV; el subsidio familiar de las cajas, si ≤ 4 SMMLV.
# * **Horas ($h_i$) y subempleo.** Horas trabajadas y deseo de trabajar más horas (subempleo por insuficiencia de horas).
# * **Ingresos no laborales.** Pensiones (alto ingreso relativo y estable), remesas, intereses/dividendos (activos financieros), ayudas institucionales (programas focalizados como Familias/Jóvenes en Acción — *señal de pobreza*, porque su focalización se basa en SISBÉN).
#
# ### 1.3 Ingreso permanente y activos
# Según la hipótesis del ingreso permanente (Friedman, 1957) y la ley de Engel–Schwabe, el gasto en vivienda es una función estable del ingreso permanente: el **arriendo pagado o estimado** es un proxy de bienestar de largo plazo menos ruidoso que el ingreso corriente. Esto es central en los *proxy means tests* (Grosh y Baker, 1995; Coady, Grosh y Hoddinott, 2004): la vivienda y los activos son difíciles de ocultar y cambian lentamente. El **hacinamiento** (personas por cuarto) es además un componente del IPM colombiano (Angulo, Díaz y Pardo, 2016).
#
# ### 1.4 Ciclo de vida, demografía y género
# El modelo de ciclo de vida (Modigliani y Brumberg, 1954) y la literatura de demografía de la pobreza (Lanjouw y Ravallion, 1995) vinculan la pobreza con la **razón de dependencia demográfica** (niños y adultos mayores por adulto en edad de trabajar), la edad del jefe y la estructura del hogar. Hogares monoparentales con jefatura femenina enfrentan restricciones de tiempo y brechas salariales (Buvinić y Gupta, 1997; Chant, 2008). Desde la perspectiva de capacidades (Sen, 1999), la educación y la salud son a la vez determinantes y dimensiones del bienestar.
#
# ### 1.5 Protección social
# El régimen **subsidiado** de salud se asigna por SISBÉN (DNP, 2021), un índice multidimensional que ya resume información de bienestar: es una *variable de focalización revelada*. Por el contrario, el régimen **contributivo** o **especial** (fuerzas armadas, magisterio, Ecopetrol) revela empleo formal o sector público.
#
# ### 1.6 Territorio
# La incidencia varía fuertemente por dominio (mercados laborales locales, costo de vida, ruralidad). La pobreza rural responde a la tenencia de tierra, la informalidad agrícola y la dispersión geográfica (De Janvry y Sadoulet, 2000). Modelamos el territorio con efectos fijos (probit) y **efectos aleatorios con agregación parcial** (GLMM, notebook 04).
#
# ## 2. Literatura de predicción de pobreza con ML
# * **Proxy means tests (PMT).** Grosh y Baker (1995); Coady, Grosh y Hoddinott (2004). Estimar log-consumo con MCO sobre proxies y focalizar por umbral. Brown, Ravallion y van de Walle (2018, *JDE*) muestran que los PMT tradicionales tienen altos errores de exclusión y proponen **regresión por cuantiles** alrededor de la línea; mejoras de 2–5 p.p. en cobertura.
# * **Validación fuera de muestra y ML.** McBride y Nichols (2018, *WBER*): el error relevante es el *fuera de muestra*; con validación cruzada y *random forests* reducen errores de focalización. Kshirsagar, Wieczorek, Ramanathan y Wells (2017): selección de pocas preguntas (L1, *stability selection*) para predecir pobreza con encuestas cortas — directamente relacionado con el criterio de "mínimo número de variables" de esta competencia.
# * **Competencia Pover-T Tests (Banco Mundial, DrivenData, 2018)**, inspiración de este PS: los ganadores usaron *gradient boosting* + redes, agregaciones de variables individuales al hogar normalizadas por tamaño y selección de variables por importancia por permutación.
# * **Datos no tradicionales.** Blumenstock, Cadamuro y On (2015, *Science*), Jean et al. (2016, *Science*), Aiken et al. (2022, *Nature*): metadatos telefónicos e imágenes satelitales para focalizar. Muestran que la **evaluación debe hacerse con las métricas de política** (errores de inclusión/exclusión), no sólo exactitud.
# * **Focalización y su evaluación.** Alatas et al. (2012, *AER*) comparan PMT vs. focalización comunitaria; Hanna y Olken (2018, *JEP*) discuten el *trade-off* inclusión–exclusión: exactamente el que resume F1.
# * **Umbral óptimo para F1.** Lipton, Elkan y Naryanaswamy (2014): con probabilidades calibradas, el umbral F1-óptimo es $F_1^*/2$; por eso el umbral **no** debe ser 0.5 (error del primer intento del equipo: F1 = 0.44).
# * **Variables generadas.** Pagan (1984); Murphy y Topel (1985): los regresores construidos en una primera etapa requieren predicción fuera de muestra para no inflar el ajuste.
# * **Modelos mixtos para áreas pequeñas.** Elbers, Lanjouw y Lanjouw (2003, *Econometrica*); Molina y Rao (2010): efectos aleatorios de área en modelos de ingreso para mapas de pobreza — base del GLMM.
#
# ## 3. Diseño de la construcción de variables
# Toda variable cumple tres condiciones: **(a)** se calcula con columnas presentes en test; **(b)** tiene una justificación teórica (bloques A–F del módulo `features`); **(c)** se agrega del individuo al hogar con sumas, proporciones por miembro de la unidad de gasto o atributos del jefe. Los faltantes de persona son estructurales y se convierten en indicadores.

# %%
import os, sys, warnings, json, time
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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pobreza.config import Rutas, SEED
from pobreza import io as pio, eda, viz, features as F, imputacion as IMP, cv as CV, metrics as M
R = Rutas(RAIZ); viz.estilo()
FIG, TAB = R.figuras, R.tablas
def guardar(fig, nombre):
    viz.guardar(fig, FIG, nombre); plt.show()
T = pio.cargar_crudos(R)
th, tp, teh, tep = T["train_hogares"], T["train_personas"], T["test_hogares"], T["test_personas"]

# %% [markdown]
# ## 4. Construcción de la tabla de modelado
# `features.construir` ejecuta: indicadores individuales → agregación por hogar → variables de vivienda y territorio → imputación del arriendo faltante (mediana por dominio × cuartos de **train**) → razones derivadas.

# %%
t0 = time.time()
Htr = F.construir(th, tp)
ref = Htr[["Dominio", "cuartos", "arriendo"]]
Hte = F.construir(teh, tep, ref_arriendo=Htr.assign(arriendo=Htr.arriendo.where(Htr.arriendo_imputado_flag == 0)))
Y = F.objetivo(th)
assert (Htr.id.values == Y.id.values).all()
print(f"Train: {Htr.shape}  Test: {Hte.shape}  ({time.time()-t0:.1f}s)")
print("Columnas en train y no en test:", set(Htr.columns) ^ set(Hte.columns))

# %% [markdown]
# ## 5. Pliegues de validación cruzada (compartidos por TODOS los modelos)
# Cinco pliegues estratificados por `Pobre × Dominio`. Se guardan para que la primera etapa, el probit y el GLMM usen exactamente la misma partición (condición necesaria para que las predicciones fuera de pliegue de la variable generada no filtren información).

# %%
folds = CV.pliegues(Y.Pobre.values, Htr.Dominio.values, k=5, semilla=SEED)
fold_id = CV.id_pliegue(len(Y), folds)
pliegues = pd.DataFrame({"id": Htr.id.values, "fold": fold_id})
pliegues.to_csv(R.processed / "pliegues.csv", index=False)
display(pd.crosstab(fold_id, Y.Pobre, normalize="index").round(4))

# %% [markdown]
# ## 6. Descomposición exacta del ingreso: arriendo imputado
# MESE define $\text{Ingtotugarr}=\text{Ingtotug}+A_h$, donde $A_h$ es el valor de uso de la vivienda propia. Verificamos la regla de imputación que reconstruye $A_h$ **sólo con variables presentes en test**: propietario sin deuda y usufructo → `P5130`; propietario pagando → $\max(\text{P5130}-\text{P5100},0)$; arrendatario, poseedor sin título y otra → 0. Si la regla es exacta, la condición de pobreza se reescribe como
# $$\text{Pobre}_h=\mathbb 1\{\text{Ingtotug}_h<R_h\},\qquad R_h=\text{Npersug}_h\cdot\text{Lp}_h-A_h,$$
# es decir, el **ingreso requerido** $R_h$ es *observable* y el problema se reduce a predecir el ingreso monetario del hogar $\text{Ingtotug}_h$ y compararlo con un umbral conocido, específico de cada hogar. Hogares con $R_h\le 0$ **no pueden ser pobres**.
#
# **Correcciones encontradas al analizar las fallas de la regla simple** (todas las fallas tienen $A_h=0$):
# 1. **Tope de MESE:** si `P5130` > 3.000.000, $A_h=0$ (exacto: 0% de imputación por encima de 3 millones, 99% por debajo).
# 2. **Propia pagando con cuota desconocida** (`P5100` ∈ {98, 99}): $A_h=0$.
# 3. **Usufructo rural:** $A_h=0$ en ≈ 33% de los casos sin regla observable (la proporción crece con `P5130`); se deja la bandera `usufructo_rural` para que los modelos aprendan esa incertidumbre.

# %%
A = Htr["arriendo_imputado_mese"].values
dif = (th.Ingtotugarr - th.Ingtotug).values
chk = pd.DataFrame({"tenencia": th.P5090.map(F.TENENCIA), "exacto": np.isclose(dif, A, atol=2)})
t_chk = chk.groupby("tenencia").exacto.agg(["mean", "size"]).rename(columns={"mean": "pct_regla_exacta", "size": "hogares"})
t_chk.to_csv(TAB / "02_regla_arriendo_imputado.csv"); display(t_chk.round(4))
p130 = pd.to_numeric(th.P5130, errors="coerce")
tope = pd.DataFrame({"P5130": pd.cut(p130, [0, 1e6, 2e6, 2.9e6, 3e6, 3.5e6, np.inf]), "A_cero": np.abs(dif) < 2})[th.P5090.isin([1, 4]).values]
display(tope.groupby("P5130", observed=True).A_cero.agg(["mean", "size"]).rename(columns={"mean": "share_A_cero"}).round(4))
print(f"Regla exacta en {chk.exacto.mean():.4%} de los hogares de train")
regla = ((th.Ingtotug < Htr.ingreso_requerido).astype(int) == th.Pobre)
print(f"Pobre == 1{{Ingtotug < R}} en {regla.mean():.4%} de los hogares | hogares con R ≤ 0: {Htr.no_puede_ser_pobre.mean():.3%} (de ellos pobres: {th.Pobre[Htr.no_puede_ser_pobre==1].mean():.3%})")

# %% [markdown]
# ## 7. Primera etapa: ingreso individual esperado (variable generada)
# Se estima $E[\text{Ingtot}_i\mid x_i, c_h]$ con *gradient boosting* (LightGBM) de pérdida Poisson (log-link, estima la media condicional en niveles y admite ceros), con árboles de 127 hojas y parada temprana sobre un 10% interno (hasta 4000 árboles). Una versión anterior con `HistGradientBoosting` y tope de 400 iteraciones quedaba sub-ajustada (varios pliegues tocaban el tope); la versión actual sube el R² OOF en niveles de 0,670 a ≈0,68 y el F1 del clasificador LightGBM en ≈0,002. $x_i$ son las características de la persona y $c_h$ el **contexto observable de su hogar** (arriendo relativo a la línea, tenencia, cuartos, perceptores y cotizantes del hogar, educación máxima, régimen subsidiado): la vivienda es el proxy de ingreso permanente más robusto en los PMT, y el ingreso de una persona está correlacionado con el de su hogar (emparejamiento selectivo). Para los hogares de train la predicción es **fuera de pliegue** con los pliegues anteriores; para test se usa el modelo entrenado con todo train. Luego
# $$\hat z^{(1)}_h=\log\frac{\sum_{i\in UG(h)}\hat E[\text{Ingtot}_i\mid x_i]+A_h}{\text{Npersug}_h\cdot \text{Lp}_h},\qquad \hat r_h=\log\Big(\sum_{i\in UG(h)}\hat E[\text{Ingtot}_i\mid x_i]\Big)-\log R_h .$$

# %%
t0 = time.time()
fold_hogar = pliegues.set_index("id").fold
oof_p, te_p, info = IMP.ingreso_individual_oof(tp, tep, fold_hogar, semilla=SEED, ctx_tr=Htr, ctx_te=Hte)
print(f"R² fuera de pliegue (niveles): {info['R2_niveles_oof']:.4f} | R² (log1p): {info['R2_log1p_oof']:.4f} | {time.time()-t0:.0f}s")
ing_tr = IMP.agregar_ingreso(tp, oof_p).reindex(Htr.id).values
ing_te = IMP.agregar_ingreso(tep, te_p).reindex(Hte.id).values
for H, ing in ((Htr, ing_tr), (Hte, ing_te)):
    H["ing_esperado_ug"] = ing
    H["log_ing_esperado_ug"] = np.log(np.maximum(ing, 1e4))
    H["lz_ingreso_esperado"] = np.log(np.maximum(ing + H.arriendo_imputado_mese, 1e3) / (H.Npersug * H.Lp))
    H["brecha_esperada"] = H["log_ing_esperado_ug"] - H["log_ingreso_requerido"]
# agregados adicionales: principal perceptor, jefe, perceptores esperados sobre la línea
for H, per, yh in ((Htr, tp, oof_p), (Hte, tep, te_p)):
    det = IMP.agregar_detalle(per, yh, H.set_index("id").Lp).reindex(H.id)
    H["log_ing_esp_max"] = np.log(np.maximum(det.ing_esp_max.values, 1e4))
    H["log_ing_esp_jefe"] = np.log(np.maximum(det.ing_esp_jefe.fillna(0).values, 1e4))
    H["n_perceptores_esp_lp"] = det.n_perceptores_esp_lp.values
    H["share_principal_perceptor"] = det.share_principal_perceptor.values
display(info["importancia"].head(20).round(4).to_frame("importancia (ganancia) 1ª etapa"))
r = stats.pearsonr(Htr.lz_ingreso_esperado, Y.z)
r2 = stats.pearsonr(Htr.log_ing_esperado_ug, Y.log_ingtotug)
print(f"corr(ẑ¹, z) = {r.statistic:.4f} (R² = {r.statistic**2:.4f}) | corr(log Î, log Ingtotug) = {r2.statistic:.4f}")
print(f"AUC(−ẑ¹ → Pobre) = {M.auc_roc(Y.Pobre, -Htr.lz_ingreso_esperado):.4f} | AUC(−r̂ → Pobre) = {M.auc_roc(Y.Pobre, -Htr.brecha_esperada):.4f}")

# %%
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
hb = ax[0].hexbin(Htr.lz_ingreso_esperado.clip(-3, 4), Y.z.clip(-3, 4), gridsize=70, cmap="magma", mincnt=2, bins="log")
ax[0].plot([-3, 4], [-3, 4], "c--", lw=1); ax[0].axhline(0, color="w", lw=.8); ax[0].axvline(0, color="w", lw=.8)
ax[0].set(title="Primera etapa (OOF): ẑ¹ vs z", xlabel="ẑ¹ = log(ingreso esperado pc / Lp)", ylabel="z observado")
b = pd.qcut(Htr.lz_ingreso_esperado, 40, duplicates="drop")
g = pd.DataFrame({"x": Htr.lz_ingreso_esperado, "y": Y.Pobre}).groupby(b, observed=True).mean()
ax[1].plot(g.x, g.y, "o", color=viz.ROJO, ms=4, label="frecuencia observada")
xs = np.linspace(g.x.min(), g.x.max(), 200)
from pobreza.probit import Probit
p1 = Probit().fit(Htr[["lz_ingreso_esperado"]].values, Y.Pobre.values)
ax[1].plot(xs, p1.predict_proba(xs[:, None]), color=viz.AZUL, label="probit univariado")
ax[1].set(title="P(pobre) vs ẑ¹ (40 bins) y ajuste probit", xlabel="ẑ¹", ylabel="P(pobre)"); ax[1].legend()
res = Y.z - Htr.lz_ingreso_esperado
ax[2].hist(res.clip(-4, 4), bins=120, color=viz.MORADO, alpha=.8, density=True)
xx = np.linspace(-4, 4, 300); ax[2].plot(xx, stats.norm.pdf(xx, res.mean(), res.std()), color="k", lw=1)
ax[2].set(title=f"Residuo z − ẑ¹: media={res.mean():.2f}, d.e.={res.std():.2f}", xlabel="residuo")
guardar(fig, "02_primera_etapa")

# %% [markdown]
# ## 8. Catálogo de variables y poder predictivo univariado
# Para cada variable construida: bloque teórico, % de faltantes, **AUC univariado** (Mann–Whitney, orientado), **información mutua** con `Pobre` y **correlación de Spearman** con $z$.

# %%
excluir = {"id", "Dominio", "Depto", "Lp", "Li", "Fex_c", "arriendo_pagado", "arriendo_estimado", "cuota_amortizacion", "ingreso_requerido",
           "jefe_oficio", "jefe_posicion", "jefe_tam_empresa", "jefe_nivel_edu", "jefe_regimen_salud", "jefe_actividad", "tenencia", "n_personas_reg"}
cands = [c for c in Htr.columns if c not in excluir and pd.api.types.is_numeric_dtype(Htr[c])]
def bloque(c):
    if c.startswith(("lz_", "ing_", "log_ing_esp", "brecha_esp", "n_perceptores_esp", "share_principal")): return "G. Variable generada (1ª etapa)"
    if any(k in c for k in ["mincer", "asalariados_formales", "horas_formales", "ocupados_superior", "ocupados_x", "dependientes_por_ocupado", "educ_jefe_x"]): return "C. Trabajo y formalidad"
    if any(k in c for k in ["arriendo", "cuarto", "dormit", "hacinam", "ten_", "amortiz"]): return "E. Vivienda"
    if any(k in c for k in ["educ", "anios"]): return "B. Capital humano"
    if any(k in c for k in ["salud", "subsid", "recibe", "pension", "transfer", "capital", "ayuda", "remesa"]): return "D. Protección social e ingresos no laborales"
    if any(k in c for k in ["ocup", "formal", "cotiza", "pos_", "asalar", "independ", "horas", "benef", "prima", "empresa", "solo", "subempleo", "segundo", "antig", "sin_rem", "busca", "pago_esp", "extra", "desocup", "inactiv", "cesante", "pet"]): return "C. Trabajo y formalidad"
    if any(k in c for k in ["Lp", "linea", "cabecera", "fex"]): return "F. Territorio y diseño"
    return "A. Demografía y escala"
cat = []
for c in cands:
    x = Htr[c]
    cat.append({"variable": c, "bloque": bloque(c), "pct_na": x.isna().mean(), "media_train": x.mean(), "media_test": Hte[c].mean(),
                "AUC": eda.auc_univariada(x, Y.Pobre), "IM": eda.informacion_mutua_discreta(x.fillna(-999), Y.Pobre),
                "spearman_z": stats.spearmanr(x, Y.z, nan_policy="omit").statistic})
cat = pd.DataFrame(cat).sort_values("AUC", ascending=False)
cat.to_csv(TAB / "02_catalogo_variables.csv", index=False)
print(f"{len(cat)} variables candidatas"); display(cat.head(40).round(4))

# %%
top = cat.head(45)
colores = {b: c for b, c in zip(sorted(cat.bloque.unique()), viz.PALETA)}
fig, ax = plt.subplots(figsize=(10, 12))
ax.barh(range(len(top)), top.AUC - .5, left=.5, color=[colores[b] for b in top.bloque])
ax.set_yticks(range(len(top))); ax.set_yticklabels(top.variable, fontsize=8); ax.invert_yaxis()
for b, c in colores.items():
    ax.barh([0], [0], color=c, label=b)
ax.legend(loc="lower right", fontsize=8); ax.set(xlim=(.5, top.AUC.max() + .02), title="AUC univariado con Pobre (top 45, color = bloque teórico)", xlabel="AUC (orientado)")
guardar(fig, "02_auc_univariado")

# %%
fig, ax = plt.subplots(figsize=(9, 5))
for b, d in cat.groupby("bloque"):
    ax.scatter(d.AUC, d.spearman_z.abs(), s=25, color=colores[b], label=b, alpha=.8)
ax.set(xlabel="AUC univariado (Pobre)", ylabel="|ρ Spearman| con z", title="Dos medidas de relevancia: clasificación vs. ordenamiento del ingreso")
ax.legend(fontsize=7)
guardar(fig, "02_auc_vs_spearman")

# %% [markdown]
# ## 9. Redundancia: correlación entre predictores
# Agrupamiento jerárquico de las 40 variables con mayor AUC según $1-|\rho_{\text{Spearman}}|$. Bloques muy correlacionados indican información redundante; para el probit conviene elegir representantes (o penalizar), lo que se formaliza con el camino L1 del notebook 03.

# %%
top40 = [c for c in cat.dropna(subset=["AUC"]).head(40).variable if Htr[c].std() > 0]
Cv = Htr[top40].rank().corr().abs().fillna(0).to_numpy(copy=True)
np.fill_diagonal(Cv, 1.0)
C = pd.DataFrame(Cv, index=top40, columns=top40)
g = sns.clustermap(C, cmap="mako", figsize=(13, 13), vmin=0, vmax=1, method="average", xticklabels=True, yticklabels=True)
g.fig.suptitle("|ρ de Spearman| entre las 40 variables más predictivas (agrupamiento jerárquico)", y=1.01, fontweight="bold")
plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=7); plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=7)
g.savefig(FIG / "02_clustermap_correlacion.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 10. Relaciones no lineales clave
# Curvas de P(pobre) por bins de cuantiles para variables continuas centrales; motivan splines en el probit.

# %%
claves = ["brecha_esperada", "log_arriendo_sobre_linea", "ocupados_pc", "educ_max_adultos", "personas_por_dormitorio", "jefe_edad", "dependencia", "share_subsidiado"]
fig, axs = plt.subplots(2, 4, figsize=(18, 8)); axs = axs.ravel()
for a, c in zip(axs, claves):
    d = pd.DataFrame({"x": Htr[c], "y": Y.Pobre}).dropna()
    if d.x.nunique() > 25:
        b = pd.qcut(d.x, 25, duplicates="drop"); gg = d.groupby(b, observed=True).agg(x=("x", "mean"), t=("y", "mean"), n=("y", "size"))
    else:
        gg = d.groupby("x").agg(x=("x", "mean"), t=("y", "mean"), n=("y", "size")); gg = gg[gg.n >= 30]
    se = np.sqrt(gg.t * (1 - gg.t) / gg.n)
    a.plot(gg.x, gg.t, "-o", ms=3, color=viz.ROJO); a.fill_between(gg.x, gg.t - 1.96 * se, gg.t + 1.96 * se, alpha=.15, color=viz.ROJO)
    # escala probit: Φ^{-1}(p) para juzgar linealidad del índice
    a2 = a.twinx(); a2.plot(gg.x, stats.norm.ppf(gg.t.clip(.002, .998)), ":", color=viz.AZUL); a2.set_ylabel("Φ⁻¹(p)", color=viz.AZUL, fontsize=8); a2.grid(False)
    a.set(title=c, ylabel="P(pobre)")
fig.suptitle("Relación con P(pobre) (rojo) y en escala probit Φ⁻¹(p) (azul punteado): curvatura ⇒ splines", fontweight="bold"); fig.tight_layout()
guardar(fig, "02_no_linealidades")

# %% [markdown]
# ## 11. Conjuntos de variables para modelar
# Se definen tres conjuntos anidados, con justificación teórica y criterio de parsimonia (la competencia premia modelos con pocas variables):
#
# * **S1 — PMT parsimonioso (≈15 variables):** un representante por mecanismo teórico.
# * **S2 — Teórico completo:** todos los bloques A–F (sin la variable generada).
# * **S3 — S2 + variable generada** (ingreso esperado de primera etapa).
#
# Las categóricas de alta cardinalidad (dominio, oficio del jefe) se tratan como efectos fijos (dummies) en el probit y como **efectos aleatorios** en el GLMM.

# %%
S1 = ["log_arriendo_sobre_linea", "personas_por_dormitorio", "ocupados_pc", "formales_pc", "educ_max_adultos", "jefe_anios_educ",
      "dependencia", "share_subsidiado", "ten_arriendo", "ten_propia_pagada", "cabecera", "log_ingreso_requerido", "no_puede_ser_pobre", "jefe_mujer_sin_pareja",
      "recibe_transferencias", "recibe_pension", "sin_ocupados", "n_empleado_domestico_hogar"]
generadas = ["lz_ingreso_esperado", "log_ing_esperado_ug", "brecha_esperada", "log_ing_esp_max", "log_ing_esp_jefe",
             "n_perceptores_esp_lp", "share_principal_perceptor"]
S2 = [c for c in cat.variable if c not in generadas + ["ing_esperado_ug", "log_fex"]]
S3 = S2 + generadas
categoricas = ["Dominio", "jefe_posicion", "jefe_nivel_edu", "jefe_regimen_salud", "jefe_actividad", "jefe_tam_empresa", "tenencia"]
conjuntos = {"S1": S1, "S2": S2, "S3": S3, "categoricas": categoricas,
             "primera_etapa": {k: v for k, v in info.items() if k != "modelo_final"}}
with open(R.processed / "conjuntos_variables.json", "w", encoding="utf-8") as f:
    def _a_json(o):
        """Convierte a tipos JSON lo que dejó la 1.ª etapa (series, arreglos, escalares de numpy)."""
        if hasattr(o, "to_dict"): return o.to_dict()
        if hasattr(o, "tolist"): return o.tolist()
        try: return float(o)
        except (TypeError, ValueError): return str(o)
    json.dump(conjuntos, f, ensure_ascii=False, indent=1, default=_a_json)
print({k: len(v) for k, v in conjuntos.items() if isinstance(v, list)})

# %% [markdown]
# ## 12. Guardar tablas de modelado

# %%
mod_tr = Htr.merge(Y, on="id").merge(pliegues, on="id")
mod_te = Hte.copy()
pio.guardar_tabla(mod_tr, R, "modelo_train"); pio.guardar_tabla(mod_te, R, "modelo_test")
print(mod_tr.shape, mod_te.shape, "→", R.processed)
