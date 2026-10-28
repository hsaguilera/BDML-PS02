# %% [markdown]
# # 02d · Variables versión 7: todo lo permitido, bloque IPM y codificación bayesiana fuera de pliegue
# **Problem Set 2 — Predicción de pobreza (MESE/DANE, GEIH 2018) · BDML Uniandes 2026-20**
#
# Este cuaderno construye, mide y audita la **versión 7** de las variables. Sigue las reglas vinculantes del plan v7
# (19-oct-2026):
#
# 1. **Sin primera etapa**: nada derivado de modelos entrenados con ingresos (las generadas de v3/v5 quedan prohibidas).
# 2. **Sí** se usan las variables de ingreso o monetarias que están **tanto en train como en test**: las preguntas sí/no
#    de recepción de ingresos y prestaciones (P6510, P6545, P6580, P6585s1–s4, P6590, P6600, P6610, P6620,
#    P6630s1–s6, P7422, P7472, P7495, P7500s2, P7500s3, P7505, P7510s1–s7), los montos de vivienda (P5100, P5130,
#    P5140) y las líneas `Lp` y `Li`. Lo que sólo está en train (montos de ingreso de personas, `Ingtot*`, `Ingpcug`,
#    `Npobres`, `Indigente`, `Estrato1`, `Cclasnr*`) **no**, salvo la etiqueta `Pobre`.
# 3. Sin datos ni constantes externas (por eso sale el índice de Mincer de `features.py`, con coeficientes de la
#    literatura); las definiciones metodológicas (umbrales del IPM, hacinamiento crítico) se documentan.
# 4. Indicación del usuario: usar **todas** las columnas comunes (de hogar y de persona, agregadas al hogar sin
#    excepción), transformaciones, funciones de varias variables y dummies; el conjunto `todas_permitidas` queda
#    documentado y la selección final la hace cada modelo en la fase D.
#
# **Estándar de la medición.** Cada familia se mide con su error estándar entre pliegues, prueba pareada con la
# varianza corregida de Nadeau y Bengio (2003), intervalo bootstrap pareado del ΔF1 y estabilidad entre semillas;
# la codificación con la etiqueta usa encogimiento bayesiano empírico jerárquico (beta-binomial, varianza entre grupos
# estimada por máxima verosimilitud marginal) calculado **dentro del pliegue**; la auditoría de fugas es formal
# (prueba de aleatorización, pruebas de etiqueta nula y de etiqueta perturbada, validación adversarial).

# %% [markdown]
# ## 0. Estimando, riesgo y por qué importan las variables
#
# Sea $Q$ la distribución de los hogares del test (los de train **fuera de Bogotá**: la validación adversarial dio
# AUC = 0,499) y $\eta(x)=P_Q(\text{Pobre}=1\mid X=x)$. Para la métrica F1, el clasificador óptimo en la población
# es un umbral sobre $\eta$: $\hat y=\mathbb 1\{\eta(x)\ge\delta^*\}$ con $\delta^*=F1^*/2$ (Lipton, Elkan y
# Naryanaswamy, 2014; Koyejo et al., 2014; Narasimhan et al., 2014). Por tanto **el valor de una familia de variables
# es lo que mejora la estimación de $\eta$**: lo medimos con una regla de puntuación propia (log-loss sin Bogotá) y
# con el F1 «plug-in» con umbral anidado (elegido en los otros pliegues, fuera de Bogotá).
#
# La regla de construcción de la etiqueta es exacta (TRASPASO §3.5, GUIA §4.1):
# $$\text{Pobre}_h=\mathbb 1\{\text{Ingtotug}_h<R_h\},\qquad R_h=\text{Npersug}_h\cdot\text{Lp}_h-A_h ,$$
# con $A_h$ el arriendo imputado que MESE suma al ingreso (reconstruible con P5090, P5100 y P5130). Si $F_h$ es la
# distribución condicional del ingreso, $\eta(x_h)=F_h(R_h\mid x_h)$: la probabilidad depende del **umbral observable**
# y de la **localización y dispersión** del ingreso dado $x$. Sin montos de ingreso, toda variable útil es un *proxy*
# de esa localización (capacidad de generar ingreso, margen extensivo de cada fuente, ingreso permanente) o del
# umbral mismo. Las familias se diseñan alrededor de esa identidad, expresando la capacidad en unidades de
# «líneas necesarias» $n_\ell=R_h/\text{Lp}_h$ (el número de líneas per cápita que el hogar debe cubrir).

# %%
import os, sys, json, time, warnings, pickle, re
warnings.filterwarnings("ignore")
HILOS = int(os.environ.get("AJUSTE_HILOS", "12"))
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
from threadpoolctl import threadpool_limits
from pobreza.config import Rutas, SEED
from pobreza import io as pio, viz, metrics as M, ajuste as AJ, busqueda as BU, permitidas as PM, variables_v7 as V
R = Rutas(RAIZ); viz.estilo()
SUB = float(os.environ.get("V7_SUBMUESTRA", "0") or 0)
PRUEBA = SUB > 0
SEMILLAS = [int(s) for s in os.environ.get("V7_SEMILLAS", "1,2,3").split(",")]
SEMILLAS_LOO = SEMILLAS[:int(os.environ.get("V7_SEMILLAS_LOO", "2"))]
REUSAR = os.environ.get("AJUSTE_REUSAR", "1") == "1"
FIG = R.figuras / "prueba_02d" if PRUEBA else R.figuras
TAB = R.tablas / "prueba_02d" if PRUEBA else R.tablas
OUT = R.processed / "prueba_02d" if PRUEBA else R.processed
for d in (FIG, TAB, OUT):
    d.mkdir(parents=True, exist_ok=True)
CACHE = __import__("pathlib").Path(os.environ.get("V7_CACHE", os.path.join(str(R.processed), "cache_02d"))) / ("prueba" if PRUEBA else "completa")
CACHE.mkdir(parents=True, exist_ok=True)
T0 = time.time()
T = pio.cargar_crudos(R)
th, tp, teh, tep = T["train_hogares"], T["train_personas"], T["test_hogares"], T["test_personas"]
pl = pd.read_csv(R.processed / "pliegues.csv")
assert (pl.id.values == th.id.values).all()
D5 = pio.leer_tabla(R, "modelo_train_v5")
assert (D5.id.values == th.id.values).all()
if PRUEBA:                                          # submuestra estratificada por pliegue × Pobre (sólo para probar)
    g = (pl.fold.astype(str) + "_" + th.Pobre.astype(int).astype(str)).values
    keep = np.sort(pd.Series(np.arange(len(th))).groupby(g).sample(frac=SUB, random_state=1).values)
    th, pl, D5 = (x.iloc[keep].reset_index(drop=True) for x in (th, pl, D5))
    tp = tp[tp.id.isin(set(th.id))].reset_index(drop=True)
    kt = np.sort(np.random.default_rng(1).choice(len(teh), int(SUB * len(teh)), replace=False))
    teh = teh.iloc[kt].reset_index(drop=True); tep = tep[tep.id.isin(set(teh.id))].reset_index(drop=True)
y = th.Pobre.values.astype(int); fold = pl.fold.values
folds = [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(5)]
sin_bog = (th.Dominio != "BOGOTA").values
conj5 = json.load(open(R.processed / "conjuntos_variables_v5.json", encoding="utf-8"))
print(f"{'PRUEBA con submuestra ' + str(SUB) if PRUEBA else 'CORRIDA COMPLETA'} | train {th.shape} | personas {tp.shape} | "
      f"test {teh.shape} / {tep.shape} | hilos {HILOS} | semillas {SEMILLAS} (LOO {SEMILLAS_LOO}) | caché {CACHE}")

# %% [markdown]
# ## 1. Construcción (`variables_v7.construir_v7`)
#
# **Garantía por construcción.** La función recibe las tablas crudas, pero lo primero que hace es quedarse con las
# 16 columnas comunes de hogar y las 63 de persona (`HOG_COMUNES`, `PER_COMUNES`). Encima de la **base permitida**
# (las 170 variables del S2 de v5 sin el índice de Mincer: bloques A–F de `features.py`, verificadas idénticas a las
# columnas homónimas de v5) se construyen nueve familias. Todo se agrega al hogar; las familias de ingreso y capacidad
# se agregan sobre la **unidad de gasto** (sin empleados domésticos ni pensionistas, P6050 ∉ {6, 7, 8}), que es la que
# define `Ingtotug` y `Npersug`; la agregación exhaustiva (familia G) usa a todas las personas.
#
# | familia | contenido | justificación económica | justificación estadística |
# |---|---|---|---|
# | **U. Umbral observable** | $R_h$, $\log n_\ell$, $R_h/\text{Npersug}$, $A_h$ en líneas, $\mathbb 1\{R_h\le 0\}$ (en la base), holgura, `Lp`, `Li`, $(\text{Lp}-\text{Li})/\text{Lp}$, montos P5100, P5130 y P5140 limpios (98/99 → faltante con indicador propio), arriendo pagado/estimado y cuota sobre la línea del hogar, cuota/arriendo | la pobreza es relativa a una línea propia de cada hogar (Ravallion, 1998); $A_h$ desplaza el umbral; el arriendo es un *proxy* del ingreso permanente (Friedman, 1957; ley de Schwabe) | $\eta(x)=F(R_h\mid x)$: el umbral entra directamente; $R_h\le0\Rightarrow\eta=0$ (exacto: 0,014 % de pobres). El «no sabe» es un faltante informativo (MNAR; Little y Rubin, 2019), por eso se conserva como indicador |
# | **I. Ingresos observables** | las 28 preguntas sí/no de recepción: «sí», «no sabe», universo, tasa, por miembro, por línea necesaria y respuesta del jefe; arriendos inferidos (P7495 = 1 sin P7500s2 ni P7500s3), tipos distintos de fuente (13), perceptores potenciales (laborales y no laborales) por miembro y por línea, prestaciones de ley, extralegales y en especie, «no sabe» total | margen extensivo de cada componente del ingreso; el auxilio de transporte sólo se paga a salarios ≤ 2 SMMLV (se usa la **respuesta**, nunca el valor del salario mínimo) y las primas de ley revelan formalidad (Perry et al., 2007); diversificación y número de perceptores (Grosh y Baker, 1995) | con un modelo de dos partes $Y=\sum_j D_jY_j$ (Cragg, 1971; Duan et al., 1983), $E[Y\mid x]=\sum_j P(D_j=1\mid x)\,E[Y_j\mid D_j=1,x]$: observar $D_j$ elimina la incertidumbre del margen extensivo, que es la mitad del problema en componentes raros (pensiones, remesas, intereses) |
# | **C. Capacidad sin montos** | ocupados, asalariados, formales, cuenta propia, patronos, horas, segunda actividad, subempleo, tamaño de empresa, antigüedad, años de educación y experiencia potencial de ocupados, del jefe y del perceptor principal; índices $\sum h_iS_i$, $\sum h_iX_i$, $\sum h_iS_i\cdot\text{formal}_i$, por miembro y por $n_\ell$; rangos en el dominio | el ingreso laboral es horas × salario y el salario depende de escolaridad y experiencia (Mincer, 1974; Becker, 1964); la formalidad y el tamaño de empresa capturan productividad (Maloney, 2004) | en lugar de imponer $\rho,\beta_1,\beta_2$ externos se entregan los **bloques suficientes** ($\sum h$, $\sum hS$, $\sum hX$) y el clasificador aprende la función; normalizar por $n_\ell$ pone capacidad y requerimiento en la misma escala |
# | **P. Bloque IPM** | 9 de los 15 indicadores del IPM oficial (ver §3), privaciones, puntaje con pesos oficiales repartidos dentro de cada dimensión, $\mathbb 1\{\text{puntaje}\ge 1/3\}$, intensidad censurada | método de conteo de Alkire y Foster (2011); IPM de Colombia (Angulo, Díaz y Pardo, 2016); privaciones como correlatos del ingreso permanente | indicadores gruesos (binarios) de variables que ya están en otras familias: su valor se mide por la **ganancia condicional** |
# | **V. Composición, vivienda, territorio, protección social** | dependencia, menores y mayores por adulto, monoparentalidad, jefatura, pirámide de edades, hacinamiento mitigable, cuartos no dormitorio, régimen subsidiado (por adulto, total, sin cotizantes), factores de expansión, dominio rural y resto urbano | ciclo de vida (Modigliani y Brumberg, 1954), economías de escala en el hogar (Lanjouw y Ravallion, 1995), jefatura femenina (Buvinić y Gupta, 1997), el régimen subsidiado está focalizado con el SISBÉN | variables verificables y baratas del PMT (Coady, Grosh y Hoddinott, 2004); los factores de expansión resumen el diseño muestral (estrato de muestreo), comunes a train y test |
# | **E. Codificación con la etiqueta** | tasa de pobreza encogida de: oficio del jefe y del perceptor principal (padre: gran grupo ocupacional), Dominio × tenencia (padre: Dominio), Depto × Clase (padre: Depto), posición del jefe | el oficio y el territorio resumen mercados laborales y costo de vida | bayes empírico jerárquico beta-binomial (§4) fuera de pliegue; encogimiento óptimo en ECM (Efron y Morris, 1975) |
# | **X. Interacciones** | 18 interacciones con motivación económica (capacidad × cabecera, valor de la vivienda × tenencia, IPM × líneas, …), posiciones relativas de v5/v6 y las candidatas guiadas por los valores de interacción SHAP | heterogeneidad de retornos urbano/rural, activos de vivienda | los árboles aproximan productos y cocientes con muchas particiones; las interacciones SHAP (Lundberg et al., 2020) señalan dónde hacerlas explícitas |
# | **G. Agregación exhaustiva** | cada columna común de persona: numéricas (suma, media, máximo, mínimo, desviación, n, jefe, perceptor principal), categóricas (conteo y proporción por nivel, faltantes, código del jefe y del perceptor principal), oficio por gran grupo, preguntas sí/no no monetarias, PET/ocupado/desocupado/inactivo | todo lo observable del hogar | un predictor de hogar es una función **simétrica** del conjunto de personas; toda función continua de conjuntos se escribe como $\rho(\sum_i\phi(x_i))$ (Zaheer et al., 2017): conteos, proporciones y momentos son la base $\phi$; jefe y perceptor principal agregan estadísticos de orden |
# | **T. Transformaciones y dummies** | log, raíz, cuadrado e inversa de 37 continuas; recorte 1 %–99 %; rango y decil dentro del dominio; dummies de Dominio, Depto y de las categóricas del jefe; tenencia × cabecera; 17 umbrales con sentido económico | concavidad de los retornos, umbrales de política (primaria, bachillerato, 2 SMMLV implícito en el auxilio) | los árboles son invariantes a transformaciones monótonas (Breiman et al., 1984), los modelos lineales y Naive Bayes no: estas columnas son para ellos |
#
# Después de construir se eliminan las columnas constantes y los **duplicados exactos** (en train y test a la vez),
# conservando la primera aparición en el orden B → U → I → C → P → V → X → G → T (la base se conserva íntegra).

# %%
t = time.time()
D, DT, META = V.construir_v7(th, tp, teh, tep)
FAM, DESC = dict(META["familia"]), dict(META["descripcion"])
COLS_CONSTRUCCION = [c for c in D.columns]
BASE = [c for c in conj5["S2"] if c not in V.CONSTANTES_EXTERNAS]
assert all(c in D.columns for c in BASE), [c for c in BASE if c not in D.columns]
comp = [c for c in BASE if c in D5.columns and c not in ("Dominio", "Depto") + tuple(V.CAT_TODAS)]
dif = [c for c in comp if not np.allclose(D[c].astype(float), D5[c].astype(float), equal_nan=True)]
print(f"Base permitida: {len(BASE)} columnas (S2 de v5 sin Mincer); comparadas con v5: {len(comp)}; distintas: {dif or 'ninguna'}")
print(f"Columnas constantes eliminadas: {len(META['constantes'])}; duplicados exactos eliminados: {len(META['duplicadas'])}")
cnt = pd.Series({c: FAM[c] for c in D.columns if c in FAM and c not in V.NO_NUMERICAS}).value_counts()
display(pd.DataFrame({"familia": [V.FAMILIAS.get(k, k) for k in cnt.index], "columnas": cnt.values}))
PROH_PERSONA = V.prohibidas_persona(tp, tep)
print(f"Columnas de persona sólo de train (prohibidas): {len(PROH_PERSONA)}")

# %% [markdown]
# ### 1.1 Mapa variable → preguntas de la encuesta (unidad de parsimonia)
# El memo metodológico (§3.2) fija la **pregunta** como unidad de costo: el costo de un conjunto $S$ es
# $c(S)=|\bigcup_{j\in S}Q(j)|$, con $Q(j)$ las preguntas que necesita la variable $j$; las líneas `Lp` y `Li`, el
# territorio (`Clase`, `Dominio`, `Depto`) y los factores de expansión no son preguntas (costo 0). El mapa no se escribe a
# mano: se obtiene por **rastreo exacto de dependencias**. En 6.000 hogares de train se reconstruye v7 permutando al
# azar una columna común cada vez (77 reconstrucciones); $c\in Q(j)$ si y sólo si la variable $j$ cambia. Como la
# construcción es determinista, una dependencia real cambia la variable en algún hogar con probabilidad ≈ 1 (salvo
# códigos rarísimos, cubiertos por una red de seguridad por nombre). P6050 (parentesco) se permuta sólo entre los no
# jefes y se añade a toda variable del jefe, de la pareja, del perceptor principal o agregada sobre la unidad de gasto.

# %%
MAPA = AJ.en_cache(CACHE / "mapa_preguntas.pkl",
                   lambda: V.mapa_preguntas_v7(th, tp, teh, tep, columnas=[c for c in D.columns if c != "id"], n_hogares=min(6000, len(th))), REUSAR)
falt = [c for c in D.columns if c != "id" and c not in MAPA]
assert not falt, falt[:20]
for v in ["ingreso_requerido", "log_arriendo_sobre_linea", "ocupados_pc", "p_puntaje", "c_hs_linea", "i_aux_transporte_por_asal"]:
    print(f"{v:28s} → {MAPA[v]}")
uso_q = pd.Series([q for c in D.columns if c in MAPA for q in MAPA[c]]).value_counts()
print(f"Preguntas (columnas comunes) usadas por alguna variable: {len(uso_q)} de {len(V.HOG_COMUNES) + len(V.PER_COMUNES) - 2}")

# %% [markdown]
# ## 2. Diccionario de las preguntas de ingreso (DANE) y su relación con la pobreza
#
# Verificado en el diccionario DANE (`ddi-documentation-spanish-608.pdf`, MESE 2018) y en los datos: 1 = sí, 2 = no
# y 9 = no sabe/no informa (las P6630s*, P7422, P7472, P7495 y P7505 sólo admiten 1/2). P7495 y P7505 son **filtros**:
# P7500s2–s3 se preguntan sólo si P7495 = 1 y P7510s* sólo si P7505 = 1 (en los datos ninguna respuesta viola el
# filtro). P7500s1 (arriendos) está sólo en train, pero se **infiere** en ambos conjuntos: P7495 = 1 sin pensión ni
# pensión alimenticia implica arriendos. El universo de P6510–P6630s6 son los asalariados (P6430 ∈ {1, 2, 3, 8});
# P7422 se pregunta a los desocupados cesantes y P7472 a los inactivos.

# %%
filas = []
mtr = sin_bog
for q, (nom, txt, univ) in V.SI_NO_INGRESO.items():
    a, b = tp[q], tep[q]
    hog_si = tp.loc[tp[q] == 1, "id"].unique()
    tiene = th.id.isin(set(hog_si)).values
    filas.append({"codigo": q, "nombre": nom, "pregunta": txt, "universo": univ,
                  "codigos_train": sorted(a.dropna().unique().astype(int).tolist()), "codigos_test": sorted(b.dropna().unique().astype(int).tolist()),
                  "pct_si_entre_resp_train": float((a == 1).sum() / max(a.notna().sum(), 1)),
                  "pct_si_entre_resp_test": float((b == 1).sum() / max(b.notna().sum(), 1)),
                  "pct_hogares_con_si_trainSB": float(tiene[mtr].mean()),
                  "pobreza_hogares_con_si": float(y[mtr & tiene].mean()) if (mtr & tiene).any() else np.nan,
                  "pobreza_hogares_sin_si": float(y[mtr & ~tiene].mean())})
dic = pd.DataFrame(filas)
dic.to_csv(TAB / "02d_diccionario_ingresos.csv", index=False)
display(dic.drop(columns=["pregunta"]).round(3))

# %% [markdown]
# ## 3. Bloque IPM: definición, lo que no se observa y relación con la pobreza monetaria
#
# El IPM colombiano (Angulo, Díaz y Pardo, 2016; método de Alkire y Foster, 2011) tiene 5 dimensiones con peso 1/5 y
# 15 indicadores; un hogar es pobre multidimensional si su puntaje ponderado es ≥ 33 %. Con las columnas comunes se
# aproximan **9 indicadores** (privación del hogar = 1 si se cumple la condición; hogares sin miembros elegibles no
# están privados en ese indicador):
#
# | dimensión (peso) | indicador (peso oficial) | aproximación con estas columnas |
# |---|---|---|
# | Educación (0,2) | bajo logro educativo (0,1) | promedio de años de educación de los miembros ≥ 15 años < 9 |
# | | analfabetismo (0,1) | algún miembro ≥ 15 sin nivel educativo (P6210 ∈ {ninguno, preescolar}) |
# | Niñez y juventud (0,2) | inasistencia escolar (0,05) | algún miembro ≤ 16 de la PET (≥ 12 urbano, ≥ 10 rural: es donde existe P6240) cuya actividad principal no es estudiar |
# | | rezago escolar (0,05) | algún 7–17 con años aprobados < edad − 6 (norma nacional); variante con un año de holgura |
# | | trabajo infantil (0,05) | algún ocupado de 12 a 17 años |
# | Trabajo (0,2) | desempleo de larga duración (0,1) | algún desocupado (no se observa la duración) |
# | | empleo informal (0,1) | algún ocupado que no cotiza a pensión ni es pensionado (P6920) |
# | Salud (0,2) | sin aseguramiento (0,1) | algún miembro con P6090 = 2 (en estos datos P6090 sólo existe para ≥ 10 años) |
# | Vivienda (0,2) | hacinamiento crítico (0,04) | más de 3 personas por dormitorio (`hacinamiento_critico` de la base) |
#
# **No observables (6):** barreras de acceso a servicios de primera infancia (0,05), barreras de acceso a salud dada
# necesidad (0,1), fuente de agua mejorada, eliminación de excretas, pisos y paredes exteriores (0,04 cada uno): las
# tablas no traen módulos de servicios, materiales ni morbilidad. **Puntaje:** cada dimensión pesa 1/5 y su peso se
# reparte por igual entre los indicadores observables (educación 0,1 + 0,1; niñez 0,2/3 × 3; trabajo 0,1 + 0,1; salud
# 0,2; vivienda 0,2). Se agrega la cota inferior del puntaje oficial (pesos oficiales de los 9 observables, máximo
# 0,69) y la intensidad censurada $\text{puntaje}\cdot\mathbb 1\{\text{puntaje}\ge 1/3\}$.

# %%
m = sin_bog
ind = [k for k in V.PESOS_IPM]
filas = []
for c in ind + ["p_rezago_holgura", "p_n_privaciones", "p_puntaje", "p_puntaje_oficial", "p_ipm_pobre", "p_ipm_oficial_33", "p_intensidad",
                "p_share_informal_pea", "p_educ_media_15"]:
    x = D[c].astype(float).values
    ok = m & ~np.isnan(x)
    fila = {"variable": c, "AUC_sinBOG": M.auc_roc(y[ok], x[ok]), "media_trainSB": float(np.nanmean(x[m])),
            "media_test": float(np.nanmean(DT[c].astype(float)))}
    if set(np.unique(x[ok])) <= {0.0, 1.0}:
        fila.update({"P(Pobre|privado)": float(y[ok & (x == 1)].mean()), "P(Pobre|no privado)": float(y[ok & (x == 0)].mean())})
        fila["razon_de_riesgo"] = fila["P(Pobre|privado)"] / max(fila["P(Pobre|no privado)"], 1e-9)
    filas.append(fila)
ipm_auc = pd.DataFrame(filas)
ipm_auc.to_csv(TAB / "02d_ipm_auc.csv", index=False)
display(ipm_auc.round(3))
ipm_m = D.loc[m, "p_ipm_pobre"].values
cr = pd.crosstab(np.where(ipm_m == 1, "pobre IPM", "no pobre IPM"), np.where(y[m] == 1, "pobre monetario", "no pobre monetario"),
                 rownames=["IPM"], colnames=["monetaria"], margins=True)
cr_tasa = pd.crosstab(np.where(ipm_m == 1, "pobre IPM", "no pobre IPM"), np.where(y[m] == 1, "pobre monetario", "no pobre monetario"),
                      rownames=["IPM"], colnames=["monetaria"], normalize="index").round(4)
kappa = (lambda po, pe: (po - pe) / (1 - pe))((D.loc[m, "p_ipm_pobre"].values == y[m]).mean(),
                                              D.loc[m, "p_ipm_pobre"].mean() * y[m].mean() + (1 - D.loc[m, "p_ipm_pobre"].mean()) * (1 - y[m].mean()))
print(f"Tabla cruzada IPM × pobreza monetaria (train sin Bogotá; κ de Cohen = {kappa:.3f}):"); display(cr); display(cr_tasa)
pd.concat({"hogares": cr, "proporcion_por_fila": cr_tasa}).to_csv(TAB / "02d_ipm_cruce.csv")
dom = pd.DataFrame({"Dominio": th.Dominio.values, "ipm": D["p_ipm_pobre"].values, "pobre": y, "punt": D["p_puntaje"].values})
por_dom = dom.groupby("Dominio").apply(lambda g: pd.Series({
    "hogares": len(g), "tasa_pobreza_monetaria": g.pobre.mean(), "tasa_pobreza_IPM": g.ipm.mean(),
    "P(pobre|IPM)": g.pobre[g.ipm == 1].mean(), "P(pobre|no IPM)": g.pobre[g.ipm == 0].mean(),
    "coincidencia": (g.ipm == g.pobre).mean(), "AUC_puntaje": M.auc_roc(g.pobre.values, g.punt.values) if g.pobre.nunique() > 1 else np.nan}))
por_dom.to_csv(TAB / "02d_ipm_por_dominio.csv")
display(por_dom.round(3))

# %%
fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
g = pd.DataFrame({"k": D.loc[m, "p_n_privaciones"].values, "y": y[m]}).groupby("k").y.agg(["mean", "size"])
ax[0].bar(g.index, g["mean"], color=viz.ROJO)
for k, (mu, n) in g.iterrows():
    ax[0].text(k, mu + .01, f"n={n:,}".replace(",", "."), ha="center", fontsize=7, color=viz.GRIS)
ax[0].set(title="Pobreza monetaria según el número de privaciones IPM", xlabel="privaciones observables (0–9)", ylabel="P(Pobre)")
b = ipm_auc[ipm_auc.variable.isin(ind + ["p_puntaje", "p_puntaje_oficial", "p_n_privaciones"])].sort_values("AUC_sinBOG")
ax[1].barh(b.variable, b.AUC_sinBOG, color=[viz.MORADO if v.startswith(("p_puntaje", "p_n_")) else viz.AZUL for v in b.variable])
ax[1].axvline(.5, color="k", lw=.7); ax[1].set_xlim(.45, max(.8, b.AUC_sinBOG.max() + .02))
ax[1].set(title="AUC de cada indicador y del puntaje (sin Bogotá)", xlabel="AUC")
pd_ = por_dom.sort_values("tasa_pobreza_monetaria")
ax[2].scatter(pd_.tasa_pobreza_IPM, pd_.tasa_pobreza_monetaria, s=pd_.hogares / 60, color=viz.VERDE, alpha=.7)
for d_, r_ in pd_.iterrows():
    ax[2].annotate(d_.title()[:12], (r_.tasa_pobreza_IPM, r_.tasa_pobreza_monetaria), fontsize=6.5, color=viz.GRIS)
ax[2].set(title="Pobreza IPM vs. monetaria por dominio", xlabel="tasa de pobreza IPM (aprox.)", ylabel="tasa de pobreza monetaria")
fig.tight_layout(); viz.guardar(fig, FIG, "02d_ipm"); plt.show()

# %% [markdown]
# ## 4. Codificación con la etiqueta: bayes empírico jerárquico beta-binomial, fuera de pliegue
#
# Para una clave $c$ (p. ej. el oficio del jefe) con padre $\pi(c)$ (su gran grupo ocupacional), con $s_c$ pobres
# entre $n_c$ hogares del conjunto de ajuste, el modelo es
# $$s_c\mid p_c\sim\text{Bin}(n_c,p_c),\qquad p_c\sim\text{Beta}\big(m\,\mu_{\pi(c)},\,m(1-\mu_{\pi(c)})\big),$$
# cuya media posterior es $\hat p_c=(s_c+m\,\mu_{\pi(c)})/(n_c+m)$: un promedio ponderado entre la tasa del grupo y la
# del padre, con peso $n_c/(n_c+m)$. La **concentración** $m$ (equivalentemente la varianza entre grupos
# $\tau^2=\mu(1-\mu)/(m+1)$) no se fija a mano: se estima por **máxima verosimilitud marginal** (bayes empírico tipo
# II), maximizando $\sum_c[\ln B(s_c+m\mu_{\pi(c)},\,n_c-s_c+m(1-\mu_{\pi(c)}))-\ln B(m\mu_{\pi(c)},\,m(1-\mu_{\pi(c)}))]$.
# El padre se encoge a su vez hacia la media global $\mu_0$ con su propia $m_1$. Es el estimador de James–Stein /
# Efron–Morris (1975) para proporciones; el suavizado de Micci-Barreca (2001) es el caso con $m$ fijo y sin jerarquía.
#
# **Fuera de pliegue.** Codificar con la etiqueta de la propia fila produce sobreajuste por construcción (fuga de
# objetivo; Kaufman et al., 2012). Por eso: (i) en las tablas guardadas, la fila del pliegue $k$ se codifica con los
# otros cuatro pliegues y el test con todo train; (ii) **para medir** (aquí y en la fase C) todo se recalcula dentro
# de cada partición con `TransformadorV7`/`codificar(train_idx, aplicar_idx)`: las filas de entrenamiento con una
# partición interna de 5 pliegues y las de validación con todo el entrenamiento, de modo que la validación nunca entra
# en su propia codificación ni en la de las filas con que se entrena el modelo (Pargent et al., 2022, muestran que
# ésta es la forma de codificación de objetivo regularizada que mejor generaliza).
#
# **Bogotá.** El test no tiene Bogotá. Las tasas por categoría que se quieren estimar son las de la población objetivo
# $Q$; incluir los 10.567 hogares de Bogotá (pobreza 9,8 % con otra línea y otro mercado laboral) en las estadísticas
# del oficio o de la posición sesga $\hat p_c$ hacia Bogotá. Excluirlos cuesta poca varianza (6,4 % de la muestra). Por
# defecto se excluyen de las **estadísticas** (las filas de Bogotá sí se codifican, para poder entrenar con ellas), y
# la decisión se valida empíricamente en §6.1.

# %%
TE_SIN_BOG = True
TEtr, TEte, HIPER = V.te_global(D, DT, y, fold, excluir_bogota=TE_SIN_BOG)
D = pd.concat([D, TEtr], axis=1); DT = pd.concat([DT, TEte], axis=1)
TE = list(TEtr.columns)
for c in TE:
    FAM[c] = "E"; DESC[c] = f"codificación bayes empírica jerárquica (beta-binomial) de {' × '.join(V.TE_ESPECIFICACIONES[c]['clave'])}"
hip = pd.DataFrame([{"particion": k, "variable": nom, **v} for k, d in HIPER.items() for nom, v in d.items()])
hip.to_csv(TAB / "02d_codificacion_hiper.csv", index=False)
display(hip[hip.particion == "todo_train"].round(4))
display(pd.DataFrame({"variable": TE, "clave": [" × ".join(V.TE_ESPECIFICACIONES[c]["clave"]) for c in TE],
                      "padre": [" × ".join(V.TE_ESPECIFICACIONES[c]["padre"]) or "—" for c in TE],
                      "niveles": [D[V.TE_ESPECIFICACIONES[c]["clave"]].astype(str).agg("|".join, axis=1).nunique() for c in TE],
                      "corr_con_Pobre_OOF": [np.corrcoef(D[c], y)[0, 1] for c in TE],
                      "AUC_sinBOG_OOF": [M.auc_roc(y[sin_bog], D[c].values[sin_bog]) for c in TE]}).round(4))

# %% [markdown]
# **Prueba formal de la propiedad fuera de pliegue (etiqueta perturbada).** Si la codificación de una fila no usa su
# propia etiqueta, cambiar las etiquetas de un pliegue no puede alterar la codificación de las filas de ese pliegue
# (sí la de los demás). Se invierten las etiquetas del pliegue 0 y se recalcula.

# %%
y_pert = y.copy(); y_pert[fold == 0] = 1 - y_pert[fold == 0]
TEp, _, _ = V.te_global(D[COLS_CONSTRUCCION], DT[COLS_CONSTRUCCION], y_pert, fold, excluir_bogota=TE_SIN_BOG)
f0 = fold == 0
cambio_propio = float(np.abs(TEp.values[f0] - TEtr.values[f0]).max())
cambio_otros = float(np.abs(TEp.values[~f0] - TEtr.values[~f0]).max())
OOF_OK = cambio_propio == 0.0 and cambio_otros > 0
print(f"Máximo cambio en las filas del pliegue perturbado: {cambio_propio:.3g} (debe ser 0); en las demás: {cambio_otros:.3g} (> 0) → "
      f"{'OK' if OOF_OK else 'FALLA'}")
assert OOF_OK

# %% [markdown]
# **Huella de la codificación cruzada.** En train cada categoría toma 5 valores (uno por pliegue) y en test un sexto
# (con todo train): los valores difieren poco pero son distintos, de modo que un árbol puede distinguir train de test por
# el valor exacto (§9 lo mide). Es el diseño estándar (el `TargetEncoder` de scikit-learn hace lo mismo) y no es una
# fuga: la magnitud relevante es cuánto se aleja la codificación fuera de pliegue de una fila de la de todo train para su
# misma categoría.

# %%
_cod_full = V.CodificadorObjetivo(D[COLS_CONSTRUCCION], y, excluir_bogota=TE_SIN_BOG)
_fit_all = np.flatnonzero(_cod_full.ok_stats)
huella = []
for c in TE:
    full = _cod_full._post(c, _fit_all, np.arange(len(D)))
    d_ = np.abs(D[c].values - full)
    huella.append({"variable": c, "media_abs_dif_OOF_vs_todo_train": float(d_.mean()), "p99_abs_dif": float(np.quantile(d_, .99)),
                   "max_abs_dif": float(d_.max()), "desv_estandar_TE": float(D[c].std()), "valores_distintos_train": int(D[c].nunique()),
                   "valores_distintos_test": int(DT[c].nunique())})
huella = pd.DataFrame(huella); huella.to_csv(TAB / "02d_codificacion_huella.csv", index=False); display(huella.round(5))

# %% [markdown]
# ### 4.1 Salida preliminar
# Para que la fase C pueda empezar, se guardan ya las tablas y un JSON preliminar (sin la familia X guiada por SHAP ni
# las mediciones). Al final del cuaderno se sobrescriben con la versión definitiva.

# %%
def guardar_tablas(D, DT, sufijo):
    S_TR = D.assign(Pobre=y, fold=fold)
    S_TE = DT.copy()
    assert not [c for c in S_TR.columns if c not in S_TE.columns and c not in ("Pobre", "fold")]
    assert not [c for c in S_TR.columns if c in V.HOG_PROHIBIDAS or c in PROH_PERSONA or c in V.OBJETIVO_DERIVADAS or c in V.CONSTANTES_EXTERNAS]
    if PRUEBA:
        S_TR.to_parquet(OUT / f"modelo_train{sufijo}.parquet", index=False); S_TE.to_parquet(OUT / f"modelo_test{sufijo}.parquet", index=False)
    else:
        pio.guardar_tabla(S_TR, R, f"modelo_train{sufijo}"); pio.guardar_tabla(S_TE, R, f"modelo_test{sufijo}")


if not PRUEBA and os.environ.get("V7_PRELIMINAR", "1") == "1":   # en re-ejecuciones (V7_PRELIMINAR=0) no se pisa el JSON definitivo
    guardar_tablas(D, DT, "_v7")
    prelim = {"version": "v7", "estado": "preliminar: faltan la familia X guiada por SHAP, las mediciones y la auditoría",
              "S1": conj5["S1"], "S2": BASE, "S3": [c for c in D.columns if c not in ("id",)], "generadas": [], "extra": [],
              "categoricas": V.CAT_TODAS, "codificacion_objetivo": {"especificaciones": V.TE_ESPECIFICACIONES, "excluir_bogota": TE_SIN_BOG},
              "especificaciones_muestra": META["especificaciones_muestra"]}
    json.dump(prelim, open(OUT / "conjuntos_variables_v7.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(f"Tablas preliminares guardadas ({(time.time() - T0) / 60:.1f} min)")

# %% [markdown]
# ## 5. Protocolo de medición
#
# * **Partición y métrica.** Los 5 pliegues de `pliegues.csv`; F1 anidado sin Bogotá (el umbral de cada pliegue se
#   elige con las predicciones fuera de pliegue de los otros cuatro, sólo fuera de Bogotá), log-loss y AUC sin Bogotá.
# * **LightGBM rápido** (el del cuaderno 09/02b): `learning_rate` = 0,1, hojas 15, `min_child_samples` 100,
#   `feature_fraction` 0,7, `bagging` 0,8, L2 = 5, parada temprana en un 10 % **interno** del pliegue de entrenamiento.
# * **Logit regularizado.** Penalización L2 fuerte (C elegido en una grilla corta, §6.4) sobre variables recortadas al
#   0,5 %–99,5 %, imputadas por la mediana con indicador de faltante y estandarizadas, B-splines cúbicos con 5 nodos en
#   cuantiles en las 5 continuas principales (las de mayor |SHAP| del LightGBM base) y one-hot de las categóricas.
#   Todo el preprocesamiento se ajusta dentro del pliegue. Los modelos lineales (y Naive Bayes) dependen mucho más de
#   la construcción de variables que los árboles; por eso se miden ambos.
# * **Todo lo que depende de la muestra se recalcula dentro de cada pliegue** (`TransformadorV7`): la codificación con
#   la etiqueta y las transformaciones de referencia (recorte, rango y decil en el dominio, posiciones relativas).
# * **Diseño de la comparación.** Para cada familia $F$: *marginal* (base + $F$ frente a base: ¿aporta sola?) y
#   *condicional* (todas frente a todas − $F$: ¿aporta dado todo lo demás?, el análogo de LOCO; Lei et al., 2018).
#   Mejora $\Delta_k$ pareada por pliegue (mismos pliegues y misma semilla), $\bar\Delta$ con
#   $\text{EE}=s_\Delta/\sqrt K$; prueba $t$ con la varianza corregida de Nadeau y Bengio (2003),
#   $\hat\sigma^2_{\text{corr}}=(1/K+n_{\text{val}}/n_{\text{ent}})\,s^2_\Delta$ con $K=5$ y $n_{\text{val}}/n_{\text{ent}}=1/4$,
#   $K-1$ grados de libertad (corrige la dependencia entre pliegues que comparten entrenamiento); IC 95 % bootstrap
#   pareado del ΔF1 global remuestreando hogares con las predicciones anidadas fijas (variabilidad de la muestra de
#   evaluación); **regla de 1 EE**: una familia aporta si mejora el F1 o el log-loss en más de 1 EE sin empeorar la
#   otra métrica más de 1 EE. **Estabilidad**: la comparación marginal se repite con 3 semillas de LightGBM (bagging,
#   submuestreo de columnas y partición interna de parada temprana) y la condicional con 2.

# %%
CATS = list(V.CAT_TODAS)
NUM_ALL = [c for c in D.columns if c not in V.NO_NUMERICAS and c not in CATS]
MINCER = list(V.CONSTANTES_EXTERNAS)[:3]
X = AJ.preparar(D, NUM_ALL + CATS, D)
for c in MINCER:                                    # sólo para medir la referencia «S2 + categóricas de v5» (con Mincer)
    X[c] = D5[c].astype("float32").values
SPECS = list(META["especificaciones_muestra"])
TRF = V.TransformadorV7(D, y, excluir_bogota=TE_SIN_BOG, grupo=SPECS, semilla=SEED)
LGB = AJ.MotorLGB(hilos=HILOS); PR = {**LGB.base, "learning_rate": 0.1}
FC = {L: [c for c in D.columns if FAM.get(c) == L and c not in BASE and c not in V.NO_NUMERICAS] for L in V.ORDEN_FAMILIAS}
print({V.FAMILIAS[L]: len(v) for L, v in FC.items()})


MEMO = {}
TRF.etiqueta = "principal"


def ev(nombre, motor, cols, params, trf=None, semilla=1, Xm=None, ym=None):
    """Evalúa un conjunto con caché en disco (reanudable) y memoria de conjuntos idénticos ya evaluados."""
    trf_ = TRF if trf is None else trf
    clave = None if ym is not None else (motor.corto, getattr(motor, "C", None), tuple(sorted(cols)), semilla, trf_.etiqueta)
    if clave in MEMO:
        r = {**MEMO[clave], "nombre": nombre}
        print(f"{motor.corto:5s} s{semilla} {nombre:52s} = {MEMO[clave]['nombre']} (mismo conjunto)", flush=True)
        return r
    slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{motor.corto}_{nombre}_s{semilla}")[:150]
    def f():
        with threadpool_limits(HILOS):
            r = V.evaluar(motor, X if Xm is None else Xm, y if ym is None else ym, folds, sin_bog, cols, params,
                          TRF if trf is None else trf, semillas=(semilla,))
        r.update(oof=r["oof"].astype("float32"), yhat=r["yhat"].astype("int8"), n_vars=len(cols), nombre=nombre,
                 motor=motor.corto, semilla=semilla, cols=list(cols))
        return r
    r = AJ.en_cache(CACHE / f"{slug}.pkl", f, REUSAR)
    if clave is not None:
        MEMO[clave] = r
    q = r["res"]
    print(f"{motor.corto:5s} s{semilla} {nombre:52s} n={len(cols):4d} F1={q['F1_anidado_sinBOG']:.4f}±{q['EE']:.4f} "
          f"LLsB={q['LogLoss_sinBOG']:.5f} AUCsB={q['AUC_sinBOG']:.4f} ({r['tiempo_s']:.0f}s, {(time.time() - T0) / 60:.0f} min)", flush=True)
    return r


FILAS = []


def fila(r, grupo, ref=None, familia=""):
    q = r["res"]
    d = {"modelo": r["motor"], "grupo": grupo, "familia": familia, "conjunto": r["nombre"], "semilla": r["semilla"],
         "n_vars": r["n_vars"], "F1_anidado_sinBOG": q["F1_anidado_sinBOG"], "EE": q["EE"], "LogLoss_sinBOG": q["LogLoss_sinBOG"],
         "LogLoss": q["LogLoss"], "AUC_sinBOG": q["AUC_sinBOG"], "Brier_sinBOG": q["Brier_sinBOG"], "tiempo_s": r["tiempo_s"]}
    if ref is not None:
        c = V.comparar(ref, r, y, sin_bog)
        d.update({"referencia": ref["nombre"], "dF1": c["dF1"], "EE_dF1": c["EE_dF1"], "t_NB_F1": c["t_NB_F1"], "p_NB_F1": c["p_NB_F1"],
                  "IC95_dF1_inf": c["IC95_dF1_boot"][0], "IC95_dF1_sup": c["IC95_dF1_boot"][1], "pliegues_mejora_F1": c["pliegues_mejora_F1"],
                  "dLogLoss_mejora": c["dLL"], "EE_dLL": c["EE_dLL"], "t_NB_LL": c["t_NB_LL"], "p_NB_LL": c["p_NB_LL"],
                  "supera_1EE": c["supera_1EE"]})
    FILAS.append(d)
    return d


def guardar_filas():
    pd.DataFrame(FILAS).to_csv(TAB / "02d_familias.csv", index=False)

# %% [markdown]
# ## 6. Medición con LightGBM
# ### 6.0 Partida
# La referencia del encargo es «sin primera etapa, S2 + categóricas» (LightGBM ajustado 0,7647 en el cuaderno 10;
# con el protocolo rápido ≈ 0,763). Ese conjunto aún tiene las 3 variables de Mincer (constantes externas): al
# quitarlas queda la **base permitida**, que es la referencia de todas las comparaciones.

# %%
S2_MINCER = BASE + MINCER
R_S2 = ev("S2 + categóricas (v5, con Mincer)", LGB, S2_MINCER, PR)
R_BASE = {s: ev("Base permitida", LGB, BASE, PR, semilla=s) for s in SEMILLAS}
fila(R_S2, "partida"); fila(R_BASE[1], "partida", R_S2, "quitar Mincer")
guardar_filas()

# %% [markdown]
# ### 6.1 ¿Bogotá en las estadísticas de la codificación?
# Base + familia E con las estadísticas calculadas **sin** Bogotá frente a **con** Bogotá (mismas semillas). Se
# conserva «sin Bogotá» (validez: la población objetivo no incluye Bogotá) salvo que «con Bogotá» mejore en más de
# 1 EE.

# %%
TRF_CB = V.TransformadorV7(D, y, excluir_bogota=False, grupo=SPECS, semilla=SEED)
TRF_CB.etiqueta = "con_bogota"
bog_filas = []
for s in SEMILLAS:
    r_sb = ev("Base + E (sin Bogotá en las estadísticas)", LGB, BASE + TE, PR, semilla=s)
    r_cb = ev("Base + E (con Bogotá en las estadísticas)", LGB, BASE + TE, PR, TRF_CB, semilla=s)
    c = V.comparar(r_sb, r_cb, y, sin_bog)
    bog_filas.append({"semilla": s, "F1_sin_Bogota": r_sb["res"]["F1_anidado_sinBOG"], "F1_con_Bogota": r_cb["res"]["F1_anidado_sinBOG"],
                      "dF1_con_menos_sin": c["dF1"], "EE": c["EE_dF1"], "dLL_mejora_con": c["dLL"], "EE_dLL": c["EE_dLL"], "con_supera_1EE": c["supera_1EE"]})
bog = pd.DataFrame(bog_filas); bog.to_csv(TAB / "02d_codificacion_bogota.csv", index=False); display(bog.round(5))
if bog.con_supera_1EE.mean() > 0.5 and bog.dF1_con_menos_sin.mean() > 0:
    TE_SIN_BOG = False
    TEtr, TEte, HIPER = V.te_global(D[COLS_CONSTRUCCION], DT[COLS_CONSTRUCCION], y, fold, excluir_bogota=False)
    for c in TE:
        D[c] = TEtr[c].values; DT[c] = TEte[c].values; X[c] = TEtr[c].values
    TRF = TRF_CB
print(f"Decisión: estadísticas de la codificación {'SIN' if TE_SIN_BOG else 'CON'} Bogotá")

# %% [markdown]
# ### 6.2 Familia X guiada por SHAP
# Un LightGBM sobre todas las variables de las familias (sin las candidatas SHAP) da la importancia E|φ_j| y un
# XGBoost sobre las 30 principales los valores de **interacción** SHAP E|φ_ij| (Lundberg et al., 2020). Con los 25
# pares más fuertes se generan productos, cocientes y diferencias de logaritmos, per cápita, posiciones relativas en
# Dominio/Depto, indicadores y un conteo de carencias (`busqueda.generar_candidatos`). Se descartan las que usan la
# codificación con la etiqueta o columnas dependientes de la muestra (no se podrían recalcular de forma consistente)
# y las transformaciones monótonas de una variable (ya están en la familia T). Filtro 1: |ρ de Spearman| con el
# residuo OOF del modelo con todas las familias; las 40 mejores entran al *pool* (familia X). Filtro 2 (informativo,
# para la fase D): selección hacia adelante por lotes de 5 con la regla de 1 EE y poda, con límite de 30 minutos.

# %%
TODAS_SIN_XS = list(dict.fromkeys(BASE + sum((FC[L] for L in V.ORDEN_FAMILIAS if L != "E"), []) + TE))
R_TSX = ev("Todas sin X-SHAP", LGB, TODAS_SIN_XS, PR)


class _Dat:
    def __init__(self):
        self.X, self.y, self.folds, self.sin_bog, self.D, self.GEN = X, y, folds, sin_bog, D, []


def _familia_xs():
    dat = _Dat()
    with threadpool_limits(HILOS):
        imp, pares, pares_todos, _, top30 = BU.importancia_e_interacciones(dat, TODAS_SIN_XS, PR, n_shap=20000, hilos=HILOS)
    specs = BU.generar_candidatos(dat, imp, pares)
    deriv = set(TE) | {s["nombre"] for s in SPECS} | set(MINCER)
    specs = [s for s in specs if not ({s.get("a"), s.get("b")} & deriv) and not s["monotona_univariada"]
             and not set(s.get("vars", [])) & deriv and s["nombre"] not in D.columns
             and not any(re.search(pt, s["nombre"].lower()) for pt in V.PATRONES_PROHIBIDOS)]
    C = V.materializar(specs, D, D)
    f1t = BU.filtro_residuo(C, y - R_TSX["oof"].astype(float), specs)
    return {"imp": imp, "pares": pares_todos, "specs": specs, "filtro": f1t}


SX = AJ.en_cache(CACHE / "familia_xs.pkl", _familia_xs, REUSAR)
SX["imp"].rename("SHAP_media_abs").to_frame().to_csv(TAB / "02d_importancia_shap.csv")
SX["pares"].to_csv(TAB / "02d_interacciones_shap.csv", index=False)
SX["filtro"].to_csv(TAB / "02d_candidatos_shap.csv", index=False)
# las 40 mejores por |ρ| con el residuo, sin duplicados exactos (p. ej. el rango de log x es el rango de x)
_spec_por_nombre = {s["nombre"]: s for s in SX["specs"]}
_hashes = {V._hash(np.r_[D[c].astype(float).values, DT[c].astype(float).values]) for c in D.columns if c not in V.NO_NUMERICAS}
SPEC_XS = []
for nom in SX["filtro"][SX["filtro"].pct_na < .5].variable:
    s = _spec_por_nombre[nom]
    v = np.r_[V.materializar([s], D, D)[nom].values, V.materializar([s], D, DT)[nom].values]
    k = V._hash(v)
    if k in _hashes or np.nanstd(v) == 0:
        continue
    _hashes.add(k); SPEC_XS.append(s)
    if len(SPEC_XS) == 40:
        break
SX["elegibles"] = [s["nombre"] for s in SPEC_XS]
Ctr, Cte = V.materializar(SPEC_XS, D, D), V.materializar(SPEC_XS, D, DT)
D = pd.concat([D, Ctr], axis=1); DT = pd.concat([DT, Cte], axis=1)
for c in Ctr.columns:
    X[c] = Ctr[c].values
for s in SPEC_XS:
    FAM[s["nombre"]] = "X"; DESC[s["nombre"]] = f"SHAP: {s.get('motivacion', s['tipo'])}"
SPEC_XS_GRUPO = [s for s in SPEC_XS if s["tipo"] in BU.GRUPALES]
SPECS = SPECS + SPEC_XS_GRUPO
TRF = V.TransformadorV7(D, y, excluir_bogota=TE_SIN_BOG, grupo=SPECS, semilla=SEED)
TRF.etiqueta = "principal" if TE_SIN_BOG else "con_bogota"   # mismos valores en las columnas previas: sólo agrega las SHAP
FC["X"] = FC["X"] + [s["nombre"] for s in SPEC_XS]
MAPA = V.mapa_derivadas(MAPA, SPEC_XS, V.TE_ESPECIFICACIONES)
assert all(c in MAPA for c in D.columns if c != "id")
print(f"{len(SX['specs'])} candidatas permitidas; {len(SPEC_XS)} al pool (familia X)")
display(SX["filtro"].head(40)[["variable", "tipo", "origen", "rho_residuo", "pct_na"]].round(4))
display(SX["imp"].head(30).rename("E|SHAP|").to_frame().round(4))

# %%
def _adelante():
    dat = _Dat(); dat.X = X
    with threadpool_limits(HILOS):
        return BU.adelante_por_lotes(LGB, dat, TODAS_SIN_XS, [s["nombre"] for s in SPEC_XS], PR, lote=5,
                                     limite_s=float(os.environ.get("V7_LIMITE_ADELANTE_MIN", "30")) * 60, transformar=TRF)


GAN_XS, camino_xs, _, _ = AJ.en_cache(CACHE / "familia_xs_adelante.pkl", _adelante, REUSAR)
camino_xs.to_csv(TAB / "02d_seleccion_shap.csv", index=False)
display(camino_xs.drop(columns="variables").round(5))
print("Candidatas SHAP aceptadas por la selección hacia adelante (informativo para la fase D):", GAN_XS)

# %% [markdown]
# ### 6.3 Aporte marginal y condicional de cada familia (LightGBM)

# %%
TODAS = list(dict.fromkeys(BASE + sum((FC[L] for L in V.ORDEN_FAMILIAS), [])))
for s in SEMILLAS:
    for L in V.ORDEN_FAMILIAS:
        r = ev(f"Base + {L}", LGB, BASE + FC[L], PR, semilla=s)
        fila(r, "marginal", R_BASE[s], L)
    guardar_filas()
R_TODAS = {}
for s in SEMILLAS_LOO:
    R_TODAS[s] = ev("Todas las permitidas", LGB, TODAS, PR, semilla=s)
    fila(R_TODAS[s], "todas", R_BASE[s], "todas")
    for L in V.ORDEN_FAMILIAS:
        r = ev(f"Todas − {L}", LGB, [c for c in TODAS if c not in set(FC[L])], PR, semilla=s)
        f_ = fila(R_TODAS[s], "condicional", r, L)
        f_["conjunto"] = f"Todas frente a todas − {L}"
    guardar_filas()

# %% [markdown]
# ### 6.4 Camino acumulado (orden del encargo) y conjuntos de referencia

# %%
acum, prev = list(BASE), R_BASE[1]
for L in V.ORDEN_FAMILIAS:
    acum = acum + FC[L]
    r = ev(f"Acumulado hasta {L}", LGB, list(acum), PR)
    fila(r, "acumulado", prev, L); prev = r
CORTA = [c for c in conj5["encuesta_corta"] if c in D.columns]
# encuesta corta v7: la de v5 + el umbral observable (sólo agrega P5100 al cuestionario) + formalidad y auxilio de transporte
CORTA_V7 = list(dict.fromkeys(CORTA + ["log_ingreso_requerido", "no_puede_ser_pobre", "u_n_lineas", "requerido_sobre_linea",
                                       "share_arriendo_imputado", "formales_pc", "i_aux_transporte_por_asal"]))
print(f"Encuesta corta v5: {len(CORTA)} variables, {V.costo_preguntas(MAPA, CORTA)[0]} preguntas; "
      f"v7: {len(CORTA_V7)} variables, {V.costo_preguntas(MAPA, CORTA_V7)[0]} preguntas")
IPM_SET = [c for c in D.columns if FAM.get(c) == "P"] + ["hacinamiento_critico"]
IPM_UMBRAL = list(dict.fromkeys(IPM_SET + ["log_ingreso_requerido", "no_puede_ser_pobre", "u_n_lineas", "Dominio", "cabecera", "Nper"]))
REF_SETS = {"Encuesta corta v5 (18)": CORTA, "Encuesta corta v7": CORTA_V7, "Bloque IPM solo": IPM_SET,
            "IPM + umbral + territorio": IPM_UMBRAL, "S1 de v5": conj5["S1"]}
R_REF = {}
for nom, cols in REF_SETS.items():
    R_REF[nom] = ev(nom, LGB, cols, PR)
    fila(R_REF[nom], "referencia", None, nom)
fila(ev("Encuesta corta v7", LGB, CORTA_V7, PR), "referencia_vs_corta", R_REF["Encuesta corta v5 (18)"], "corta v7 vs v5")
fila(ev("IPM + umbral + territorio", LGB, IPM_UMBRAL, PR), "referencia_vs_ipm", R_REF["Bloque IPM solo"], "umbral sobre IPM")
guardar_filas()

# %% [markdown]
# ## 7. Medición con el Logit regularizado
# ### 7.1 Splines y fuerza de la penalización
# Las 5 continuas con splines son las de mayor E|SHAP| de un LightGBM sobre la base. C (inverso de la penalización
# L2 sobre variables estandarizadas) se elige en {0,003; 0,01; 0,03; 0,1} por log-loss sin Bogotá en la base y en
# todas; se usa el C óptimo de «todas» (el conjunto con más riesgo de sobreajuste) en todas las comparaciones y se
# reporta la sensibilidad.

# %%
def _shap_base():
    from pobreza.modelos_ml import _split_es
    fit_m, es_m = _split_es(y, 5)
    with threadpool_limits(HILOS):
        mod, n = LGB.ajustar(X.loc[fit_m, BASE], y[fit_m], X.loc[es_m, BASE], y[es_m], PR, semilla=1)
        idx = np.random.default_rng(1).choice(len(y), min(20000, len(y)), replace=False)
        phi = LGB.shap(mod, X.iloc[idx][BASE], n)
    return pd.Series(np.abs(phi).mean(0), index=BASE).sort_values(ascending=False)


IMP_BASE = AJ.en_cache(CACHE / "shap_base.pkl", _shap_base, REUSAR)
SPLINES = [c for c in IMP_BASE.index if c not in CATS and D[c].nunique() > 50][:5]
print("Splines en:", SPLINES)
grid = []
R_LOG_GRID = {}
for C_ in (0.001, 0.003, 0.01, 0.03, 0.1):
    lg = V.MotorLogitV7(C=C_, splines=SPLINES)
    for nom, cols in (("Base permitida", BASE), ("Todas las permitidas", TODAS)):
        r = ev(f"{nom} (C={C_})", lg, cols, {})
        R_LOG_GRID[(nom, C_)] = r
        grid.append({"conjunto": nom, "C": C_, "F1_anidado_sinBOG": r["res"]["F1_anidado_sinBOG"], "EE": r["res"]["EE"],
                     "LogLoss_sinBOG": r["res"]["LogLoss_sinBOG"], "AUC_sinBOG": r["res"]["AUC_sinBOG"]})
grid = pd.DataFrame(grid); grid.to_csv(TAB / "02d_logit_grilla_C.csv", index=False); display(grid.round(5))
C_OPT = float(grid[grid.conjunto == "Todas las permitidas"].sort_values("LogLoss_sinBOG").C.iloc[0])
C_BASE = float(grid[grid.conjunto == "Base permitida"].sort_values("LogLoss_sinBOG").C.iloc[0])
print(f"C elegido (todas): {C_OPT}; óptimo en la base: {C_BASE}")
LOG = V.MotorLogitV7(C=C_OPT, splines=SPLINES)

# %% [markdown]
# ### 7.2 Aporte marginal y condicional de cada familia (Logit)
# El Logit es determinista dada la partición (la única aleatoriedad es la partición interna de la codificación, fija),
# así que se mide con una semilla.

# %%
RL_BASE = R_LOG_GRID[("Base permitida", C_OPT)]
RL_TODAS = R_LOG_GRID[("Todas las permitidas", C_OPT)]
fila(ev("S2 + categóricas (v5, con Mincer)", LOG, S2_MINCER, {}), "partida")
fila(RL_BASE, "partida")
fila(RL_TODAS, "todas", RL_BASE, "todas")
for L in V.ORDEN_FAMILIAS:
    fila(ev(f"Base + {L}", LOG, BASE + FC[L], {}), "marginal", RL_BASE, L)
    guardar_filas()
for L in V.ORDEN_FAMILIAS:
    r = ev(f"Todas − {L}", LOG, [c for c in TODAS if c not in set(FC[L])], {})
    f_ = fila(RL_TODAS, "condicional", r, L); f_["conjunto"] = f"Todas frente a todas − {L}"
    guardar_filas()
for nom, cols in REF_SETS.items():
    fila(ev(nom, LOG, cols, {}), "referencia", None, nom)
guardar_filas()

# %% [markdown]
# ## 8. Resultados por familia
# La tabla `02_outputs/tables/02d_familias.csv` trae todas las filas (modelo, diseño, semilla, F1 ± EE, log-loss, AUC,
# ΔF1 con EE, prueba de Nadeau–Bengio, IC bootstrap y regla de 1 EE). Abajo, el resumen: para LightGBM se promedian
# las semillas (y se reporta la desviación entre semillas como medida de estabilidad).

# %%
TF = pd.DataFrame(FILAS)
TF.to_csv(TAB / "02d_familias.csv", index=False)
comp = TF[TF.grupo.isin(["marginal", "condicional"])].copy()
res = (comp.groupby(["modelo", "grupo", "familia"])
       .agg(dF1=("dF1", "mean"), EE_dF1=("EE_dF1", "mean"), sd_dF1_semillas=("dF1", "std"), dLL=("dLogLoss_mejora", "mean"),
            EE_dLL=("EE_dLL", "mean"), p_NB_F1=("p_NB_F1", "median"), p_NB_LL=("p_NB_LL", "median"),
            semillas=("semilla", "nunique"), semillas_1EE=("supera_1EE", "sum"), semillas_dF1_pos=("dF1", lambda s: int((s > 0).sum())))
       .reset_index())
res["nombre_familia"] = res.familia.map(V.FAMILIAS)
res["aporta"] = res.semillas_1EE >= np.ceil(res.semillas / 2 + 1e-9)
res.to_csv(TAB / "02d_familias_resumen.csv", index=False)
est = comp[(comp.modelo == "LGBM") & (comp.grupo == "marginal")].pivot_table(index="familia", columns="semilla", values="dF1")
est["media"] = est.mean(1); est["sd_entre_semillas"] = est.iloc[:, :len(SEMILLAS)].std(1)
est.to_csv(TAB / "02d_estabilidad_semillas.csv")
display(res.round(5))
print("Estabilidad del ΔF1 marginal (LightGBM) entre semillas:"); display(est.round(5))
lin = ["**Resumen por familia (cifras de esta corrida; ΔF1 en puntos de F1, ± 1 EE entre pliegues; p de Nadeau–Bengio).**"]
for mod, nm in (("LGBM", "LightGBM"), ("LOGIT", "Logit")):
    for gr in ("marginal", "condicional"):
        s = res[(res.modelo == mod) & (res.grupo == gr)].set_index("familia").reindex(V.ORDEN_FAMILIAS)
        partes = [f"{L} {r.dF1:+.4f}±{r.EE_dF1:.4f} (p={r.p_NB_F1:.2f}{', aporta' if r.aporta else ''})" for L, r in s.iterrows() if pd.notna(r.dF1)]
        lin.append(f"*{nm}, {gr}*: " + "; ".join(partes) + ".")
display(Markdown("\n\n".join(lin)))

# %% [markdown]
# ### 8.0 Costo en preguntas de cada conjunto
# $c(S)$ = número de preguntas reales del cuestionario (costo 1) que exige el conjunto; las líneas, el territorio y los
# factores de expansión cuestan 0.

# %%
CONJ_COSTO = {"Base permitida": BASE, "Todas las permitidas": TODAS, **{f"Base + {L}": BASE + FC[L] for L in V.ORDEN_FAMILIAS},
              **{f"Sólo familia {L}": FC[L] for L in V.ORDEN_FAMILIAS}, **REF_SETS}
cst = []
for nom, cols in CONJ_COSTO.items():
    k, qs = V.costo_preguntas(MAPA, cols)
    cst.append({"conjunto": nom, "n_vars": len(cols), "c_S_preguntas": k, "preguntas": " ".join(q for q in qs if V.PREGUNTAS.get(q, ("", "", 1))[2] == 1),
                "datos_costo_cero": " ".join(q for q in qs if V.PREGUNTAS.get(q, ("", "", 1))[2] == 0)})
cst = pd.DataFrame(cst); cst.to_csv(TAB / "02d_costo_preguntas.csv", index=False)
display(cst.drop(columns=["preguntas"]))

# %%
fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=True)
orden = V.ORDEN_FAMILIAS
for i, mod in enumerate(["LGBM", "LOGIT"]):
    for j, gr in enumerate(["marginal", "condicional"]):
        ax = axes[i, j]
        s = res[(res.modelo == mod) & (res.grupo == gr)].set_index("familia").reindex(orden)
        yy = np.arange(len(orden))
        col = [viz.VERDE if a else viz.GRIS for a in s.aporta.fillna(False)]
        ax.barh(yy - .2, s.dF1, xerr=s.EE_dF1, height=.4, color=col, capsize=2, label="ΔF1 anidado (± 1 EE)")
        ax.barh(yy + .2, s.dLL * 10, xerr=s.EE_dLL * 10, height=.4, color=viz.MORADO, alpha=.7, capsize=2, label="−Δ log-loss × 10 (± 1 EE)")
        pts = comp[(comp.modelo == mod) & (comp.grupo == gr)]
        for k, L in enumerate(orden):
            v = pts[pts.familia == L].dF1.values
            ax.scatter(v, np.full(len(v), k - .2), color="k", s=10, zorder=3)
        ax.set_yticks(yy); ax.set_yticklabels([V.FAMILIAS[L][3:] for L in orden], fontsize=8); ax.invert_yaxis()
        ax.axvline(0, color="k", lw=.7)
        ax.set(title=f"{'LightGBM rápido' if mod == 'LGBM' else 'Logit L2 + splines'}: aporte {gr} "
                     f"({'base + F vs base' if gr == 'marginal' else 'todas vs todas − F'})", xlabel="mejora (positivo = mejor)")
        if i == 0 and j == 0:
            ax.legend(fontsize=8, loc="lower right")
fig.suptitle("Aporte de cada familia de variables v7 (verde: supera la regla de 1 EE; puntos: semillas)", fontsize=12, fontweight="bold")
fig.tight_layout(); viz.guardar(fig, FIG, "02d_familias"); plt.show()

# %%
acu = TF[(TF.grupo == "acumulado")]
fig, ax = plt.subplots(figsize=(11, 4.5))
xx = np.arange(len(acu) + 1)
f1s = [R_BASE[1]["res"]["F1_anidado_sinBOG"]] + acu.F1_anidado_sinBOG.tolist()
ees = [R_BASE[1]["res"]["EE"]] + acu.EE.tolist()
ax.errorbar(xx, f1s, yerr=ees, marker="o", color=viz.AZUL, capsize=3)
ax.set_xticks(xx); ax.set_xticklabels(["base"] + [f"+{L}" for L in V.ORDEN_FAMILIAS])
ax.axhline(R_S2["res"]["F1_anidado_sinBOG"], color=viz.GRIS, ls=":", label="S2 + categóricas v5 (con Mincer)")
ax.set(title="Camino acumulado (LightGBM rápido, semilla 1)", ylabel="F1 anidado sin Bogotá (± 1 EE)"); ax.legend(fontsize=8)
fig.tight_layout(); viz.guardar(fig, FIG, "02d_camino_acumulado"); plt.show()

# %% [markdown]
# ### 8.1 Ganancia condicional del bloque IPM
# El IPM resume privaciones que ya están, más finas, en otras familias (educación, informalidad, hacinamiento). Su
# valor predictivo se lee en tres números: su AUC aislado (§3), el aporte **marginal** sobre la base y el aporte
# **condicional** dado todo lo demás.

# %%
ipm_res = res[res.familia == "P"][["modelo", "grupo", "dF1", "EE_dF1", "dLL", "EE_dLL", "p_NB_F1", "p_NB_LL", "semillas_1EE", "semillas", "aporta"]]
display(ipm_res.round(5))
ia = ipm_auc.set_index("variable")
g_ = lambda mod, gr, k: float(ipm_res[(ipm_res.modelo == mod) & (ipm_res.grupo == gr)][k].iloc[0])
display(Markdown(
    f"**IPM y pobreza monetaria.** En train sin Bogotá, {ipm_m.mean():.1%} de los hogares son pobres IPM (aproximado) y {y[m].mean():.1%} "
    f"pobres monetarios; P(pobre monetario | pobre IPM) = {cr_tasa.loc['pobre IPM', 'pobre monetario']:.1%} frente a "
    f"{cr_tasa.loc['no pobre IPM', 'pobre monetario']:.1%} entre los no pobres IPM; κ de Cohen = {kappa:.3f} (concordancia baja: "
    f"son conceptos distintos, Bourguignon y Chakravarty, 2003). El puntaje tiene AUC = {ia.loc['p_puntaje', 'AUC_sinBOG']:.3f} y el número de "
    f"privaciones {ia.loc['p_n_privaciones', 'AUC_sinBOG']:.3f}; el indicador más discriminante es "
    f"{ia.loc[ind, 'AUC_sinBOG'].idxmax()} (AUC = {ia.loc[ind, 'AUC_sinBOG'].max():.3f}). Con LightGBM, el bloque solo da F1 = "
    f"{R_REF['Bloque IPM solo']['res']['F1_anidado_sinBOG']:.4f}; su aporte marginal sobre la base es ΔF1 = {g_('LGBM', 'marginal', 'dF1'):+.4f} ± "
    f"{g_('LGBM', 'marginal', 'EE_dF1'):.4f} y el condicional {g_('LGBM', 'condicional', 'dF1'):+.4f} ± {g_('LGBM', 'condicional', 'EE_dF1'):.4f}; "
    f"con el Logit, {g_('LOGIT', 'marginal', 'dF1'):+.4f} ± {g_('LOGIT', 'marginal', 'EE_dF1'):.4f} y {g_('LOGIT', 'condicional', 'dF1'):+.4f} ± "
    f"{g_('LOGIT', 'condicional', 'EE_dF1'):.4f}."))

# %% [markdown]
# ## 9. Auditoría formal de fugas
#
# 1. **Medibilidad (prueba de aleatorización).** Sea $\mathcal C$ el conjunto de columnas comunes. Si cada variable
#    construida es función sólo de $\mathcal C$, entonces es invariante a cualquier permutación de las columnas no
#    comunes de train (ingresos, montos de persona, `Estrato1`, `Cclasnr*` y la propia etiqueta). Se reconstruye v7
#    con todas ellas permutadas al azar y se exige identidad exacta, columna por columna.
# 2. **Correlación con lo prohibido.** Máxima |correlación| de Pearson y de Spearman (pares completos, al menos 1.000
#    hogares) de cada columna con: las columnas de ingreso del hogar y sus transformaciones (`Ingtotug`, `Ingtotugarr`,
#    `Ingpcug`, `Indigente`, `Npobres`, `Nindigentes`, log y razones con la línea), **cada** columna de persona sólo de
#    train que es un monto de ingreso, agregada por hogar (suma, máximo y valor del jefe; «no se preguntó» = 0) y el
#    estrato (`Estrato1`). Falla si supera 0,95, salvo **compuerta común**: el monto prohibido $b$ sólo es positivo
#    cuando la variable permitida $a$ lo es ($P(a\neq 0\mid b\neq 0)\ge 0{,}99$) y dentro del soporte común
#    $|\rho(a,b)|<0{,}95$: $a$ reproduce el margen extensivo (la pregunta sí/no que filtra el monto, permitida por la
#    regla 2), no el valor. Las preguntas de **seguimiento** sólo de train que no son montos («¿incluyó este valor…?»,
#    meses a que corresponde lo recibido, motivos, banderas de imputación) se reportan aparte: son casi funciones de
#    preguntas permitidas (p. ej. P6760 se hace a los independientes y vale 1 casi siempre), y la dependencia va de lo
#    permitido a ellas, no al revés. |corr con `Pobre`| > 0,9 se trataría como fuga.
# 3. **Nombre y mapa.** Ningún nombre con patrón de variable generada o de ingreso, y las preguntas fuente de cada
#    variable (§1.1) son todas columnas comunes.
# 4. **Etiqueta nula.** Con la etiqueta permutada al azar, el protocolo completo (codificación dentro del pliegue
#    incluida) debe dar AUC ≈ 0,5: un AUC significativamente mayor delataría información de la etiqueta de
#    validación filtrada a sus propias variables (Ojala y Garriga, 2010).
# 5. **Etiqueta perturbada** (ya en §4): la codificación de una fila no cambia al invertir su propia etiqueta.
# 6. **Distribución train sin Bogotá vs test.** KS de dos muestras por variable (el test es una muestra aleatoria de la
#    misma población; con ~1.150 pruebas se usa Bonferroni) y faltantes en cada conjunto; y **validación adversarial**
#    multivariada con LightGBM (AUC ≈ 0,5 si las variables se construyen igual en ambos conjuntos).

# %%
del TRF_CB
t = time.time()
inv_ok, inv_dif, inv_info = V.invariancia_v7(th, tp, teh, tep, D=D[COLS_CONSTRUCCION])
print(f"(1) Invariancia a permutar {len(inv_info['hogar_permutadas'])} columnas no comunes de hogar y {inv_info['persona_permutadas']} de persona: "
      f"{'OK' if inv_ok else 'FALLA ' + str(inv_dif[:20])} ({time.time() - t:.0f} s)")
PRO, TIPOS_PRO = V.matriz_prohibida_v7(th, tp, tep)
print("Referencia prohibida por tipo:", pd.Series(TIPOS_PRO).value_counts().to_dict())
COLS_AUD = [c for c in D.columns if c not in V.NO_NUMERICAS]
aud = V.auditar_v7(D, COLS_AUD, PRO, TIPOS_PRO, y, FAM, set(TODAS), mapa=MAPA, umbral=0.95, umbral_y=0.9)
ks = V.ks_train_test(D, DT, COLS_AUD, sin_bog)
aud = aud.merge(ks, on="variable", how="left")
aud["invariante_a_columnas_no_comunes"] = [c not in set(inv_dif) for c in aud.variable]
aud["depende_de_la_etiqueta"] = aud.variable.isin(TE)
aud["dependiente_de_muestra_X"] = aud.variable.isin({s["nombre"] for s in SPECS})
aud.loc[~aud.invariante_a_columnas_no_comunes & ~aud.depende_de_la_etiqueta, "estado"] = "FALLA"
alfa_bonf = 0.05 / len(aud)
aud["KS_rechaza_Bonferroni"] = aud.p_KS < alfa_bonf
aud["descripcion"] = aud.variable.map(DESC)
aud = aud.sort_values("max_abs_spearman_prohibida", ascending=False).reset_index(drop=True)
aud.to_csv(TAB / "02d_auditoria.csv", index=False)
print(f"(2–3) Prohibidas de referencia: {PRO.shape[1]} columnas | auditadas: {len(aud)} | estados: {aud.estado.value_counts().to_dict()}")
print(f"      máx. |Pearson| con prohibidas {aud.max_abs_pearson_prohibida.max():.4f}; máx. |Spearman| {aud.max_abs_spearman_prohibida.max():.4f}; "
      f"máx. |corr con Pobre| {aud.corr_pearson_con_Pobre.abs().max():.4f}; todas las variables con preguntas fuente comunes: {bool(aud.preguntas_comunes.all())}")
print(f"      (informativo) máx. |Spearman| con preguntas de seguimiento sólo de train: {aud.max_abs_spearman_seguimiento.max():.4f} "
      f"({aud.loc[aud.max_abs_spearman_seguimiento.idxmax(), 'variable']} ~ {aud.loc[aud.max_abs_spearman_seguimiento.idxmax(), 'seguimiento_spearman']})")
print(f"(6)   KS: máx. D = {aud.KS_trainSB_vs_test.max():.4f}; variables que rechazan igualdad con Bonferroni (α = {alfa_bonf:.1e}): "
      f"{int(aud.KS_rechaza_Bonferroni.sum())}; máx. diferencia de faltantes = {aud.dif_na.abs().max():.4f}")
display(aud.head(20)[["variable", "familia", "max_abs_pearson_prohibida", "prohibida_pearson", "max_abs_spearman_prohibida", "prohibida_spearman",
                      "contencion_soporte", "max_abs_corr_en_soporte", "corr_pearson_con_Pobre", "n_preguntas_costo", "estado"]].round(4))
display(aud.sort_values("max_abs_spearman_seguimiento", ascending=False).head(8)[
    ["variable", "max_abs_spearman_seguimiento", "seguimiento_spearman", "max_abs_pearson_seguimiento", "seguimiento_pearson"]].round(4))
display(aud.sort_values("KS_trainSB_vs_test", ascending=False).head(12)[["variable", "familia", "KS_trainSB_vs_test", "p_KS", "pct_na_train", "pct_na_test"]].round(4))
FALLAS = aud[aud.estado == "FALLA"]
assert inv_ok, inv_dif
assert FALLAS.empty, FALLAS[["variable", "max_abs_pearson_prohibida", "prohibida_pearson", "max_abs_spearman_prohibida", "prohibida_spearman",
                             "contencion_soporte", "max_abs_corr_en_soporte", "corr_pearson_con_Pobre", "preguntas_comunes"]].to_string()

# %%
# (4) Etiqueta nula: protocolo completo con la etiqueta permutada
rng = np.random.default_rng(SEED)
y_nula = rng.permutation(y)
TRF_NULA = V.TransformadorV7(D, y_nula, excluir_bogota=TE_SIN_BOG, grupo=SPECS, semilla=SEED)
TRF_NULA.etiqueta = "nula"
r_nula = ev("Etiqueta nula (todas)", LGB, TODAS, PR, TRF_NULA, ym=y_nula)
n1 = int(y_nula[sin_bog].sum()); n0 = int(sin_bog.sum() - n1)
se_auc = np.sqrt((n0 + n1 + 1) / (12 * n0 * n1))
z_nula = (r_nula["res"]["AUC_sinBOG"] - 0.5) / se_auc
from scipy.stats import norm as _norm
oof_n = r_nula["oof"].astype(float)
auc_k_nula = np.array([M.auc_roc(y_nula[va][sin_bog[va]], oof_n[va][sin_bog[va]]) for _, va in folds])
prev_va_nula = np.array([y_nula[va][sin_bog[va]].mean() for _, va in folds])
pred_va_nula = np.array([oof_n[va][sin_bog[va]].mean() for _, va in folds])
p_fuga_nula = float(1 - _norm.cdf(z_nula))                     # prueba unilateral de fuga: H1 AUC > 0,5
print(f"(4) Etiqueta nula: AUC sin Bogotá = {r_nula['res']['AUC_sinBOG']:.4f} (EE bajo H0 {se_auc:.4f}, z = {z_nula:.2f}; p unilateral de fuga "
      f"[H1: AUC > 0,5] = {p_fuga_nula:.3f}); AUC por pliegue = {np.round(auc_k_nula, 4)} (media {auc_k_nula.mean():.4f} ± "
      f"{auc_k_nula.std(ddof=1) / np.sqrt(len(auc_k_nula)):.4f}); corr(predicción media del pliegue, prevalencia de validación) = "
      f"{np.corrcoef(pred_va_nula, prev_va_nula)[0, 1]:.2f}; F1 = {r_nula['res']['F1_anidado_sinBOG']:.4f} "
      f"(F1 de predecir todos pobres ≈ {2 * y_nula[sin_bog].mean() / (1 + y_nula[sin_bog].mean()):.4f})")
del TRF_NULA

# %%
# (6b) Validación adversarial: train sin Bogotá (0) vs test (1) con todas las variables v7
from sklearn.model_selection import StratifiedKFold
def _te_test_cruzado(semilla=SEED):
    """Placebo: codifica el test EXACTAMENTE como train (cada fila de test a un pliegue al azar y estadísticas de los
    otros cuatro). Si la validación adversarial vuelve a ≈ 0,5, el AUC = 1 con la codificación guardada es la huella del
    sexto valor (todo train), no una diferencia de distribución."""
    cod = V.CodificadorObjetivo(D[COLS_CONSTRUCCION], y, DT=DT[COLS_CONSTRUCCION], excluir_bogota=TE_SIN_BOG)
    ft = np.random.default_rng(semilla).integers(0, 5, len(DT))
    out = {c: np.zeros(len(DT), "float32") for c in TE}
    for k in range(5):
        fit = np.flatnonzero((fold != k) & cod.ok_stats)
        idx = np.flatnonzero(ft == k)
        for c in TE:
            out[c][idx] = cod._post(c, fit, len(D) + idx)
    return pd.DataFrame(out, index=DT.index)


def _adversarial(modo):
    cols = [c for c in TODAS if modo != "sin_TE" or c not in TE]
    DTm = DT
    if modo == "con_TE_test_cruzado":
        DTm = DT.copy(); TTc = _te_test_cruzado()
        for c in TE:
            DTm[c] = TTc[c].values
    XT = AJ.preparar(DTm, cols, D)
    Xa = pd.concat([X.loc[sin_bog, cols].reset_index(drop=True), XT[cols]], ignore_index=True)
    ya = np.r_[np.zeros(int(sin_bog.sum()), int), np.ones(len(XT), int)]
    fa = [(a, b) for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(ya)), ya)]
    with threadpool_limits(HILOS):
        r = AJ.cv(LGB, Xa, ya, fa, PR, AJ.Conjunto("adv", cols))
    imp_ = None
    return {"AUC": M.auc_roc(ya, r["oof"]), "n_train": int(sin_bog.sum()), "n_test": len(XT)}


ADV = {nom: AJ.en_cache(CACHE / f"adversarial_{nom}.pkl", lambda: _adversarial(nom), REUSAR) for nom in ("sin_TE", "con_TE", "con_TE_test_cruzado")}
for nom, v in ADV.items():
    n0_, n1_ = v["n_train"], v["n_test"]
    se_ = np.sqrt((n0_ + n1_ + 1) / (12 * n0_ * n1_))
    v["z"] = (v["AUC"] - .5) / se_
    print(f"(6b) Validación adversarial {nom}: AUC = {v['AUC']:.4f} (EE bajo H0 {se_:.4f}, z = {v['z']:.2f})")

# %%
compuerta = aud[aud.estado == "OK (compuerta común)"]
top_ks = aud.sort_values("KS_trainSB_vs_test", ascending=False).head(5)
txt = ["**Lectura de la auditoría (cifras de esta corrida).**",
       f"(i) *Aleatorización*: {'ninguna columna construida cambia' if inv_ok else f'{len(inv_dif)} columnas cambian (FALLA)'} al permutar las "
       f"{len(inv_info['hogar_permutadas'])} columnas no comunes de hogar y las {inv_info['persona_permutadas']} de persona (incluida la etiqueta).",
       f"(ii) *Correlación con lo prohibido*: máximo |Pearson| = {aud.max_abs_pearson_prohibida.max():.3f} y |Spearman| = "
       f"{aud.max_abs_spearman_prohibida.max():.3f} frente a {PRO.shape[1]} columnas prohibidas (umbral 0,95). "
       + (f"{len(compuerta)} columnas lo superan sólo por **compuerta común** (el monto es positivo sólo cuando la variable permitida lo es en "
          f"≥ {compuerta.contencion_soporte.min():.1%} de los casos y |ρ| dentro del soporte ≤ {compuerta.max_abs_corr_en_soporte.max():.3f}): son "
          f"conteos de respuestas «sí» frente a la suma del monto que esa misma pregunta filtra ({', '.join(compuerta.variable.head(6))}…): "
          f"comparten el margen extensivo, no el valor."
          if len(compuerta) else "Ninguna columna supera el umbral.")
       + f" Con las preguntas de seguimiento sólo de train (no montos) el máximo |Spearman| es {aud.max_abs_spearman_seguimiento.max():.3f} "
         f"({aud.loc[aud.max_abs_spearman_seguimiento.idxmax(), 'variable']} ~ {aud.loc[aud.max_abs_spearman_seguimiento.idxmax(), 'seguimiento_spearman']}): "
         f"esas columnas son casi funciones de preguntas permitidas. Todas las variables tienen preguntas fuente comunes: {bool(aud.preguntas_comunes.all())}.",
       f"(iii) *Etiqueta*: máximo |corr con Pobre| = {aud.corr_pearson_con_Pobre.abs().max():.3f} (una fuga daría > 0,9); con la etiqueta "
       f"permutada el protocolo completo da AUC = {r_nula['res']['AUC_sinBOG']:.4f} (z = {z_nula:.2f}; por pliegue {auc_k_nula.mean():.4f} ± "
       f"{auc_k_nula.std(ddof=1) / np.sqrt(len(auc_k_nula)):.4f}). Una fuga empujaría el AUC **por encima** de 0,5 (p unilateral = "
       f"{p_fuga_nula:.3f}); un valor algo **menor** que 0,5 es el sesgo conocido de la validación cruzada bajo la nula por permutación: con el "
       f"número de pobres fijo, las sumas de etiquetas de entrenamiento y validación en cualquier región que el modelo aprende están "
       f"correlacionadas negativamente, y lo que el modelo aprende del ruido de entrenamiento predice al revés en validación (Jamalabadi et al., "
       f"2016; para el AUC agrupado, Airola et al., 2011: aquí la predicción media de cada pliegue tiene correlación "
       f"{np.corrcoef(pred_va_nula, prev_va_nula)[0, 1]:.2f} con la prevalencia de su validación). La codificación de una fila no cambia al "
       f"invertir su propia etiqueta ({'OK' if OOF_OK else 'FALLA'}).",
       f"(iv) *Train sin Bogotá vs test*: KS máximo = {aud.KS_trainSB_vs_test.max():.4f} ({', '.join(top_ks.variable)}); "
       f"{int(aud.KS_rechaza_Bonferroni.sum())} de {len(aud)} variables rechazan la igualdad con Bonferroni; la validación adversarial multivariada da "
       f"AUC = {ADV['sin_TE']['AUC']:.4f} (z = {ADV['sin_TE']['z']:.2f}) sin la codificación con la etiqueta y {ADV['con_TE']['AUC']:.4f} "
       f"con la codificación guardada (test con todo train). Placebo: si el test se codifica **como train** (pliegue al azar, estadísticas de los "
       f"otros cuatro), el AUC es {ADV['con_TE_test_cruzado']['AUC']:.4f} (z = {ADV['con_TE_test_cruzado']['z']:.2f}; p bilateral con Bonferroni "
       f"por las 3 pruebas adversariales = {min(1.0, 3 * 2 * (1 - _norm.cdf(abs(ADV['con_TE_test_cruzado']['z'])))):.2f}; tamaño de efecto "
       f"despreciable). Luego la separación con la "
       f"codificación guardada es la **huella** del esquema cruzado (5 valores por categoría en train y un sexto en test; diferencia media con la "
       f"codificación de todo train = {huella.media_abs_dif_OOF_vs_todo_train.max():.4f}, §4): el árbol reconoce el valor exacto, no una "
       f"diferencia de distribución. Es el diseño estándar (`TargetEncoder` de scikit-learn); se documenta como límite de la validación "
       f"adversarial con codificaciones cruzadas, y la fase C recalcula la codificación dentro de su propio esquema con `codificar`."]
display(Markdown("\n\n".join(txt)))

# %% [markdown]
# ## 10. Salidas
# * `00_data/processed/modelo_train_v7.parquet` / `modelo_test_v7.parquet`: todas las columnas v7 (codificación con la
#   etiqueta: train fuera de pliegue con `pliegues.csv`, test con todo train; columnas dependientes de la muestra con
#   las X de todo train), `Pobre` y `fold`. **No** contienen ingresos, `z`, `brecha`, generadas ni Mincer.
# * `00_data/processed/conjuntos_variables_v7.json`: `todas_permitidas`, base, familias, base + familia, aceptadas por
#   la regla de 1 EE (LightGBM y Logit), encuesta corta, IPM, categóricas, columnas dependientes de la etiqueta y de
#   la muestra con sus especificaciones (para recalcularlas dentro de cada pliegue con `variables_v7.activar` o
#   `variables_v7.codificar(train_idx, aplicar_idx)`), descripciones y la **lista explícita de prohibidas**.
# * `00_data/processed/v7_LISTO.json` con las cifras; tablas `02_outputs/tables/02d_*.csv` y figuras `02d_*.png`.

# %%
if os.environ.get("V7_REESCRIBIR_TABLAS", "1") == "1":            # la construcción es determinista: re-ejecutar no cambia las tablas
    guardar_tablas(D, DT, "_v7")
PROHIBIDAS = {
    "etiqueta_unica_permitida": V.ETIQUETA, "hogar_ingresos": V.HOG_PROHIBIDAS, "persona_solo_train": PROH_PERSONA,
    "derivadas_del_objetivo_v3_v5": V.OBJETIVO_DERIVADAS, "generadas_primera_etapa_v3_v5": conj5.get("generadas", []),
    "constantes_externas": V.CONSTANTES_EXTERNAS, "patrones_de_nombre": V.PATRONES_PROHIBIDOS}
ap = lambda mod, gr: [L for L in V.ORDEN_FAMILIAS if bool(res[(res.modelo == mod) & (res.grupo == gr) & (res.familia == L)].aporta.any())]
ACEP_LGB = sorted(set(ap("LGBM", "marginal")) | set(ap("LGBM", "condicional")), key=V.ORDEN_FAMILIAS.index)
ACEP_LOG = sorted(set(ap("LOGIT", "marginal")) | set(ap("LOGIT", "condicional")), key=V.ORDEN_FAMILIAS.index)
fam_cols = {V.FAMILIAS[L]: FC[L] for L in V.ORDEN_FAMILIAS}
conj7 = {
    "version": "v7", "estado": "definitivo",
    "descripcion": "Versión 7 (cuaderno 02d): todas las columnas comunes agregadas al hogar, familias U, I, C, P (IPM), V, E (codificación "
                   "bayes empírica fuera de pliegue), X, G y T; sin primera etapa ni constantes externas",
    "S1": conj5["S1"], "S2": BASE, "S3": TODAS, "generadas": [], "extra": [], "extra_gen": [],
    "todas_permitidas": TODAS, "completo": TODAS, "base_permitida": BASE,
    "sin_codificacion_objetivo": [c for c in TODAS if c not in TE],
    "familias": fam_cols, "familias_letras": V.FAMILIAS,
    "base_mas_familia": {V.FAMILIAS[L]: BASE + FC[L] for L in V.ORDEN_FAMILIAS},
    "familias_aceptadas_1EE": {"LightGBM": [V.FAMILIAS[L] for L in ACEP_LGB], "Logit": [V.FAMILIAS[L] for L in ACEP_LOG]},
    "aceptadas_lgbm": list(dict.fromkeys(BASE + sum((FC[L] for L in ACEP_LGB), []))),
    "aceptadas_logit": list(dict.fromkeys(BASE + sum((FC[L] for L in ACEP_LOG), []))),
    "x_shap_aceptadas_adelante": GAN_XS,
    "encuesta_corta": CORTA, "encuesta_corta_v7": CORTA_V7, "ipm": IPM_SET, "ipm_umbral_territorio": IPM_UMBRAL,
    "ipm_no_observables": V.IPM_NO_OBSERVABLES, "ipm_pesos": {k: {"dimension": d, "peso_oficial": w} for k, (d, w) in V.PESOS_IPM.items()},
    "categoricas": CATS,
    "codigos_numericos": [c for c in TODAS if c.endswith(("_jefe", "_principal")) and c.startswith("g_")] + ["principal_oficio", "principal_posicion"],
    "splines_logit": SPLINES, "logit_C": C_OPT,
    "bloques": {c: FAM.get(c, "B") for c in TODAS}, "nombres_bloques": V.FAMILIAS,
    "bloques_teoricos": {c: (AJ.NOMBRE_BLOQUE[AJ.bloque_variable(c)] if FAM.get(c, "B") == "B" else V.FAMILIAS[FAM[c]]) for c in TODAS},
    "mapa_preguntas": {c: MAPA[c] for c in TODAS},
    "mapa_preguntas_nota": ("Q(j): columnas comunes de las que depende la variable j, por rastreo exacto de dependencias (permutación de cada "
                            "columna común y reconstrucción). Las codificaciones con la etiqueta (te_*) además usan la etiqueta Pobre de "
                            "entrenamiento; en el despliegue sólo necesitan las preguntas de su clave. Costo c(S) = número de preguntas con "
                            "costo 1 en la unión; Lp, Li, Clase, Dominio, Depto y factores de expansión cuestan 0."),
    "preguntas": {q: {"descripcion": d, "tipo": t_, "costo": k, "n_variables": int(sum(q in MAPA[c] for c in TODAS))}
                  for q, (d, t_, k) in V.PREGUNTAS.items() if any(q in MAPA[c] for c in TODAS)},
    "costo_preguntas": {r_.conjunto: {"n_vars": int(r_.n_vars), "c_S": int(r_.c_S_preguntas)} for r_ in cst.itertuples()},
    "descripciones": {c: DESC.get(c, "base v5 (features.py)") for c in TODAS},
    "codificacion_objetivo": {"especificaciones": V.TE_ESPECIFICACIONES, "metodo": "bayes empírico jerárquico beta-binomial (MV marginal de la concentración)",
                              "excluir_bogota": TE_SIN_BOG, "k_interno": V.TE_K_INTERNO, "columnas": TE,
                              "hiperparametros_todo_train": HIPER["todo_train"],
                              "nota": "train de las tablas: fuera de pliegue con pliegues.csv; test: todo train. En CV recalcular dentro de cada "
                                      "pliegue con variables_v7.activar(dat, conj) / TransformadorV7 o variables_v7.codificar(train_idx, aplicar_idx)."},
    "especificaciones_muestra": SPECS, "especificaciones_shap": SPEC_XS,
    "prohibidas": PROHIBIDAS,
    "uso": "dat = ajuste.Datos(R, sufijo='_v7', conj_json='conjuntos_variables_v7.json'); variables_v7.activar(dat, c); "
           "ajuste.cv(..., transformar=dat.transformar). Columnas: c['todas_permitidas'] (o c['familias'], c['aceptadas_lgbm'], …).",
}
json.dump(conj7, open(OUT / "conjuntos_variables_v7.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
F1 = lambda r: float(r["res"]["F1_anidado_sinBOG"])
listo = {
    "listo": True, "prueba_submuestra": SUB if PRUEBA else None, "fecha": time.strftime("%Y-%m-%d %H:%M"), "cuaderno": "02d_variables_v7",
    "archivos": {"train": str(OUT / "modelo_train_v7.parquet"), "test": str(OUT / "modelo_test_v7.parquet"),
                 "conjuntos": str(OUT / "conjuntos_variables_v7.json"), "familias": str(TAB / "02d_familias.csv"),
                 "familias_resumen": str(TAB / "02d_familias_resumen.csv"), "auditoria": str(TAB / "02d_auditoria.csv"),
                 "ipm": [str(TAB / f) for f in ("02d_ipm_auc.csv", "02d_ipm_cruce.csv", "02d_ipm_por_dominio.csv")],
                 "figuras": [str(FIG / f"{f}.png") for f in ("02d_familias", "02d_camino_acumulado", "02d_ipm")]},
    "n_vars": {"base": len(BASE), "todas_permitidas": len(TODAS), **{V.FAMILIAS[L]: len(FC[L]) for L in V.ORDEN_FAMILIAS}},
    "LightGBM": {"F1_S2_cat_con_Mincer": F1(R_S2), "F1_base": {s: F1(r) for s, r in R_BASE.items()},
                 "F1_todas": {s: F1(r) for s, r in R_TODAS.items()}, "EE_todas": float(R_TODAS[1]["res"]["EE"]),
                 "LogLoss_sinBOG_todas": float(R_TODAS[1]["res"]["LogLoss_sinBOG"]), "AUC_sinBOG_todas": float(R_TODAS[1]["res"]["AUC_sinBOG"]),
                 "F1_referencias": {k: F1(v) for k, v in R_REF.items()}},
    "Logit": {"C": C_OPT, "splines": SPLINES, "F1_base": F1(RL_BASE), "F1_todas": F1(RL_TODAS), "EE_todas": float(RL_TODAS["res"]["EE"])},
    "familias_resumen": res.to_dict(orient="records"),
    "familias_aceptadas_1EE": conj7["familias_aceptadas_1EE"],
    "costo_preguntas": conj7["costo_preguntas"], "n_preguntas_usadas": len(conj7["preguntas"]),
    "ipm": {"tabla_cruce": cr.to_dict(), "kappa": float(kappa), "AUC_puntaje": float(ipm_auc.set_index("variable").loc["p_puntaje", "AUC_sinBOG"]),
            "F1_bloque_solo_LGBM": F1(R_REF["Bloque IPM solo"]), "resumen": ipm_res.to_dict(orient="records")},
    "codificacion": {"excluir_bogota": TE_SIN_BOG, "decision_bogota": bog.to_dict(orient="records"), "oof_etiqueta_perturbada_ok": OOF_OK},
    "auditoria": {"invariancia_ok": bool(inv_ok), "columnas_auditadas": len(aud), "prohibidas_de_referencia": int(PRO.shape[1]),
                  "fallas": int(len(FALLAS)), "estados": aud.estado.value_counts().to_dict(),
                  "max_abs_pearson": float(aud.max_abs_pearson_prohibida.max()), "max_abs_spearman": float(aud.max_abs_spearman_prohibida.max()),
                  "max_abs_corr_Pobre": float(aud.corr_pearson_con_Pobre.abs().max()),
                  "ok_por_compuerta_comun": aud[aud.estado == "OK (compuerta común)"][["variable", "prohibida_spearman", "max_abs_spearman_prohibida", "max_abs_corr_en_soporte"]].to_dict(orient="records"),
                  "KS_max": float(aud.KS_trainSB_vs_test.max()), "KS_rechazos_Bonferroni": int(aud.KS_rechaza_Bonferroni.sum()),
                  "etiqueta_nula_AUC": float(r_nula["res"]["AUC_sinBOG"]), "etiqueta_nula_z": float(z_nula),
                  "etiqueta_nula_AUC_por_pliegue": auc_k_nula.tolist(), "etiqueta_nula_p_unilateral_fuga": p_fuga_nula,
                  "adversarial": {k: {"AUC": float(v["AUC"]), "z": float(v["z"])} for k, v in ADV.items()}},
    "uso": conj7["uso"], "tiempo_min": (time.time() - T0) / 60,
}
json.dump(listo, open(OUT / "v7_LISTO.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: listo[k] for k in ("n_vars", "LightGBM", "Logit", "familias_aceptadas_1EE", "codificacion", "tiempo_min")}, ensure_ascii=False, indent=1, default=float))
