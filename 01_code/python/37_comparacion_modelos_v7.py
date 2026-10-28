# %% [markdown]
# > **Repositorio BDML-PS02 · script 37.** Lo ejecuta `01_code/37_comparacion_modelos_v7.R`. Es el cuaderno interno `39_comparacion_v7`:
# > sus tablas y figuras conservan el prefijo `39_` y quedan en `02_outputs/tables/37_comparacion_modelos_v7/` y
# > `02_outputs/figures/37_comparacion_modelos_v7/`; los CSV de envío, con el nombre con que se generaron (y, si se subieron,
# > con el que se subieron a Kaggle), en `02_outputs/predictions/`.

# %% [markdown]
# # 39 · Comparación de los modelos v7, disyuntiva de política y propuesta de envíos
# **Problem Set 2 — Predicción de pobreza (MESE/DANE) · BDML Uniandes 2026-20 · ruta de clasificación**
#
# Este cuaderno **no entrena nada**: lee lo que dejaron los cuadernos 30–38 del protocolo v7 (señales `3x_LISTO.json`,
# registro común `00_data/processed/modelos/` y caché `00_data/processed/cache_v7/`) y produce la comparación formal entre
# algoritmos, la comparación en igualdad de condiciones que pide el enunciado (§4, *compare*), la disyuntiva
# filtración/subcobertura (§3.4) y la propuesta de envíos a Kaggle (§2.1). Es reejecutable: cada vez que termine otro
# cuaderno (32 Elastic Net, 34 Random Forest, 36 XGBoost, 37 CatBoost), basta volver a correrlo. **No sube nada a Kaggle.**
#
# ## Qué se estima y cómo se compara (memo §2)
#
# * **Estimando.** El F1 de la clase pobre en la población del test: los 24 dominios distintos de Bogotá ($\mathcal T$).
#   Todas las métricas se calculan sólo con hogares de $\mathcal T$ de los pliegues de validación (CV ponderada por
#   importancia con pesos exactos; memo §1.2).
# * **F1 agregado anidado.** Se suman TP, FP y FN de todos los pliegues y se calcula un cociente (consistente para el F1
#   poblacional; Forman y Scholz, 2010); el umbral del pliegue $k$ se elige en las OOF de los otros pliegues (Koyejo et
#   al., 2014). EE por función de influencia, $\psi_i=\{2t_i-F(2t_i+f_i+g_i)\}/D$, e IC bootstrap de hogares.
# * **Comparación de procedimientos.** Con las diferencias por (partición, pliegue) $d_j$, $J=R\cdot K$ en las MISMAS
#   particiones (`pliegues_v7.csv`): t corregido de Nadeau y Bengio (2003), $t=\bar d/\sqrt{(1/J+1/(K-1))\hat\sigma_d^2}$
#   con $J-1$ g.l. (incertidumbre del procedimiento, incluido el entrenamiento); prueba bayesiana correlacionada (Corani y
#   Benavoli, 2015; Benavoli et al., 2017) con ROPE de 0,002 en F1, 0,0005 en log-loss y 0,0002 en Brier; **Holm** sobre
#   todos los pares (Demšar, 2006).
# * **Comparación de los clasificadores ajustados.** Bootstrap pareado **conjunto** de hogares (los mismos remuestreos
#   para todos los modelos) sobre las predicciones anidadas de la partición 1: IC de cada diferencia, $P(A>B)$ y
#   $P(\text{el modelo es el mejor})$ — responde «¿le ganaría A a B en el tablero privado?» (memo §2.4).
# * **Conjunto de confianza de modelos** (CVC; Lei, 2020) al 90 %: para cada candidato se contrasta «no es peor que
#   ningún otro» con bootstrap gaussiano de multiplicadores sobre pérdidas por hogar. Para el F1 (un cociente, no un
#   promedio) se usan los pseudo-valores de la linealización, $L_i=-(\hat F+\psi_i)$, cuya media es $-\hat F$ y cuyas
#   diferencias tienen la varianza correcta $\mathrm{sd}(\psi^A-\psi^B)^2$; además log-loss y Brier por hogar.
# * **Igualdad de condiciones** (enunciado §4): cada cuaderno evaluó, con sus hiperparámetros elegidos, el conjunto
#   común `todas_permitidas` (1.152 variables) y la encuesta corta (25 variables, 19 preguntas), ambos con $\lambda_B=1$
#   y $R\times K$ particiones comunes (Paso 4 del protocolo); aquí se tabulan y se comparan.
#
# **Ejecución.** Usa ≤ 2 hilos y no carga las 1.152 columnas (sólo `id`, `Pobre`, `Dominio`). Variables de entorno
# opcionales: `TALLER2_ROOT`, `P7_HILOS` (2 por defecto), `E39_B` (remuestreos bootstrap, 2.000 por defecto).

# %%
import os, sys, json, glob, hashlib, zipfile, pickle, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
HILOS = int(os.environ.get("P7_HILOS", "2"))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(HILOS)


def _raiz() -> Path:
    if os.environ.get("TALLER2_ROOT"):
        return Path(os.environ["TALLER2_ROOT"])
    p = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for q in [p, *p.parents]:
        if (q / ".here").exists():
            return q
    raise FileNotFoundError("No encuentro la raíz del repositorio (archivo .here); defina TALLER2_ROOT.")


RAIZ = _raiz()
sys.path.insert(0, str(RAIZ / "01_code" / "python"))
import numpy as np
import pandas as pd
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from IPython.display import display, Markdown
except Exception:                                                   # ejecución como script
    def display(o):
        print(o.to_string(max_rows=60, max_colwidth=60) if isinstance(o, pd.DataFrame) else o)
    Markdown = str
from pobreza.p7 import metricas as MT
from pobreza.p7 import comparar as CP
from pobreza.p7 import cvmotor as _cvm                               # noqa: F401 (clases de las cachés en pickle)

from pobreza.config import Rutas

_R = Rutas(RAIZ)                                                    # plantilla del repositorio (00_data, 02_outputs)
PROC, RAW = _R.processed, _R.raw
TABLAS, FIGS, ENVIOS = _R.tablas, _R.figuras, _R.submissions        # salidas propias: subcarpeta de este script
TABLAS_BASE = _R.tablas_base                                        # señales de los modelos: una subcarpeta por script
MODELOS, CACHE = PROC / "modelos", PROC / "cache_v7"
XLSX = TABLAS / "resultados_modelos_v7.xlsx"


def _rel(p) -> str:
    """Ruta relativa a la raíz del repositorio (con /), para que las tablas no guarden rutas de la máquina."""
    try:
        return Path(p).resolve().relative_to(Path(RAIZ).resolve()).as_posix()
    except ValueError:
        return str(p)


def _listo(pref: str) -> Path:
    """Señal `{pref}_LISTO.json` del modelo (está en la subcarpeta de su script en 02_outputs/tables)."""
    return next(iter(sorted(TABLAS_BASE.rglob(f"{pref}_LISTO.json"))))
B_BOOT = int(os.environ.get("E39_B", "2000"))
B_CVC = 1000
SEMILLA = 20262
K = 5
T0 = time.time()

# Algoritmos del protocolo v7 (prefijo del cuaderno → nombre corto y algoritmo del enunciado)
ALG = {"30": ("30_lpm", "LPM", "Linear Probability Model"), "31": ("31_logit", "Logit", "Logit"),
       "32": ("32_elastic_net", "Elastic Net", "Elastic Net"), "33": ("33_cart", "CART", "CART"),
       "34": ("34_random_forest", "Random Forest", "Random Forest"), "35": ("35_lightgbm", "LightGBM", "Boosting"),
       "36": ("36_xgboost", "XGBoost", "Boosting"), "37": ("37_catboost", "CatBoost", "Boosting"),
       "38": ("38_naive_bayes", "Naive Bayes", "Naive Bayes")}
NB_CUADERNO = {v[0]: k for k, v in ALG.items()}
# Los tres envíos v5 del 19 oct (con primera etapa: zona gris; se comparan aparte)
V5_SUBIDOS = [("10", "LGBM_v5_principal_8_vars", "LightGBM v5", "Boosting", "principal", 0.77),
              ("11", "XGB_v5_principal_8_vars", "XGBoost v5", "Boosting", "principal", 0.77),
              ("12", "RF_v5_parsimoniosa_20_vars", "Random Forest v5", "Random Forest", "parsimoniosa", 0.77)]
# Estado de Kaggle informado por el coordinador (20 oct): 10 envíos, 3 algoritmos de la lista
ENVIOS_HECHOS = {"Elastic Net": 4, "Random Forest": 2, "Boosting": 4}
FECHA_PLAN = "2026-10-20"
DIAS_PLAN = ["2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23", "2026-10-24", "2026-10-25"]
ENVIOS_POR_DIA = 3
print(f"Raíz {RAIZ} · {HILOS} hilos · B = {B_BOOT}", flush=True)

# %% [markdown]
# ## 1. Datos mínimos: etiqueta, dominio, particiones comunes y `sample_submission`

# %%
D = pd.read_parquet(PROC / "modelo_train_v7.parquet", columns=["id", "Pobre", "Dominio"])
DT = pd.read_parquet(PROC / "modelo_test_v7.parquet", columns=["id", "Dominio"])
PL = pd.read_csv(PROC / "pliegues_v7.csv")
assert len(PL) == len(D) and (PL.id.values == D.id.values).all(), "pliegues_v7.csv no está alineado con modelo_train_v7"
y = D.Pobre.values.astype(int)
dom = D.Dominio.astype(str).values
mT = dom != "BOGOTA"
yT = y[mT]
nT = int(mT.sum())


def _folds(r):
    f = PL[f"fold_r{r}"].values
    return [(np.flatnonzero(f != k), np.flatnonzero(f == k)) for k in sorted(np.unique(f))]


FOLDS = {r: _folds(r) for r in (1, 2, 3, 4)}
PREV = MT.prevalencia_esperada_test(y, dom, mT, DT.Dominio.astype(str).values)
PESOS = PREV["pesos_dominio"]
if (RAW / "sample_submission.csv").exists():
    SAMPLE = pd.read_csv(RAW / "sample_submission.csv")
else:
    with zipfile.ZipFile(RAW / "uniandes-bdml-2026-20-ps-2.zip") as z, z.open("sample_submission.csv") as f:
        SAMPLE = pd.read_csv(f)
assert len(SAMPLE) == 66168 and SAMPLE.id.is_unique
print(f"Train {len(y):,} hogares; 𝒯 = {nT:,} (Bogotá {int((~mT).sum()):,}); prevalencia 𝒯 {yT.mean():.4f}; "
      f"esperada en test {PREV['prevalencia_test']:.4f}; sample_submission {len(SAMPLE):,} filas (del zip)")

# %% [markdown]
# ## 2. Modelos disponibles: señales de listo y registro común

# %%
LISTOS = {}
for p in sorted(TABLAS_BASE.rglob("3[0-8]_LISTO.json"), key=lambda q: q.name):
    LISTOS[p.name[:2]] = json.loads(p.read_text(encoding="utf-8"))
PENDIENTES = [f"{k} {v[1]}" for k, v in ALG.items() if k not in LISTOS]
print("Terminados:", ", ".join(f"{k} {ALG[k][1]}" for k in LISTOS), "| pendientes:", ", ".join(PENDIENTES) or "ninguno")


def cargar_registro(slug):
    meta = json.loads((MODELOS / f"{slug}.json").read_text(encoding="utf-8"))
    oof = pd.read_parquet(MODELOS / f"{slug}.parquet").set_index("id").oof.reindex(D.id.values).values.astype(float)
    pt = MODELOS / f"{slug}_test.parquet"
    te = pd.read_parquet(pt).set_index("id").score.reindex(SAMPLE.id.values).values.astype(float) if pt.exists() else None
    return meta, oof, te


# entradas v7 del registro (principal, parsimoniosa, encuesta corta) de los cuadernos terminados
REG = {}
for pj in sorted(MODELOS.glob("*.json")):
    meta = json.loads(pj.read_text(encoding="utf-8"))
    pref = NB_CUADERNO.get(meta.get("notebook", ""))
    ver = (meta.get("params") or {}).get("version")
    if pref in LISTOS and ver:
        REG[(pref, ver)] = pj.stem
assert all((p, "principal") in REG for p in LISTOS), "falta la versión principal de algún cuaderno en el registro"
print(f"{len(REG)} entradas v7 en el registro:", ", ".join(f"{ALG[p][1]}/{v}" for p, v in REG))

# %% [markdown]
# ## 3. Tabla maestra (partición 1, OOF del predictor final; otras particiones desde la caché del protocolo)
#
# Las métricas de la partición 1 se **recalculan** aquí con las OOF del registro (y se verifica que coincidan con la
# señal de listo). El F1 de las particiones 2 y 3 es el de la configuración principal en la caché del protocolo
# (pasos de Bogotá o de monotonía); el de la partición 4 es la re-evaluación en una partición nueva (Paso 8).

# %%
def cache_de(pref, L):
    cands = [d for d in CACHE.iterdir() if d.is_dir() and d.name.startswith(f"{pref}_v7_{L.get('presupuesto', '')}_")
             and "_sub" not in d.name and (d / "paso_final.pkl").exists()]
    if L.get("pool_candidatas"):
        cands = [d for d in cands if "_pool" in d.name] or cands
    else:
        cands = [d for d in cands if "_pool" not in d.name] or cands
    return sorted(cands, key=lambda d: (d / "paso_final.pkl").stat().st_mtime)[-1] if cands else None


def leer_pkl(ruta):
    with open(ruta, "rb") as f:
        return pickle.load(f)


def rk_principal(pref, L):
    """Vectores por (partición, pliegue) de la configuración principal (params, variables, λ_B) evaluada con R×K×S en
    el protocolo, y OOF por partición localizadas en la caché `cv_*.npz` (para log-loss y Brier de todos)."""
    d = cache_de(pref, L)
    mono = L.get("monotonia") or {}
    if mono.get("adoptada") and (d / "paso_monotonia.pkl").exists():
        o = leer_pkl(d / "paso_monotonia.pkl"); rk = o["rk"]["monótona"]; fuente = "paso de monotonía (variante adoptada)"
    else:
        o = leer_pkl(d / "paso_bogota.pkl"); lam = float(L["peso_bogota"]); rk = o["rk"][f"λ_B = {lam:g}"]
        fuente = f"paso de Bogotá (λ_B = {lam:g})"
    etiq = [tuple(e) for e in rk["etiquetas"]]
    F_r = {}
    for (r, k), f in zip(etiq, rk["F1"]):
        F_r.setdefault(r, []).append(f)
    # OOF de cada partición: archivos de la caché con la misma λ_B y el mismo número de candidatas cuyo F1 por pliegue
    # reproduce el del protocolo (identificación por contenido, no por la clave)
    lam = float(L["peso_bogota"]); ncand = int(L["n_candidatas"])
    oofs, dif = {}, {}
    for p in sorted(d.glob("cv_*.npz")):
        try:
            with np.load(p) as z:
                meta = json.loads(str(z["meta"]))
            r = int(meta["r"])
            if r not in F_r or abs(float(meta["lam_B"]) - lam) > 1e-9 or int(meta.get("extra", {}).get("n_vars", -1)) != ncand:
                continue
            with np.load(p) as z:
                s = z["oof"].astype(float)
            fk = MT.f1_anidado(y, s, FOLDS[r], mT)["F1_pliegues"]
            e = float(np.max(np.abs(fk - np.asarray(F_r[r]))))
            if e < 5e-4 and e < dif.get(r, 1.0):
                oofs[r], dif[r] = s, e
        except Exception:
            continue
    LL, BR = [], []
    for (r, k) in etiq:
        if r in oofs:
            tr, va = FOLDS[r][k]; v = va[mT[va]]
            LL.append(MT.logloss(y[v], MT.recortar(oofs[r][v]))); BR.append(MT.brier(y[v], np.clip(oofs[r][v], 0, 1)))
        else:
            LL.append(np.nan); BR.append(np.nan)
    return {"F1": np.asarray(rk["F1"], float), "LL": np.asarray(LL), "BR": np.asarray(BR), "etiq": etiq,
            "F1_r": {int(r): float(v) for r, v in rk["F1_por_particion"].items()}, "fuente": fuente,
            "oof_encontradas": sorted(oofs), "dif_max": max(dif.values()) if dif else None, "cache": d.name}


def metricas_r1(oof):
    m = MT.metricas_particion(y, oof, FOLDS[1], mT, B=B_BOOT, semilla=SEMILLA, dominio=dom, pesos_dominio=PESOS)
    an = MT.f1_anidado(y, oof, FOLDS[1], mT)
    return m, an["yhat"][mT].astype(int)


def mil(v, d=0):
    s = f"{v:,.{d}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def coma(v, d=4):
    return f"{v:.{d}f}".replace(".", ",")


FILAS, PRED, RK, OOF_R1, TEST, META = [], {}, {}, {}, {}, {}
for pref, L in LISTOS.items():
    _, corto, fam = ALG[pref]
    RK[corto] = rk_principal(pref, L)
    for (p, ver), slug in REG.items():
        if p != pref:
            continue
        meta, oof, te = cargar_registro(slug)
        m, yhat = metricas_r1(oof)
        nombre = corto if ver == "principal" else f"{corto} · {ver.replace('_', ' ')}"
        PRED[nombre], OOF_R1[nombre], TEST[nombre], META[nombre] = yhat, oof, te, meta
        if ver == "principal":
            assert abs(m["F1_anidado"] - L["F1_anidado_sinBOG"]) < 1e-9, (nombre, m["F1_anidado"], L["F1_anidado_sinBOG"])
            umb, regla, tasa = L["umbral"], L["regla_umbral"], L["tasa_predicha_test"]
            nv, nq, lam, arch = L["n_vars"], L["n_preguntas"], L["peso_bogota"], L["archivo"]
            f2, f3 = RK[corto]["F1_r"].get(2), RK[corto]["F1_r"].get(3)
            f4, tmin = L.get("F1_reevaluacion_r4"), L.get("tiempo_total_min")
        else:
            ov = next((o for o in L.get("otras_versiones", []) if o.get("version") == ver), {})
            if ov:
                assert abs(m["F1_anidado"] - ov["F1_anidado_sinBOG"]) < 1e-9, (nombre, m["F1_anidado"])
            umb, regla = meta["params"].get("umbral_test"), meta["params"].get("regla_umbral")
            tasa = float((te >= umb).mean()) if te is not None else np.nan
            nv, nq, lam, arch = ov.get("n_vars", meta.get("n_vars")), ov.get("n_preguntas"), ov.get("lam_B"), meta.get("archivo_envio")
            f2 = f3 = f4 = tmin = None
        FILAS.append({"grupo": "v7 sin primera etapa", "modelo": nombre, "algoritmo": fam, "cuaderno": ALG[pref][0],
                      "versión": ver, "F1": m["F1_anidado"], "IC95 inf": m["IC95_F1"][0], "IC95 sup": m["IC95_F1"][1],
                      "EE ψ": m["EE_psi"], "F1 r2": f2, "F1 r3": f3, "F1 r4 (re-evaluación)": f4,
                      "F1 post-estratificado": m.get("F1_postestratificado"), "log-loss": m["LogLoss"], "Brier": m["Brier"],
                      "ACE": m["ACE"], "AUC": m["AUC_ROC"], "AUC-PR": m["AUC_PR"], "precisión": m["precision"],
                      "exhaustividad": m["exhaustividad"], "MCC": m["MCC"], "tasa predicha OOF": m["tasa_predicha_anidada"],
                      "tasa predicha test": tasa, "umbral": umb, "regla del umbral": regla, "n_vars": nv, "n_preguntas": nq,
                      "λ_B (peso de Bogotá)": lam, "tiempo (min)": tmin, "archivo CSV": arch, "slug registro": slug})

# referencias v5 (con primera etapa: zona gris)
for pref5, slug, nombre, fam, ver, pub in V5_SUBIDOS:
    if not (MODELOS / f"{slug}.json").exists():
        continue
    L5 = json.loads(_listo(pref5).read_text(encoding="utf-8"))
    meta, oof, te = cargar_registro(slug)
    m, yhat = metricas_r1(oof)
    PRED[nombre], OOF_R1[nombre], TEST[nombre] = yhat, oof, te
    ov = L5 if ver == "principal" else next((o for o in L5.get("otras_versiones", []) if o.get("version") == ver), {})
    FILAS.append({"grupo": "v5 con primera etapa (zona gris)", "modelo": nombre, "algoritmo": fam,
                  "cuaderno": f"{pref5} (v5)", "versión": ver, "F1": m["F1_anidado"], "IC95 inf": m["IC95_F1"][0],
                  "IC95 sup": m["IC95_F1"][1], "EE ψ": m["EE_psi"],
                  "F1 r2": L5.get("F1_segunda_particion") if ver == "principal" else None, "F1 r3": None, "F1 r4 (re-evaluación)": None,
                  "F1 post-estratificado": m.get("F1_postestratificado"), "log-loss": m["LogLoss"], "Brier": m["Brier"],
                  "ACE": m["ACE"], "AUC": m["AUC_ROC"], "AUC-PR": m["AUC_PR"], "precisión": m["precision"],
                  "exhaustividad": m["exhaustividad"], "MCC": m["MCC"], "tasa predicha OOF": m["tasa_predicha_anidada"],
                  "tasa predicha test": ov.get("tasa_predicha_test"), "umbral": ov.get("umbral"),
                  "regla del umbral": "F1-óptimo OOF sin Bogotá", "n_vars": ov.get("n_vars", meta.get("n_vars")),
                  "n_preguntas": None, "λ_B (peso de Bogotá)": L5.get("peso_bogota"),
                  "tiempo (min)": L5.get("tiempo_total_min") if ver == "principal" else None,
                  "archivo CSV": meta.get("archivo_envio"), "slug registro": slug, "Kaggle público": pub})

RES = pd.DataFrame(FILAS)
RES["_g"] = RES.grupo.str.startswith("v5").astype(int)
RES = RES.sort_values(["_g", "F1"], ascending=[True, False]).drop(columns="_g").reset_index(drop=True)
PRINC = [ALG[p][1] for p in LISTOS]                                  # modelos principales permitidos
PRINC = sorted(PRINC, key=lambda n: -float(RES.loc[RES.modelo == n, "F1"].iloc[0]))
MEJOR = PRINC[0]
print("Mejor modelo permitido (F1 anidado, partición 1):", MEJOR)
for n in PRINC:
    print(f"  {n:14s} {RK[n]['fuente']}; OOF de la caché encontradas para r = {RK[n]['oof_encontradas']} "
          f"(máx. |ΔF1 pliegue| = {RK[n]['dif_max']})")
print(f"[t = {(time.time() - T0) / 60:.1f} min] tabla maestra lista", flush=True)
display(RES[["grupo", "modelo", "F1", "IC95 inf", "IC95 sup", "F1 r2", "F1 r4 (re-evaluación)", "log-loss", "AUC",
             "n_vars", "n_preguntas", "λ_B (peso de Bogotá)"]].round(4))

# %% [markdown]
# ## 4. Comparaciones pareadas entre todos los pares de modelos principales
#
# Para cada par $(A,B)$, con $A$ el de mayor F1: t corregido y prueba bayesiana sobre las $J$ diferencias por
# (partición, pliegue) comunes (F1 anidado, log-loss y Brier), bootstrap pareado conjunto de hogares (partición 1),
# EE de la diferencia por función de influencia y Holm sobre los p-valores del F1 de todos los pares.

# %%
def bootstrap_conjunto(yv, H, B=B_BOOT, semilla=SEMILLA, lote=50):
    """F1 de cada columna de H (predicciones 0/1 fijas) en B remuestreos de hogares COMUNES a todos los modelos."""
    yv = np.asarray(yv).astype(int); H = np.asarray(H).astype(int); n, M = H.shape
    T = np.concatenate([((yv == 1)[:, None] & (H == 1)), ((yv == 0)[:, None] & (H == 1)),
                        ((yv == 1)[:, None] & (H == 0))], 1).astype(np.float32)
    rng = np.random.default_rng(semilla); F = np.empty((B, M))
    for b0 in range(0, B, lote):
        m = min(lote, B - b0)
        C = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(m)]).astype(np.float32)
        V = (C @ T).astype(float)                       # sumas enteras < 2^24: exactas en float32
        tp, fp, fn = V[:, :M], V[:, M:2 * M], V[:, 2 * M:]
        F[b0:b0 + m] = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)
    return F


def psi_de(nombre):
    return MT.influencia_f1(yT, PRED[nombre])[2]


def pareada(A, B, rkA, rkB, FbA=None, FbB=None, psiA=None, psiB=None):
    ea, eb = rkA["etiq"], rkB["etiq"]; sb = set(eb)
    comunes = [e for e in ea if e in sb]
    ia = [ea.index(e) for e in comunes]; ib = [eb.index(e) for e in comunes]
    out = {"A": A, "B": B, "J": len(comunes), "particiones": ",".join(str(r) for r in sorted({r for r, _ in comunes}))}
    for met, rope, nom in (("F1", MT.ROPE_F1, "ΔF1"), ("LL", MT.ROPE_LL, "Δlog-loss"), ("BR", MT.ROPE_BRIER, "ΔBrier")):
        d = np.asarray(rkA[met])[ia] - np.asarray(rkB[met])[ib]
        if np.isnan(d).any():
            continue
        nb = CP.nadeau_bengio(d, K); ba = CP.bayes_correlacionado(d, rope, K)
        out.update({f"{nom}": nb["media"], f"{nom} EE corregido": nb["EE_corregido"], f"{nom} IC95 inf": nb["IC95"][0],
                    f"{nom} IC95 sup": nb["IC95"][1], f"{nom} t": nb["t"], f"{nom} p": nb["p"]})
        if met == "F1":
            out.update({"P(A mejor)": ba["P_der"], "P(ROPE)": ba["P_rope"], "P(B mejor)": ba["P_izq"],
                        "pliegues con A mejor": int((d > 0).sum()), "EE ingenuo": nb["EE_ingenuo"]})
        else:
            out.update({f"{nom} P(A mejor)": ba["P_izq"], f"{nom} P(ROPE)": ba["P_rope"], f"{nom} P(B mejor)": ba["P_der"]})
    if FbA is not None:
        db = FbA - FbB
        out.update({"ΔF1 r1 (hogares fijos)": float(MT.f1(yT, PRED[A]) - MT.f1(yT, PRED[B])),
                    "bootstrap IC95 inf": float(np.quantile(db, .025)), "bootstrap IC95 sup": float(np.quantile(db, .975)),
                    "bootstrap P(A > B)": float((db > 0).mean()),
                    "EE ψ de la diferencia": float((psiA - psiB).std(ddof=1) / np.sqrt(len(psiA))),
                    "corr(ψA, ψB)": float(np.corrcoef(psiA, psiB)[0, 1])})
    return out


HP = np.column_stack([PRED[n] for n in PRINC])
FB = bootstrap_conjunto(yT, HP)
PSI = {n: psi_de(n) for n in PRINC}
P_MEJOR = {n: float((FB.argmax(1) == j).mean()) for j, n in enumerate(PRINC)}
PARES = []
for i, A in enumerate(PRINC):
    for j in range(i + 1, len(PRINC)):
        B = PRINC[j]
        PARES.append(pareada(A, B, RK[A], RK[B], FB[:, i], FB[:, j], PSI[A], PSI[B]))
COMP = pd.DataFrame(PARES)
if len(COMP):
    COMP["ΔF1 p Holm (todos los pares)"] = CP.holm(COMP["ΔF1 p"].values)
    COMP["distinguibles (Holm 5 %)"] = np.where(COMP["ΔF1 p Holm (todos los pares)"] < 0.05, "sí", "no")
    COMP["equivalentes en la práctica (P(ROPE) > 0,95)"] = np.where(COMP["P(ROPE)"] > 0.95, "sí", "no")
VS = COMP[COMP.A == MEJOR].copy()
if len(VS):
    VS["ΔF1 p Holm (vs mejor)"] = CP.holm(VS["ΔF1 p"].values)

# conjunto de confianza de modelos (CVC, Lei 2020) al 90 %
def pseudo_f1(n):
    F = MT.f1(yT, PRED[n]); return -(F + PSI[n])


CVC = {}
for crit, Lfun in (("F1 (pseudo-valores ψ)", pseudo_f1),
                   ("log-loss", lambda n: MT.perdida_obs(yT, OOF_R1[n][mT], "logloss")),
                   ("Brier", lambda n: MT.perdida_obs(yT, OOF_R1[n][mT], "brier"))):
    Lm = np.column_stack([Lfun(n) for n in PRINC])
    c = CP.cvc(Lm, alfa=0.10, B=B_CVC, semilla=SEMILLA, lote=20)
    CVC[crit] = {"p": dict(zip(PRINC, c["p_valores"])), "conjunto": [PRINC[k] for k in c["conjunto"]]}
for crit, c in CVC.items():
    print(f"CVC 90 % ({crit}): {c['conjunto']}  p = " + ", ".join(f"{k} {v:.3f}" for k, v in c["p"].items()))

# columnas de la comparación en la tabla maestra
for n in PRINC:
    i = RES.index[RES.modelo == n][0]
    RES.loc[i, "P(mejor) bootstrap"] = P_MEJOR[n]
    for crit, c in CVC.items():
        RES.loc[i, f"CVC 90 % {crit}: p"] = c["p"][n]
    if n == MEJOR:
        RES.loc[i, "vs mejor"] = "— (mejor)"
    else:
        f = VS[VS.B == n].iloc[0]
        RES.loc[i, "ΔF1 vs mejor (t corregido)"] = -f["ΔF1"]
        RES.loc[i, "p Holm vs mejor"] = f["ΔF1 p Holm (vs mejor)"]
        RES.loc[i, "vs mejor"] = ("indistinguible" if f["ΔF1 p Holm (vs mejor)"] >= 0.05 else "peor (Holm 5 %)") + \
            ("; equivalente práctico" if f["P(ROPE)"] > 0.95 else "")

INDIST = [n for n in PRINC if n != MEJOR and (VS[VS.B == n]["ΔF1 p Holm (vs mejor)"].iloc[0] >= 0.05)]
display(VS[["A", "B", "J", "ΔF1", "ΔF1 IC95 inf", "ΔF1 IC95 sup", "ΔF1 p", "ΔF1 p Holm (vs mejor)", "P(ROPE)",
            "bootstrap IC95 inf", "bootstrap IC95 sup", "bootstrap P(A > B)"]].round(5))
print("P(mejor) bootstrap:", {k: round(v, 3) for k, v in P_MEJOR.items()})
print("Indistinguibles del mejor (t corregido + Holm 5 %):", INDIST or "ninguno")

print(f"[t = {(time.time() - T0) / 60:.1f} min] comparaciones listas", flush=True)
# referencia: mejor v7 frente a los tres envíos v5 (con primera etapa), bootstrap pareado de hogares en la partición 1
V5N = [n for _, _, n, *_ in V5_SUBIDOS if n in PRED]
if V5N:
    Fv = bootstrap_conjunto(yT, np.column_stack([PRED[MEJOR]] + [PRED[n] for n in V5N]))
    filas = []
    for j, n in enumerate(V5N, start=1):
        db = Fv[:, 0] - Fv[:, j]; ps = psi_de(n)
        filas.append({"A": MEJOR + " (v7)", "B": n + " — con primera etapa", "ΔF1 r1": float(MT.f1(yT, PRED[MEJOR]) - MT.f1(yT, PRED[n])),
                      "bootstrap IC95 inf": float(np.quantile(db, .025)), "bootstrap IC95 sup": float(np.quantile(db, .975)),
                      "bootstrap P(A > B)": float((db > 0).mean()),
                      "EE ψ de la diferencia": float((PSI[MEJOR] - ps).std(ddof=1) / np.sqrt(nT))})
    V5COMP = pd.DataFrame(filas)
    display(V5COMP.round(5))
else:
    V5COMP = pd.DataFrame()

# %% [markdown]
# ## 5. Igualdad de condiciones (enunciado §4: «differences in scores should reflect the algorithm»)
#
# Cada cuaderno evaluó, con SUS hiperparámetros elegidos, dos conjuntos comunes con $\lambda_B=1$ y las mismas
# particiones: **todas las variables v7** (1.152 = `todas_permitidas`) y la **encuesta corta** (25 variables, 19
# preguntas). Los hiperparámetros se ajustaron con el conjunto propio de cada algoritmo (no se re-ajustaron para el
# común), así que la comparación favorece levemente a los algoritmos cuyo conjunto propio coincide con el común (LPM,
# CART, LightGBM). Se reportan además el esfuerzo de búsqueda (configuraciones y minutos; Sculley et al., 2018).

# %%
N_TODAS = len(json.loads((PROC / "conjuntos_variables_v7.json").read_text(encoding="utf-8"))["todas_permitidas"])
IGU, RKI, OOFI = [], {}, {}
for pref, L in LISTOS.items():
    corto = ALG[pref][1]
    d = cache_de(pref, L)
    if d is None or not (d / "paso_conjuntos.pkl").exists():
        continue
    o = leer_pkl(d / "paso_conjuntos.pkl")
    bus = leer_pkl(d / "paso_busqueda.pkl") if (d / "paso_busqueda.pkl").exists() else {}
    for var, rr in o.get("resultados", {}).items():
        nv = int(rr[min(rr)].extra.get("n_vars", -1))
        cond = "todas (1.152 variables)" if nv == N_TODAS else ("encuesta corta (25 variables)" if var == "corta" else None)
        if cond is None:
            continue
        Rs = sorted(rr)
        F_r, LL_r, BR_r, F_k, LL_k, BR_k, etiq = {}, {}, {}, [], [], [], []
        for r in Rs:
            s = rr[r].oof
            an = MT.f1_anidado(y, s, FOLDS[r], mT)
            F_r[r] = an["F1"]; F_k += list(an["F1_pliegues"])
            LL_r[r] = MT.logloss(yT, MT.recortar(s[mT])); BR_r[r] = MT.brier(yT, np.clip(s[mT], 0, 1))
            LL_k += list(MT.por_pliegue(y, np.clip(s, MT.EPS, 1 - MT.EPS), FOLDS[r], mT, "logloss"))
            BR_k += list(MT.por_pliegue(y, np.clip(s, 0, 1), FOLDS[r], mT, "brier"))
            etiq += [(r, k) for k in range(len(FOLDS[r]))]
        clave = (cond, corto)
        RKI[clave] = {"F1": np.asarray(F_k), "LL": np.asarray(LL_k), "BR": np.asarray(BR_k), "etiq": etiq}
        s1 = rr[Rs[0]].oof; OOFI[clave] = s1
        h1 = MT.f1_anidado(y, s1, FOLDS[Rs[0]], mT)["yhat"][mT].astype(int)
        _, ee, _ = MT.influencia_f1(yT, h1)
        sesgo = (bus.get("sesgo_tt") or {}) if isinstance(bus, dict) else {}
        IGU.append({"condición": cond, "modelo": corto, "algoritmo": ALG[pref][2], "variante del cuaderno": var,
                    "n_vars": nv, "λ_B": float(rr[Rs[0]].lam_B), "particiones": len(Rs), "F1 medio particiones": float(np.mean(list(F_r.values()))),
                    **{f"F1 r{r}": v for r, v in F_r.items()}, "EE ψ (r1)": ee,
                    "log-loss": float(np.mean(list(LL_r.values()))), "Brier": float(np.mean(list(BR_r.values()))),
                    "AUC (r1)": MT.auc_roc(yT, s1[mT]), "AUC-PR (r1)": MT.auc_pr(yT, s1[mT]),
                    "F1 selección propia (principal)": float(np.mean(list(RK[corto]["F1_r"].values()))),
                    "configuraciones evaluadas en la búsqueda": sesgo.get("n_config"),
                    "minutos de búsqueda": (L.get("tiempos_min") or {}).get("busqueda")})
IG = pd.DataFrame(IGU)
IGC = []
if len(IG):
    IG["Δ selección propia − común"] = IG["F1 selección propia (principal)"] - IG["F1 medio particiones"]
    for cond, g in IG.groupby("condición", sort=False):
        orden = g.sort_values("F1 medio particiones", ascending=False).modelo.tolist()
        mejor_c = orden[0]
        if len(orden) > 1:
            Hc = np.column_stack([MT.f1_anidado(y, OOFI[(cond, n)], FOLDS[1], mT)["yhat"][mT] for n in orden])
            Fc = bootstrap_conjunto(yT, Hc)
            pm = {n: float((Fc.argmax(1) == j).mean()) for j, n in enumerate(orden)}
            ps_c = {n: MT.influencia_f1(yT, Hc[:, j])[2] for j, n in enumerate(orden)}
            filas = []
            for j, n in enumerate(orden[1:], start=1):
                PRED["_a"], PRED["_b"] = Hc[:, 0], Hc[:, j]
                f = pareada("_a", "_b", RKI[(cond, mejor_c)], RKI[(cond, n)], Fc[:, 0], Fc[:, j], ps_c[mejor_c], ps_c[n])
                f.update({"condición": cond, "A": mejor_c, "B": n})
                filas.append(f)
            PRED.pop("_a", None); PRED.pop("_b", None)
            t = pd.DataFrame(filas); t["ΔF1 p Holm (vs mejor de la condición)"] = CP.holm(t["ΔF1 p"].values)
            IGC.append(t)
            for n in orden:
                IG.loc[(IG["condición"] == cond) & (IG.modelo == n), "P(mejor) bootstrap"] = pm[n]
        IG.loc[IG["condición"] == cond, "rango"] = IG.loc[IG["condición"] == cond, "F1 medio particiones"].rank(ascending=False).values
IGC = pd.concat(IGC, ignore_index=True) if IGC else pd.DataFrame()
if len(IGC):                                                       # comparación con el mejor de cada condición, en la misma fila
    m_ = IGC.rename(columns={"A": "mejor de la condición", "B": "modelo", "ΔF1": "ΔF1 mejor − modelo (t corregido)",
                             "ΔF1 IC95 inf": "ΔF1 IC95 inf", "ΔF1 IC95 sup": "ΔF1 IC95 sup"})
    keep = ["condición", "modelo", "mejor de la condición", "J", "ΔF1 mejor − modelo (t corregido)", "ΔF1 IC95 inf",
            "ΔF1 IC95 sup", "ΔF1 p", "ΔF1 p Holm (vs mejor de la condición)", "P(ROPE)", "Δlog-loss", "Δlog-loss p",
            "bootstrap IC95 inf", "bootstrap IC95 sup", "bootstrap P(A > B)"]
    IG = IG.merge(m_[[c for c in keep if c in m_.columns]], on=["condición", "modelo"], how="left")
print(f"[t = {(time.time() - T0) / 60:.1f} min] igualdad lista", flush=True)
if len(IG):
    IG = IG.sort_values(["condición", "F1 medio particiones"], ascending=[False, False]).reset_index(drop=True)
    display(IG[["condición", "modelo", "n_vars", "F1 medio particiones", "EE ψ (r1)", "log-loss", "AUC (r1)",
                "F1 selección propia (principal)", "Δ selección propia − común"]].round(4))

# %% [markdown]
# ## 6. Disyuntiva de política: filtración y subcobertura por cada 1.000 hogares (enunciado §3.4)
#
# Falsos positivos = **filtración** (hogares no pobres que entrarían al programa); falsos negativos = **subcobertura**
# (hogares pobres excluidos). Tres reglas de decisión, todas **anidadas** (el umbral del pliegue $k$ se fija con las
# OOF de los otros pliegues y se aplica al pliegue $k$; partición 1, sólo $\mathcal T$):
# 1. **F1-óptimo** (lo que premia la competencia);
# 2. **cobertura del 90 %** de los pobres (un gobierno que teme más la exclusión);
# 3. **tasa = prevalencia** (presupuesto para tantos cupos como pobres hay; FP = FN por construcción).
#
# Con probabilidades calibradas, el umbral $t$ equivale a una razón implícita de costos $c_{FN}/c_{FP}=(1-t)/t$
# (regla de Bayes: clasificar pobre si $p\ge c_{FP}/(c_{FP}+c_{FN})$).

# %%
def umbral_criterio(yv, sv, crit):
    if crit == "F1-óptimo":
        return MT.umbral_argmax(yv, sv)[0]
    if crit == "cobertura del 90 % de los pobres":
        sp = np.sort(sv[yv == 1])[::-1]
        return float(sp[int(np.ceil(0.90 * len(sp))) - 1])
    if crit == "tasa predicha = prevalencia":
        return float(np.quantile(sv, 1 - yv.mean()))
    raise ValueError(crit)


CRITERIOS = ["F1-óptimo", "cobertura del 90 % de los pobres", "tasa predicha = prevalencia"]


def politica(nombre, s, crit):
    yhat = np.full(len(y), -1); umbs = []
    for tr, va in FOLDS[1]:
        trT, vaT = tr[mT[tr]], va[mT[va]]
        t = umbral_criterio(y[trT], s[trT], crit); umbs.append(t)
        yhat[vaT] = (s[vaT] >= t)
    h = yhat[mT]
    tp, fp, fn, tn = MT.confusion(yT, h)
    t_med = float(np.mean(umbs))
    por = lambda c: 1000 * c / nT
    ee = lambda c: 1000 * np.sqrt((c / nT) * (1 - c / nT) / nT)
    return {"modelo": nombre, "criterio": crit, "umbral (media de pliegues)": t_med,
            "razón implícita c_FN/c_FP": (1 - t_med) / t_med if 0 < t_med < 1 else np.nan,
            "pobres atendidos (TP) por 1.000": por(tp), "filtración: no pobres incluidos (FP) por 1.000": por(fp),
            "EE FP por 1.000": ee(fp), "subcobertura: pobres excluidos (FN) por 1.000": por(fn), "EE FN por 1.000": ee(fn),
            "no pobres excluidos (TN) por 1.000": por(tn), "cupos asignados por 1.000": por(tp + fp),
            "tasa de filtración FP/(TP+FP)": fp / max(tp + fp, 1), "tasa de subcobertura FN/(TP+FN)": fn / max(tp + fn, 1),
            "F1": MT.f1(yT, h), "precisión": tp / max(tp + fp, 1), "exhaustividad": tp / max(tp + fn, 1)}


corto_mejor = MEJOR
pref_mejor = next(p for p in LISTOS if ALG[p][1] == MEJOR)
CORTA = f"{MEJOR} · encuesta corta" if f"{MEJOR} · encuesta corta" in OOF_R1 else None
POL = []
for nombre in [MEJOR] + ([CORTA] if CORTA else []):
    for crit in CRITERIOS:
        POL.append({"panel": "mejor modelo y su encuesta corta", **politica(nombre, OOF_R1[nombre], crit)})
for nombre in PRINC + [n for n in OOF_R1 if "encuesta corta" in n and n != CORTA]:
    POL.append({"panel": "todos los modelos v7 (umbral F1-óptimo)", **politica(nombre, OOF_R1[nombre], "F1-óptimo")})
POL = pd.DataFrame(POL)
for nombre in POL.modelo.unique():
    i = RES.index[RES.modelo == nombre][0]
    POL.loc[POL.modelo == nombre, "n_vars"] = RES.loc[i, "n_vars"]
    POL.loc[POL.modelo == nombre, "n_preguntas"] = RES.loc[i, "n_preguntas"]
    POL.loc[POL.modelo == nombre, "ACE (calibración)"] = RES.loc[i, "ACE"]
POL["razón de costos interpretable"] = np.where(POL["ACE (calibración)"] <= 0.01, "sí (ACE ≤ 0,01)", "no: probabilidades mal calibradas")
print(f"[t = {(time.time() - T0) / 60:.1f} min] política lista")
display(POL[POL.panel.str.startswith("mejor")][["modelo", "criterio", "umbral (media de pliegues)", "razón implícita c_FN/c_FP",
                                                 "filtración: no pobres incluidos (FP) por 1.000",
                                                 "subcobertura: pobres excluidos (FN) por 1.000", "F1"]].round(3))

# %% [markdown]
# ## 7. Propuesta de envíos a Kaggle (sólo propuesta: **no se sube nada**)
#
# Regla del enunciado (§2.1): ≥ 15 envíos con ≥ 5 algoritmos distintos de la lista; Kaggle permite 3 por día hasta el
# 25 de octubre. Cada CSV propuesto se verifica: `id,pobre`; 66.168 filas; ids y orden de `sample_submission` (leído del
# zip de `00_data/raw/`); valores 0/1; tasa predicha igual a la de la señal y coherente con $\hat q$ de las OOF (±0,003) y
# con la prevalencia esperada en test (0,2075); y **reproducción exacta** desde el puntaje de test del registro con el
# umbral declarado. El envío final se elige por CV, nunca por el tablero público (memo §2.1 y Paso 13).

# %%
V5_CSV = {n: ENVIOS / a for (_, s, n, *_), a in zip(V5_SUBIDOS, [
    "LGBM_leaves_10_lr_0_0253_mcs_22_ff_0_442_l2_0_00326_n_8.csv",
    "XGB_depth_3_eta_0_0473_mcw_1_68_sub_0_561_col_0_876_n_8_bog0_5.csv",
    "RF_mtry_0_292_minleaf_156_ms_0_443_depth_None_cw_none_trees_360_n_20_pars_bog0_5.csv"])}
_V5_PRED = {n: pd.read_csv(p).set_index("id").pobre.reindex(SAMPLE.id.values).values for n, p in V5_CSV.items() if p.exists()}


def verificar_csv(nombre, archivo, umbral, tasa_declarada, q_hat):
    ruta = ENVIOS / archivo
    out = {"archivo": archivo, "ruta": _rel(ruta)}
    if not ruta.exists():
        return {**out, "verificación": "FALTA EL ARCHIVO"}
    raw = ruta.read_bytes()
    csv = pd.read_csv(ruta)
    ok = {"columnas id,pobre": list(csv.columns) == ["id", "pobre"],
          "66.168 filas": len(csv) == 66168 == len(SAMPLE),
          "ids y orden de sample_submission": len(csv) == len(SAMPLE) and bool((csv.id.values == SAMPLE.id.values).all()),
          "ids únicos": bool(csv.id.is_unique),
          "valores 0/1 sin vacíos": bool(csv.pobre.notna().all() and set(np.unique(csv.pobre)) <= {0, 1})}
    tasa = float(csv.pobre.mean())
    ok["tasa = la declarada"] = tasa_declarada is not None and abs(tasa - tasa_declarada) < 1e-9
    ok["tasa/prevalencia esperada en [1,00; 1,35]"] = 1.0 <= tasa / PREV["prevalencia_test"] <= 1.35
    # umbral con precisión completa del registro (el de la señal está redondeado a 10 cifras: con la regla de tasa igualada
    # el umbral ES un puntaje del test y el redondeo podría cambiar el hogar del borde)
    umb_reg = (META.get(nombre, {}).get("params") or {}).get("umbral_test", umbral)
    te = TEST.get(nombre)
    avisos, empates = [], 0
    if te is not None and umb_reg is not None:
        ok["reproduce registro (puntaje ≥ umbral)"] = bool(((te >= umb_reg).astype(int) == csv.pobre.values).all())
        empates = int(np.isclose(te, umb_reg, rtol=0, atol=1e-12).sum())
    brecha = abs(tasa - q_hat) if q_hat is not None else np.inf
    if brecha > 0.003 + 1e-9 and brecha <= 0.003 + empates / len(csv):
        avisos.append(f"tasa a {coma(brecha, 4)} de q̂: {empates} hogares empatados en el umbral (puntajes discretos) impiden "
                      "igualarla; aceptable")
    else:
        ok["tasa coherente con q̂ OOF (±0,003)"] = brecha <= 0.003 + 1e-9
    fallos = [k for k, v in ok.items() if not v]
    out.update({"verificación": ("OK" if not fallos else "REVISAR: " + "; ".join(fallos)) + ("" if not avisos else " · AVISO: " + "; ".join(avisos)),
                "umbral (registro, precisión completa)": umb_reg, "empates en el umbral": empates, "tasa predicha CSV": tasa,
                "tasa / prevalencia esperada": tasa / PREV["prevalencia_test"], "md5": hashlib.md5(raw).hexdigest(),
                **{f"coincide con {n} (%)": 100 * float((csv.pobre.values == v).mean()) for n, v in _V5_PRED.items()}})
    return out


def fmt_hp(pref, L, hp):
    g = lambda k, d=2: coma(float(hp[k]), d) if k in hp and hp[k] is not None else "?"
    c = ALG[pref][1]
    if c == "LightGBM":
        arb = (L.get("diagnostico_umbral") or {}).get("arboles_finales")
        sem = (L.get("diagnostico_umbral") or {}).get("semillas_finales")
        mon = len(hp.get("monotonas") or {})
        return (f"hojas={hp.get('num_leaves')}; lr={g('learning_rate')}; {mil(arb) if arb else '?'} árboles × {sem} semillas; "
                f"min_child={hp.get('min_child_samples')}; feature_frac={g('feature_fraction')}; bagging={g('bagging_fraction')}; "
                f"L1={g('lambda_l1')}; L2={g('lambda_l2')}" + (f"; {mon} monótonas" if mon else ""))
    if c == "Logit":
        return "máxima verosimilitud (cresta 1e-5); selección lasso dentro del pliegue; splines en 5 continuas" if hp.get("splines") \
            else "máxima verosimilitud (cresta 1e-5); selección lasso dentro del pliegue"
    if c == "LPM":
        return "MCO" + ("; splines en 5 continuas" if hp.get("splines") else "")
    if c == "CART":
        return (f"criterio={hp.get('criterion')}; max_depth={hp.get('max_depth')}; min_leaf={hp.get('min_samples_leaf')}; "
                f"min_split={hp.get('min_samples_split')}; ccp_alpha={float(hp.get('ccp_alpha', 0)):.2e} (poda por CV)").replace(".", ",")
    if c == "Naive Bayes":
        return (f"{hp.get('tipo')}; {hp.get('n_bins')} intervalos por {hp.get('estrategia')}; suavizado alfa={g('alpha')}; "
                "selección hacia adelante por preguntas")
    return "; ".join(f"{k}={coma(v, 3) if isinstance(v, float) else v}" for k, v in list(hp.items())[:7]
                     if not isinstance(v, (dict, list)))


def descripcion(pref, L, ver="principal", ov=None):
    nombre_alg = {"LightGBM": "Boosting (LightGBM)", "XGBoost": "Boosting (XGBoost)", "CatBoost": "Boosting (CatBoost)",
                  "LPM": "Linear Probability Model"}.get(ALG[pref][1], ALG[pref][1])
    hp = L.get("hiperparametros") or {}
    if ver == "principal":
        nv, nq, lam, umb, regla = L["n_vars"], L["n_preguntas"], L["peso_bogota"], L["umbral"], L["regla_umbral"]
        f1, ic = L["F1_anidado_sinBOG"], L["IC95"]
        etiqueta = "v7"
    else:
        nv, nq, lam, umb = ov["n_vars"], ov["n_preguntas"], ov["lam_B"], ov["umbral"]
        fila = RES[RES["archivo CSV"] == ov["archivo"]].iloc[0]
        regla, f1, ic = fila["regla del umbral"], ov["F1_anidado_sinBOG"], [fila["IC95 inf"], fila["IC95 sup"]]
        etiqueta = "v7 " + {"encuesta_corta": "encuesta corta", "parsimoniosa": "parsimoniosa"}.get(ver, ver)
    regla_c = "tasa igualada" if "igualada" in str(regla) else ("fijo, tasa verificada" if "fijo" in str(regla) else str(regla))
    return (f"{nombre_alg} {etiqueta}, sin primera etapa. {fmt_hp(pref, L, hp)}. {mil(nv)} vars ({nq} preguntas); "
            f"λ_B={lam:g}; umbral {coma(umb, 3)} ({regla_c}). F1 CV sin Bogotá {coma(f1)} "
            f"[{coma(ic[0])}–{coma(ic[1])}]")


def fila_envio(pref, ver="principal"):
    L = LISTOS[pref]; corto = ALG[pref][1]
    if ver == "principal":
        nombre, arch, umb, tasa, q = corto, L["archivo"], L["umbral"], L["tasa_predicha_test"], L["tasa_predicha_oof_sinBOG"]
        f1, nv, nq = L["F1_anidado_sinBOG"], L["n_vars"], L["n_preguntas"]
        ov = None
    else:
        ov = next(o for o in L["otras_versiones"] if o["version"] == ver)
        nombre = f"{corto} · {ver.replace('_', ' ')}"
        arch, umb, tasa, f1, nv, nq = ov["archivo"], ov["umbral"], ov["tasa_predicha_test"], ov["F1_anidado_sinBOG"], ov["n_vars"], ov["n_preguntas"]
        # q̂ como en `final.umbral_y_tasa`: mediana bootstrap (B = 500, misma semilla) del umbral F1-óptimo en las OOF de 𝒯
        oo = OOF_R1[nombre][mT]
        q = float((oo >= MT.umbral_bootstrap(yT, oo, B=500)["umbral"]).mean())
    return {"algoritmo (enunciado)": ALG[pref][2], "modelo": nombre, "versión": ver, "F1 CV anidado": f1,
            "n_vars": nv, "n_preguntas": nq, "umbral": umb, "q̂ OOF": q,
            **verificar_csv(nombre, arch, umb, tasa, q), "descripción para Kaggle": descripcion(pref, L, ver, ov)}


# --- plan: hoy = el mejor permitido + los dos mejores algoritmos que faltan; luego cubrir todo y llegar a ≥ 15 / ≥ 5
pref_de = {ALG[p][1]: p for p in LISTOS}
faltan = sorted({ALG[p][2] for p in LISTOS} - set(ENVIOS_HECHOS), key=lambda a: -max(
    LISTOS[p]["F1_anidado_sinBOG"] for p in LISTOS if ALG[p][2] == a))
mejor_de_alg = {a: max((p for p in LISTOS if ALG[p][2] == a), key=lambda p: LISTOS[p]["F1_anidado_sinBOG"]) for a in faltan}
cola = [(pref_mejor, "principal", "mejor modelo permitido (mayor F1 CV); da la puntuación pública del «Best Model Overview»")]
cola += [(mejor_de_alg[a], "principal", f"cubre un algoritmo que falta ({a}); mejor versión por CV de ese algoritmo")
         for a in faltan]
PLAN = []
for i, (p, v, por_que) in enumerate(cola):
    PLAN.append({"día": DIAS_PLAN[min(i // ENVIOS_POR_DIA, len(DIAS_PLAN) - 1)], "orden": i + 1, "estado": "listo (CSV verificado)",
                 "justificación": por_que, **fila_envio(p, v)})
n_listos = len(PLAN)
algs_tras = set(ENVIOS_HECHOS) | {ALG[p][2] for p, _, _ in cola}
# pendientes y extras, en orden de prioridad
extras = []
SUSTITUTO = "si no está listo ese día: subir el siguiente envío «listo» de la lista y correr éste al día siguiente"
def _v7(p, nom, por_que):
    cola_ = [(c[0], c[1]) for c in cola]
    if p in LISTOS:
        if (p, "principal") not in cola_:
            extras.append(("listo", p, "principal", por_que))
    else:
        extras.append(("pendiente", p, None, f"{nom}: {por_que}. Subir cuando termine {ALG[p][0]} y se re-ejecute este cuaderno "
                                             f"(la verificación del CSV se hace aquí); {SUSTITUTO}"))
_v7("32", "Elastic Net v7 principal", "mejor versión v7 de Elastic Net (los 4 envíos EN previos son anteriores al protocolo v7)")
_v7("34", "Random Forest v7 principal", "mejor versión v7 de Random Forest (comparación por algoritmo en el mismo protocolo)")
if CORTA:
    extras.append(("listo", pref_mejor, "encuesta_corta", "versión de política: encuesta corta (25 variables, 19 preguntas) "
                                                         "del mejor algoritmo; mide en el público el costo de la parsimonia (§3.4)"))
_v7("36", "XGBoost v7 principal", "segundo motor de Boosting; candidato a mejor si entra en la ROPE de LightGBM")
_v7("37", "CatBoost v7 principal", "tercer motor de Boosting; candidato a mejor si entra en la ROPE de LightGBM")
for p in LISTOS:
    for o in LISTOS[p].get("otras_versiones", []):
        if o["version"] == "parsimoniosa":
            extras.append(("listo", p, "parsimoniosa", f"{ALG[p][1]} parsimonioso (regla 1-EE de la poda): menos variables"))
for p in sorted(LISTOS, key=lambda p: -LISTOS[p]["F1_anidado_sinBOG"]):
    if p != pref_mejor and any(o["version"] == "encuesta_corta" for o in LISTOS[p].get("otras_versiones", [])) and ALG[p][1] in ("Logit",):
        extras.append(("listo", p, "encuesta_corta", f"{ALG[p][1]} con la encuesta corta: modelo interpretable y barato (reserva)"))
for estado, p, v, por_que in extras:
    i = len(PLAN)
    base = {"día": DIAS_PLAN[min(i // ENVIOS_POR_DIA, len(DIAS_PLAN) - 1)], "orden": i + 1, "justificación": por_que}
    if estado == "listo":
        PLAN.append({**base, "estado": "listo (CSV verificado)", **fila_envio(p, v)})
    else:
        PLAN.append({**base, "estado": "pendiente (el cuaderno sigue corriendo o está en pausa)",
                     "algoritmo (enunciado)": ALG[p][2], "modelo": ALG[p][1], "versión": "principal"})
ult = DIAS_PLAN.index(PLAN[-1]["día"]) if PLAN else 0
for dia, txt in ((DIAS_PLAN[min(ult + 1, len(DIAS_PLAN) - 2)],
                  "reserva: re-ejecutar este cuaderno con todos los modelos terminados; si XGBoost o CatBoost superan a LightGBM "
                  "o quedan dentro de su ROPE, subir su versión parsimoniosa o la de encuesta corta; si no, ninguno"),
                 (DIAS_PLAN[-1], "sin envíos planeados (margen ante fallos del día anterior). En Kaggle, MARCAR a mano el envío "
                                 "final elegido por CV (mayor F1 CV permitido; dentro de la ROPE, el más parsimonioso): si no se marca, "
                                 "Kaggle toma el de mayor puntaje público, que hoy es un envío v5 con primera etapa (zona gris)")):
    PLAN.append({"día": dia, "orden": len(PLAN) + 1, "estado": "reserva / decisión", "justificación": txt})
PLAN = pd.DataFrame(PLAN)
PLAN.insert(3, "envío n.º del equipo", (PLAN["orden"] + sum(ENVIOS_HECHOS.values())).astype(float))
PLAN.loc[PLAN.estado.str.startswith("reserva"), "envío n.º del equipo"] = np.nan
hoy = PLAN[(PLAN["día"] == FECHA_PLAN) & PLAN.estado.str.startswith("listo")]
print(f"Envíos hechos: {sum(ENVIOS_HECHOS.values())} ({len(ENVIOS_HECHOS)} algoritmos). Con los {len(hoy)} de hoy: "
      f"{sum(ENVIOS_HECHOS.values()) + len(hoy)} envíos y "
      f"{len(set(ENVIOS_HECHOS) | set(hoy['algoritmo (enunciado)']))} algoritmos.")
display(PLAN[["día", "orden", "envío n.º del equipo", "estado", "modelo", "F1 CV anidado", "verificación"]])
for _, f in hoy.iterrows():
    print(f"\n[{f['modelo']}] {f['ruta']}\n  {f['descripción para Kaggle']}  ({len(f['descripción para Kaggle'])} caracteres)")

# %% [markdown]
# ## 8. Figuras de publicación (`02_outputs/figures/39_*.png`)
#
# Paleta categórica validada (azul, naranja, aguamarina; gris = referencia con primera etapa); ejes y rejilla tenues;
# identidad siempre en la etiqueta del eje además del color.

# %%
AZUL, NARANJA, AGUA, GRIS, TINTA, TINTA2, REJ = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985", "#0b0b0b", "#52514e", "#e6e5e1"
ROJO_D, AZUL_D, NEUTRO = "#c2410c", "#2a78d6", "#f0efec"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 220, "font.size": 9.5, "axes.titlesize": 11.5,
                     "axes.titleweight": "bold", "axes.labelsize": 10, "axes.edgecolor": "#b9b8b3", "axes.labelcolor": TINTA,
                     "xtick.color": TINTA2, "ytick.color": TINTA2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": REJ, "grid.linewidth": 0.6, "grid.linestyle": "-",
                     "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white"})
FIGURAS = []


from matplotlib.ticker import FuncFormatter
def _coma_fmt(dec):
    return FuncFormatter(lambda v, _: f"{v:.{dec}f}".replace("-", "−").replace(".", ","))


def ejes_coma(ax, x=True, y=False, dec=2):
    """Separador decimal en coma (convención en español) en los ejes numéricos."""
    if x:
        ax.xaxis.set_major_formatter(_coma_fmt(dec))
    if y:
        ax.yaxis.set_major_formatter(_coma_fmt(dec))


def guardar(fig, nombre):
    ruta = FIGS / f"39_{nombre}.png"
    fig.savefig(ruta, bbox_inches="tight", facecolor="white")
    plt.close(fig); FIGURAS.append(_rel(ruta))
    return ruta


def etiqueta_modelo(f):
    nq = f["n_preguntas"]
    nq = "" if nq is None or (isinstance(nq, float) and np.isnan(nq)) else f", {int(nq)} preg."
    return f"{f['modelo']}  ({mil(f['n_vars'])} vars{nq})"


# (1) F1 con IC 95 % de todos los modelos
R1 = RES.copy()
R1["serie"] = np.where(R1.grupo.str.startswith("v5"), "v5 con primera etapa (zona gris; referencia)",
                       np.where(R1["versión"] == "principal", "v7 principal",
                                np.where(R1["versión"] == "encuesta_corta", "v7 encuesta corta (25 vars, 19 preguntas)",
                                         "v7 parsimoniosa")))
COL_SERIE = {"v7 principal": AZUL, "v7 encuesta corta (25 vars, 19 preguntas)": NARANJA, "v7 parsimoniosa": AGUA,
             "v5 con primera etapa (zona gris; referencia)": GRIS}
R1 = R1.sort_values("F1").reset_index(drop=True)                    # de abajo hacia arriba: v5 abajo, mejor arriba
R1 = pd.concat([R1[R1.serie.str.startswith("v5")], R1[~R1.serie.str.startswith("v5")]], ignore_index=True)
fig, ax = plt.subplots(figsize=(8.6, 0.36 * len(R1) + 1.6))
yy = np.arange(len(R1))
for s, c in COL_SERIE.items():
    g = R1[R1.serie == s]
    if not len(g):
        continue
    ax.errorbar(g.F1, g.index, xerr=[g.F1 - g["IC95 inf"], g["IC95 sup"] - g.F1], fmt="o", ms=6.5, color=c,
                mfc=c if not s.startswith("v5") else "white", mec=c, elinewidth=1.6, capsize=0, label=s, zorder=3)
ax.set_yticks(yy); ax.set_yticklabels([etiqueta_modelo(f) for _, f in R1.iterrows()], fontsize=8.6)
fm = float(RES.loc[RES.modelo == MEJOR, "F1"].iloc[0])
ax.axvline(fm, color=AZUL, lw=0.8, alpha=0.6, zorder=1)
ax.axvspan(fm - MT.ROPE_F1, fm + MT.ROPE_F1, color=AZUL, alpha=0.07, lw=0, zorder=0)
n5 = int(R1.serie.str.startswith("v5").sum())
if n5:
    ax.axhline(n5 - 0.5, color="#b9b8b3", lw=0.8)
    ax.text(R1["IC95 inf"].min(), n5 - 0.62, "referencia: envíos v5 del 19 oct con primera etapa (zona gris)",
            fontsize=7.8, color=TINTA2, va="top")
ax.annotate(f"mejor permitido: {MEJOR} {coma(fm)}\n(franja: ROPE ± 0,002)", xy=(fm, len(R1) - 1), xytext=(fm - 0.006, len(R1) - 1.5),
            ha="right", va="center", fontsize=8, color=TINTA2, arrowprops=dict(arrowstyle="-", color=TINTA2, lw=0.6))
ax.set_xlabel("F1 de la clase pobre — agregado, umbral anidado, sólo hogares fuera de Bogotá (OOF, partición 1)")
ax.set_title("F1 fuera de muestra de todos los modelos (IC 95 % bootstrap de hogares)", loc="left")
ax.grid(axis="y", visible=False)
ax.set_ylim(-0.7, len(R1) - 0.3)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.09), fontsize=8.2, ncol=2)
ejes_coma(ax)
guardar(fig, "f1_modelos")

# (2) diferencias frente al mejor: t corregido (procedimiento) y bootstrap pareado (clasificadores ajustados)
if len(VS):
    V = VS.copy().iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.6, 0.62 * len(V) + 1.7))
    yy = np.arange(len(V))
    ax.axvspan(-MT.ROPE_F1, MT.ROPE_F1, color=GRIS, alpha=0.15, lw=0, label="ROPE ± 0,002 (equivalencia práctica)")
    ax.axvline(0, color=TINTA2, lw=0.8)
    d = -V["ΔF1"]; lo, hi = -V["ΔF1 IC95 sup"], -V["ΔF1 IC95 inf"]
    ax.errorbar(d, yy + 0.13, xerr=[d - lo, hi - d], fmt="o", ms=6.5, color=AZUL, elinewidth=1.8, capsize=0,
                label=f"t corregido de Nadeau–Bengio (IC 95 %, J = {int(V.J.max())} particiones × pliegues)", zorder=3)
    db = -V["ΔF1 r1 (hogares fijos)"]; blo, bhi = -V["bootstrap IC95 sup"], -V["bootstrap IC95 inf"]
    ax.errorbar(db, yy - 0.13, xerr=[db - blo, bhi - db], fmt="s", ms=5.5, color=NARANJA, elinewidth=1.4, capsize=0,
                label="bootstrap pareado de hogares (IC 95 %, partición 1)", zorder=3)
    for k, f in V.iterrows():
        txt = f"p Holm {'< 0,001' if f['ΔF1 p Holm (vs mejor)'] < 0.001 else coma(f['ΔF1 p Holm (vs mejor)'], 3)} · P(ROPE) {coma(f['P(ROPE)'], 2)}"
        ax.text(0.0025, k, txt, va="center", fontsize=7.8, color=TINTA2)
    ax.set_yticks(yy); ax.set_yticklabels([f"{b} − {a}" for a, b in zip(V.A, V.B)], fontsize=8.8)
    ax.set_xlabel(f"Diferencia de F1 frente al mejor ({MEJOR}); negativo = peor")
    ax.set_xlim(min(lo.min(), blo.min()) - 0.004, 0.0135)
    ax.set_title(f"¿Algún algoritmo se distingue de {MEJOR}? Diferencias pareadas en las mismas particiones", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13 - 0.02 * (6 - len(V))), fontsize=8, ncol=1)
    ejes_coma(ax)
    guardar(fig, "diferencias_vs_mejor")

# (3) matriz de diferencias pareadas (fila − columna) con significación de Holm
M = len(PRINC)
if M > 1:
    Dm = np.full((M, M), np.nan); Sm = np.full((M, M), "", dtype=object)
    for _, f in COMP.iterrows():
        i, j = PRINC.index(f.A), PRINC.index(f.B)
        Dm[i, j], Dm[j, i] = f["ΔF1"], -f["ΔF1"]
        s = "*" if f["ΔF1 p Holm (todos los pares)"] < 0.05 else ""
        Sm[i, j] = Sm[j, i] = s
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", [ROJO_D, NEUTRO, AZUL_D])
    lim = float(np.nanmax(np.abs(Dm)))
    fig, ax = plt.subplots(figsize=(1.05 * M + 2.6, 0.85 * M + 1.6))
    im = ax.imshow(1000 * Dm, cmap=cmap, norm=TwoSlopeNorm(0, -1000 * lim, 1000 * lim))
    for i in range(M):
        for j in range(M):
            if i != j:
                v = 1000 * Dm[i, j]
                ax.text(j, i, f"{coma(v, 1)}{Sm[i, j]}", ha="center", va="center", fontsize=8.6,
                        color="white" if abs(v) > 0.6 * 1000 * lim else TINTA)
            else:
                ax.text(j, i, "—", ha="center", va="center", color=TINTA2)
    ax.set_xticks(range(M)); ax.set_xticklabels(PRINC, rotation=30, ha="right"); ax.set_yticks(range(M)); ax.set_yticklabels(PRINC)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("ΔF1 fila − columna (milésimas)"); cb.outline.set_visible(False)
    ax.set_title("Diferencias pareadas de F1 entre modelos principales\n(* = distinta de cero con t corregido y Holm al 5 %)", loc="left")
    guardar(fig, "matriz_pareada")

# (4) igualdad de condiciones
if len(IG):
    algos = IG.groupby("modelo")["F1 selección propia (principal)"].first().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(8.6, 0.55 * len(algos) + 1.8))
    yy = np.arange(len(algos))
    series = [("selección propia (versión principal)", AZUL, "o", 0.18),
              ("todas (1.152 variables)", AGUA, "D", 0.0), ("encuesta corta (25 variables)", NARANJA, "s", -0.18)]
    for nom, c, mk, off in series:
        xs, es, ys = [], [], []
        for k, a in enumerate(algos):
            if nom.startswith("selección"):
                xs.append(np.mean(list(RK[a]["F1_r"].values()))); es.append(float(RES.loc[RES.modelo == a, "EE ψ"].iloc[0])); ys.append(k + off)
            else:
                g = IG[(IG["condición"] == nom) & (IG.modelo == a)]
                if len(g):
                    xs.append(g["F1 medio particiones"].iloc[0]); es.append(g["EE ψ (r1)"].iloc[0]); ys.append(k + off)
        lab = {"selección propia (versión principal)": "selección propia de cada algoritmo (versión principal)",
               "todas (1.152 variables)": "conjunto común: todas las variables v7 (1.152), λ_B = 1",
               "encuesta corta (25 variables)": "conjunto común: encuesta corta (25 variables, 19 preguntas), λ_B = 1"}[nom]
        ax.errorbar(xs, ys, xerr=1.96 * np.asarray(es), fmt=mk, ms=6, color=c, elinewidth=1.4, capsize=0, label=lab, zorder=3)
    ax.set_yticks(yy); ax.set_yticklabels(algos)
    ax.set_xlabel("F1 agregado anidado fuera de Bogotá (media de las particiones comunes; barras: IC 95 % por ψ)")
    ax.set_title("Igualdad de condiciones: mismas variables, mismo trato de Bogotá, mismas particiones", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14 - 0.03 * (6 - len(algos))), fontsize=8)
    ejes_coma(ax)
    guardar(fig, "igualdad_condiciones")

# (5) disyuntiva de política: filtración vs subcobertura por cada 1.000 hogares
fig, ax = plt.subplots(figsize=(7.6, 5.4))
marc = {"F1-óptimo": "o", "cobertura del 90 % de los pobres": "s", "tasa predicha = prevalencia": "^"}
for nombre, c in [(MEJOR, AZUL)] + ([(CORTA, NARANJA)] if CORTA else []):
    s = OOF_R1[nombre][mT]
    thr, tp, fp, P, f = MT.curva_f1(yT, s)
    fn = P - tp
    ok = (tp + fp) / nT <= 0.6
    nq = RES.loc[RES.modelo == nombre, "n_preguntas"].iloc[0]
    ax.plot(1000 * fn[ok] / nT, 1000 * fp[ok] / nT, color=c, lw=2,
            label=f"{nombre} ({mil(RES.loc[RES.modelo == nombre, 'n_vars'].iloc[0])} variables, {int(nq)} preguntas)")
    for _, g in POL[(POL.modelo == nombre) & POL.panel.str.startswith("mejor")].iterrows():
        x, yv = g["subcobertura: pobres excluidos (FN) por 1.000"], g["filtración: no pobres incluidos (FP) por 1.000"]
        ax.plot(x, yv, marc[g.criterio], ms=8.5, color=c, mec="white", mew=1.6, zorder=4)
for crit, mk in marc.items():
    ax.plot([], [], mk, color=TINTA2, ms=7, label=f"umbral: {crit}")
pi_ = 1000 * yT.mean()
ax.set_xlabel(f"Subcobertura: hogares pobres excluidos por cada 1.000 hogares (de {coma(pi_, 0)} pobres)")
ax.set_ylabel("Filtración: hogares no pobres incluidos por cada 1.000 hogares")
mk = POL[POL.panel.str.startswith("mejor")]
ax.set_xlim(0, max(120, 1.25 * mk["subcobertura: pobres excluidos (FN) por 1.000"].max()))
ax.set_ylim(0, 1.12 * mk["filtración: no pobres incluidos (FP) por 1.000"].max())
ejes_coma(ax, dec=0)
ax.set_title("Disyuntiva de focalización: cada punto de la curva es un umbral\n(OOF fuera de Bogotá; marcadores con umbral anidado)", loc="left")
ax.legend(loc="upper right", fontsize=8)
guardar(fig, "disyuntiva_politica")
print("Figuras:", *FIGURAS, sep="\n  ")

# %% [markdown]
# ## 9. Tablas, libro de Excel y señal

# %%
RES.to_csv(TABLAS / "39_resumen.csv", index=False)
COMP.to_csv(TABLAS / "39_comparaciones.csv", index=False)
VS.to_csv(TABLAS / "39_vs_mejor.csv", index=False)
IG.to_csv(TABLAS / "39_igualdad.csv", index=False)
IGC.to_csv(TABLAS / "39_igualdad_comparaciones.csv", index=False)
POL.to_csv(TABLAS / "39_politica.csv", index=False)
PLAN.to_csv(TABLAS / "39_envios.csv", index=False)
CVCT = pd.DataFrame([{"criterio": c, "modelo": n, "p": v["p"][n], "en el conjunto (α = 0,10)": n in v["conjunto"]}
                     for c, v in CVC.items() for n in PRINC])
CVCT.to_csv(TABLAS / "39_cvc.csv", index=False)
if len(V5COMP):
    V5COMP.to_csv(TABLAS / "39_v7_vs_v5.csv", index=False)

pref_sorted = sorted(LISTOS)
LEEME = [
    f"Resultados de los modelos v7 (protocolo sin primera etapa) — generado {time.strftime('%Y-%m-%d %H:%M')} por 01_code/python/39_comparacion_v7.py.",
    f"Modelos terminados: {', '.join(ALG[p][1] for p in pref_sorted)}. Pendientes: {', '.join(PENDIENTES) or 'ninguno'} (re-ejecutar el cuaderno 39 cuando terminen).",
    "Estimando: F1 de la clase pobre en los 24 dominios del test (sin Bogotá). F1 = agregado (TP, FP, FN sumados en los pliegues) con umbral anidado (cada pliegue usa el umbral elegido en los otros). Particiones comunes: pliegues_v7.csv (r1 = pliegues.csv; r2, r3 semillas 20263–20264; r4 = 20265, re-evaluación).",
    "Hoja Resumen: métricas de la partición 1 recalculadas con las OOF del predictor final (registro común) y verificadas contra las señales 3x_LISTO.json; F1 r2/r3 = configuración principal en la caché del protocolo (para boosting, a la tasa de búsqueda 0,05); F1 r4 = re-evaluación en partición nueva. IC95 = bootstrap percentil de hogares (B = 2.000); EE ψ = función de influencia.",
    "Filas «v5 con primera etapa (zona gris)»: los tres envíos del 19 oct (LightGBM 8 vars, XGBoost 8 vars, RF 20 vars; público 0,77). Usan predicciones de un modelo de ingreso (primera etapa), prohibido por la regla 1 del usuario; se muestran sólo como referencia.",
    "Hoja Comparaciones: todos los pares de modelos principales. ΔF1 = F1(A) − F1(B) con A el de mayor F1. t corregido de Nadeau–Bengio (factor 1/J + 1/(K−1), J − 1 g.l.), prueba bayesiana correlacionada (ROPE 0,002 en F1; 0,0005 log-loss; 0,0002 Brier; ρ = 1/K), Holm sobre todos los pares, bootstrap pareado conjunto de hogares (partición 1) y EE ψ de la diferencia.",
    "CVC (Lei, 2020) al 90 %: conjunto de modelos que no se rechazan como «no peores que ninguno»; para F1 con pseudo-valores de la linealización (−F̂ − ψ_i), además log-loss y Brier por hogar. P(mejor) = fracción de remuestreos bootstrap en que el modelo tiene el mayor F1.",
    "Hoja Igualdad: el mismo algoritmo con sus hiperparámetros elegidos, evaluado con conjuntos COMUNES (todas las 1.152 variables v7; encuesta corta de 25 variables / 19 preguntas), λ_B = 1 y las mismas particiones. Δ selección propia − común mide cuánto aporta la selección de variables de cada algoritmo. Hiperparámetros no re-ajustados al conjunto común (declarado).",
    "Hoja Política: filtración (FP: no pobres incluidos) y subcobertura (FN: pobres excluidos) por cada 1.000 hogares fuera de Bogotá, con tres reglas de umbral anidadas: F1-óptimo, cobertura del 90 % de los pobres y tasa predicha = prevalencia. Razón implícita c_FN/c_FP = (1 − t)/t si las probabilidades están calibradas (ver ACE).",
    f"Hoja Envíos: propuesta (NO subida) para cumplir ≥ 15 envíos con ≥ 5 algoritmos; estado de partida informado: {sum(ENVIOS_HECHOS.values())} envíos ({', '.join(f'{k} ×{v}' for k, v in ENVIOS_HECHOS.items())}); 3 envíos por día hasta el 25 oct. Cada CSV verificado (columnas, 66.168 filas, ids y orden de sample_submission del zip, 0/1, tasa, reproducción exacta desde el registro, md5). La descripción para Kaggle está lista para pegar.",
    "Elección del envío final: por F1 de CV (nunca por el tablero público, EE ≈ 0,006); dentro de la ROPE, el más parsimonioso. Hay que MARCARLO a mano en Kaggle: si no, Kaggle toma el de mayor puntaje público, que hoy es un envío v5 con primera etapa (zona gris). Nada se sube a Kaggle desde aquí.",
    f"Costo de la regla «sin primera etapa»: el mejor v7 ({MEJOR}) queda {coma(-float(V5COMP['ΔF1 r1'].mean()), 4) if len(V5COMP) else '?'} de F1 por debajo de los envíos v5 (bootstrap pareado, partición 1; hoja Comparaciones, filas de referencia).",
    "Figuras: 02_outputs/figures/39_f1_modelos.png, 39_diferencias_vs_mejor.png, 39_matriz_pareada.png, 39_igualdad_condiciones.png, 39_disyuntiva_politica.png.",
]


def escribir_excel(ruta, hojas, notas):
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(ruta, engine="openpyxl") as xw:
        pd.DataFrame({"Léeme": notas}).to_excel(xw, sheet_name="Léeme", index=False)
        for nom, (df, congelar) in hojas.items():
            df.to_excel(xw, sheet_name=nom, index=False)
        wb = xw.book
        azul = PatternFill("solid", fgColor="1F3864")
        for ws in wb.worksheets:
            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF"); c.fill = azul
                c.alignment = Alignment(wrap_text=True, vertical="center")
            ws.row_dimensions[1].height = 48
            if ws.title == "Léeme":
                ws.column_dimensions["A"].width = 170; ws.freeze_panes = "A2"
                for i in range(2, ws.max_row + 1):
                    ws.cell(i, 1).alignment = Alignment(wrap_text=True, vertical="top")
                continue
            ws.freeze_panes = hojas[ws.title][1]
            enc = [c.value for c in ws[1]]
            for j, h in enumerate(enc, start=1):
                L_ = get_column_letter(j)
                vals = [ws.cell(i, j).value for i in range(2, ws.max_row + 1)]
                txt = [len(str(v)) for v in vals if v is not None]
                larga = str(h).startswith(("descripción", "justificación", "ruta", "verificación"))
                ws.column_dimensions[L_].width = 70 if larga else min(max(10, len(str(h)) * 0.55, max(txt or [8]) * 0.95 + 1), 42)
                num = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
                if num and any(isinstance(v, float) for v in num):
                    hs = str(h)
                    fmtn = ("0.0E+00" if (hs.endswith(" p") or " p Holm" in hs or hs.startswith("p ") or hs.endswith(": p")) else
                            "0.000" if "P(" in hs else
                            "0.0" if ("por 1.000" in hs or "(%)" in hs or "minutos" in hs or "(min)" in hs) else "0.0000")
                    for i in range(2, ws.max_row + 1):
                        if isinstance(ws.cell(i, j).value, float):
                            ws.cell(i, j).number_format = fmtn
                for i in range(2, ws.max_row + 1):
                    ws.cell(i, j).alignment = Alignment(wrap_text=larga, vertical="top")
                hs = str(h)
                if (hs.startswith("F1") or hs.startswith("AUC")) and num and ws.max_row > 2:
                    ws.conditional_formatting.add(f"{L_}2:{L_}{ws.max_row}", ColorScaleRule(
                        start_type="min", start_color="F8696B", mid_type="percentile", mid_value=50, mid_color="FFEB84",
                        end_type="max", end_color="63BE7B"))
                elif hs in ("ΔF1", "ΔF1 r1 (hogares fijos)", "ΔF1 vs mejor (t corregido)", "ΔF1 mejor − modelo (t corregido)") \
                        and num and ws.max_row > 2:                    # tamaño de la diferencia: escala de un solo tono
                    ws.conditional_formatting.add(f"{L_}2:{L_}{ws.max_row}", ColorScaleRule(
                        start_type="min", start_color="FFFFFF", end_type="max", end_color="5B9BD5"))
    return ruta


cols_res = ["grupo", "modelo", "algoritmo", "versión", "F1", "IC95 inf", "IC95 sup", "EE ψ", "F1 r2", "F1 r3",
            "F1 r4 (re-evaluación)", "F1 post-estratificado", "log-loss", "Brier", "ACE", "AUC", "AUC-PR", "precisión",
            "exhaustividad", "MCC", "tasa predicha OOF", "tasa predicha test", "umbral", "regla del umbral", "n_vars",
            "n_preguntas", "λ_B (peso de Bogotá)", "tiempo (min)", "P(mejor) bootstrap", "vs mejor",
            "ΔF1 vs mejor (t corregido)", "p Holm vs mejor"] + [c for c in RES.columns if c.startswith("CVC")] + \
           ["Kaggle público", "archivo CSV", "cuaderno", "slug registro"]
cols_res = [c for c in cols_res if c in RES.columns]
ENV_COLS = ["día", "orden", "envío n.º del equipo", "estado", "algoritmo (enunciado)", "modelo", "versión", "F1 CV anidado",
            "n_vars", "n_preguntas", "umbral", "umbral (registro, precisión completa)", "empates en el umbral", "q̂ OOF", "tasa predicha CSV", "tasa / prevalencia esperada", "verificación",
            "descripción para Kaggle", "justificación", "archivo", "ruta", "md5"] + [c for c in PLAN.columns if c.startswith("coincide")]
ENV_COLS = [c for c in ENV_COLS if c in PLAN.columns]
COMP_X = COMP.copy()
if len(V5COMP):
    COMP_X = pd.concat([COMP_X, V5COMP.rename(columns={"ΔF1 r1": "ΔF1 r1 (hogares fijos)"}).assign(
        J=None, particiones="referencia v5 (sólo bootstrap, partición 1)")], ignore_index=True)
hojas = {"Resumen": (RES[cols_res], "C2"), "Comparaciones": (COMP_X, "C2"), "Igualdad": (IG, "C2"),
         "Política": (POL, "C2"), "Envíos": (PLAN[ENV_COLS], "G2")}
escribir_excel(XLSX, hojas, LEEME)
print("Excel:", XLSX)

resumen = {"generado": time.strftime("%Y-%m-%d %H:%M"), "modelos": PRINC, "mejor": MEJOR, "pendientes": PENDIENTES,
           "F1": {n: float(RES.loc[RES.modelo == n, "F1"].iloc[0]) for n in PRINC},
           "P_mejor_bootstrap": P_MEJOR, "indistinguibles_del_mejor_Holm": INDIST,
           "CVC90": {c: v["conjunto"] for c, v in CVC.items()},
           "envios_hoy": hoy[["modelo", "ruta", "verificación", "descripción para Kaggle"]].to_dict("records"),
           "figuras": FIGURAS, "excel": _rel(XLSX), "minutos": (time.time() - T0) / 60}
(TABLAS / "39_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(f"Listo en {(time.time() - T0) / 60:.1f} min")
