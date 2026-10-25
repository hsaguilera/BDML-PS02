"""XGBoost (regresión) para pobreza con ingeniería de variables: "regress-then-threshold".

En lugar de clasificar, se predice el ingreso per cápita relativo a la línea de pobreza,
z = log(Ingpcug / Lp) recortado a [log 0,05; log 20], con el ingreso que SOLO existe en
train; el hogar se clasifica como pobre si z_hat < tau (z < 0 es la definición de pobreza).
El umbral tau se elige para maximizar F1 en OOF. Ventajas: usa la información continua
(qué tan lejos está el hogar de la línea) y no sólo el 0/1; ver Brown, Ravallion y van de
Walle (2018) para la comparación con la clasificación directa.

XGBoost: boosting de gradiente con regularización L1/L2 explícita y poda por `gamma`,
árboles por niveles (profundidad máxima), histogramas.
Protocolo: búsqueda aleatoria (lr alto + parada temprana) y reentrenamiento con lr bajo y
rondas fijas, con los 5 pliegues compartidos; test = promedio de los 5 modelos.
Uso: python harold_xgboost_reg.py <hogares.csv> <scores.csv> <busqueda.csv> <params.json>
La puntuación guardada es s = -z_hat (mayor = más probable ser pobre).
"""
import sys, json, os, time
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score

f_data, f_scores, f_search, f_params = sys.argv[1:5]
NJOBS = int(os.environ.get("PY_THREADS", "2"))
SEED = 2026
df = pd.read_csv(f_data)
feats = pd.read_csv(f_data.replace(".csv", "_vars.csv"))["variable"].tolist()
for c in ["Dominio", "tenure"]:
    if c in feats:
        df[c] = df[c].astype("category")
tr = df[df["sample"] == "train"].reset_index(drop=True)
te = df[df["sample"] == "test"].reset_index(drop=True)
y, z = tr["poor"].values, tr["z"].values
nb = (tr["bogota"] == 0).values
print("train", tr.shape, "test", te.shape, "vars", len(feats), flush=True)
dte = xgb.DMatrix(te[feats], enable_categorical=True, nthread=NJOBS)

def best_f1(yy, s):
    ts = np.arange(-0.8, 0.8, 0.01)
    f = np.array([f1_score(yy, s >= t) for t in ts])
    return float(ts[f.argmax()]), float(f.max())

def cv_run(params, lr, rounds=None, es=True, predict_test=False):
    s_oof = np.zeros(len(tr)); s_te = np.zeros(len(te)); iters = []
    for k in range(1, 6):
        itr, iva = np.where(tr["fold"] != k)[0], np.where(tr["fold"] == k)[0]
        dtr = xgb.DMatrix(tr.loc[itr, feats], z[itr], enable_categorical=True, nthread=NJOBS)
        dva = xgb.DMatrix(tr.loc[iva, feats], z[iva], enable_categorical=True, nthread=NJOBS)
        P = dict(tree_method="hist", learning_rate=lr, nthread=NJOBS, seed=SEED, max_cat_to_onehot=1, **params)
        if es:
            m = xgb.train(P, dtr, 3000, evals=[(dva, "va")], early_stopping_rounds=40, verbose_eval=False)
            iters.append(m.best_iteration + 1)
            s_oof[iva] = -m.predict(dva, iteration_range=(0, m.best_iteration + 1))
        else:
            m = xgb.train(P, dtr, rounds, verbose_eval=False)
            s_oof[iva] = -m.predict(dva)
        if predict_test:
            s_te += -m.predict(dte) / 5
    return s_oof, s_te, iters

rng = np.random.default_rng(SEED)
space = dict(max_depth=[4, 6, 8, 10], min_child_weight=[1, 5, 20, 50], subsample=[0.7, 0.85, 1.0],
             colsample_bytree=[0.4, 0.6, 0.8, 1.0], reg_lambda=[1.0, 10.0, 50.0], reg_alpha=[0.0, 0.5, 2.0],
             gamma=[0.0, 0.1, 0.5], objective=["reg:squarederror", "reg:pseudohubererror"])
configs = [{k: v[int(rng.integers(len(v)))] for k, v in space.items()} for _ in range(8)]
configs.insert(0, dict(max_depth=6, min_child_weight=5, subsample=0.85, colsample_bytree=0.7,
                       reg_lambda=10.0, reg_alpha=0.0, gamma=0.0, objective="reg:squarederror"))

rows = []
for i, cfg in enumerate(configs):
    t0 = time.time()
    s, _, iters = cv_run(cfg, lr=0.08)
    t, f1 = best_f1(y[nb], s[nb])
    rows.append(dict(config=i, **cfg, rondas_media=float(np.mean(iters)), umbral=t, f1=f1,
                     auc=float(roc_auc_score(y[nb], s[nb])), rmse_z=float(np.sqrt(np.mean((-s[nb] - z[nb]) ** 2))),
                     segundos=time.time() - t0))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(f_search, index=False)

S = pd.DataFrame(rows)
best = S.loc[S["f1"].idxmax()]
cfg = {k: best[k] for k in space}
for k in ("max_depth", "min_child_weight"):
    cfg[k] = int(cfg[k])
for k in ("subsample", "colsample_bytree", "reg_lambda", "reg_alpha", "gamma"):
    cfg[k] = float(cfg[k])
LR_FINAL = 0.03
rounds = int(round(best["rondas_media"] * 0.08 / LR_FINAL))
print("mejor configuración:", cfg, "rondas finales:", rounds, flush=True)
s_oof, s_te, _ = cv_run(cfg, lr=LR_FINAL, rounds=rounds, es=False, predict_test=True)

out = pd.concat([pd.DataFrame({"id": tr["id"], "sample": "train", "fold": tr["fold"], "score": s_oof}),
                 pd.DataFrame({"id": te["id"], "sample": "test", "fold": 0, "score": s_te})])
out.to_csv(f_scores, index=False)
json.dump(dict(params=cfg, learning_rate=LR_FINAL, rondas=rounds, n_vars=len(feats)), open(f_params, "w"))
print("listo", flush=True)
