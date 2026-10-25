"""LightGBM (clasificación) para pobreza con ingeniería de variables.

Boosting de árboles con crecimiento por hojas, binning por histogramas y manejo nativo de
categóricas y faltantes; destaca en datos tabulares con relaciones no lineales e
interacciones, que es justo lo que se espera de las variables de ingreso imputado.

Protocolo (idéntico al de los demás modelos):
  1) búsqueda aleatoria de hiperparámetros con validación cruzada de 5 pliegues compartidos
     (columna `fold`), lr alto y parada temprana; se elige por F1 con umbral óptimo en OOF
     (solo hogares fuera de Bogotá);
  2) con la mejor configuración se reentrena con lr bajo y número de árboles fijo (sin parada
     temprana, para que el OOF no se vea favorecido) y se promedian los 5 modelos en test.
Uso: python harold_lightgbm.py <hogares.csv> <salida_scores.csv> <salida_busqueda.csv> <salida_params.json>
"""
import sys, json, os, time
import numpy as np, pandas as pd, lightgbm as lgb
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
y = tr["poor"].values
nb = (tr["bogota"] == 0).values
print("train", tr.shape, "test", te.shape, "vars", len(feats), flush=True)

def best_f1(yy, p):
    ts = np.arange(0.10, 0.80, 0.01)
    f = np.array([f1_score(yy, p >= t) for t in ts])
    return float(ts[f.argmax()]), float(f.max())

def cv_run(params, lr, rounds=None, es=True, predict_test=False):
    oof = np.zeros(len(tr)); pte = np.zeros(len(te)); iters = []
    for k in range(1, 6):
        itr, iva = np.where(tr["fold"] != k)[0], np.where(tr["fold"] == k)[0]
        dtr = lgb.Dataset(tr.loc[itr, feats], y[itr])
        P = dict(objective="binary", learning_rate=lr, n_jobs=NJOBS, verbose=-1, seed=SEED,
                 bagging_freq=1, **params)
        if es:
            dva = lgb.Dataset(tr.loc[iva, feats], y[iva])
            m = lgb.train(P, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(40, verbose=False)])
            iters.append(m.best_iteration)
        else:
            m = lgb.train(P, dtr, rounds)
        oof[iva] = m.predict(tr.loc[iva, feats])
        if predict_test:
            pte += m.predict(te[feats]) / 5
    return oof, pte, iters

rng = np.random.default_rng(SEED)
space = dict(num_leaves=[15, 31, 63, 127], min_child_samples=[20, 50, 100, 200],
             feature_fraction=[0.4, 0.6, 0.8, 1.0], bagging_fraction=[0.7, 0.85, 1.0],
             lambda_l2=[0.0, 5.0, 20.0, 50.0], scale_pos_weight=[1.0, 1.5, 2.5],
             max_depth=[-1, 6, 8])
configs = [{k: v[int(rng.integers(len(v)))] for k, v in space.items()} for _ in range(10)]
configs.insert(0, dict(num_leaves=31, min_child_samples=50, feature_fraction=0.7, bagging_fraction=0.85,
                       lambda_l2=5.0, scale_pos_weight=1.0, max_depth=-1))   # configuración de referencia

rows = []
for i, cfg in enumerate(configs):
    t0 = time.time()
    oof, _, iters = cv_run(cfg, lr=0.08)
    t, f1 = best_f1(y[nb], oof[nb])
    p = np.clip(oof, 1e-12, 1 - 1e-12)
    ll = float(-np.mean(y[nb] * np.log(p[nb]) + (1 - y[nb]) * np.log(1 - p[nb])))
    rows.append(dict(config=i, **cfg, rondas_media=float(np.mean(iters)), umbral=t, f1=f1,
                     auc=float(roc_auc_score(y[nb], oof[nb])), logloss=ll, segundos=time.time() - t0))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(f_search, index=False)

S = pd.DataFrame(rows)
best = S.loc[S["f1"].idxmax()]
cfg = {k: best[k] for k in space}
for k in ("num_leaves", "min_child_samples", "max_depth"):
    cfg[k] = int(cfg[k])
for k in ("feature_fraction", "bagging_fraction", "lambda_l2", "scale_pos_weight"):
    cfg[k] = float(cfg[k])
LR_FINAL = 0.03
rounds = int(round(best["rondas_media"] * 0.08 / LR_FINAL))
print("mejor configuración:", cfg, "rondas finales:", rounds, flush=True)
oof, pte, _ = cv_run(cfg, lr=LR_FINAL, rounds=rounds, es=False, predict_test=True)

out = pd.concat([pd.DataFrame({"id": tr["id"], "sample": "train", "fold": tr["fold"], "score": oof}),
                 pd.DataFrame({"id": te["id"], "sample": "test", "fold": 0, "score": pte})])
out.to_csv(f_scores, index=False)
json.dump(dict(params=cfg, learning_rate=LR_FINAL, rondas=rounds, n_vars=len(feats)), open(f_params, "w"))
print("listo", flush=True)
