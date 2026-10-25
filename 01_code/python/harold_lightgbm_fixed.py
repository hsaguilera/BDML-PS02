"""LightGBM con hiperparámetros fijos (los elegidos en `harold_lightgbm.py`) sobre cualquier conjunto de variables.

Sirve para comparar algoritmos con los MISMOS predictores y para medir cuánto se pierde con encuestas
más cortas: no hay búsqueda, solo 5 pliegues compartidos (`fold`), el número de rondas del modelo final y
el promedio de los 5 modelos sobre test.
Uso: python harold_lightgbm_fixed.py <hogares.csv> <params.json> <salida_scores.csv>
"""
import sys, json, os
import numpy as np, pandas as pd, lightgbm as lgb

f_data, f_params, f_scores = sys.argv[1:4]
SEED = 2026
prm = json.load(open(f_params))
df = pd.read_csv(f_data)
feats = pd.read_csv(f_data.replace(".csv", "_vars.csv"))["variable"].tolist()
for c in ["Dominio", "tenure"]:
    if c in feats:
        df[c] = df[c].astype("category")
tr = df[df["sample"] == "train"].reset_index(drop=True)
te = df[df["sample"] == "test"].reset_index(drop=True)
y = tr["poor"].values
P = dict(objective="binary", learning_rate=prm["learning_rate"], n_jobs=int(os.environ.get("PY_THREADS", "2")),
         verbose=-1, seed=SEED, bagging_freq=1, **prm["params"])
oof, pte = np.zeros(len(tr)), np.zeros(len(te))
for k in range(1, 6):
    itr, iva = np.where(tr["fold"] != k)[0], np.where(tr["fold"] == k)[0]
    m = lgb.train(P, lgb.Dataset(tr.loc[itr, feats], y[itr]), prm["rondas"])
    oof[iva] = m.predict(tr.loc[iva, feats])
    pte += m.predict(te[feats]) / 5
    print("pliegue", k, flush=True)
out = pd.concat([pd.DataFrame({"id": tr["id"], "sample": "train", "fold": tr["fold"], "score": oof}),
                 pd.DataFrame({"id": te["id"], "sample": "test", "fold": 0, "score": pte})])
out.to_csv(f_scores, index=False)
print("listo", len(feats), "variables", flush=True)
