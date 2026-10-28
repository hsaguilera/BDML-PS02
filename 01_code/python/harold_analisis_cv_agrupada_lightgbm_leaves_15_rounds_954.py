"""Prueba de estrés para LightGBM: validación cruzada AGRUPADA por dominio geográfico.

Cada pliegue deja fuera ciudades/dominios completos (columna `gfold`), así que el modelo se
evalúa en regiones que no vio, una situación más dura que los pliegues aleatorios. Usa los
hiperparámetros del modelo final (json del script 09).
Uso: python harold_analisis_cv_agrupada_lightgbm_leaves_15_rounds_954.py <hogares.csv> <params.json> <salida_scores.csv>
"""
import sys, os, json
import numpy as np, pandas as pd, lightgbm as lgb

f_data, f_par, f_out = sys.argv[1:4]
NJOBS = int(os.environ.get("PY_THREADS", "2"))
df = pd.read_csv(f_data)
feats = pd.read_csv(f_data.replace(".csv", "_vars.csv"))["variable"].tolist()
for c in ["Dominio", "tenure"]:
    if c in feats:
        df[c] = df[c].astype("category")
par = json.load(open(f_par))
tr = df[df["sample"] == "train"].reset_index(drop=True)
P = dict(objective="binary", learning_rate=par["learning_rate"], n_jobs=NJOBS, verbose=-1, seed=2026,
         bagging_freq=1, **par["params"])
score = np.zeros(len(tr))
for k in sorted(tr["gfold"].unique()):
    itr, iva = np.where(tr["gfold"] != k)[0], np.where(tr["gfold"] == k)[0]
    m = lgb.train(P, lgb.Dataset(tr.loc[itr, feats], tr.loc[itr, "poor"].values), par["rondas"])
    score[iva] = m.predict(tr.loc[iva, feats])
    print("pliegue", k, "listo", flush=True)
pd.DataFrame({"id": tr["id"], "gfold": tr["gfold"], "score": score}).to_csv(f_out, index=False)
