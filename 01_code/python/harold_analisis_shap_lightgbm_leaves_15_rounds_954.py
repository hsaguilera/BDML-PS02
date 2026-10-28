"""Importancia de variables del LightGBM final (interpretación, no genera envíos).

Reentrena el LightGBM con los hiperparámetros elegidos en `harold_modelo_lightgbm_fe_leaves_15_rounds_954.py` sobre TODO el
train, y calcula (i) la importancia por ganancia y (ii) los valores SHAP (contribuciones por
árbol de LightGBM, `pred_contrib`, en escala log-odds) sobre una muestra aleatoria de hogares.
Uso: python harold_analisis_shap_lightgbm_leaves_15_rounds_954.py <hogares.csv> <params.json> <salida_importancia.csv> <salida_shap_muestra.csv>
"""
import sys, json, os
import numpy as np, pandas as pd, lightgbm as lgb

f_data, f_params, f_imp, f_shap = sys.argv[1:5]
SEED, N_SHAP, N_TOP, N_PTS = 2026, 15000, 6, 4000
prm = json.load(open(f_params))
df = pd.read_csv(f_data)
feats = pd.read_csv(f_data.replace(".csv", "_vars.csv"))["variable"].tolist()
for c in ["Dominio", "tenure"]:
    if c in feats:
        df[c] = df[c].astype("category")
tr = df[df["sample"] == "train"].reset_index(drop=True)

P = dict(objective="binary", learning_rate=prm["learning_rate"], n_jobs=int(os.environ.get("PY_THREADS", "2")),
         verbose=-1, seed=SEED, bagging_freq=1, **prm["params"])
m = lgb.train(P, lgb.Dataset(tr[feats], tr["poor"].values), prm["rondas"])

rng = np.random.default_rng(SEED)
idx = rng.choice(len(tr), size=min(N_SHAP, len(tr)), replace=False)
X = tr.loc[idx, feats]
contrib = m.predict(X, pred_contrib=True)[:, :-1]            # última columna = valor base
imp = pd.DataFrame({"variable": feats,
                    "shap_media_abs": np.abs(contrib).mean(axis=0),
                    "ganancia": m.feature_importance("gain")})
imp["ganancia"] = imp["ganancia"] / imp["ganancia"].sum()
imp = imp.sort_values("shap_media_abs", ascending=False)
imp.to_csv(f_imp, index=False)

top = imp["variable"].head(N_TOP).tolist()
sel = rng.choice(len(X), size=min(N_PTS, len(X)), replace=False)
rows = []
for v in top:
    j = feats.index(v)
    xv = X[v].iloc[sel]
    rows.append(pd.DataFrame({"variable": v, "valor": pd.to_numeric(xv, errors="coerce").values,
                              "shap": contrib[sel, j]}))
pd.concat(rows).to_csv(f_shap, index=False)
print(imp.head(10).to_string(index=False), flush=True)
