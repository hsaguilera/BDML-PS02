"""Modelo auxiliar de ingreso individual (LightGBM), parte de la ingeniería de variables.

Idea: el ingreso del hogar es la suma del ingreso de sus personas (Ingtot). En test no
hay ingresos, pero sí las respuestas de la GEIH sobre cada persona (ocupación, posición
ocupacional, tamaño de la empresa, si recibe primas, subsidios, pensión, remesas, horas,
educación, etc.). Se entrena en train un modelo que predice el ingreso individual con
esas variables y se usa su predicción (fuera de muestra) como variable del hogar.

Para evitar fuga: pliegues por hogar (los mismos `fold` de los modelos del hogar). Cada
persona de train se predice con el modelo que NO vio su hogar; las de test se predicen con
el promedio de los 5 modelos.

Tres cabezas:  p_pos   = P(Ingtot > 0)                      (clasificación)
               mu_log  = E[log(Ingtot) | Ingtot > 0]        (regresión)
               mean_raw = E[Ingtot]                          (regresión L2, ingreso recortado)
Uso: python harold_person_income.py <entrada.csv> <salida.csv>
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

SEED = 2026
f_in, f_out = sys.argv[1], sys.argv[2]
df = pd.read_csv(f_in)
print("personas:", df.shape, flush=True)

cat_cols = ["Dominio", "Depto", "Oficio", "P6210", "P6240", "P6430", "P6050", "P6870", "P6920", "P5090"]
for c in cat_cols:
    df[c] = df[c].astype("category")

drop = {"id", "Orden", "sample", "fold", "Ingtot"}
feats = [c for c in df.columns if c not in drop]
# Contexto del hogar: la línea de pobreza regional (Lp) identifica el costo de vida del lugar.
df["is_train"] = (df["sample"] == "train")
df["y"] = df["Ingtot"].fillna(0.0).clip(lower=0)

common = dict(learning_rate=0.05, num_leaves=63, min_child_samples=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, n_jobs=2, verbose=-1, seed=SEED)
N_TREES = 500

out = df[["id", "Orden"]].copy()
for c in ["p_pos", "mu_log", "mean_raw"]:
    out[c] = 0.0
out["bin_head"] = (df["P6050"].astype(float) == 1).astype(int)

cap = df.loc[df.is_train, "y"].quantile(0.999)
test_idx = np.where(~df["is_train"])[0]
acc = {k: np.zeros(len(test_idx)) for k in ["p_pos", "mu_log", "mean_raw"]}

for k in range(1, 6):
    tr = np.where(df.is_train & (df["fold"] != k))[0]
    va = np.where(df.is_train & (df["fold"] == k))[0]
    X_tr, X_va = df.loc[tr, feats], df.loc[va, feats]
    y_tr = df.loc[tr, "y"].values
    pos = y_tr > 0

    m1 = lgb.train({**common, "objective": "binary"}, lgb.Dataset(X_tr, (y_tr > 0).astype(int)), N_TREES)
    m2 = lgb.train({**common, "objective": "regression"}, lgb.Dataset(X_tr[pos], np.log(y_tr[pos])), N_TREES)
    m3 = lgb.train({**common, "objective": "regression"}, lgb.Dataset(X_tr, np.clip(y_tr, 0, cap) / 1e6), N_TREES)

    out.loc[va, "p_pos"] = m1.predict(X_va)
    out.loc[va, "mu_log"] = m2.predict(X_va)
    out.loc[va, "mean_raw"] = np.clip(m3.predict(X_va), 0, None) * 1e6
    X_te = df.loc[test_idx, feats]
    acc["p_pos"] += m1.predict(X_te) / 5
    acc["mu_log"] += m2.predict(X_te) / 5
    acc["mean_raw"] += np.clip(m3.predict(X_te), 0, None) * 1e6 / 5
    print(f"fold {k} listo", flush=True)

for c in acc:
    out.loc[test_idx, c] = acc[c]
out.drop(columns=["Orden"]).to_csv(f_out, index=False)
print("guardado", f_out, flush=True)
