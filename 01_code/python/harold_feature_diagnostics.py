"""Diagnóstico de variables antes de modelar (train, hogares).

Salidas (CSV en <out_dir>):
  harold_fe_univariate.csv : por variable -> AUC univariado, correlación de Pearson y
                             Spearman con 'pobre', AUC no lineal (20 cuantiles), ganancia
                             no lineal, información mutua, % ceros / faltantes.
  harold_fe_importance.csv : importancia de LightGBM (ganancia) y |SHAP| medio (pred_contrib)
                             con todas las variables, validación en el pliegue 1.
  harold_fe_ablation.csv   : F1 (umbral óptimo en OOF), AUC y PR-AUC fuera de muestra por
                             conjunto de variables (A base, A+B, A+C, A+D, A+B+C+D). Se
                             evalúa SIN Bogotá (no existe en test).
Uso: python harold_feature_diagnostics.py <hogares_fe.csv> <grupos.csv> <out_dir>
"""
import sys, os
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr

f_data, f_groups, out_dir = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)
SEED = 2026
df = pd.read_csv(f_data)
groups = pd.read_csv(f_groups)                       # columnas: variable, grupo
tr = df[df["sample"] == "train"].reset_index(drop=True)
y = tr["poor"].values
nb = (tr["bogota"] == 0).values                      # hogares fuera de Bogotá
print("train:", tr.shape, "pobreza:", y.mean().round(4), flush=True)

num_feats = groups["variable"].tolist()
for c in ["Dominio", "tenure"]:
    if c in tr: tr[c] = tr[c].astype("category")

# ------------------------------------------------------------------ 1. univariado
rng = np.random.default_rng(SEED)
sub = rng.choice(len(tr), 60000, replace=False)
rows = []
for v in num_feats:
    x = pd.to_numeric(tr[v].astype(float), errors="coerce") if v not in ("Dominio", "tenure") else tr[v].cat.codes.astype(float)
    xf = x.fillna(x.median())
    if xf.nunique() < 2:
        continue
    auc = roc_auc_score(y, xf)
    q = pd.qcut(xf.rank(method="first"), 20, labels=False, duplicates="drop")
    rate = pd.Series(y).groupby(q.values).transform("mean")
    auc_bin = roc_auc_score(y, rate)
    mi = mutual_info_classif(xf.values[sub].reshape(-1, 1), y[sub], random_state=SEED)[0]
    rows.append(dict(variable=v, grupo=groups.set_index("variable").loc[v, "grupo"],
                     auc=auc, auc_lineal=max(auc, 1 - auc), auc_binned=auc_bin,
                     ganancia_no_lineal=auc_bin - max(auc, 1 - auc),
                     pearson=np.corrcoef(xf, y)[0, 1], spearman=spearmanr(xf, y)[0],
                     info_mutua=mi, pct_cero=float((x == 0).mean()), pct_na=float(x.isna().mean())))
uni = pd.DataFrame(rows).sort_values("auc_binned", ascending=False)
uni.to_csv(os.path.join(out_dir, "harold_fe_univariate.csv"), index=False)
print("univariado listo", flush=True)

# ------------------------------------------------------------------ 2. ablación
P = dict(objective="binary", learning_rate=0.06, num_leaves=31, min_child_samples=50,
         feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
         n_jobs=2, verbose=-1, seed=SEED)
G = lambda *g: groups.loc[groups["grupo"].isin(g), "variable"].tolist()
sets = {"A base": G("A"), "A + B (ratios e interacciones)": G("A", "B"),
        "A + C (estructura de la regla)": G("A", "C"), "A + D (ingreso imputado)": G("A", "D"),
        "A + C + D": G("A", "C", "D"), "A + B + C + D (todas)": G("A", "B", "C", "D")}

def best_f1(yy, p):
    ts = np.arange(0.10, 0.80, 0.01)
    f = [f1_score(yy, p >= t) for t in ts]
    return ts[int(np.argmax(f))], max(f)

abl = []
imp_rows = None
for name, feats in sets.items():
    oof = np.zeros(len(tr))
    for k in range(1, 6):
        i_tr, i_va = np.where(tr["fold"] != k)[0], np.where(tr["fold"] == k)[0]
        m = lgb.train(P, lgb.Dataset(tr.loc[i_tr, feats], y[i_tr]), 400)
        oof[i_va] = m.predict(tr.loc[i_va, feats])
        if name.endswith("(todas)") and k == 1:
            imp = pd.DataFrame({"variable": feats,
                                "gain": m.feature_importance("gain") / m.feature_importance("gain").sum()})
            contrib = m.predict(tr.loc[i_va[:20000], feats], pred_contrib=True)[:, :-1]
            imp["shap_abs"] = np.abs(contrib).mean(0)
            imp["grupo"] = imp["variable"].map(groups.set_index("variable")["grupo"])
            imp_rows = imp.sort_values("shap_abs", ascending=False)
    t, f1 = best_f1(y[nb], oof[nb])
    abl.append(dict(conjunto=name, n_vars=len(feats), umbral=t, f1=f1,
                    auc=roc_auc_score(y[nb], oof[nb]), pr_auc=average_precision_score(y[nb], oof[nb]),
                    f1_umbral_050=f1_score(y[nb], oof[nb] >= 0.5)))
    print(abl[-1], flush=True)
pd.DataFrame(abl).to_csv(os.path.join(out_dir, "harold_fe_ablation.csv"), index=False)
imp_rows.to_csv(os.path.join(out_dir, "harold_fe_importance.csv"), index=False)
print("listo", flush=True)
