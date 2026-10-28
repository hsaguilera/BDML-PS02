"""Pruebas del protocolo v7 (Paso 14 del memo) con datos SINTÉTICOS: señal conocida y un «dominio extra» (BOGOTA) con
otra prevalencia. Nada se lee ni se escribe en data/, reports/ ni 02_outputs/predictions/ reales (todo en un directorio temporal).

Uso: python -m unittest pobreza.tests_v7.test_protocolo_v7 -v   (desde src/), o
     python "01_code/python/pobreza/tests_v7/test_protocolo_v7.py"
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve()
sys.path.insert(0, str(AQUI.parents[2]))
warnings.filterwarnings("ignore")

from pobreza.p7 import comparar as CP  # noqa: E402
from pobreza.p7 import glm as GL  # noqa: E402
from pobreza.p7 import metricas as MT  # noqa: E402
from pobreza.p7 import motores as MO  # noqa: E402
from pobreza.p7.cvmotor import cv  # noqa: E402
from pobreza.p7.datos import Datos7  # noqa: E402
from pobreza.p7.entorno import Entorno, presupuesto  # noqa: E402

DOMS = ["BOGOTA", "MEDELLIN", "CALI", "BARRANQUILLA", "PASTO", "RURAL"]


def escribir_sinteticos(carpeta: Path, n=6000, n_test=2000, semilla=0):
    """Tablas con el esquema de v7: señal logística en x1–x3, ruido en x4–x6, una categórica y BOGOTA con prevalencia
    menor y valores de Lp propios (soporte disjunto, como en los datos reales)."""
    rng = np.random.default_rng(semilla)

    def tabla(m, train):
        dom = rng.choice(DOMS if train else DOMS[1:], m, p=[.12, .22, .2, .18, .14, .14] if train else [.25, .23, .2, .16, .16])
        X = rng.standard_normal((m, 6)).astype(np.float32)
        lp = np.where(dom == "BOGOTA", 400.0, 300.0) + rng.integers(0, 3, m) * (7 if train else 5)
        eta = -1.6 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] * X[:, 0] - 0.7 * (dom == "BOGOTA")
        df = pd.DataFrame({"id": [f"{'tr' if train else 'te'}{i:06d}" for i in range(m)], "Dominio": dom,
                           "Depto": rng.choice(["05", "08", "11", "76"], m), "Lp": lp,
                           **{f"x{j + 1}": X[:, j] for j in range(6)},
                           "tenencia": rng.choice(["1", "2", "3"], m), "cabecera": (dom != "RURAL").astype(float)})
        df.loc[rng.random(m) < 0.05, "x4"] = np.nan
        if train:
            df["Pobre"] = (rng.random(m) < 1 / (1 + np.exp(-eta))).astype(int)
            df["fold"] = rng.integers(0, 5, m)
        return df
    D, DT = tabla(n, True), tabla(n_test, False)
    D.to_parquet(carpeta / "modelo_train_v7.parquet", index=False); DT.to_parquet(carpeta / "modelo_test_v7.parquet", index=False)
    todas = ["x1", "x2", "x3", "x4", "x5", "x6", "Dominio", "Depto", "Lp", "tenencia", "cabecera"]
    conj = {"version": "v7", "estado": "sintético", "todas_permitidas": todas, "categoricas": ["Dominio", "Depto", "tenencia"],
            "encuesta_corta_v7": ["x1", "x2", "Dominio"],
            "mapa_preguntas": {"x1": ["Q1"], "x2": ["Q2"], "x3": ["Q1", "Q3"], "x4": ["Q4"], "x5": ["Q5"], "x6": ["Q5", "Q6"],
                               "Dominio": ["Dominio"], "Depto": ["Depto"], "Lp": ["Lp"], "tenencia": ["Q7"], "cabecera": ["Clase"]},
            "preguntas": {q: {"descripcion": q, "costo": 0 if q in ("Dominio", "Depto", "Lp", "Clase") else 1}
                          for q in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Dominio", "Depto", "Lp", "Clase"]},
            "bloques": {c: ("A" if c in ("x1", "x2", "x3") else "B") for c in todas}, "prohibidas": {}}
    (carpeta / "conjuntos_variables_v7.json").write_text(json.dumps(conj), encoding="utf-8")
    pd.DataFrame({"id": DT.id, "pobre": 0}).to_csv(carpeta / "sample_submission.csv", index=False)
    return D, DT


class ProtocoloV7(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="p7_prueba_"))
        (cls.tmp / "datos").mkdir()
        escribir_sinteticos(cls.tmp / "datos")
        cls.ent = Entorno(raiz=AQUI.parents[3], salida=cls.tmp / "salida", datos=cls.tmp / "datos", raw=cls.tmp / "datos",
                          cache=cls.tmp / "cache", hilos=2, extra={"n_test_esperado": None})
        cls.d = Datos7(cls.ent, fuente="v7", verbose=False)
        cls.logit = MO.MotorLogit(hilos=2)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ---------------------------------------------------------------- datos y particiones
    def test_paso0_y_particiones(self):
        d = self.d
        self.assertTrue(d.verificacion["ok"], d.verificacion)
        self.assertTrue(d.verificacion["b_Lp_bogota_ausente_en_test"])
        P = d.particiones_df
        self.assertTrue((P.fold_r1.values == pd.read_parquet(self.tmp / "datos" / "modelo_train_v7.parquet").fold.values).all())
        for r in (2, 3, 4):
            self.assertEqual(set(P[f"fold_r{r}"]), set(range(5)))
            # estratificación por Pobre × Dominio: tasas por pliegue casi iguales
            tasas = pd.Series(d.y).groupby(P[f"fold_r{r}"].values).mean()
            self.assertLess(tasas.max() - tasas.min(), 0.03)
        self.assertTrue((self.ent.procesados / "pliegues_v7.csv").exists())
        self.assertEqual(d.MP.costo(["x1", "x3"]), 2)            # Q1 compartida: costo marginal cero
        self.assertEqual(d.MP.costo(["Dominio", "Lp", "cabecera"]), 0)

    # ---------------------------------------------------------------- (a) ninguna métrica usa filas de Bogotá
    def test_a_metricas_sin_bogota(self):
        d = self.d
        r = cv(self.logit, d, {}, d.todas, 1.0, 1, (1,))
        m1 = MT.metricas_particion(d.y, r.oof, d.folds(1), d.mT, B=50)
        s2 = r.oof.copy(); s2[d.bog] = np.random.default_rng(1).random(d.bog.sum())   # basura en Bogotá
        m2 = MT.metricas_particion(d.y, s2, d.folds(1), d.mT, B=50)
        for k in ("F1_anidado", "LogLoss", "Brier", "AUC_ROC", "AUC_PR", "umbrales_pliegues", "F1_pliegues"):
            self.assertEqual(m1[k], m2[k], k)
        # con λ_B = 0 las etiquetas de Bogotá no influyen en las OOF de 𝒯
        import copy
        d2 = copy.copy(d); d2.y = d.y.copy(); d2.y[d.bog] = 1 - d2.y[d.bog]
        a = cv(self.logit, d, {}, d.todas, 0.0, 1, (1,)); b = cv(self.logit, d2, {}, d.todas, 0.0, 1, (1,))
        np.testing.assert_allclose(a.oof[d.mT], b.oof[d.mT], rtol=1e-9, atol=1e-12)

    # ---------------------------------------------------------------- (b) umbral anidado sin el pliegue evaluado
    def test_b_umbral_anidado(self):
        d = self.d
        r = cv(self.logit, d, {}, d.todas, 1.0, 1, (1,))
        f = d.folds(1)
        base = MT.f1_anidado(d.y, r.oof, f, d.mT)["umbrales"]
        s2 = r.oof.copy(); s2[f[2][1]] = 1 - s2[f[2][1]]                       # se altera sólo el pliegue 2
        nuevo = MT.f1_anidado(d.y, s2, f, d.mT)["umbrales"]
        self.assertAlmostEqual(base[2], nuevo[2], places=12)                    # el umbral del pliegue 2 no lo usa
        self.assertTrue(any(abs(base[k] - nuevo[k]) > 0 for k in (0, 1, 3, 4)))

    # ---------------------------------------------------------------- (c) etiquetas permutadas → F1 no informativo
    def test_c_etiquetas_permutadas(self):
        import copy
        d = self.d
        d2 = copy.copy(d); d2.y = np.random.default_rng(7).permutation(d.y)
        r = cv(self.logit, d2, {}, d.todas, 1.0, 1, (1,))
        F = MT.f1_anidado(d2.y, r.oof, d.folds(1), d.mT)["F1"]
        pi = d2.y[d.mT].mean()
        self.assertLess(abs(F - 2 * pi / (1 + pi)), 0.04, (F, 2 * pi / (1 + pi)))

    # ---------------------------------------------------------------- (d) reproducibilidad con las mismas semillas
    def test_d_reproducible(self):
        d = self.d
        m = MO.MotorLGB(hilos=2, lr=0.1, paciencia=20, max_arboles=300)
        a = cv(m, d, {"num_leaves": 15}, d.todas, 0.5, 2, (1, 2))
        b = cv(m, d, {"num_leaves": 15}, d.todas, 0.5, 2, (1, 2))
        np.testing.assert_array_equal(a.oof, b.oof)

    # ---------------------------------------------------------------- incertidumbre y comparaciones
    def test_ee_influencia_vs_bootstrap(self):
        rng = np.random.default_rng(3); n = 20000
        y = (rng.random(n) < .2).astype(int); yh = np.where(rng.random(n) < .8, y, 1 - y)
        F, ee, _ = MT.influencia_f1(y, yh)
        bt = MT.bootstrap_f1(y, yh, B=400)
        self.assertLess(abs(ee - bt["EE_boot"]) / ee, 0.15)

    def test_umbral_argmax_fuerza_bruta(self):
        rng = np.random.default_rng(4); y = (rng.random(800) < .3).astype(int); s = np.round(rng.random(800) * 0.6 + 0.4 * y, 2)
        t, f = MT.umbral_argmax(y, s)
        mejor = max(MT.f1(y, s >= u) for u in np.unique(s))
        self.assertAlmostEqual(f, mejor, places=12); self.assertAlmostEqual(MT.f1(y, s >= t), mejor, places=12)

    def test_nadeau_bengio_y_bayes(self):
        d = np.array([0.001, 0.002, 0.0015, 0.0005, 0.0012])
        nb = CP.nadeau_bengio(d, 5)
        self.assertAlmostEqual(nb["EE_corregido"], np.sqrt((1 / 5 + 1 / 4) * d.var(ddof=1)))
        b = CP.bayes_correlacionado(d, 0.002, 5)
        self.assertAlmostEqual(b["P_izq"] + b["P_rope"] + b["P_der"], 1.0)
        E = np.array([[0.30, 0.31, 0.29], [0.305, 0.30, 0.295]])
        self.assertGreaterEqual(CP.tibshirani_tibshirani(E)["sesgo"], 0.0)

    # ---------------------------------------------------------------- solucionadores y poda
    def test_glm_contra_sklearn(self):
        from sklearn.linear_model import Lasso, LogisticRegression
        rng = np.random.default_rng(0); n, p = 3000, 12
        Z = rng.standard_normal((n, p)).astype(np.float32); y = (rng.random(n) < 1 / (1 + np.exp(1 - Z[:, 0] + Z[:, 1]))).astype(float)
        est = GL.EstadisticosMCO.de(Z, y); lam = GL.grilla_lambda(GL.lambda_max_mco(est), 8, 1e-2)
        B, b0 = GL.ruta_lasso_mco(est, lam)
        sk = Lasso(alpha=lam[5], tol=1e-10, max_iter=100000).fit(Z.astype(float), y)
        np.testing.assert_allclose(B[:, 5], sk.coef_, atol=1e-4)
        b0n, bn, _ = GL.logit_newton(Z, y, cresta=0.0)
        sk = LogisticRegression(penalty=None, tol=1e-10, max_iter=5000).fit(Z, y)
        np.testing.assert_allclose(bn, sk.coef_[0], atol=1e-5)

    def test_poda_cart_igual_a_sklearn(self):
        from sklearn.tree import DecisionTreeClassifier
        rng = np.random.default_rng(1); X = rng.random((3000, 4)); y = (rng.random(3000) < X[:, 0] * .6).astype(int)
        t = DecisionTreeClassifier(min_samples_leaf=20, random_state=0).fit(X, y)
        a = float(t.cost_complexity_pruning_path(X, y).ccp_alphas[-4]) * 1.0000001
        P = MO.predecir_podado(t, X, [a], m=0.0, prior=0.2)[:, 0]
        t2 = DecisionTreeClassifier(min_samples_leaf=20, random_state=0, ccp_alpha=a).fit(X, y)
        np.testing.assert_allclose(P, t2.predict_proba(X)[:, 1], atol=1e-9)

    # ---------------------------------------------------------------- CSV verificado
    def test_csv(self):
        from pobreza.p7 import final as FI
        d = self.d
        ruta = FI.escribir_csv(self.ent, d.ids_test, np.zeros(len(d.ids_test), int), "prueba_v7.csv")
        ver = FI.verificar_csv(ruta, pd.read_csv(self.tmp / "datos" / "sample_submission.csv"), None)
        self.assertTrue(ver["verificado"])
        self.assertTrue(str(ruta).startswith(str(self.tmp)))           # nunca en 02_outputs/predictions/ reales


if __name__ == "__main__":
    unittest.main(verbosity=2)
