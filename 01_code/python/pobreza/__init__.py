"""Paquete `pobreza` — Problem Set 2 (BDML, Uniandes 2026-20).

Predicción de pobreza monetaria de hogares colombianos (MESE/DANE) para la
competencia Kaggle `uniandes-bdml-2026-20-ps-2` (métrica: F1).

Módulos
-------
config      Rutas del proyecto (Colab/Drive o local), semillas y constantes.
io          Lectura de los datos crudos (zip o csv) y tipado de columnas.
eda         Utilidades del análisis exploratorio.
features    Construcción de variables a nivel hogar (teoría económica/social).
metrics     F1 y métricas de focalización, calibración y umbral óptimo.
cv          Validación cruzada estratificada y anidada.
probit      Probit (MV, penalizado L1/L2, heterocedástico y latente-gaussiano).
glmm        Probit de efectos mixtos (GLMM) con efectos aleatorios cruzados (Laplace).
viz         Estilo gráfico común.
nbtools     Construcción de notebooks .ipynb desde fuentes .py.
"""
__version__ = "1.0.0"
