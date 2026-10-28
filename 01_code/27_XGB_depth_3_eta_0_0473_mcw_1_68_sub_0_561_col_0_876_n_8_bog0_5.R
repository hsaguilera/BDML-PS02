# ==============================================================================
# 27_XGB_depth_3_eta_0_0473_mcw_1_68_sub_0_561_col_0_876_n_8_bog0_5.R
# Modelo: Boosting (XGBoost) · v5, con primera etapa: el ingreso esperado fuera de pliegue entra como variable (zona gris, ver README).
# Envío a Kaggle: XGB_depth_3_eta_0_0473_mcw_1_68_sub_0_561_col_0_876_n_8_bog0_5.csv (2026-10-19; F1 público 0,77).
#
# Ejecuta 01_code/python/27_XGB_depth_3_eta_0_0473_mcw_1_68_sub_0_561_col_0_876_n_8_bog0_5.py (cuaderno interno 11_xgboost).
# 6 hilos, como en la corrida que produjo el envío.
# Si faltan las bases que necesita en 00_data/processed, primero las construye desde 00_data/raw con
# 01_code/python/datos_cadena.py (cadena 00 → 01 → 02 → 09 → 02b).
# Salidas: 02_outputs/predictions/ (CSV de envío; nombre = modelo e hiperparámetros) y
#          02_outputs/tables/27_XGB_v5/, 02_outputs/figures/27_XGB_v5/ (prefijo 11_).
# Tiempo aproximado: 32 min con 6 hilos (más la cadena de datos la primera vez).
# ==============================================================================
# Raíz del repositorio (carpeta con el archivo .here) y funciones para llamar a Python
raiz <- normalizePath(getwd(), winslash = "/")
while (!file.exists(file.path(raiz, ".here"))) {
  if (dirname(raiz) == raiz) stop("Ejecute este script desde el repositorio (no encuentro el archivo .here).")
  raiz <- dirname(raiz)
}
source(file.path(raiz, "01_code", "functions", "fun_python_modelos.R"), encoding = "UTF-8")

py_correr_modelo(
  script      = "27_XGB_depth_3_eta_0_0473_mcw_1_68_sub_0_561_col_0_876_n_8_bog0_5.py",
  carpeta     = "27_XGB_v5",
  etapa_datos = "v5",
  presupuesto = "",
  hilos       = 6
)
