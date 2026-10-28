# ==============================================================================
# 26_LGBM_leaves_10_lr_0_0253_mcs_22_ff_0_442_l2_0_00326_n_8.R
# Autor: Jhon (Jhon Sebastián Vela Salcedo)
# Modelo: Boosting (LightGBM) · v5, con primera etapa: el ingreso esperado fuera de pliegue entra como variable (zona gris, ver README).
# Envío a Kaggle: LGBM_leaves_10_lr_0_0253_mcs_22_ff_0_442_l2_0_00326_n_8.csv (2026-10-19; F1 público 0,77).
#
# Ejecuta 01_code/python/26_LGBM_leaves_10_lr_0_0253_mcs_22_ff_0_442_l2_0_00326_n_8.py (cuaderno interno 10_lightgbm).
# 6 hilos, como en la corrida que produjo el envío.
# Si faltan las bases que necesita en 00_data/processed, primero las construye desde 00_data/raw con
# 01_code/python/datos_cadena.py (cadena 00 → 01 → 02 → 09 → 02b).
# Salidas: 02_outputs/predictions/ (CSV de envío; nombre = modelo e hiperparámetros) y
#          02_outputs/tables/26_LGBM_v5/, 02_outputs/figures/26_LGBM_v5/ (prefijo 10_).
# Tiempo aproximado: 22 min con 6 hilos (más la cadena de datos la primera vez).
# ==============================================================================
AUTOR <- "Jhon"

# Raíz del repositorio (carpeta con el archivo .here) y funciones para llamar a Python
raiz <- normalizePath(getwd(), winslash = "/")
while (!file.exists(file.path(raiz, ".here"))) {
  if (dirname(raiz) == raiz) stop("Ejecute este script desde el repositorio (no encuentro el archivo .here).")
  raiz <- dirname(raiz)
}
source(file.path(raiz, "01_code", "functions", "fun_python_modelos.R"), encoding = "UTF-8")

py_correr_modelo(
  script      = "26_LGBM_leaves_10_lr_0_0253_mcs_22_ff_0_442_l2_0_00326_n_8.py",
  carpeta     = "26_LGBM_v5",
  etapa_datos = "v5",
  presupuesto = "",
  hilos       = 6
)
