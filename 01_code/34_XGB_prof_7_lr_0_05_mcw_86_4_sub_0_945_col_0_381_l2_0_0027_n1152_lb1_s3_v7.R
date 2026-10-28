# ==============================================================================
# 34_XGB_prof_7_lr_0_05_mcw_86_4_sub_0_945_col_0_381_l2_0_0027_n1152_lb1_s3_v7.R
# Modelo: Boosting (XGBoost) · v7, sin primera etapa: 1.152 variables candidatas de 64 preguntas, nada derivado de modelos de ingreso.
# Envío a Kaggle: XGB_prof_7_lr_0_05_mcw_86_4_sub_0_945_col_0_381_l2_0_0027_n1152_lb1_s3_v7.csv (2026-10-23; F1 público 0,77).
#
# Ejecuta 01_code/python/34_XGB_prof_7_lr_0_05_mcw_86_4_sub_0_945_col_0_381_l2_0_0027_n1152_lb1_s3_v7.py (cuaderno interno 36_xgboost).
# Presupuesto de búsqueda «minimo» (01_code/python/pobreza/p7/entorno.py) y 12 hilos, como en la
# corrida que produjo el envío.
# Selección restringida a las variables candidatas del LightGBM v7 (script 29): 02_outputs/tables/29_LGBM_v7/35_LISTO.json.
# Si faltan las bases que necesita en 00_data/processed, primero las construye desde 00_data/raw con
# 01_code/python/datos_cadena.py (cadena 00 → 01 → 02 → 09 → 02b → 02d).
# Salidas: 02_outputs/predictions/ (CSV de envío; nombre = modelo e hiperparámetros) y
#          02_outputs/tables/34_XGB_v7/, 02_outputs/figures/34_XGB_v7/ (prefijo 36_).
# Tiempo aproximado: 7,7 h con 12 hilos (más la cadena de datos la primera vez).
# ==============================================================================
# Raíz del repositorio (carpeta con el archivo .here) y funciones para llamar a Python
raiz <- normalizePath(getwd(), winslash = "/")
while (!file.exists(file.path(raiz, ".here"))) {
  if (dirname(raiz) == raiz) stop("Ejecute este script desde el repositorio (no encuentro el archivo .here).")
  raiz <- dirname(raiz)
}
source(file.path(raiz, "01_code", "functions", "fun_python_modelos.R"), encoding = "UTF-8")

py_correr_modelo(
  script      = "34_XGB_prof_7_lr_0_05_mcw_86_4_sub_0_945_col_0_381_l2_0_0027_n1152_lb1_s3_v7.py",
  carpeta     = "34_XGB_v7",
  etapa_datos = "v7",
  presupuesto = "minimo",
  hilos       = 12,
  pool        = "02_outputs/tables/29_LGBM_v7/35_LISTO.json"
)
