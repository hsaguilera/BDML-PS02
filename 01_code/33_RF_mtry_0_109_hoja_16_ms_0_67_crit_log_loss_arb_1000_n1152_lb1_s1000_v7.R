# ==============================================================================
# 33_RF_mtry_0_109_hoja_16_ms_0_67_crit_log_loss_arb_1000_n1152_lb1_s1000_v7.R
# Modelo: Random Forest · v7, sin primera etapa: 1.152 variables candidatas de 64 preguntas, nada derivado de modelos de ingreso.
# Envío a Kaggle: RF_mtry_0_109_hoja_16_ms_0_67_crit_log_loss_arb_1000_n1152_lb1_s1000_v7.csv (2026-10-23; F1 público 0,74).
#
# Ejecuta 01_code/python/33_RF_mtry_0_109_hoja_16_ms_0_67_crit_log_loss_arb_1000_n1152_lb1_s1000_v7.py (cuaderno interno 34_random_forest).
# Presupuesto de búsqueda «express» (01_code/python/pobreza/p7/entorno.py) y 12 hilos, como en la
# corrida que produjo el envío.
# Si faltan las bases que necesita en 00_data/processed, primero las construye desde 00_data/raw con
# 01_code/python/datos_cadena.py (cadena 00 → 01 → 02 → 09 → 02b → 02d).
# Salidas: 02_outputs/predictions/ (CSV de envío; nombre = modelo e hiperparámetros) y
#          02_outputs/tables/33_RF_v7/, 02_outputs/figures/33_RF_v7/ (prefijo 34_).
# Tiempo aproximado: 4,0 h con 12 hilos (más la cadena de datos la primera vez).
# ==============================================================================
# Raíz del repositorio (carpeta con el archivo .here) y funciones para llamar a Python
raiz <- normalizePath(getwd(), winslash = "/")
while (!file.exists(file.path(raiz, ".here"))) {
  if (dirname(raiz) == raiz) stop("Ejecute este script desde el repositorio (no encuentro el archivo .here).")
  raiz <- dirname(raiz)
}
source(file.path(raiz, "01_code", "functions", "fun_python_modelos.R"), encoding = "UTF-8")

py_correr_modelo(
  script      = "33_RF_mtry_0_109_hoja_16_ms_0_67_crit_log_loss_arb_1000_n1152_lb1_s1000_v7.py",
  carpeta     = "33_RF_v7",
  etapa_datos = "v7",
  presupuesto = "express",
  hilos       = 12
)
