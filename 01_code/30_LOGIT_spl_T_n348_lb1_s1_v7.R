# ==============================================================================
# 30_LOGIT_spl_T_n348_lb1_s1_v7.R
# Modelo: Logit · v7, sin primera etapa: 1.152 variables candidatas de 64 preguntas, nada derivado de modelos de ingreso.
# Envío a Kaggle: LOGIT_spl_T_n348_lb1_s1_v7.csv (2026-10-21; F1 público 0,76).
#
# Ejecuta 01_code/python/30_LOGIT_spl_T_n348_lb1_s1_v7.py (cuaderno interno 31_logit).
# Presupuesto de búsqueda «reducido» (01_code/python/pobreza/p7/entorno.py) y 3 hilos, como en la
# corrida que produjo el envío.
# Si faltan las bases que necesita en 00_data/processed, primero las construye desde 00_data/raw con
# 01_code/python/datos_cadena.py (cadena 00 → 01 → 02 → 09 → 02b → 02d).
# Salidas: 02_outputs/predictions/ (CSV de envío; nombre = modelo e hiperparámetros) y
#          02_outputs/tables/30_LOGIT_v7/, 02_outputs/figures/30_LOGIT_v7/ (prefijo 31_).
# Tiempo aproximado: 5,1 h con 3 hilos (más la cadena de datos la primera vez).
# ==============================================================================
# Raíz del repositorio (carpeta con el archivo .here) y funciones para llamar a Python
raiz <- normalizePath(getwd(), winslash = "/")
while (!file.exists(file.path(raiz, ".here"))) {
  if (dirname(raiz) == raiz) stop("Ejecute este script desde el repositorio (no encuentro el archivo .here).")
  raiz <- dirname(raiz)
}
source(file.path(raiz, "01_code", "functions", "fun_python_modelos.R"), encoding = "UTF-8")

py_correr_modelo(
  script      = "30_LOGIT_spl_T_n348_lb1_s1_v7.py",
  carpeta     = "30_LOGIT_v7",
  etapa_datos = "v7",
  presupuesto = "reducido",
  hilos       = 3
)
