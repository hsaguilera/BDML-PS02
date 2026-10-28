# ==============================================================================
# 31_LPM_spl_T_n1152_lb0_s1_v7.R
# Autor: Jhon (Jhon Sebastián Vela Salcedo)
# Modelo: Linear Probability Model · v7, sin primera etapa: 1.152 variables candidatas de 64 preguntas, nada derivado de modelos de ingreso.
# Envío a Kaggle: LPM_spl_T_n1152_lb0_s1_v7.csv (2026-10-23; F1 público 0,74).
#
# Ejecuta 01_code/python/31_LPM_spl_T_n1152_lb0_s1_v7.py (cuaderno interno 30_lpm).
# Presupuesto de búsqueda «reducido» (01_code/python/pobreza/p7/entorno.py) y 2 hilos, como en la
# corrida que produjo el envío.
# Si faltan las bases que necesita en 00_data/processed, primero las construye desde 00_data/raw con
# 01_code/python/datos_cadena.py (cadena 00 → 01 → 02 → 09 → 02b → 02d).
# Salidas: 02_outputs/predictions/ (CSV de envío; nombre = modelo e hiperparámetros) y
#          02_outputs/tables/31_LPM_v7/, 02_outputs/figures/31_LPM_v7/ (prefijo 30_).
# Tiempo aproximado: 76 min con 2 hilos (más la cadena de datos la primera vez).
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
  script      = "31_LPM_spl_T_n1152_lb0_s1_v7.py",
  carpeta     = "31_LPM_v7",
  etapa_datos = "v7",
  presupuesto = "reducido",
  hilos       = 2
)
