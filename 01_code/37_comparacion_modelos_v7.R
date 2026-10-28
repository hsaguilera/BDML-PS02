# ==============================================================================
# 37_comparacion_modelos_v7.R
# Autor: Jhon (Jhon Sebastián Vela Salcedo)
# Comparación de los ocho modelos v7 (scripts 29–36) y, aparte, de los tres v5 con primera etapa (26–28).
#
# Ejecuta 01_code/python/37_comparacion_modelos_v7.py (cuaderno interno 39_comparacion_v7). No entrena modelos: lee las señales
# *_LISTO.json de 02_outputs/tables/<script>/ y las predicciones fuera de pliegue que esos scripts dejan en
# 00_data/processed (modelos/ y cache_v7/), así que debe correr después de ellos.
# Hace: tabla maestra (F1 anidado sin Bogotá con IC 95 %, re-evaluación en una partición nueva, log-loss, AUC…),
# comparaciones pareadas (t corregido de Nadeau–Bengio, prueba bayesiana correlacionada con ROPE 0,002, bootstrap
# pareado, Holm), conjunto de confianza de modelos, comparación en igualdad de condiciones (mismas 1.152 variables
# y encuesta corta) y disyuntiva de política (filtración y subcobertura por 1.000 hogares).
# Salidas: 02_outputs/tables/37_comparacion_modelos_v7/ (39_*.csv, resultados_modelos_v7.xlsx) y 02_outputs/figures/37_comparacion_modelos_v7/.
# Tiempo aproximado: 15 min.
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
  script      = "37_comparacion_modelos_v7.py",
  carpeta     = "37_comparacion_modelos_v7",
  etapa_datos = "v7",
  hilos       = 4
)
