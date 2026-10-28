# ==============================================================================
# 00_rundirectory.R  --  Script maestro (PS2: Predicting Poverty)
# Uso:  source("01_code/00_rundirectory.R")   (desde la raíz del repositorio)
# Ejecuta, en orden, todos los scripts numerados y reproduce las salidas.
#
# Camino corto (opcional): por defecto corre TODO (01-37, más de 30 horas). Con variables de entorno
# se puede acotar el rango de scripts, p. ej. solo la cadena de R y las diapositivas (01-18):
#   Sys.setenv(PS2_HASTA = "19")    # corre 01-18 (omite los modelos 20-37)
#   Sys.setenv(PS2_DESDE = "13")    # corre desde el 13 (si ya existen los insumos de 00_data/processed)
# ==============================================================================

# --- Entorno común: raíz, paquetes (con versiones), parámetros y funciones -----
if (!requireNamespace("here", quietly = TRUE)) install.packages("here")
source(file.path(here::here(), "01_code", "00_setup.R"), encoding = "UTF-8")

# --- Pipeline -------------------------------------------------------------------
scripts <- sort(list.files("01_code", pattern = "^[0-9]{2}_.*\\.R$", full.names = TRUE))
scripts <- scripts[!grepl("^01_code/00_", scripts)]
num     <- as.integer(substr(basename(scripts), 1, 2))
desde   <- as.integer(Sys.getenv("PS2_DESDE", "0")); hasta <- as.integer(Sys.getenv("PS2_HASTA", "99"))
scripts <- scripts[num >= desde & num <= hasta]
for (s in scripts) {
  message("\n>>> Ejecutando ", s)
  source(s, local = new.env(parent = globalenv()), encoding = "UTF-8")
}
# --- Registro del entorno (versión de R y paquetes) ----------------------------
py_ver <- tryCatch(system2(Sys.getenv("PYTHON_BIN", "python3"), c("-I", "-c", shQuote(
  "import sys,numpy,pandas,scipy,sklearn,lightgbm,xgboost;print('Python',sys.version.split()[0]);[print(m.__name__,m.__version__) for m in (numpy,pandas,scipy,sklearn,lightgbm,xgboost)]")),
  stdout = TRUE, stderr = FALSE), error = function(e) "Python no disponible")
writeLines(c(capture.output(sessionInfo()), "", py_ver), file.path("02_outputs", "session_info.txt"))
message("\nPipeline completo.")
