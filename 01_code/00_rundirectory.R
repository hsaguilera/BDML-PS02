# ==============================================================================
# 00_rundirectory.R  --  Script maestro (PS2: Predicting Poverty)
# Uso:  source("01_code/00_rundirectory.R")   (desde la raíz del repositorio)
# Ejecuta, en orden, todos los scripts numerados y reproduce las salidas.
# ==============================================================================

# --- Raíz del proyecto: sin rutas absolutas -----------------------------------
if (!requireNamespace("here", quietly = TRUE)) install.packages("here")
ROOT <- here::here()
setwd(ROOT)

# --- Paquetes (pacman instala los que falten) ---------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(tidyverse, data.table, glmnet, caret, rpart, ranger, gbm, e1071, ragg,
               pROC, xtable, jsonlite, ggrepel, scales, here)

# --- Parámetros globales -------------------------------------------------------
SEED      <- 2026
N_FOLDS   <- 5
DIR_RAW   <- "00_data/raw"
DIR_PROC  <- "00_data/processed"
DIR_FIG   <- "02_outputs/figures"
DIR_TAB   <- "02_outputs/tables"
DIR_PRED  <- "02_outputs/predictions"
set.seed(SEED)

# --- Funciones reutilizables ----------------------------------------------------
for (f in list.files("01_code/functions", pattern = "\\.R$", full.names = TRUE)) source(f, encoding = "UTF-8")

# --- Pipeline -------------------------------------------------------------------
scripts <- sort(list.files("01_code", pattern = "^[0-9]{2}_.*\\.R$", full.names = TRUE))
scripts <- scripts[!grepl("00_rundirectory", scripts)]
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
