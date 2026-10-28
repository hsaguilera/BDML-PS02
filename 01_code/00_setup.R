# ==============================================================================
# 00_setup.R  --  Entorno común de los scripts 01-18 (rutas, paquetes, parámetros y funciones)
# Lo carga 00_rundirectory.R y, si se corre un script 01-18 por separado, el propio script
# (primera línea de código: `if (!exists("DIR_PROC")) ...`). Los scripts 20-37 construyen
# su propio entorno y no lo necesitan.
# Sin rutas absolutas: la raíz del proyecto es la carpeta que contiene el archivo `.here`.
# Versiones: `requirements_R.txt` fija las versiones con las que se produjeron los resultados;
# si una difiere, se avisa. `PS2_INSTALAR_VERSIONES=1` instala exactamente esas versiones.
# ==============================================================================

# --- Raíz del proyecto ---------------------------------------------------------
if (!requireNamespace("here", quietly = TRUE)) install.packages("here")
ROOT <- here::here()
setwd(ROOT)

# --- Versiones de paquetes de R (requirements_R.txt: paquete==versión) ----------
pins <- character()
if (file.exists("requirements_R.txt")) {
  l <- trimws(readLines("requirements_R.txt", warn = FALSE))
  l <- l[nzchar(l) & !startsWith(l, "#") & grepl("==", l, fixed = TRUE)]
  pins <- setNames(sub(".*==", "", l), sub("==.*", "", l))
}
if (Sys.getenv("PS2_INSTALAR_VERSIONES") == "1" && length(pins)) {
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  for (p in names(pins))
    if (!requireNamespace(p, quietly = TRUE) || as.character(utils::packageVersion(p)) != pins[[p]])
      remotes::install_version(p, version = pins[[p]], upgrade = "never")
}

# --- Paquetes (pacman instala los que falten) -----------------------------------
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(tidyverse, data.table, glmnet, caret, rpart, ranger, gbm, e1071, ragg,
               pROC, xtable, jsonlite, ggrepel, scales, here)
dif <- names(pins)[vapply(names(pins), function(p)
  !requireNamespace(p, quietly = TRUE) || as.character(utils::packageVersion(p)) != pins[[p]], logical(1))]
if (length(dif)) message("Aviso: versión distinta de la fijada en requirements_R.txt para: ", paste(dif, collapse = ", "),
                         " (los resultados pueden variar levemente).")

# --- Parámetros globales ---------------------------------------------------------
SEED      <- 2026
N_FOLDS   <- 5
DIR_RAW   <- "00_data/raw"
DIR_PROC  <- "00_data/processed"
DIR_FIG   <- "02_outputs/figures"
DIR_TAB   <- "02_outputs/tables"
DIR_PRED  <- "02_outputs/predictions"
set.seed(SEED)

# --- Funciones reutilizables --------------------------------------------------------
for (f in list.files("01_code/functions", pattern = "\\.R$", full.names = TRUE)) source(f, encoding = "UTF-8")
