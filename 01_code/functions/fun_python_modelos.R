# ==============================================================================
# fun_python_modelos.R  --  Funciones para ejecutar desde R los scripts de Python 26–37
# Autor: Jhon (Jhon Sebastián Vela Salcedo)
#
# Las usan los scripts 26 a 37 de 01_code/. Cada uno llama a un script de Python de 01_code/python/ (mismo nombre,
# extensión .py), que usa el paquete 01_code/python/pobreza/. Requisitos de Python: requirements.txt.
#
# Variables de entorno opcionales:
#   PYTHON_BIN   ruta del intérprete de Python (por defecto, el primero que encuentre: python o python3)
#   PY_HILOS   hilos de cómputo (por defecto, los que se usaron para obtener los resultados del repositorio)
# ==============================================================================

# Raíz del repositorio: la carpeta que contiene el archivo .here (sube desde el directorio de trabajo).
py_raiz <- function() {
  d <- normalizePath(getwd(), winslash = "/")
  while (!file.exists(file.path(d, ".here"))) {
    if (dirname(d) == d) stop("No encuentro la raíz del repositorio (archivo .here). Ejecute desde el repositorio.")
    d <- dirname(d)
  }
  d
}

# Intérprete de Python.
py_interprete <- function() {
  py <- Sys.getenv("PYTHON_BIN", "")
  if (nzchar(py)) return(py)
  for (cand in c("python", "python3")) if (nzchar(Sys.which(cand))) return(cand)
  stop("No encuentro Python; defina PYTHON_BIN con la ruta del intérprete.")
}

# Ejecuta 01_code/python/<script> con variables de entorno temporales (se restauran al salir) y se detiene si el
# script falla. MPLBACKEND = Agg: las figuras se guardan en disco sin abrir ventanas.
py_ejecutar <- function(script, args = character(), entorno = character()) {
  raiz <- py_raiz()
  entorno <- c(TALLER2_ROOT = raiz, PYTHONUTF8 = "1", MPLBACKEND = "Agg", entorno)
  previo <- Sys.getenv(names(entorno), unset = NA, names = TRUE)
  do.call(Sys.setenv, as.list(entorno))
  on.exit({
    for (k in names(previo)) {
      if (is.na(previo[[k]])) Sys.unsetenv(k) else do.call(Sys.setenv, stats::setNames(list(previo[[k]]), k))
    }
  }, add = TRUE)
  ruta <- file.path(raiz, "01_code", "python", script)
  message(">>> Python: ", script, if (length(args)) paste0(" ", paste(args, collapse = " ")))
  estado <- system2(py_interprete(), c("-X", "utf8", shQuote(ruta), args))
  if (!identical(as.integer(estado), 0L)) stop("Falló ", script, " (código ", estado, ").")
  invisible(TRUE)
}

# Corre un modelo: primero construye, si faltan, las bases que necesita (etapa "v5" o "v7", ver
# 01_code/python/datos_cadena.py) y luego el script, con el presupuesto e hilos con que se obtuvieron los resultados.
#   carpeta     subcarpeta de 02_outputs/tables y 02_outputs/figures donde quedan sus tablas y figuras
#   presupuesto presupuesto de búsqueda del protocolo v7 (p7/entorno.py); vacío en v5 y en la comparación
#   pool        (opcional) señal LISTO cuyas variables restringen la selección (XGBoost parte de las de LightGBM)
py_correr_modelo <- function(script, carpeta, etapa_datos, presupuesto = "", hilos = 4, pool = "") {
  hilos <- as.character(as.integer(Sys.getenv("PY_HILOS", hilos)))
  py_ejecutar("datos_cadena.py", args = etapa_datos, entorno = c(AJUSTE_HILOS = hilos, P7_HILOS = hilos))
  entorno <- c(SALIDAS_CARPETA = carpeta, AJUSTE_HILOS = hilos, P7_HILOS = hilos, OMP_NUM_THREADS = hilos,
               MKL_NUM_THREADS = hilos, OPENBLAS_NUM_THREADS = hilos)
  if (nzchar(presupuesto)) entorno <- c(entorno, P7_PRESUPUESTO = presupuesto)
  if (nzchar(pool)) entorno <- c(entorno, P7_POOL = file.path(py_raiz(), pool))
  py_ejecutar(script, entorno = entorno)
}
