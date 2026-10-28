# ==============================================================================
# 23_XGB_fe_depth_6_eta_005_nr_541_umbral_0515.R
# Autor: Cesar
# Modelo: XGBoost (clasificación, binary:logistic, hist) con variables nuevas ("fe").
#
# Lee 00_data/raw/*.csv, construye la base de hogares, entrena, elige el umbral de decisión y genera el
# archivo de envío. No depende de otros scripts ni del runner (00); detecta la raíz del proyecto.
#
# El nombre del archivo recoge el modelo ganador: conjunto fe, max_depth 6, eta 0.05, 541 árboles y
# umbral 0.515.
#
# Parámetro de ejecución (sección "PARÁMETROS DE EJECUCIÓN"):
#   USAR_PARAMETROS_GANADORES = TRUE  entrena la configuración ganadora, sin búsqueda;
#                               FALSE repite el screening y la búsqueda aleatoria completa.
# ==============================================================================

# ============================================================
# PARÁMETROS DE EJECUCIÓN
# ============================================================
AUTOR <- "Cesar"
USAR_PARAMETROS_GANADORES <- TRUE

# Configuración ganadora: hiperparámetros y 541 árboles de la descripción del modelo (coinciden con
# `params_base`). La estrategia de desbalance no figura en el nombre ni en la descripción: se supone
# "pesos_raiz", porque con "ninguno" el umbral F1-óptimo de un clasificador como este ronda 0.35, y un
# umbral de 0.515 solo cuadra con probabilidades infladas por scale_pos_weight (la raíz de n_No/n_Yes ~ 2
# lleva el óptimo a ~0.5). `desbalance` es el campo que corresponde ajustar si la estrategia real difiere.
GANADOR <- list(variables = "fe", desbalance = "pesos_raiz", nrounds = 541L, umbral_ref = 0.515)

# ============================================================
# PAQUETES Y RAÍZ DEL PROYECTO
# ============================================================
# pacman::p_load carga cada paquete y lo instala si falta. No se carga el meta-paquete tidyverse
# (arrastra ragg): el script solo usa sus componentes.
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  readr, dplyr, tidyr, purrr, tibble, stringr,   # datos y modelo
  xgboost,                                       # gradient boosting
  caret,                                         # dummyVars, createFolds y confusionMatrix
  MLmetrics,                                     # PRAUC y AUC
  here                                           # raíz del proyecto
)

# Raíz del proyecto: carpeta que contiene 00_data/raw; las rutas posteriores son relativas a ella.
# Se busca, en orden, en RAIZ_PROYECTO, la carpeta del script (source() o RStudio), el directorio de
# trabajo y here::here().
RAIZ_PROYECTO <- ""   # ruta opcional de la raíz; vacío = detección automática
buscar_raiz <- function(desde) {
  d <- normalizePath(desde, winslash = "/", mustWork = FALSE)
  for (i in 1:6) {                                    # sube hasta 6 niveles buscando 00_data/raw
    if (dir.exists(file.path(d, "00_data", "raw"))) return(d)
    padre <- dirname(d)
    if (identical(padre, d)) break
    d <- padre
  }
  NA_character_
}
ruta_script <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)            # source("ruta/script.R")
if (is.null(ruta_script) && requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable())
  ruta_script <- tryCatch(rstudioapi::getSourceEditorContext()$path, error = function(e) NULL)   # RStudio
candidatos <- c(RAIZ_PROYECTO, if (length(ruta_script) && nzchar(ruta_script)) dirname(ruta_script),
                getwd(), here::here())
ROOT <- NA_character_
for (cand in candidatos[nzchar(candidatos)]) { ROOT <- buscar_raiz(cand); if (!is.na(ROOT)) break }
if (is.na(ROOT)) stop("No encuentro la carpeta del proyecto (la que contiene 00_data/raw). Define RAIZ_PROYECTO ",
                      "con su ruta o ejecuta el script desde la carpeta del proyecto. Directorio actual: ", getwd())
setwd(ROOT)
message("Raíz del proyecto: ", ROOT)
dir.create("salidas", showWarnings = FALSE)

# ============================================================
# BASE ANALÍTICA DE HOGARES
# ============================================================
#
# Construye, desde los CSV originales (00_data/raw/, solo lectura), una fila por hogar con:
#   - variables originales del hogar que existen en test (vivienda, tamaño, zona);
#   - variables construidas desde hogares;
#   - variables construidas desde personas y agregadas al hogar;
#   - 4 banderas de recepción de ingresos (condicionales: están en test, pero vienen de preguntas
#     de ingresos).
# Produce `db_train` (con la respuesta Pobre) y `db_test`, que usa el modelo. Se construyen en memoria:
# no se escriben archivos intermedios.
#
# Supuestos de limpieza (no verificados con la documentación del DANE):
#   - códigos 1, 2, 3... en el orden de las opciones y 9 = no sabe;
#   - 98 y 99 en las variables de vivienda son "no sabe / no informa", no montos: pasan a NA.

# Lee solo las columnas pedidas. Todo es numérico; las llaves y la geografía se leen como texto
# (conservan ceros a la izquierda, p. ej. Depto "05").
leer_raw <- function(archivo, columnas) {
  read_csv(file.path("00_data", "raw", archivo), col_select = all_of(columnas), progress = FALSE,
           col_types = cols(.default = col_double(), id = col_character(), Clase = col_character(),
                            Dominio = col_character(), Depto = col_character()))
}

# Variables del hogar: originales de test y construidas.
# P5100, P5130 y P5140 son NA por diseño según la tenencia: no se imputan.
features_hogar <- function(h) {
  # La cuota solo la paga quien tiene vivienda propia en pago: su NA es estructural. Se guarda antes
  # de limpiar 98/99 para no confundir "no sabe" con "no paga".
  cuota_estructural <- is.na(h$P5100)
  cuota_cruda       <- h$P5100
  for (v in c("P5000", "P5010", "P5100", "P5130", "P5140")) h[[v]][h[[v]] %in% c(98, 99)] <- NA_real_

  h |>
    mutate(
      nper = pmax(Nper, 1), npug = pmax(Npersug, 1),                  # evitan dividir por cero
      rural = as.integer(Clase == "2"),                               # Clase 2 = resto (rural)
      hacinamiento = nper / pmax(P5010, 1),                           # personas por cuarto de dormir
      cuartos_pc   = P5000 / nper,                                    # cuartos totales por persona
      # Seis dummies de tenencia (propia pagada, propia pagando, arriendo, usufructo, posesión, otra).
      tenencia_propia_pagada  = as.integer(P5090 %in% 1),
      tenencia_propia_pagando = as.integer(P5090 %in% 2),
      tenencia_arriendo       = as.integer(P5090 %in% 3),
      tenencia_usufructo      = as.integer(P5090 %in% 4),
      tenencia_posesion       = as.integer(P5090 %in% 5),
      tenencia_otra           = as.integer(P5090 %in% 6),
      # Arriendo pagado si arrienda; estimado en los demás casos.
      valor_arriendo          = if_else(P5090 %in% 3, P5140, P5130),
      valor_arriendo_faltante = as.integer(is.na(valor_arriendo)),
      renta_pc                = valor_arriendo / nper,
      renta_pc_lp             = renta_pc / Lp,
      carga_arriendo_lp       = if_else(P5090 %in% 3, P5140 / (Lp * npug), 0),
      # Crédito de vivienda: la bandera usa el valor crudo (98 también implica que paga cuota).
      tiene_credito_vivienda  = as.integer(!cuota_estructural & cuota_cruda > 0),
      cuota_pc                = if_else(cuota_estructural, 0, P5100 / nper),
      log_nper = log(nper), hogar_unipersonal = as.integer(Nper == 1), razon_ug = npug / nper) |>
    rename(peso = Fex_c) |>                                           # peso muestral: metadato, no predictor
    select(id, any_of("Pobre"), peso, Dominio, Depto, rural, P5000, P5010, P5090, P5100, P5130, P5140,
           Nper, Npersug, Lp, hacinamiento, cuartos_pc, starts_with("tenencia_"), valor_arriendo,
           valor_arriendo_faltante, renta_pc, renta_pc_lp, carga_arriendo_lp, tiene_credito_vivienda,
           cuota_pc, log_nper, hogar_unipersonal, razon_ug)
}

# Variables de personas agregadas al hogar y banderas de recepción de ingresos. Todo son
# conteos simples (Fex_c es constante dentro del hogar); con denominador cero el cociente vale 0 y lo
# marca `sin_ocupados`; horas_prom_ocupados queda NA si el hogar no tiene ocupados.
features_personas <- function(p) {
  p <- p |>
    mutate(across(c(Pet, Oc, Des, Ina), ~ replace_na(.x, 0))) |>      # NA = fuera de la PET / no aplica
    mutate(
      pet = Pet == 1, ocupado = Oc == 1, desocupado = Des == 1,
      adulto_18_65   = between(P6040, 18, 65),
      inactivo_18_65 = adulto_18_65 & Ina == 1,
      menor_15 = P6040 < 15, edad_15_64 = between(P6040, 15, 64), mayor_65 = P6040 >= 65,
      adulto_18 = P6040 >= 18,
      educ_media_o_mas = adulto_18 & P6210 %in% 5:6,                  # 5 media, 6 superior
      subsidiado = P6100 %in% 3, sin_salud = P6090 %in% 2,
      ocup_no_cotiza = ocupado & P6920 %in% 2,
      ocup_cp_jornal = ocupado & P6430 %in% c(4, 8),                  # cuenta propia o jornalero
      ocup_micro     = ocupado & P6870 %in% 1:2,                      # trabaja solo o con 2 a 3 personas
      ocup_mas_horas = ocupado & P7090 %in% 1,
      es_hijo = P6050 %in% 3, es_conyuge = P6050 %in% 2,
      horas_ocupado = if_else(ocupado, P6800, NA_real_))

  agg <- p |>
    group_by(id) |>
    summarize(
      n_personas = n(),
      across(c(pet, ocupado, desocupado, adulto_18_65, inactivo_18_65, menor_15, edad_15_64, mayor_65,
               adulto_18, educ_media_o_mas, subsidiado, sin_salud, ocup_no_cotiza, ocup_cp_jornal,
               ocup_micro, ocup_mas_horas, es_hijo, es_conyuge), sum),
      horas_prom_ocupados = mean(horas_ocupado, na.rm = TRUE),
      hogar_recibe_pension     = as.integer(any(P7500s2 %in% 1)),
      hogar_recibe_remesas     = as.integer(any(P7510s2 %in% 1)),
      hogar_recibe_ayudas_inst = as.integer(any(P7510s3 %in% 1)),
      hogar_recibe_subsidio    = as.integer(any(c(P6585s1, P6585s2, P6585s3, P6585s4) %in% 1)),
      .groups = "drop") |>
    mutate(horas_prom_ocupados = if_else(is.nan(horas_prom_ocupados), NA_real_, horas_prom_ocupados))

  # Atributos del jefe o jefa del hogar (P6050 == 1).
  jefe <- p |>
    filter(P6050 %in% 1) |>
    distinct(id, .keep_all = TRUE) |>
    transmute(id, jefe_mujer = as.integer(P6020 %in% 2), edad_jefe = P6040, educ_jefe = P6210,
              jefe_ocupado = as.integer(ocupado))

  den <- function(x) pmax(x, 1)
  agg |>
    left_join(jefe, by = "id") |>
    transmute(
      id, n_ocupados = ocupado, ocupados_pc = ocupado / n_personas, sin_ocupados = as.integer(ocupado == 0),
      tasa_ocupacion_pet = ocupado / den(pet), prop_inactivos_18_65 = inactivo_18_65 / den(adulto_18_65),
      tasa_desocupacion = desocupado / den(ocupado + desocupado),
      razon_dependencia = (menor_15 + mayor_65) / den(edad_15_64),
      prop_menores_15 = menor_15 / n_personas, prop_mayores_65 = mayor_65 / n_personas,
      hogar_monoparental = as.integer(es_conyuge == 0 & es_hijo > 0),
      jefe_mujer, edad_jefe, educ_jefe, jefe_ocupado,
      prop_adultos_media_o_mas = educ_media_o_mas / den(adulto_18),
      prop_subsidiado = subsidiado / n_personas, prop_sin_salud = sin_salud / n_personas,
      prop_ocup_no_cotiza = ocup_no_cotiza / den(ocupado), prop_ocup_cp_jornal = ocup_cp_jornal / den(ocupado),
      prop_ocup_micro = ocup_micro / den(ocupado), prop_ocup_mas_horas = ocup_mas_horas / den(ocupado),
      horas_prom_ocupados, hogar_recibe_pension, hogar_recibe_remesas, hogar_recibe_ayudas_inst,
      hogar_recibe_subsidio, aux_n_personas = n_personas)       # auxiliar de control: se descarta al unir
}

# Une hogares con el agregado de personas de un lado ("train" o "test") y devuelve un data.frame.
armar_base <- function(lado) {
  cols_hog <- c("id", "Clase", "Dominio", "Depto", "P5000", "P5010", "P5090", "P5100", "P5130", "P5140",
                "Nper", "Npersug", "Lp", "Fex_c", if (lado == "train") "Pobre")   # Pobre solo existe en train
  cols_per <- c("id", "P6020", "P6040", "P6050", "P6090", "P6100", "P6210", "P6430", "P6800", "P6870",
                "P6920", "P7090", "Pet", "Oc", "Des", "Ina", "P6585s1", "P6585s2", "P6585s3", "P6585s4",
                "P7500s2", "P7510s2", "P7510s3")
  h <- leer_raw(paste0(lado, "_hogares.csv"), cols_hog)
  s <- left_join(features_hogar(h), features_personas(leer_raw(paste0(lado, "_personas.csv"), cols_per)), by = "id")
  # Un hogar por fila y todos con personas (la unión no cambia el número de filas).
  stopifnot(nrow(s) == nrow(h), !anyDuplicated(s$id), !anyNA(s$aux_n_personas))
  as.data.frame(select(s, -aux_n_personas))
}

db_train <- armar_base("train")
db_test  <- armar_base("test")
stopifnot(ncol(db_train) == 59L, ncol(db_test) == 58L)        # dimensiones esperadas de la base
message("Base de hogares: db_train ", nrow(db_train), " x ", ncol(db_train),
        " | db_test ", nrow(db_test), " x ", ncol(db_test))

# ============================================================
# MODELO: XGBOOST
# ============================================================
#
# XGBoost (clasificación de pobreza, F1). Usa los mismos folds (semilla 2026), la misma población de
# evaluación (sin Bogotá) y la misma elección del umbral (suavizado + bootstrap) que los demás modelos.
#
# Definiciones:
#   - Sin splines, imputación por mediana ni estandarización: XGBoost maneja NA de forma nativa y no
#     depende de la escala ni de transformaciones monótonas.
#   - Variables: "base" (las de la base de hogares) y "fe" (base + variables nuevas de la sección 3.3,
#     las mismas que usa Elastic Net). La matriz "fe" es común a los demás algoritmos; la única
#     diferencia es que aquí las dummies no usan fullRank.
#   - Hiperparámetros: búsqueda aleatoria con xgb.cv y parada temprana (nrounds sale de la validación).
#     Se prefiere sobre una grilla porque XGBoost tiene 8 o más hiperparámetros relevantes.
#   - Desbalance: "ninguno", "pesos" (scale_pos_weight = n_No/n_Yes), "pesos_raiz" (su raíz
#     cuadrada), "up" y "down" (remuestreo dentro de cada fold de entrenamiento).
#   - Validación: la búsqueda se rankea con PR-AUC (no depende del umbral). Los mejores candidatos
#     se evalúan con F1 fuera de muestra (OOF) en su mejor umbral, calculado a mano fold por fold.
#   - La predicción de test es el promedio de los 5 modelos de los folds (no hay refit): sus
#     probabilidades son comparables con las del OOF, de donde sale el umbral.
#
# Entradas: db_train y db_test.
# Salidas (carpeta salidas/):
#   XGB_<variables>_<desbalance>_depth_*_eta_*_nr_*_umbral_*.csv   envío a Kaggle
#   XGB_screening.csv                                              variables x desbalance
#   XGB_busqueda.csv                                               búsqueda aleatoria
# Salidas (carpeta modelos/, sección 19):
#   XGB_folds/XGB_fold1.json ... XGB_fold5.json   los 5 modelos cuyo promedio predice test
#   XGB_modelo_imp.json                           modelo auxiliar con todo train (importancia)
#   XGB_pipeline.rds                              config, hiperparámetros, umbral, dv, keep y referencias
#
# Dependencias: xgboost, caret y MLmetrics (cargados con pacman::p_load). Usa funciones comunes a las
# versiones 1.x a 3.x de xgboost: xgb.DMatrix, xgb.cv, xgb.train, predict y xgb.importance.
#
# Recursos: el costo lo dominan N_BUSQUEDA y NROUNDS_MAX (una corrida corta admite N_BUSQUEDA = 10 y
# DESBALANCE_SCREENING = c("ninguno", "pesos")).
#
# Desbalance: en boosting, "up" y los pesos tienden a dar resultados casi iguales; "up" duplica filas y
# puede sobreajustar. Con "up"/"down" se reutiliza el nrounds del modelo "ninguno" (xgb.cv no permite
# remuestrear dentro del fold), por lo que su comparación es aproximada. Con cualquiera de estas
# estrategias las probabilidades quedan sesgadas hacia arriba y el umbral óptimo sube; la sección 11
# lo ajusta con el OOF.


# ============================================================
# 1. PARÁMETROS DEL MODELO
# ============================================================

# ------------------------------------------------------------
# 1.1. Configuración de la búsqueda
# ------------------------------------------------------------

# Hilos de XGBoost (el paralelismo es interno al paquete, no por fold).
N_THREADS <- max(1L, parallel::detectCores() - 1L)

# Conjuntos de variables y estrategias de desbalance del screening (sección 8).
VARIABLES_SCREENING  <- c("base", "fe")
DESBALANCE_SCREENING <- c("ninguno", "pesos", "pesos_raiz", "up")

# Búsqueda aleatoria de hiperparámetros (sección 9).
N_BUSQUEDA <- 30     # combinaciones muestreadas
TOP_K      <- 5      # cuántas se reevalúan con F1 OOF
NROUNDS_MAX <- 2000  # tope de árboles
PARADA_TEMPRANA <- 50

# Réplicas bootstrap para la estabilidad del umbral (sección 11).
B_BOOT <- 200


# ============================================================
# 2. BASES DE TRAIN Y TEST
# ============================================================

# db_train (incluye la variable objetivo Pobre, 0/1) y db_test (sin Pobre; base sobre la que se producen
# las predicciones finales) ya están en memoria (parte 1): no se leen archivos. id, Dominio y Depto son texto.

# Train y test deben tener las mismas columnas; train solo agrega la respuesta.
stopifnot(setequal(setdiff(names(db_train), names(db_test)), "Pobre"),
          length(setdiff(names(db_test), names(db_train))) == 0)

# Un hogar por fila: id no puede repetirse.
stopifnot(!anyDuplicated(db_train$id), !anyDuplicated(db_test$id))


# ============================================================
# 3. PREPARACIÓN DE LAS BASES
# ============================================================

# ------------------------------------------------------------
# 3.1. Variable objetivo y factores
# ------------------------------------------------------------

# 1 = pobre; 0 = no pobre. El primer nivel es la clase positiva.
db_train <- db_train |>
  mutate(Pobre = factor(Pobre, levels = c(1, 0), labels = c("Yes", "No")))

# Los niveles de la geografía se fijan solo con train.
niveles_dominio <- sort(unique(db_train$Dominio))
niveles_depto   <- sort(unique(db_train$Depto))
stopifnot(all(db_test$Dominio %in% niveles_dominio), all(db_test$Depto %in% niveles_depto))

a_factores <- function(.db) {
  .db |> mutate(Dominio = factor(Dominio, levels = niveles_dominio),
                Depto   = factor(Depto,   levels = niveles_depto))
}
db_train <- a_factores(db_train)
db_test  <- a_factores(db_test)

# Población de evaluación. Si Bogotá está en train y no en test, el F1 y el umbral se miden
# solo sobre los hogares no-Bogotá del OOF: es la población que se parece a test.
bogota_en_train <- any(db_train$Dominio == "BOGOTA")
bogota_en_test  <- any(db_test$Dominio  == "BOGOTA")
idx_eval <- if (bogota_en_train && !bogota_en_test) {
  which(db_train$Dominio != "BOGOTA")
} else {
  seq_len(nrow(db_train))
}
message("Población de evaluación OOF: ", length(idx_eval), " de ", nrow(db_train),
        " hogares (", if (length(idx_eval) < nrow(db_train)) "sin Bogotá" else "todos", ")")

# ------------------------------------------------------------
# 3.2. Faltantes estructurales: cero más bandera
# ------------------------------------------------------------

# Se mantiene igual que en Elastic Net para que todos los algoritmos vean los mismos predictores.
# Los NA que quedan se dejan: XGBoost los envía por la rama que más reduce la pérdida.
vars_cero <- intersect(c("P5100", "P5130", "P5140", "horas_prom_ocupados"), names(db_train))

a_cero <- function(.db) .db |> mutate(across(all_of(vars_cero), ~ replace_na(.x, 0)))
db_train <- a_cero(db_train)
db_test  <- a_cero(db_test)

colSums(is.na(db_train))[colSums(is.na(db_train)) > 0]

# No son predictores: id (llave) y peso (factor de expansión del diseño muestral).
vars_no_predictoras <- c("id", "peso")

# ------------------------------------------------------------
# 3.3. Variables nuevas (mismas que Elastic Net, sin splines)
# ------------------------------------------------------------

# Referencias que dependen de los datos: se calculan SOLO con train y se aplican igual a test.
# Son no supervisadas (no usan Pobre), así que no filtran la respuesta.
calcular_ref <- function(.db) {
  ref <- list()
  if (all(c("valor_arriendo", "Dominio") %in% names(.db))) {
    ref$med_log_arr_dom <- .db |>
      group_by(Dominio) |>
      summarise(m = median(log1p(pmax(valor_arriendo, 0)), na.rm = TRUE), .groups = "drop") |>
      deframe()
  }
  if (all(c("educ_jefe", "Dominio") %in% names(.db))) {
    ref$med_educ_dom <- .db |>
      group_by(Dominio) |>
      summarise(m = mean(educ_jefe, na.rm = TRUE), .groups = "drop") |>
      deframe()
  }
  if ("educ_jefe" %in% names(.db) && n_distinct(.db$educ_jefe, na.rm = TRUE) <= 12) {
    ref$niveles_educ <- sort(unique(.db$educ_jefe[!is.na(.db$educ_jefe)]))
  }
  ref
}

crear_features <- function(.db, ref) {
  d <- .db
  omitidas <- character()

  # Crea la variable `nom` solo si existen todas las columnas de `req`. Si la base no trae alguna,
  # la variable se omite y queda registrada (no se rompe el script).
  agregar <- function(nom, req, f) {
    if (all(req %in% names(d))) d[[nom]] <<- f(d) else omitidas <<- c(omitidas, nom)
  }

  # --- Umbral de ingreso del hogar y carga de vivienda sobre ese umbral ---------------
  agregar("log_umbral_hogar", c("Lp", "Npersug"),
          \(x) log(x$Lp * pmax(x$Npersug, 1)))
  agregar("arriendo_sobre_umbral", c("valor_arriendo", "Lp", "Npersug"),
          \(x) pmin(pmax(x$valor_arriendo, 0) / (x$Lp * pmax(x$Npersug, 1)), 3))
  agregar("arriendo_por_cuarto_log", c("valor_arriendo", "P5000"),
          \(x) log1p(pmax(x$valor_arriendo, 0) / pmax(x$P5000, 1)))

  # --- Composición del hogar -----------------------------------------------------------
  agregar("n_menores", c("Nper", "prop_menores_15"), \(x) round(x$Nper * x$prop_menores_15))
  agregar("n_mayores65", c("Nper", "prop_mayores_65"), \(x) round(x$Nper * x$prop_mayores_65))
  agregar("n_adultos_15_64", c("Nper", "n_menores", "n_mayores65"),
          \(x) pmax(x$Nper - x$n_menores - x$n_mayores65, 0))
  agregar("dependientes_por_ocupado", c("Nper", "n_ocupados"),
          \(x) log1p(pmax(x$Nper - x$n_ocupados, 0) / pmax(x$n_ocupados, 1)))
  agregar("menores_por_ocupado", c("n_menores", "n_ocupados"),
          \(x) x$n_menores / pmax(x$n_ocupados, 1))

  # --- Calidad del empleo en términos per cápita ---------------------------------------
  agregar("informales_pc", c("n_ocupados", "prop_ocup_no_cotiza", "Nper"),
          \(x) x$n_ocupados * x$prop_ocup_no_cotiza / pmax(x$Nper, 1))
  agregar("formales_pc", c("n_ocupados", "prop_ocup_no_cotiza", "Nper"),
          \(x) x$n_ocupados * (1 - x$prop_ocup_no_cotiza) / pmax(x$Nper, 1))
  agregar("cp_jornal_pc", c("n_ocupados", "prop_ocup_cp_jornal", "Nper"),
          \(x) x$n_ocupados * x$prop_ocup_cp_jornal / pmax(x$Nper, 1))
  agregar("micro_pc", c("n_ocupados", "prop_ocup_micro", "Nper"),
          \(x) x$n_ocupados * x$prop_ocup_micro / pmax(x$Nper, 1))
  agregar("horas_totales_pc", c("n_ocupados", "horas_prom_ocupados", "Nper"),
          \(x) x$n_ocupados * x$horas_prom_ocupados / pmax(x$Nper, 1))
  agregar("log_horas_totales_pc", c("horas_totales_pc"), \(x) log1p(x$horas_totales_pc))
  agregar("educ_x_ocupados", c("prop_adultos_media_o_mas", "ocupados_pc"),
          \(x) x$prop_adultos_media_o_mas * x$ocupados_pc)

  # --- Diferencias ---------------------------------------------------------------------
  agregar("dif_ocup_pet", c("tasa_ocupacion_pet", "ocupados_pc"),
          \(x) x$tasa_ocupacion_pet - x$ocupados_pc)
  agregar("dif_nper_npersug", c("Nper", "Npersug"), \(x) x$Nper - x$Npersug)
  agregar("dif_cuartos_dormitorios", c("P5000", "P5010"), \(x) x$P5000 - x$P5010)

  # --- Relativas al mercado local (Dominio), con referencias de train ------------------
  if (!is.null(ref$med_log_arr_dom)) {
    agregar("arriendo_rel_dominio", c("valor_arriendo", "Dominio"),
            \(x) log1p(pmax(x$valor_arriendo, 0)) - unname(ref$med_log_arr_dom[as.character(x$Dominio)]))
  }
  if (!is.null(ref$med_educ_dom)) {
    agregar("educ_jefe_rel_dominio", c("educ_jefe", "Dominio"),
            \(x) x$educ_jefe - unname(ref$med_educ_dom[as.character(x$Dominio)]))
  }

  # --- Interacciones (los árboles las encuentran solos, pero se conservan para que el
  #     conjunto "fe" sea idéntico al de los modelos lineales) -------------------------
  if ("rural" %in% names(d)) {
    for (v in c("Nper", "hacinamiento", "educ_jefe", "ocupados_pc",
                "prop_menores_15", "prop_ocup_no_cotiza")) {
      agregar(paste0("rural_x_", v), c("rural", v), \(x) x$rural * x[[v]])
    }
  }
  agregar("jefe_mujer_x_monoparental", c("jefe_mujer", "hogar_monoparental"),
          \(x) x$jefe_mujer * x$hogar_monoparental)
  agregar("jefe_mujer_x_sin_ocupados", c("jefe_mujer", "sin_ocupados"),
          \(x) x$jefe_mujer * x$sin_ocupados)
  agregar("hacin_x_menores", c("hacinamiento", "prop_menores_15"),
          \(x) x$hacinamiento * x$prop_menores_15)
  agregar("sin_ocupados_x_menores", c("sin_ocupados", "prop_menores_15"),
          \(x) x$sin_ocupados * x$prop_menores_15)

  # --- Educación del jefe como factor --------------------------------------------------
  if (!is.null(ref$niveles_educ) && "educ_jefe" %in% names(d)) {
    d$educ_jefe_f <- factor(d$educ_jefe, levels = ref$niveles_educ)
  }

  # --- Transformaciones: log1p de montos (no cambian los árboles, pero mantienen el mismo
  #     conjunto que Elastic Net) -------------------------------------------------------
  a_log <- intersect(c("P5100", "P5130", "P5140", "valor_arriendo", "renta_pc", "cuota_pc",
                       "renta_pc_lp", "carga_arriendo_lp"), names(d))
  for (v in a_log) d[[paste0("log_", v)]] <- log1p(pmax(d[[v]], 0))
  d <- d |> select(-all_of(a_log))

  attr(d, "omitidas") <- omitidas
  d
}

ref_train <- calcular_ref(db_train)
db_train_fe <- crear_features(db_train, ref_train)
db_test_fe  <- crear_features(db_test,  ref_train)

omitidas <- attr(db_train_fe, "omitidas")
if (length(omitidas) > 0) message("Variables nuevas omitidas por columnas ausentes: ",
                                   paste(omitidas, collapse = ", "))

# Columnas nuevas respecto de la base de hogares.
setdiff(names(db_train_fe), names(db_train))

# ------------------------------------------------------------
# 3.4. Matrices numéricas por conjunto de variables
# ------------------------------------------------------------

# Dummies ajustadas solo con train. Para árboles no se usa fullRank (una dummy por nivel). Los NA se
# conservan (na.pass). Se descartan columnas constantes y se sanean los nombres (XGBoost no acepta
# algunos caracteres en nombres de variables).
armar_matrices <- function(.tr, .te) {
  x_tr_df <- .tr |> select(-any_of(c(vars_no_predictoras, "Pobre")))
  x_te_df <- .te |> select(-any_of(vars_no_predictoras))
  stopifnot(identical(names(x_tr_df), names(x_te_df)))
  dv <- dummyVars(~ ., data = x_tr_df, fullRank = FALSE)
  xt <- predict(dv, newdata = x_tr_df, na.action = na.pass)
  xs <- predict(dv, newdata = x_te_df, na.action = na.pass)
  sd_col <- apply(xt, 2, sd, na.rm = TRUE)
  keep <- !is.na(sd_col) & sd_col > 0
  xt <- xt[, keep, drop = FALSE]
  xs <- xs[, keep, drop = FALSE]
  nombres <- make.names(colnames(xt), unique = TRUE)
  colnames(xt) <- nombres
  colnames(xs) <- nombres
  # dv y keep se devuelven para poder construir las mismas columnas con datos nuevos (sección 19).
  list(train = xt, test = xs, dv = dv, keep = keep)
}

y_train <- db_train$Pobre
y01     <- as.integer(y_train == "Yes")

# Con los parámetros ganadores solo se arma el conjunto ganador (no los dos del screening).
variables_usadas <- if (USAR_PARAMETROS_GANADORES) GANADOR$variables else VARIABLES_SCREENING
conjuntos <- list()
if ("base" %in% variables_usadas) conjuntos$base <- armar_matrices(db_train,    db_test)
if ("fe"   %in% variables_usadas) conjuntos$fe   <- armar_matrices(db_train_fe, db_test_fe)

# Número de columnas (con dummies) por conjunto de variables.
sapply(conjuntos, \(m) ncol(m$train))


# ============================================================
# 4. REGLA DE REFERENCIA
# ============================================================

db_referencia <- tibble(
  Hogares = length(y_train),
  proporcion_pobres = mean(y01),
  proporcion_pobres_eval = mean(y01[idx_eval]),
  Accuracy = mean(y01 == 0)
)
db_referencia

# Razón de clases (n_No / n_Yes): base de scale_pos_weight.
RAZON <- sum(y01 == 0) / sum(y01 == 1)


# ============================================================
# 5. VALIDACIÓN CRUZADA
# ============================================================

# Mismos folds que los scripts de Elastic Net (semilla 2026, estratificados por Pobre).
set.seed(2026)
folds <- createFolds(y_train, k = 5, returnTrain = TRUE)

# Versión para xgb.cv: índices de validación de cada fold.
folds_val <- unname(lapply(folds, function(tr) setdiff(seq_along(y01), tr)))


# ============================================================
# 6. FUNCIONES: F1 POR UMBRAL, PARÁMETROS, CV Y OOF
# ============================================================

# ------------------------------------------------------------
# 6.1. Curva de F1 por umbral (un solo recorrido de los datos)
# ------------------------------------------------------------

# Para cada umbral u en `us` (ordenado), F1 de la clase positiva al predecir "Yes" si p >= u.
f1_curva <- function(p, y, us) {
  b      <- findInterval(p, us)
  tp_bin <- tabulate(b[y] + 1L, nbins = length(us) + 1L)
  n_bin  <- tabulate(b + 1L,    nbins = length(us) + 1L)
  tp <- rev(cumsum(rev(tp_bin)))[-1]
  pp <- rev(cumsum(rev(n_bin)))[-1]
  2 * tp / (pp + sum(y))
}

# Umbrales amplios: con pesos o remuestreo las probabilidades suben y el óptimo puede pasar de 0.5.
us_fino <- seq(0.05, 0.95, by = 0.005)

# F1 del OOF agrupado en su mejor umbral, sobre la población comparable con test.
evaluar_oof <- function(p) {
  y <- y01[idx_eval] == 1
  pe <- p[idx_eval]
  f <- f1_curva(pe, y, us_fino)
  tibble(F1_oof = max(f), umbral = us_fino[which.max(f)],
         PRAUC = MLmetrics::PRAUC(pe, as.numeric(y)),
         AUC   = MLmetrics::AUC(pe, as.numeric(y)))
}

# ------------------------------------------------------------
# 6.2. Parámetros de XGBoost
# ------------------------------------------------------------

PARAMS_XGB <- c("max_depth", "eta", "min_child_weight", "subsample",
                "colsample_bytree", "gamma", "lambda", "alpha")

# Valores de partida (razonables para datos tabulares de ~150 mil filas).
params_base <- tibble(max_depth = 6, eta = 0.05, min_child_weight = 5, subsample = 0.8,
                      colsample_bytree = 0.7, gamma = 0, lambda = 1, alpha = 0)

# Lista de parámetros lista para xgboost. Solo "pesos" y "pesos_raiz" cambian scale_pos_weight;
# "up" y "down" remuestrean las filas (ver oof_xgb) y usan peso 1.
armar_params <- function(p, desbalance = "ninguno") {
  spw <- switch(desbalance, pesos = RAZON, pesos_raiz = sqrt(RAZON), 1)
  c(as.list(p[PARAMS_XGB]),
    list(objective = "binary:logistic", eval_metric = "aucpr", tree_method = "hist",
         nthread = N_THREADS, seed = 2026, scale_pos_weight = spw))
}

# ------------------------------------------------------------
# 6.3. nrounds por parada temprana (xgb.cv con los mismos folds)
# ------------------------------------------------------------

# Devuelve el número de árboles con mejor PR-AUC de validación y ese PR-AUC. Para "up" y "down"
# xgb.cv no puede remuestrear dentro del fold, así que se usa el esquema "ninguno".
nrounds_xgb <- function(x, p, desbalance = "ninguno") {
  modo <- if (desbalance %in% c("up", "down")) "ninguno" else desbalance
  cv <- xgb.cv(
    params = armar_params(p, modo),
    data = xgb.DMatrix(x, label = y01),
    nrounds = NROUNDS_MAX,
    folds = folds_val,
    early_stopping_rounds = PARADA_TEMPRANA,
    maximize = TRUE,
    verbose = 0
  )
  reg <- as.data.frame(cv$evaluation_log)
  i <- which.max(reg$test_aucpr_mean)
  list(nrounds = reg$iter[i], aucpr = reg$test_aucpr_mean[i],
       en_tope = reg$iter[i] >= NROUNDS_MAX)
}

# ------------------------------------------------------------
# 6.4. OOF con nrounds fijo (y promedio de los 5 modelos sobre test)
# ------------------------------------------------------------

remuestrear <- function(idx, modo) {
  pos <- idx[y01[idx] == 1]
  neg <- idx[y01[idx] == 0]
  if (modo == "up") {
    c(neg, pos, sample(pos, length(neg) - length(pos), replace = TRUE))   # clase pobre hasta 1:1
  } else {
    c(pos, sample(neg, length(pos)))                                      # clase No hasta 1:1
  }
}

# Entrena un modelo por fold en el 80 % de train (remuestreado si corresponde) y predice el 20 %
# restante. Si se entrega x_test, también promedia las predicciones de los 5 modelos sobre test.
# Si se entrega dir_modelos, guarda cada modelo de fold en JSON (xgb.save): son los modelos que
# producen la predicción de test.
oof_xgb <- function(x, p, nrounds, desbalance = "ninguno", x_test = NULL, dir_modelos = NULL) {
  oof <- numeric(nrow(x))
  d_test <- if (!is.null(x_test)) xgb.DMatrix(x_test) else NULL
  pred_test <- if (!is.null(x_test)) numeric(nrow(x_test)) else NULL
  prm <- armar_params(p, desbalance)

  for (k in seq_along(folds)) {
    tr <- folds[[k]]
    va <- setdiff(seq_along(y01), tr)
    set.seed(2026 + k)
    tr_m <- if (desbalance %in% c("up", "down")) remuestrear(tr, desbalance) else tr
    m <- xgb.train(params = prm,
                   data = xgb.DMatrix(x[tr_m, , drop = FALSE], label = y01[tr_m]),
                   nrounds = nrounds, verbose = 0)
    if (!is.null(dir_modelos)) {
      dir.create(dir_modelos, showWarnings = FALSE, recursive = TRUE)
      xgb.save(m, file.path(dir_modelos, sprintf("XGB_fold%d.json", k)))
    }
    oof[va] <- predict(m, xgb.DMatrix(x[va, , drop = FALSE]))
    if (!is.null(d_test)) pred_test <- pred_test + predict(m, d_test) / length(folds)
  }
  list(oof = oof, test = pred_test)
}


# ============================================================
# 7. ESPACIO DE BÚSQUEDA DE HIPERPARÁMETROS
# ============================================================

# Muestreo aleatorio (escala logarítmica donde el efecto es multiplicativo).
#   max_depth, min_child_weight, gamma: complejidad de cada árbol.
#   eta: tasa de aprendizaje (con parada temprana, más baja = más árboles y más estable).
#   subsample, colsample_bytree: aleatorización tipo bosque (reduce varianza).
#   lambda, alpha: regularización L2 y L1 de los pesos de las hojas.
muestrear_params <- function(n) {
  tibble(
    max_depth        = sample(3:9, n, replace = TRUE),
    eta              = 10^runif(n, log10(0.02), log10(0.15)),
    min_child_weight = 10^runif(n, 0, log10(50)),
    subsample        = runif(n, 0.6, 1),
    colsample_bytree = runif(n, 0.4, 1),
    gamma            = ifelse(runif(n) < 0.4, 0, 10^runif(n, -3, log10(5))),
    lambda           = 10^runif(n, -1, log10(20)),
    alpha            = 10^runif(n, -3, log10(5))
  )
}


# ============================================================
# 8. SCREENING: VARIABLES x DESBALANCE (con parámetros de partida)
# ============================================================

dir.create("salidas", showWarnings = FALSE)

# USAR_PARAMETROS_GANADORES = FALSE: screening (8) + búsqueda aleatoria (9).
# TRUE: se omite lo anterior y se usa la configuración ganadora (rama `else` al final de la sección 9).
if (!USAR_PARAMETROS_GANADORES) {

orden_desbalance <- c("ninguno", "pesos", "pesos_raiz", "up", "down")
configs <- expand.grid(variables = VARIABLES_SCREENING, desbalance = DESBALANCE_SCREENING,
                       stringsAsFactors = FALSE) |>
  arrange(match(desbalance, orden_desbalance), variables)

nr_ref <- list()   # nrounds del esquema "ninguno" por conjunto de variables (para up/down)
resultados <- vector("list", nrow(configs))

for (i in seq_len(nrow(configs))) {
  cfg <- configs[i, ]
  message(sprintf("[%d/%d] variables = %s | desbalance = %s", i, nrow(configs),
                  cfg$variables, cfg$desbalance))
  t0 <- Sys.time()
  x <- conjuntos[[cfg$variables]]$train

  if (cfg$desbalance %in% c("up", "down")) {
    if (is.null(nr_ref[[cfg$variables]])) {
      nr_ref[[cfg$variables]] <- nrounds_xgb(x, params_base, "ninguno")$nrounds
    }
    nr <- nr_ref[[cfg$variables]]
    aucpr_cv <- NA_real_
  } else {
    r <- nrounds_xgb(x, params_base, cfg$desbalance)
    nr <- r$nrounds
    aucpr_cv <- r$aucpr
    if (cfg$desbalance == "ninguno") nr_ref[[cfg$variables]] <- nr
  }

  p_oof <- oof_xgb(x, params_base, nr, cfg$desbalance)$oof
  resultados[[i]] <- bind_cols(
    tibble(variables = cfg$variables, desbalance = cfg$desbalance,
           n_cols = ncol(x), nrounds = nr, PRAUC_cv = aucpr_cv),
    evaluar_oof(p_oof),
    tibble(minutos = as.numeric(difftime(Sys.time(), t0, units = "mins")))
  )
  rm(x, p_oof); gc(verbose = FALSE)
}

screening <- bind_rows(resultados) |> arrange(desc(F1_oof))
write.csv(screening, file.path("salidas", "XGB_screening.csv"), row.names = FALSE)

# F1_oof: F1 del OOF agrupado, en el umbral óptimo, sobre la población comparable con test.
# Con una sola corrida, diferencias menores a 0.002 son ruido.
print(screening |> mutate(across(where(is.numeric), ~ round(.x, 5))), width = Inf)

mejor_cfg <- screening[1, ]


# ============================================================
# 9. BÚSQUEDA ALEATORIA DE HIPERPARÁMETROS (sobre la mejor configuración)
# ============================================================

x_train <- conjuntos[[mejor_cfg$variables]]$train
x_test  <- conjuntos[[mejor_cfg$variables]]$test
stopifnot(identical(colnames(x_train), colnames(x_test)))

# Para "up" y "down" la búsqueda se hace con el esquema "ninguno" (limitación de xgb.cv).
modo_busqueda <- if (mejor_cfg$desbalance %in% c("up", "down")) "ninguno" else mejor_cfg$desbalance

set.seed(2026)
candidatos <- muestrear_params(N_BUSQUEDA)

busqueda <- vector("list", N_BUSQUEDA)
for (i in seq_len(N_BUSQUEDA)) {
  message(sprintf("Búsqueda %d/%d", i, N_BUSQUEDA))
  r <- nrounds_xgb(x_train, candidatos[i, ], modo_busqueda)
  busqueda[[i]] <- bind_cols(candidatos[i, ],
                             tibble(nrounds = r$nrounds, PRAUC_cv = r$aucpr, en_tope = r$en_tope))
}
busqueda <- bind_rows(busqueda) |> arrange(desc(PRAUC_cv))

# Reevaluación de los TOP_K con F1 OOF (y de los parámetros de partida como referencia). Se evalúa
# con el desbalance elegido, que puede ser "up" o "down" aunque la búsqueda no lo use.
top <- busqueda |> slice_head(n = TOP_K)
top$F1_oof <- NA_real_
top$umbral_oof <- NA_real_
for (i in seq_len(nrow(top))) {
  message(sprintf("F1 OOF del candidato %d/%d", i, nrow(top)))
  ev <- evaluar_oof(oof_xgb(x_train, top[i, ], top$nrounds[i], mejor_cfg$desbalance)$oof)
  top$F1_oof[i]     <- ev$F1_oof
  top$umbral_oof[i] <- ev$umbral
}
top <- top |> arrange(desc(F1_oof))

write.csv(busqueda, file.path("salidas", "XGB_busqueda.csv"), row.names = FALSE)
print(top |> mutate(across(where(is.numeric), ~ round(.x, 5))), width = Inf)

# Se compara contra los parámetros de partida (ya evaluados en el screening). Si no los supera
# por más de 0.001, se conservan: evita premiar el ruido de la búsqueda.
if (top$F1_oof[1] > mejor_cfg$F1_oof + 0.001) {
  params_final  <- top[1, ]
  nrounds_final <- top$nrounds[1]
} else {
  message("La búsqueda no superó a los parámetros de partida: se conservan.")
  params_final  <- params_base
  nrounds_final <- mejor_cfg$nrounds
}
as.data.frame(params_final[PARAMS_XGB])
nrounds_final

} else {
  # Parámetros ganadores (sin búsqueda). Define lo mismo que las secciones 8 y 9.
  message("Parámetros ganadores: conjunto ", GANADOR$variables, ", desbalance ", GANADOR$desbalance,
          ", ", GANADOR$nrounds, " árboles")
  mejor_cfg <- tibble(variables = GANADOR$variables, desbalance = GANADOR$desbalance, nrounds = GANADOR$nrounds)
  x_train <- conjuntos[[mejor_cfg$variables]]$train
  x_test  <- conjuntos[[mejor_cfg$variables]]$test
  stopifnot(identical(colnames(x_train), colnames(x_test)))
  params_final  <- params_base                     # max_depth 6, eta 0.05, etc., como en el nombre
  nrounds_final <- GANADOR$nrounds
}


# ============================================================
# 10. OOF FINAL Y PREDICCIÓN SOBRE TEST
# ============================================================

# Se repite el OOF con la configuración elegida, esta vez promediando los 5 modelos sobre test.
# Las probabilidades de test y las del OOF vienen del mismo tipo de modelo, así que el umbral que
# se elige con el OOF les aplica a ambas.
fin <- oof_xgb(x_train, params_final, nrounds_final, mejor_cfg$desbalance, x_test = x_test,
               dir_modelos = file.path("modelos", "XGB_folds"))

oof <- tibble(rowIndex = seq_along(y01), obs = y_train, Yes = fin$oof)
prob_pobre <- fin$test
stopifnot(length(prob_pobre) == nrow(db_test), !anyNA(prob_pobre))


# ============================================================
# 11. UMBRAL DE DECISIÓN
# ============================================================

#   - Población: solo hogares no-Bogotá del OOF si Bogotá no está en test (idx_eval).
#   - Estabilidad: la curva de F1 es casi plana cerca del máximo. Se suaviza con una media móvil
#     (±0.015) y se reporta la variabilidad por bootstrap.
#   - Referencia teórica: con probabilidades calibradas (esquema "ninguno") el umbral que maximiza
#     F1 es cercano a F*/2. Con pesos o remuestreo el óptimo sube porque las probabilidades se
#     inflan; el umbral lo corrige.
oof_eval <- oof[oof$rowIndex %in% idx_eval, ]
y_eval   <- oof_eval$obs == "Yes"

f1_u <- f1_curva(oof_eval$Yes, y_eval, us_fino)

f1_suave <- as.numeric(stats::filter(f1_u, rep(1 / 7, 7), sides = 2))
f1_suave[is.na(f1_suave)] <- f1_u[is.na(f1_suave)]

umbral_puntual <- us_fino[which.max(f1_u)]
umbral_opt     <- us_fino[which.max(f1_suave)]

set.seed(2026)
n_eval <- length(y_eval)
u_boot <- replicate(B_BOOT, {
  i <- sample.int(n_eval, n_eval, replace = TRUE)
  us_fino[which.max(f1_curva(oof_eval$Yes[i], y_eval[i], us_fino))]
})

resumen_umbral <- tibble(
  umbral_puntual = umbral_puntual,
  umbral_suave = umbral_opt,
  boot_p25 = quantile(u_boot, 0.25), boot_mediana = median(u_boot), boot_p75 = quantile(u_boot, 0.75),
  F1_en_0.50 = f1_u[which.min(abs(us_fino - 0.5))],
  F1_en_umbral = f1_u[which.min(abs(us_fino - umbral_opt))],
  F_estrella_medio = max(f1_u) / 2
)
resumen_umbral


# ============================================================
# 12. CONVERSIÓN DE PROBABILIDAD A CLASE
# ============================================================

umbral <- umbral_opt
if (USAR_PARAMETROS_GANADORES) message(sprintf("Umbral obtenido %.3f | umbral de referencia (nombre del archivo) %.3f", umbral, GANADOR$umbral_ref))
clase_pobre <- ifelse(prob_pobre >= umbral, "Yes", "No")

# Tasa de hogares predichos como pobres en test frente al OOF comparable: si difieren mucho,
# puede deberse a la composición por dominio (sección 18.3).
c(tasa_pred_test = mean(prob_pobre >= umbral),
  tasa_pred_oof  = mean(oof_eval$Yes >= umbral),
  prevalencia_oof = mean(y_eval))


# ============================================================
# 13. PREPARACIÓN DE PREDICCIONES PARA KAGGLE
# ============================================================

predictSample <- db_test |>
  mutate(pobre_lab = clase_pobre) |>
  select(id, pobre_lab)

head(predictSample)


# ============================================================
# 14. RECODIFICACIÓN FINAL DE LA VARIABLE PREDICHA
# ============================================================

predictSample <- predictSample |>
  mutate(pobre = if_else(pobre_lab == "Yes", 1L, 0L)) |>
  select(id, pobre)

stopifnot(nrow(predictSample) == nrow(db_test),
          !anyDuplicated(predictSample$id),
          !anyNA(predictSample),
          all(predictSample$pobre %in% c(0L, 1L)),
          setequal(predictSample$id, db_test$id))


# ============================================================
# 15. CONSTRUCCIÓN DEL NOMBRE DEL ARCHIVO
# ============================================================

fmt <- function(v, d = 3) gsub(".", "", as.character(signif(v, d)), fixed = TRUE)
umbral_str <- gsub(".", "", sprintf("%.3f", umbral), fixed = TRUE)

name <- paste0(AUTOR, "_XGB_", mejor_cfg$variables, "_", mejor_cfg$desbalance,
               "_depth_", params_final$max_depth, "_eta_", fmt(params_final$eta),
               "_nr_", nrounds_final, "_umbral_", umbral_str, ".csv")


# ============================================================
# 16. INSPECCIÓN DEL ARCHIVO FINAL
# ============================================================

head(predictSample)
name


# ============================================================
# 17. EXPORTACIÓN DEL ARCHIVO
# ============================================================

write.csv(predictSample, file.path("salidas", name), row.names = FALSE)


# ============================================================
# 18. DIAGNÓSTICOS: IMPORTANCIA, ERRORES Y POBLACIÓN
# ============================================================

# ------------------------------------------------------------
# 18.1. Importancia de variables
# ------------------------------------------------------------

# Modelo auxiliar entrenado con todo train, solo para la importancia (la predicción de test sale de
# los modelos por fold). Con "up" o "down" se usa el esquema "ninguno" para esto. Gain = aporte
# promedio a la reducción de la pérdida cuando la variable se usa en un corte.
modelo_imp <- xgb.train(
  params = armar_params(params_final,
                        if (mejor_cfg$desbalance %in% c("up", "down")) "ninguno" else mejor_cfg$desbalance),
  data = xgb.DMatrix(x_train, label = y01),
  nrounds = nrounds_final, verbose = 0
)
importancia <- as_tibble(as.data.frame(xgb.importance(model = modelo_imp)))
importancia |> slice_head(n = 25)

# ------------------------------------------------------------
# 18.2. Matriz de confusión del OOF (población de evaluación): corte 0.5 frente al umbral elegido
# ------------------------------------------------------------

cm_umbral <- function(d, u) {
  p <- factor(ifelse(d$Yes >= u, "Yes", "No"), levels = c("Yes", "No"))
  confusionMatrix(p, d$obs, positive = "Yes", mode = "prec_recall")
}

cm_opt <- cm_umbral(oof_eval, umbral)
cm_opt$table                                      # filas = predicho, columnas = observado
prop.table(cm_opt$table, margin = 2)              # diagonal: recall de "Yes" y especificidad

# Subcobertura (falsos negativos) y filtración (falsos positivos): lectura de política (Etapa 4).
do.call(rbind, lapply(c(`0.50` = 0.5, elegido = umbral), \(u) {
  cm <- cm_umbral(oof_eval, u)
  c(umbral = u, TP = cm$table["Yes", "Yes"], FP = cm$table["Yes", "No"],
    FN = cm$table["No", "Yes"], TN = cm$table["No", "No"],
    cm$byClass[c("Precision", "Recall", "F1")])
})) |> round(3)

# ------------------------------------------------------------
# 18.3. Bogotá frente al resto (OOF completo) y composición de test
# ------------------------------------------------------------

oof |>
  mutate(bogota = as.character(db_train$Dominio[rowIndex]) == "BOGOTA") |>
  group_by(bogota) |>
  summarise(n = n(), prevalencia = mean(obs == "Yes"), prob_media = mean(Yes),
            tasa_pred = mean(Yes >= umbral),
            F1 = F_meas(factor(ifelse(Yes >= umbral, "Yes", "No"), levels = c("Yes", "No")),
                        obs, relevant = "Yes"),
            .groups = "drop")

# Tasa predicha por dominio: OOF (sin Bogotá si aplica) frente a test.
tasa_dom <- bind_rows(
  tibble(Dominio = as.character(db_train$Dominio[oof_eval$rowIndex]), p = oof_eval$Yes, fuente = "OOF"),
  tibble(Dominio = as.character(db_test$Dominio), p = prob_pobre, fuente = "Test")
) |>
  group_by(Dominio, fuente) |>
  summarise(tasa_pred = mean(p >= umbral), prob_media = mean(p), .groups = "drop") |>
  tidyr::pivot_wider(names_from = fuente, values_from = c(tasa_pred, prob_media))
tasa_dom


# ============================================================
# 19. GUARDADO DEL MEJOR MODELO
# ============================================================

# La predicción de Kaggle es el promedio de los 5 modelos de los folds, guardados en
# modelos/XGB_folds/ por oof_xgb (sección 10). El umbral aplica a ese promedio.
# xgb.save (JSON) y no saveRDS: el objeto de xgboost guarda un puntero que no sobrevive entre sesiones.
dir.create("modelos", showWarnings = FALSE)

# Modelo auxiliar entrenado con todo train (el de la importancia, 18.1). Sirve para inspección,
# no para reproducir el envío a Kaggle.
xgb.save(modelo_imp, file.path("modelos", "XGB_modelo_imp.json"))

# Todo lo necesario para reconstruir las columnas y decidir la clase con datos nuevos.
saveRDS(list(
  config          = mejor_cfg,                                # variables y desbalance ganadores
  params          = params_final,                             # hiperparámetros elegidos
  nrounds         = nrounds_final,
  umbral          = umbral,
  columnas        = colnames(x_train),                        # orden exacto de las columnas del modelo
  dv              = conjuntos[[mejor_cfg$variables]]$dv,      # codificador de dummies
  keep            = conjuntos[[mejor_cfg$variables]]$keep,    # columnas no constantes
  ref_train       = ref_train,                                # medianas por Dominio (3.3)
  niveles_dominio = niveles_dominio,
  niveles_depto   = niveles_depto,
  vars_cero       = vars_cero
), file.path("modelos", "XGB_pipeline.rds"))

# Reutilización del pipeline guardado (ejemplo):
#   pipe    <- readRDS(file.path("modelos", "XGB_pipeline.rds"))
#   modelos <- lapply(1:5, \(k) xgb.load(file.path("modelos", "XGB_folds", sprintf("XGB_fold%d.json", k))))
#   prob    <- rowMeans(sapply(modelos, \(m) predict(m, xgb.DMatrix(x_nuevo))))
#   clase   <- as.integer(prob >= pipe$umbral)
#
# x_nuevo se construye igual que en train: a_factores(), a_cero(), crear_features(db, pipe$ref_train),
# predict(pipe$dv, ...), luego [, pipe$keep] y colnames(x_nuevo) <- pipe$columnas.

# ============================================================
# 20. DESCRIPCIÓN DE RESULTADOS
# ============================================================

# Dos prints y una tabla del MEJOR modelo. Todo se calcula sobre el OOF de los hogares fuera de Bogotá
# (la población que se parece a test) y en el umbral elegido:
#   print 1: hiperparámetros y resultados de la validación cruzada (F1 y umbral);
#   print 2: detalle (matriz de confusión, métricas, class weight, proporción de pobres, variables);
#   tabla  : salidas/Cesar_XGB_mejor_modelo.csv, una fila con toda esa información en columnas.
# Hay dos F1: el del umbral elegido fuera de cada fold (el de referencia) y el del umbral
# elegido en todo el OOF (optimista). No es un CV anidado: el desbalance, las variables y los
# hiperparámetros se eligieron con los mismos folds.

# ------------------------------------------------------------
# 20.1. Insumos y funciones de métricas
# ------------------------------------------------------------

p_eval    <- oof_eval$Yes                 # probabilidad OOF de la población de evaluación
y_eval_l  <- as.logical(y_eval)           # TRUE = pobre
# Pliegue en que cada hogar queda fuera de entrenamiento (folds guarda los índices de entrenamiento).
fold_id <- integer(nrow(db_train))
for (k in seq_along(folds)) fold_id[setdiff(seq_len(nrow(db_train)), folds[[k]])] <- k
fold_eval <- fold_id[oof_eval$rowIndex]   # pliegue de cada hogar evaluado (mismo orden que p_eval)

# AUC y precisión promedio (área bajo precisión-cobertura) en numérico: MLmetrics::AUC desborda el
# entero de R con tantas observaciones y devuelve NA.
auc_num <- function(p, y) {
  n1 <- as.numeric(sum(y)); n0 <- as.numeric(sum(!y))
  (sum(rank(p)[y]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
ap_num <- function(p, y) {
  o <- order(p, decreasing = TRUE); yy <- y[o]; pp <- p[o]
  fin_g <- c(which(diff(pp) != 0), length(pp)); tp_c <- cumsum(yy)[fin_g]
  sum(diff(c(0, tp_c)) / sum(y) * tp_c / fin_g)
}

# Matriz de confusión y métricas en el umbral `u` (conteos en numérico para que los productos no desborden).
metricas_umbral <- function(p, y, u) {
  pred <- p >= u
  tp <- as.numeric(sum(pred & y)); fp <- as.numeric(sum(pred & !y))
  fn <- as.numeric(sum(!pred & y)); tn <- as.numeric(sum(!pred & !y))
  tibble(TP = tp, FP = fp, FN = fn, TN = tn,
         accuracy = (tp + tn) / (tp + fp + fn + tn),
         balanced_accuracy = (tp / (tp + fn) + tn / (tn + fp)) / 2,
         precision = tp / (tp + fp),            # valor predictivo positivo
         recall = tp / (tp + fn),               # sensibilidad (tasa de verdaderos positivos)
         specificity = tn / (tn + fp),          # tasa de verdaderos negativos
         FPR = fp / (fp + tn),                  # tasa de falsos positivos
         FNR = fn / (fn + tp),                  # tasa de falsos negativos (subcobertura)
         FDR = fp / (tp + fp),                  # filtración: % de beneficiarios que no son pobres
         NPV = tn / (tn + fn),
         F1 = 2 * tp / (2 * tp + fp + fn),
         MCC = (tp * tn - fp * fn) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
}

cm_res <- metricas_umbral(p_eval, y_eval_l, umbral)
f1_pliegues <- vapply(sort(unique(fold_eval)), function(k)
  metricas_umbral(p_eval[fold_eval == k], y_eval_l[fold_eval == k], umbral)$F1, numeric(1))
auc_oof <- auc_num(p_eval, y_eval_l)
ap_oof  <- ap_num(p_eval, y_eval_l)

# F1 con el umbral de cada fold elegido FUERA de ese fold (con los otros 4, curva suavizada con media
# móvil de 7 puntos). Los aciertos y errores se acumulan sobre los 5 folds.
suavizar7 <- function(f) {
  fs <- as.numeric(stats::filter(f, rep(1 / 7, 7), sides = 2))
  fs[is.na(fs)] <- f[is.na(fs)]
  fs
}
tp_h <- fp_h <- fn_h <- 0
umbrales_fold <- numeric(length(folds))
for (k in seq_along(folds)) {
  otros <- fold_eval != k
  u_k <- us_fino[which.max(suavizar7(f1_curva(p_eval[otros], y_eval_l[otros], us_fino)))]
  umbrales_fold[k] <- u_k
  pred_k <- p_eval[!otros] >= u_k; yy_k <- y_eval_l[!otros]
  tp_h <- tp_h + sum(pred_k & yy_k); fp_h <- fp_h + sum(pred_k & !yy_k); fn_h <- fn_h + sum(!pred_k & yy_k)
}
f1_honesto <- 2 * tp_h / (2 * tp_h + fp_h + fn_h)

matriz_conf <- matrix(c(cm_res$TP, cm_res$FN, cm_res$FP, cm_res$TN), 2, 2,
                      dimnames = list(Predicho = c("Pobre", "No pobre"), Observado = c("Pobre", "No pobre")))

# Hiperparámetros del modelo final (los de params_final más árboles y estrategia de desbalance).
hp <- c(as.list(params_final[PARAMS_XGB]),
        list(nrounds = nrounds_final, desbalance = mejor_cfg$desbalance))
fmt_hp <- function(v) if (is.numeric(v)) format(signif(v, 4)) else as.character(v)
hp_txt <- paste(names(hp), vapply(hp, fmt_hp, character(1)), sep = " = ")

# Variables del modelo: las columnas de la matriz de diseño (con dummies, sin las constantes).
vars_modelo_final <- colnames(x_train)

# Class weight: "pesos" usa scale_pos_weight = n_No/n_Yes; "pesos_raiz", su raíz cuadrada; "up" y "down"
# remuestrean las filas de entrenamiento de cada fold (peso 1); "ninguno" no corrige el desbalance.
usa_cw <- mejor_cfg$desbalance != "ninguno"
spw_usado <- switch(mejor_cfg$desbalance, pesos = RAZON, pesos_raiz = sqrt(RAZON), 1)
modo_txt <- if (USAR_PARAMETROS_GANADORES) "parámetros ganadores (sin búsqueda)" else "búsqueda completa"

# ------------------------------------------------------------
# 20.2. Primer print: hiperparámetros y validación cruzada
# ------------------------------------------------------------

linea_igual <- strrep("=", 78)
imprimir_resumen <- function() {
  cat(c(linea_igual,
        "DESCRIPCIÓN DE RESULTADOS (1/2): MEJOR MODELO, HIPERPARÁMETROS Y VALIDACIÓN CRUZADA",
        linea_igual,
        sprintf("Modelo: XGBoost (binary:logistic, hist) | conjunto de variables: %s | ejecución: %s",
                mejor_cfg$variables, modo_txt),
        "Hiperparámetros:",
        strwrap(paste(hp_txt, collapse = " | "), width = 90, indent = 2, exdent = 2),
        sprintf("  (parada temprana sobre PR-AUC; predicción de test = promedio de los %d modelos de los folds)", length(folds)),
        "",
        sprintf("Validación cruzada (%d pliegues; OOF de %d hogares fuera de Bogotá, %.2f%% pobres):",
                length(folds), length(y_eval_l), 100 * mean(y_eval_l)),
        sprintf("  Umbral elegido: %.3f (F1-óptimo OOF suavizado; bootstrap p25-p75: %.3f - %.3f)",
                umbral, resumen_umbral$boot_p25, resumen_umbral$boot_p75),
        sprintf("  F1, umbral elegido fuera del fold (cifra a reportar): %.4f", f1_honesto),
        sprintf("  F1, umbral elegido en todo el OOF (optimista):        %.4f", cm_res$F1),
        sprintf("  F1 por pliegue en el umbral final: media %.4f | DE %.4f | mín %.4f | máx %.4f",
                mean(f1_pliegues), sd(f1_pliegues), min(f1_pliegues), max(f1_pliegues)),
        sprintf("  AUC %.4f | PR-AUC (precisión promedio) %.4f", auc_oof, ap_oof),
        linea_igual), sep = "\n")
}
imprimir_resumen()

# ------------------------------------------------------------
# 20.3. Segundo print: detalle del mejor modelo
# ------------------------------------------------------------

imprimir_detalle <- function() {
  cat(c("", linea_igual,
        "DESCRIPCIÓN DE RESULTADOS (2/2): DETALLE DEL MEJOR MODELO",
        linea_igual,
        sprintf("Matriz de confusión (OOF sin Bogotá, umbral %.3f; filas = predicho, columnas = observado):", umbral)),
      sep = "\n")
  print(matriz_conf)
  cat("\nMatriz de confusión por columna (% de cada clase observada):\n")
  print(round(100 * prop.table(matriz_conf, margin = 2), 2))
  cat(c("",
        "Métricas:",
        sprintf("  %-40s %.4f", c("Accuracy", "Balanced accuracy", "Precisión (PPV)", "Recall (sensibilidad, TPR)",
                                  "Especificidad (TNR)", "FP rate (FPR)", "FN rate (FNR, subcobertura)",
                                  "FDR (filtración)", "NPV", "F1", "MCC"),
                unlist(cm_res[c("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "FPR",
                                "FNR", "FDR", "NPV", "F1", "MCC")])),
        "",
        "Class weight:",
        sprintf("  desbalance = %s (scale_pos_weight = %s) -> %s", mejor_cfg$desbalance, fmt_hp(spw_usado),
                if (usa_cw) "SÍ se corrigió el desbalance de clases" else "NO se corrigió el desbalance (peso 1)"),
        "  XGBoost no pondera las filas de Bogotá: entran con peso 1 al entrenamiento.",
        "",
        "Proporción de pobres:",
        sprintf("  Train (todos los hogares):          %.2f%%", 100 * mean(y01)),
        sprintf("  Población de evaluación (sin Bogotá): %.2f%%", 100 * mean(y_eval_l)),
        sprintf("  Predicha pobre en el OOF:           %.2f%%", 100 * mean(p_eval >= umbral)),
        sprintf("  Predicha pobre en test:             %.2f%%", 100 * mean(prob_pobre >= umbral)),
        "",
        sprintf("Variables: %d en el modelo final (columnas con dummies, sin constantes)", length(vars_modelo_final)),
        strwrap(paste(vars_modelo_final, collapse = ", "), width = 90, indent = 2, exdent = 2),
        linea_igual), sep = "\n")
}
imprimir_detalle()

# ------------------------------------------------------------
# 20.4. Tabla del mejor modelo (una fila, todo en columnas)
# ------------------------------------------------------------

tabla_mejor <- bind_cols(
  tibble(modelo = "XGBoost", autor = AUTOR, ejecucion = modo_txt, conjunto_variables = mejor_cfg$variables),
  as_tibble(setNames(hp, paste0("hp_", names(hp)))),
  tibble(cv_pliegues = length(folds), cv_hogares_evaluados = length(y_eval_l),
         umbral = umbral, umbral_boot_p25 = resumen_umbral$boot_p25, umbral_boot_p75 = resumen_umbral$boot_p75,
         F1_umbral_fuera_del_fold = f1_honesto, F1_umbral_todo_oof = cm_res$F1,
         F1_pliegue_media = mean(f1_pliegues), F1_pliegue_de = sd(f1_pliegues),
         AUC = auc_oof, PR_AUC = ap_oof),
  select(cm_res, TP, FP, FN, TN, accuracy, balanced_accuracy, precision, recall, specificity, FPR, FNR, FDR, NPV, MCC),
  tibble(class_weight = usa_cw, scale_pos_weight = spw_usado,
         peso_bogota = 1,
         prop_pobres_train = mean(y01), prop_pobres_evaluacion = mean(y_eval_l),
         prop_pred_pobre_oof = mean(p_eval >= umbral), prop_pred_pobre_test = mean(prob_pobre >= umbral),
         n_variables = length(vars_modelo_final), variables = paste(vars_modelo_final, collapse = "; "),
         archivo_envio = name))
write.csv(tabla_mejor, file.path("salidas", paste0(AUTOR, "_XGB_mejor_modelo.csv")), row.names = FALSE)

# Los dos prints también quedan en un archivo de texto.
writeLines(capture.output(imprimir_resumen(), imprimir_detalle()), file.path("salidas", paste0(AUTOR, "_XGB_descripcion.txt")))
cat("Tabla del mejor modelo: salidas/", AUTOR, "_XGB_mejor_modelo.csv\n", sep = "")
