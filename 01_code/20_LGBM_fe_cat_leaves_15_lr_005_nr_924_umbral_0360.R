# ==============================================================================
# 20_LGBM_fe_cat_leaves_15_lr_005_nr_924_umbral_0360.R
# Autor: Cesar
# Modelo: LightGBM con selección de variables dentro del pliegue.
#
# Lee 00_data/raw/*.csv, construye la base de hogares, entrena, elige el umbral de decisión y genera el
# archivo de envío. No depende de otros scripts ni del runner (00); detecta la raíz del proyecto.
#
# El nombre del archivo recoge el modelo ganador: conjunto fe_cat, 15 hojas, learning_rate 0.05,
# 924 árboles y umbral 0.360.
#
# Parámetro de ejecución (sección "PARÁMETROS DE EJECUCIÓN"):
#   USAR_PARAMETROS_GANADORES = TRUE  entrena la configuración ganadora, sin búsqueda;
#                               FALSE repite el screening, la selección de K y la búsqueda completa.
# ==============================================================================

# ============================================================
# PARÁMETROS DE EJECUCIÓN
# ============================================================
AUTOR <- "Cesar"
USAR_PARAMETROS_GANADORES <- TRUE

# Configuración ganadora. learning_rate 0.05 y 15 hojas salen del nombre del archivo y coinciden con
# `params_base` (los parámetros de partida): la búsqueda no los superó. K y el peso de Bogotá NO
# figuran en el nombre: se suponen todas las variables (K = NULL) y peso 1. Con FALSE no se usa.
GANADOR <- list(variables = "fe_cat", peso_bogota = 1, K = NULL, nrounds = 924L, umbral_ref = 0.360)

# ============================================================
# PAQUETES Y RAÍZ DEL PROYECTO
# ============================================================
# pacman::p_load carga cada paquete y lo instala si falta. No se carga el meta-paquete tidyverse
# (arrastra ragg): el script solo usa sus componentes.
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  readr, dplyr, tidyr, purrr, tibble, stringr,   # datos y modelo
  lightgbm,                                      # gradient boosting
  caret,                                         # dummyVars y confusionMatrix
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
# MODELO: LIGHTGBM
# ============================================================
#
# Clasificador de gradient boosting pequeño y regularizado (pocas hojas, feature_fraction bajo y
# penalización L1) en lugar de árboles grandes sobre más de 100 columnas.
#
# Definiciones:
#   - Selección de variables dentro del pliegue (sección 9): una primera etapa ordena las variables por
#     ganancia con el 80 % de train de cada fold y la segunda usa las top-K; K se elige por F1 OOF. La
#     selección no ve las filas de validación de su fold.
#   - Peso de Bogotá (sección 8): test no incluye Bogotá; se comparan pesos 1, 0.5 y 0 (excluirla del
#     entrenamiento). La evaluación es siempre sin Bogotá.
#   - scale_pos_weight continuo dentro de la búsqueda (sección 10).
#   - Categóricas nativas (conjunto "fe_cat"): Dominio, Depto y educ_jefe_f entran como categóricas
#     de LightGBM en lugar de dummies.
#   - Bolsa de semillas (N_SEMILLAS) en el modelo final: promedia el efecto del bagging y de
#     feature_fraction.
#   - Búsqueda en dos etapas: la estructura (hojas, regularización, fracciones) se explora con
#     learning_rate alto y solo los TOP_K finalistas se reentrenan con learning_rate bajo. El espacio
#     cubre modelos pequeños (4 a 48 hojas, lambda_l1 hasta 15, feature_fraction 0.3 a 0.9) e incluye
#     max_bin, path_smooth y extra_trees.
#   - F1 con umbral elegido fuera del pliegue (sección 12): el umbral de cada fold se elige con los
#     otros 4; el F1 con el umbral elegido en todo el OOF es optimista.
#   - Diagnósticos: F1 por dominio y perfil de variables en los errores (sección 16).
#
# Entradas: db_train y db_test.
# Salidas (carpeta salidas/):
#   LGBM_<variables>_K<K>_pb<peso>_leaves_*_lr_*_nr_*_umbral_*.csv   envío a Kaggle
#   LGBM_screening.csv, LGBM_seleccionK.csv, LGBM_busqueda.csv
#   LGBM_descripcion.txt                                             descripción de resultados
# Salidas (carpeta modelos/):
#   LGBM_folds/LGBM_fold{k}_s{semilla}.txt   modelos que promedian la predicción de test
#   LGBM_modelo_imp.txt                      modelo auxiliar con todo train (importancia)
#   LGBM_pipeline.rds                        config, hiperparámetros, umbral, variables por fold, dv, keep
#
# Dependencias: lightgbm, caret y MLmetrics (cargados con pacman::p_load). Escrito para lightgbm 3.x y 4.x.
#
# Recursos: el costo lo dominan N_BUSQUEDA, NROUNDS_MAX y N_SEMILLAS (una corrida corta admite
# N_BUSQUEDA = 10, K_CANDIDATOS = c(8, 20) y N_SEMILLAS = 2).
#
# Los conjuntos "base" y "fe" usan la misma matriz que Elastic Net y XGBoost; "fe_cat" y la selección
# de variables son decisiones propias de LightGBM y forman parte de su especificación.


# ============================================================
# 1. PARÁMETROS DEL MODELO
# ============================================================

# ------------------------------------------------------------
# 1.1. Configuración de la búsqueda
# ------------------------------------------------------------

# Hilos de LightGBM (el paralelismo es interno al paquete).
N_THREADS <- max(1L, parallel::detectCores() - 1L)

# Screening (sección 8): conjuntos de variables y peso de las filas de Bogotá en el entrenamiento.
#   "base"   = variables de la base de hogares sin transformar | "fe" = base + variables nuevas (dummies)
#   "fe_cat" = fe con Dominio, Depto y educ_jefe_f como categóricas nativas de LightGBM
VARIABLES_SCREENING <- c("base", "fe", "fe_cat")
PESO_BOGOTA         <- c(1, 0.5, 0)   # 0 = Bogotá no entra al entrenamiento

# Selección de variables dentro del pliegue (sección 9). La opción "todas" siempre se evalúa.
K_CANDIDATOS     <- c(8, 12, 20, 30, 50)
STAGE1_NROUNDS   <- 150   # árboles de la primera etapa (ranking por ganancia)

# Búsqueda aleatoria (sección 10).
N_BUSQUEDA       <- 30     # combinaciones muestreadas
TOP_K            <- 5      # finalistas que se reentrenan con learning_rate bajo y se evalúan con F1 OOF
LR_BUSQUEDA      <- 0.08   # learning_rate durante la búsqueda de estructura
LR_FINAL         <- 0.025  # learning_rate de los finalistas y del modelo final
NROUNDS_MAX      <- 3000
PARADA_TEMPRANA  <- 50

# Bolsa de semillas del modelo final (sección 11).
N_SEMILLAS <- 5

# Réplicas bootstrap para la estabilidad del umbral (sección 12).
B_BOOT <- 200


# ============================================================
# 2. BASES DE TRAIN Y TEST
# ============================================================

# db_train (con la respuesta Pobre) y db_test. id, Dominio y Depto son texto.

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
es_bogota <- db_train$Dominio == "BOGOTA"
message("Población de evaluación OOF: ", length(idx_eval), " de ", nrow(db_train),
        " hogares (", if (length(idx_eval) < nrow(db_train)) "sin Bogotá" else "todos", ")")

# ------------------------------------------------------------
# 3.2. Faltantes estructurales: cero más bandera
# ------------------------------------------------------------

# Igual que en los otros scripts, para que todos los algoritmos vean los mismos predictores.
# Los NA que quedan se dejan: LightGBM los envía por la rama que más reduce la pérdida.
vars_cero <- intersect(c("P5100", "P5130", "P5140", "horas_prom_ocupados"), names(db_train))

a_cero <- function(.db) .db |> mutate(across(all_of(vars_cero), ~ replace_na(.x, 0)))
db_train <- a_cero(db_train)
db_test  <- a_cero(db_test)

colSums(is.na(db_train))[colSums(is.na(db_train)) > 0]

# No son predictores: id (llave) y peso (factor de expansión del diseño muestral).
vars_no_predictoras <- c("id", "peso")

# ------------------------------------------------------------
# 3.3. Variables nuevas (mismas que Elastic Net y XGBoost, sin splines)
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

  # --- Fuentes de ingreso no laboral (banderas de recepción; solo si la base las trae) ------
  fuentes <- intersect(c("hogar_recibe_pension", "hogar_recibe_remesas",
                         "hogar_recibe_ayudas_inst", "hogar_recibe_subsidio"), names(d))
  if (length(fuentes) >= 2) {
    d$n_fuentes_no_laborales <- rowSums(as.matrix(d[fuentes]) == 1, na.rm = TRUE)
  } else {
    omitidas <- c(omitidas, "n_fuentes_no_laborales")
  }

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

  # --- Interacciones (se conservan para que "fe" sea idéntico al de los modelos lineales) ---
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

  # --- Transformaciones: log1p de montos -------------------------------------------------
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

# Versión con dummies (una por nivel, sin fullRank). Los NA se conservan (na.pass). Se descartan
# columnas constantes y se sanean los nombres (LightGBM no acepta caracteres especiales de JSON).
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
  list(train = xt, test = xs, dv = dv, keep = keep, cats = character(0))
}

# Versión con categóricas nativas: los factores pasan como códigos enteros y se declaran en
# `cats`. Los niveles de test deben ser los de train (ya se fijaron en 3.1 y 3.3).
armar_matrices_cat <- function(.tr, .te) {
  x_tr_df <- .tr |> select(-any_of(c(vars_no_predictoras, "Pobre")))
  x_te_df <- .te |> select(-any_of(vars_no_predictoras))
  stopifnot(identical(names(x_tr_df), names(x_te_df)))
  es_fac <- vapply(x_tr_df, is.factor, logical(1))
  for (v in names(x_tr_df)[es_fac]) {
    stopifnot(identical(levels(x_tr_df[[v]]), levels(x_te_df[[v]])))
  }
  xt <- data.matrix(x_tr_df)
  xs <- data.matrix(x_te_df)
  sd_col <- apply(xt, 2, sd, na.rm = TRUE)
  keep <- !is.na(sd_col) & sd_col > 0
  xt <- xt[, keep, drop = FALSE]
  xs <- xs[, keep, drop = FALSE]
  nombres <- make.names(colnames(xt), unique = TRUE)
  cats <- nombres[names(x_tr_df)[keep] %in% names(x_tr_df)[es_fac]]
  colnames(xt) <- nombres
  colnames(xs) <- nombres
  list(train = xt, test = xs, dv = NULL, keep = keep, cats = cats)
}

y_train <- db_train$Pobre
y01     <- as.integer(y_train == "Yes")

# Con los parámetros ganadores solo se arma el conjunto ganador (no los tres del screening).
variables_usadas <- if (USAR_PARAMETROS_GANADORES) GANADOR$variables else VARIABLES_SCREENING
conjuntos <- list()
if ("base"   %in% variables_usadas) conjuntos$base   <- armar_matrices(db_train,    db_test)
if ("fe"     %in% variables_usadas) conjuntos$fe     <- armar_matrices(db_train_fe, db_test_fe)
if ("fe_cat" %in% variables_usadas) conjuntos$fe_cat <- armar_matrices_cat(db_train_fe, db_test_fe)

# Número de columnas por conjunto de variables.
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

# Razón de clases (n_No / n_Yes): tope de scale_pos_weight en la búsqueda.
RAZON <- sum(y01 == 0) / sum(y01 == 1)


# ============================================================
# 5. VALIDACIÓN CRUZADA
# ============================================================

# Mismos folds que los otros scripts (semilla 2026, estratificados por Pobre).
set.seed(2026)
folds <- createFolds(y_train, k = 5, returnTrain = TRUE)

# Versión para lgb.cv: índices de validación de cada fold.
folds_val <- unname(lapply(folds, function(tr) setdiff(seq_along(y01), tr)))

# Fold de validación al que pertenece cada hogar.
fold_id <- integer(length(y01))
for (k in seq_along(folds_val)) fold_id[folds_val[[k]]] <- k


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

us_fino <- seq(0.05, 0.95, by = 0.005)

# Media móvil de 7 puntos (±0.015); en los bordes se deja el valor original.
suavizar <- function(f) {
  fs <- as.numeric(stats::filter(f, rep(1 / 7, 7), sides = 2))
  fs[is.na(fs)] <- f[is.na(fs)]
  fs
}

# F1 del OOF agrupado en su mejor umbral, sobre la población comparable con test.
evaluar_oof <- function(p) {
  y <- y01[idx_eval] == 1
  pe <- p[idx_eval]
  f <- f1_curva(pe, y, us_fino)
  tibble(F1_oof = max(f), umbral = us_fino[which.max(f)],
         PRAUC = MLmetrics::PRAUC(pe, as.numeric(y)),
         AUC   = MLmetrics::AUC(pe, as.numeric(y)))
}

# F1 con el umbral de cada fold elegido FUERA de ese fold (con los otros 4, suavizado). Los
# aciertos y errores se acumulan sobre los 5 folds. Es una estimación sin el optimismo de elegir
# el umbral en las mismas predicciones que se evalúan.
f1_umbral_fuera <- function(p) {
  y   <- y01[idx_eval] == 1
  pe  <- p[idx_eval]
  fid <- fold_id[idx_eval]
  tp <- fp <- fn <- 0
  umbrales <- numeric(length(folds))
  for (k in seq_along(folds)) {
    otros <- fid != k
    u <- us_fino[which.max(suavizar(f1_curva(pe[otros], y[otros], us_fino)))]
    umbrales[k] <- u
    pred <- pe[!otros] >= u
    yy   <- y[!otros]
    tp <- tp + sum(pred & yy); fp <- fp + sum(pred & !yy); fn <- fn + sum(!pred & yy)
  }
  list(F1 = 2 * tp / (2 * tp + fp + fn), umbrales = umbrales)
}

# ------------------------------------------------------------
# 6.2. Parámetros de LightGBM
# ------------------------------------------------------------

PARAMS_LGB <- c("learning_rate", "num_leaves", "max_depth", "min_data_in_leaf",
                "feature_fraction", "bagging_fraction", "lambda_l1", "lambda_l2",
                "min_gain_to_split", "scale_pos_weight", "max_bin", "path_smooth", "extra_trees")

# Valores de partida: modelo pequeño y regularizado (la región del modelo de referencia).
# min_data_in_leaf equivale a min_child_samples.
params_base <- tibble(learning_rate = 0.05, num_leaves = 15L, max_depth = -1L, min_data_in_leaf = 50L,
                      feature_fraction = 0.6, bagging_fraction = 0.8, lambda_l1 = 1, lambda_l2 = 1,
                      min_gain_to_split = 0, scale_pos_weight = 1, max_bin = 255L,
                      path_smooth = 0, extra_trees = FALSE)

# Primera etapa de la selección de variables: modelo rápido y fijo para rankear por ganancia.
params_stage1 <- params_base |> mutate(learning_rate = 0.1, num_leaves = 31L)

# bagging_freq = 1 activa el submuestreo de filas en cada árbol; feature_pre_filter = FALSE evita
# errores al reutilizar datasets con distintos min_data_in_leaf.
armar_params <- function(p) {
  c(as.list(p[PARAMS_LGB]),
    list(objective = "binary", metric = "average_precision", bagging_freq = 1L,
         num_threads = N_THREADS, seed = 2026L, feature_pre_filter = FALSE, verbose = -1L))
}

# Dataset de LightGBM sobre las filas `idx` y las columnas de `x`. Las categóricas que no estén en
# las columnas seleccionadas se descartan.
crear_dataset <- function(x, idx, w, cats) {
  cats <- intersect(cats, colnames(x))
  lgb.Dataset(x[idx, , drop = FALSE], label = y01[idx], weight = w[idx],
              categorical_feature = if (length(cats) > 0) cats else NULL)
}

# ------------------------------------------------------------
# 6.3. nrounds por parada temprana (lgb.cv con los mismos folds)
# ------------------------------------------------------------

# Devuelve el número de árboles con mejor average precision de validación y ese valor.
nrounds_lgb <- function(x, p, w, cats) {
  cv <- lgb.cv(
    params = armar_params(p),
    data = crear_dataset(x, seq_len(nrow(x)), w, cats),
    nrounds = NROUNDS_MAX,
    folds = folds_val,
    early_stopping_rounds = PARADA_TEMPRANA,
    verbose = -1L
  )
  bi <- cv$best_iter
  if (is.null(bi) || is.na(bi) || bi < 1) bi <- NROUNDS_MAX     # sin parada temprana: se llegó al tope
  bs <- cv$best_score
  if (is.null(bs) || length(bs) != 1 || is.na(bs)) bs <- NA_real_
  list(nrounds = as.integer(bi), aucpr = as.numeric(bs), en_tope = bi >= NROUNDS_MAX)
}

# ------------------------------------------------------------
# 6.4. Ranking de variables por fold (primera etapa)
# ------------------------------------------------------------

# Para cada fold entrena un modelo rápido con el 80 % de train de ESE fold (nunca ve su
# validación) y ordena las variables por ganancia. Devuelve una lista de 5 vectores de nombres.
ranking_por_fold <- function(x, w, cats) {
  lapply(seq_along(folds), function(k) {
    tr <- folds[[k]]
    tr <- tr[w[tr] > 0]
    m <- lgb.train(params = armar_params(params_stage1),
                   data = crear_dataset(x, tr, w, cats),
                   nrounds = STAGE1_NROUNDS, verbose = -1L)
    imp <- as.data.frame(lgb.importance(m, percentage = TRUE))
    imp$Feature[order(-imp$Gain)]
  })
}

# ------------------------------------------------------------
# 6.5. OOF con nrounds fijo (y promedio de modelos sobre test)
# ------------------------------------------------------------

# Por fold: selecciona las top-K variables del ranking de ese fold (si K es NULL usa todas),
# entrena con el 80 % de train (sin las filas de peso 0) y predice el 20 % restante. Con varias
# semillas promedia sus predicciones. Si se entrega x_test, promedia también las predicciones de
# todos los modelos sobre test. Si se entrega dir_modelos, guarda cada modelo (lgb.save).
oof_lgb <- function(x, p, nrounds, w, cats, ranking = NULL, K = NULL, x_test = NULL,
                    semillas = 2026L, dir_modelos = NULL) {
  oof <- numeric(nrow(x))
  pred_test <- if (!is.null(x_test)) numeric(nrow(x_test)) else NULL
  cols_folds <- vector("list", length(folds))
  n_modelos <- length(folds) * length(semillas)

  for (k in seq_along(folds)) {
    tr <- folds[[k]]
    va <- setdiff(seq_along(y01), tr)
    tr <- tr[w[tr] > 0]
    cols <- if (is.null(K) || is.null(ranking)) colnames(x) else head(ranking[[k]], K)
    cols_folds[[k]] <- cols
    d_tr <- crear_dataset(x[, cols, drop = FALSE], tr, w, cats)

    for (s in semillas) {
      prm <- armar_params(p)
      prm$seed <- as.integer(s)
      m <- lgb.train(params = prm, data = d_tr, nrounds = nrounds, verbose = -1L)
      if (!is.null(dir_modelos)) {
        dir.create(dir_modelos, showWarnings = FALSE, recursive = TRUE)
        lgb.save(m, file.path(dir_modelos, sprintf("LGBM_fold%d_s%d.txt", k, s)))
      }
      oof[va] <- oof[va] + predict(m, x[va, cols, drop = FALSE]) / length(semillas)
      if (!is.null(x_test)) pred_test <- pred_test + predict(m, x_test[, cols, drop = FALSE]) / n_modelos
    }
  }
  list(oof = oof, test = pred_test, cols_folds = cols_folds)
}


# ============================================================
# 7. ESPACIO DE BÚSQUEDA DE HIPERPARÁMETROS
# ============================================================

# Muestreo aleatorio orientado a modelos pequeños y muy regularizados.
#   num_leaves 4 a 48; si max_depth > 0 se acota a 2^max_depth.
#   min_data_in_leaf 10 a 200 (= min_child_samples).
#   feature_fraction 0.3 a 0.9, bagging_fraction 0.6 a 1.
#   lambda_l1 0.01 a 15, lambda_l2 0.001 a 30, min_gain_to_split 0 o 0.001 a 1.
#   scale_pos_weight entre 1 y n_No/n_Yes (log-uniforme): el desbalance como hiperparámetro.
#   max_bin, path_smooth y extra_trees: regularización adicional.
muestrear_params <- function(n, lr) {
  md <- sample(c(-1L, 4L, 5L, 6L, 8L, 10L), n, replace = TRUE)
  nl <- round(10^runif(n, log10(4), log10(48)))
  nl <- ifelse(md > 0, pmin(nl, 2^md), nl)
  tibble(
    learning_rate     = lr,
    num_leaves        = as.integer(nl),
    max_depth         = md,
    min_data_in_leaf  = as.integer(round(10^runif(n, 1, log10(200)))),
    feature_fraction  = runif(n, 0.3, 0.9),
    bagging_fraction  = runif(n, 0.6, 1),
    lambda_l1         = 10^runif(n, -2, log10(15)),
    lambda_l2         = 10^runif(n, -3, log10(30)),
    min_gain_to_split = ifelse(runif(n) < 0.4, 0, 10^runif(n, -3, 0)),
    scale_pos_weight  = 10^runif(n, 0, log10(RAZON)),
    max_bin           = sample(c(63L, 127L, 255L, 511L), n, replace = TRUE),
    path_smooth       = ifelse(runif(n) < 0.5, 0, 10^runif(n, 0, log10(20))),
    extra_trees       = runif(n) < 0.25
  )
}


# ============================================================
# 8. SCREENING: VARIABLES x PESO DE BOGOTÁ (parámetros de partida, todas las variables)
# ============================================================

dir.create("salidas", showWarnings = FALSE)

# Peso de cada fila: Bogotá recibe `pb`; el resto, 1. La evaluación siempre es sin Bogotá.
pesos_filas <- function(pb) ifelse(es_bogota, pb, 1)

# USAR_PARAMETROS_GANADORES = FALSE: screening + selección de K (9) + búsqueda aleatoria (10).
# TRUE: se omite lo anterior y se usa la configuración ganadora (rama `else` al final de la sección 10).
if (!USAR_PARAMETROS_GANADORES) {

configs <- expand.grid(variables = VARIABLES_SCREENING, peso_bogota = PESO_BOGOTA,
                       stringsAsFactors = FALSE)
resultados <- vector("list", nrow(configs))

for (i in seq_len(nrow(configs))) {
  cfg <- configs[i, ]
  message(sprintf("[%d/%d] variables = %s | peso Bogotá = %s", i, nrow(configs),
                  cfg$variables, cfg$peso_bogota))
  t0 <- Sys.time()
  x <- conjuntos[[cfg$variables]]$train
  cats <- conjuntos[[cfg$variables]]$cats
  w <- pesos_filas(cfg$peso_bogota)

  r <- nrounds_lgb(x, params_base, w, cats)
  p_oof <- oof_lgb(x, params_base, r$nrounds, w, cats)$oof
  resultados[[i]] <- bind_cols(
    tibble(variables = cfg$variables, peso_bogota = cfg$peso_bogota,
           n_cols = ncol(x), nrounds = r$nrounds, AP_cv = r$aucpr),
    evaluar_oof(p_oof),
    tibble(minutos = as.numeric(difftime(Sys.time(), t0, units = "mins")))
  )
  rm(x, p_oof); gc(verbose = FALSE)
}

screening <- bind_rows(resultados) |> arrange(desc(F1_oof))
write.csv(screening, file.path("salidas", "LGBM_screening.csv"), row.names = FALSE)

# F1_oof: F1 del OOF agrupado, en el umbral óptimo, sobre la población comparable con test.
# Con una sola corrida, diferencias menores a 0.002 son ruido.
print(screening |> mutate(across(where(is.numeric), ~ round(.x, 5))), width = Inf)

mejor_cfg <- screening[1, ]
x_train <- conjuntos[[mejor_cfg$variables]]$train
x_test  <- conjuntos[[mejor_cfg$variables]]$test
cats    <- conjuntos[[mejor_cfg$variables]]$cats
w       <- pesos_filas(mejor_cfg$peso_bogota)
stopifnot(identical(colnames(x_train), colnames(x_test)))


# ============================================================
# 9. SELECCIÓN DE VARIABLES DENTRO DEL PLIEGUE (top-K por ganancia)
# ============================================================

# Primera etapa: ranking por ganancia calculado en cada fold solo con su 80 % de train.
# Segunda etapa: cada fold entrena con sus top-K variables; las predicciones de validación
# (OOF) y de test usan las columnas de ese fold. La opción "todas" viene del screening.
# Para estimar nrounds con lgb.cv (que usa las mismas columnas en todos los folds) se toma la unión
# de las top-K de los 5 folds: filtra levemente, pero solo afecta el número de árboles.
message("Ranking de variables por fold (primera etapa)")
ranking <- ranking_por_fold(x_train, w, cats)

sel <- vector("list", length(K_CANDIDATOS) + 1)
for (i in seq_along(K_CANDIDATOS)) {
  K <- K_CANDIDATOS[i]
  message(sprintf("Selección K = %d (%d/%d)", K, i, length(K_CANDIDATOS)))
  cols_u <- unique(unlist(lapply(ranking, head, K)))
  r <- nrounds_lgb(x_train[, cols_u, drop = FALSE], params_base, w, cats)
  p_oof <- oof_lgb(x_train, params_base, r$nrounds, w, cats, ranking, K)$oof
  sel[[i]] <- bind_cols(tibble(K = K, n_union = length(cols_u), nrounds = r$nrounds),
                        evaluar_oof(p_oof))
}
sel[[length(sel)]] <- bind_cols(
  tibble(K = NA_real_, n_union = ncol(x_train), nrounds = mejor_cfg$nrounds),
  mejor_cfg |> select(F1_oof, umbral, PRAUC, AUC)
)
seleccionK <- bind_rows(sel)
write.csv(seleccionK, file.path("salidas", "LGBM_seleccionK.csv"), row.names = FALSE)
print(seleccionK |> mutate(across(where(is.numeric), ~ round(.x, 5))), width = Inf)

# Parsimonia: entre las opciones a menos de 0.001 del mejor F1, se toma la de menos variables.
ok <- seleccionK$F1_oof >= max(seleccionK$F1_oof) - 0.001
sel_fila <- seleccionK[ok, ][which.min(seleccionK$n_union[ok]), ]
K_final <- if (is.na(sel_fila$K)) NULL else sel_fila$K
cols_union <- if (is.null(K_final)) colnames(x_train) else unique(unlist(lapply(ranking, head, K_final)))
message("K elegido: ", if (is.null(K_final)) "todas las variables" else K_final,
        " (", length(cols_union), " variables distintas entre folds)")


# ============================================================
# 10. BÚSQUEDA ALEATORIA DE HIPERPARÁMETROS (dos etapas de learning_rate)
# ============================================================

# Etapa A: estructura con learning_rate alto (rápida), rankeada por average precision.
# Etapa B: los TOP_K finalistas se reentrenan con learning_rate bajo, se recalcula nrounds y se
# evalúan con F1 OOF usando la selección de variables por fold.
x_b <- x_train[, cols_union, drop = FALSE]

set.seed(2026)
candidatos <- muestrear_params(N_BUSQUEDA, LR_BUSQUEDA)

busqueda <- vector("list", N_BUSQUEDA)
for (i in seq_len(N_BUSQUEDA)) {
  message(sprintf("Búsqueda %d/%d", i, N_BUSQUEDA))
  r <- nrounds_lgb(x_b, candidatos[i, ], w, cats)
  busqueda[[i]] <- bind_cols(candidatos[i, ],
                             tibble(nrounds = r$nrounds, AP_cv = r$aucpr, en_tope = r$en_tope))
}
busqueda <- bind_rows(busqueda) |> arrange(desc(AP_cv))
write.csv(busqueda, file.path("salidas", "LGBM_busqueda.csv"), row.names = FALSE)

top <- busqueda |> slice_head(n = TOP_K) |> mutate(learning_rate = LR_FINAL)
top$F1_oof <- NA_real_
top$umbral_oof <- NA_real_
for (i in seq_len(nrow(top))) {
  message(sprintf("Finalista %d/%d (learning_rate %s)", i, nrow(top), LR_FINAL))
  r <- nrounds_lgb(x_b, top[i, ], w, cats)
  top$nrounds[i] <- r$nrounds
  top$AP_cv[i]   <- r$aucpr
  top$en_tope[i] <- r$en_tope
  ev <- evaluar_oof(oof_lgb(x_train, top[i, ], r$nrounds, w, cats, ranking, K_final)$oof)
  top$F1_oof[i]     <- ev$F1_oof
  top$umbral_oof[i] <- ev$umbral
}
top <- top |> arrange(desc(F1_oof))
print(top |> mutate(across(where(is.numeric), ~ round(.x, 5))), width = Inf)

# Si el mejor finalista no supera por más de 0.001 al modelo de partida (con el mismo K), se
# conserva el de partida: evita premiar el ruido de la búsqueda.
if (top$F1_oof[1] > sel_fila$F1_oof + 0.001) {
  params_final  <- top[1, ]
  nrounds_final <- top$nrounds[1]
} else {
  message("La búsqueda no superó a los parámetros de partida: se conservan.")
  params_final  <- params_base
  nrounds_final <- sel_fila$nrounds
}
as.data.frame(params_final[PARAMS_LGB])
nrounds_final

} else {
  # Parámetros ganadores (sin búsqueda). Define lo mismo que las secciones 8 a 10.
  message("Parámetros ganadores: conjunto ", GANADOR$variables, ", peso Bogotá ", GANADOR$peso_bogota,
          ", ", GANADOR$nrounds, " árboles")
  mejor_cfg <- tibble(variables = GANADOR$variables, peso_bogota = GANADOR$peso_bogota,
                      nrounds = GANADOR$nrounds)
  x_train <- conjuntos[[mejor_cfg$variables]]$train
  x_test  <- conjuntos[[mejor_cfg$variables]]$test
  cats    <- conjuntos[[mejor_cfg$variables]]$cats
  w       <- pesos_filas(mejor_cfg$peso_bogota)
  stopifnot(identical(colnames(x_train), colnames(x_test)))

  K_final       <- GANADOR$K                       # NULL = todas las variables
  ranking       <- if (is.null(K_final)) NULL else ranking_por_fold(x_train, w, cats)
  cols_union    <- if (is.null(K_final)) colnames(x_train) else unique(unlist(lapply(ranking, head, K_final)))
  params_final  <- params_base                     # learning_rate 0.05 y 15 hojas, como en el nombre
  nrounds_final <- GANADOR$nrounds
}


# ============================================================
# 11. OOF FINAL Y PREDICCIÓN SOBRE TEST (bolsa de semillas)
# ============================================================

# Se repite el OOF con la configuración elegida, promediando N_SEMILLAS semillas por fold y
# promediando todos los modelos sobre test. Las probabilidades de test y las del OOF vienen del
# mismo tipo de modelo, así que el umbral elegido con el OOF aplica a ambas.
semillas <- 2026L + 100L * (0:(N_SEMILLAS - 1L))

fin <- oof_lgb(x_train, params_final, nrounds_final, w, cats, ranking, K_final,
               x_test = x_test, semillas = semillas,
               dir_modelos = file.path("modelos", "LGBM_folds"))

oof <- tibble(rowIndex = seq_along(y01), obs = y_train, Yes = fin$oof)
prob_pobre <- fin$test
stopifnot(length(prob_pobre) == nrow(db_test), !anyNA(prob_pobre))


# ============================================================
# 12. UMBRAL DE DECISIÓN
# ============================================================

#   - Población: solo hogares no-Bogotá del OOF si Bogotá no está en test (idx_eval).
#   - Estabilidad: se suaviza la curva de F1 y se reporta la variabilidad por bootstrap.
#   - Honestidad: f1_umbral_fuera() elige el umbral de cada fold con los otros 4. Esa es la cifra
#     a reportar; el F1 en el umbral elegido sobre todo el OOF es algo optimista.
oof_eval <- oof[oof$rowIndex %in% idx_eval, ]
y_eval   <- oof_eval$obs == "Yes"

f1_u     <- f1_curva(oof_eval$Yes, y_eval, us_fino)
f1_suave <- suavizar(f1_u)

umbral_puntual <- us_fino[which.max(f1_u)]
umbral_opt     <- us_fino[which.max(f1_suave)]

set.seed(2026)
n_eval <- length(y_eval)
u_boot <- replicate(B_BOOT, {
  i <- sample.int(n_eval, n_eval, replace = TRUE)
  us_fino[which.max(f1_curva(oof_eval$Yes[i], y_eval[i], us_fino))]
})

honesto <- f1_umbral_fuera(oof$Yes)

resumen_umbral <- tibble(
  umbral_puntual = umbral_puntual,
  umbral_suave = umbral_opt,
  boot_p25 = quantile(u_boot, 0.25), boot_mediana = median(u_boot), boot_p75 = quantile(u_boot, 0.75),
  F1_en_0.50 = f1_u[which.min(abs(us_fino - 0.5))],
  F1_en_umbral = f1_u[which.min(abs(us_fino - umbral_opt))],
  F1_umbral_fuera_del_fold = honesto$F1,
  F_estrella_medio = max(f1_u) / 2
)
resumen_umbral
honesto$umbrales      # umbral de cada fold, elegido con los otros 4


# ============================================================
# 13. CONVERSIÓN DE PROBABILIDAD A CLASE
# ============================================================

umbral <- umbral_opt
if (USAR_PARAMETROS_GANADORES) message(sprintf("Umbral obtenido %.3f | umbral de referencia (nombre del archivo) %.3f", umbral, GANADOR$umbral_ref))
clase_pobre <- ifelse(prob_pobre >= umbral, "Yes", "No")

# Tasa de hogares predichos como pobres en test frente al OOF comparable: una diferencia grande
# puede deberse a la composición por dominio (sección 16.3).
c(tasa_pred_test = mean(prob_pobre >= umbral),
  tasa_pred_oof  = mean(oof_eval$Yes >= umbral),
  prevalencia_oof = mean(y_eval))


# ============================================================
# 14. PREDICCIONES PARA KAGGLE
# ============================================================

predictSample <- db_test |>
  mutate(pobre_lab = clase_pobre) |>
  select(id, pobre_lab) |>
  mutate(pobre = if_else(pobre_lab == "Yes", 1L, 0L)) |>
  select(id, pobre)

stopifnot(nrow(predictSample) == nrow(db_test),
          !anyDuplicated(predictSample$id),
          !anyNA(predictSample),
          all(predictSample$pobre %in% c(0L, 1L)),
          setequal(predictSample$id, db_test$id))

head(predictSample)


# ============================================================
# 15. NOMBRE Y EXPORTACIÓN DEL ARCHIVO
# ============================================================

fmt <- function(v, d = 3) gsub(".", "", as.character(signif(v, d)), fixed = TRUE)
umbral_str <- gsub(".", "", sprintf("%.3f", umbral), fixed = TRUE)

name <- paste0(AUTOR, "_LGBM_", mejor_cfg$variables, "_K", if (is.null(K_final)) "todas" else K_final,
               "_pb", gsub(".", "", as.character(mejor_cfg$peso_bogota), fixed = TRUE),
               "_leaves_", params_final$num_leaves, "_lr_", fmt(params_final$learning_rate),
               "_nr_", nrounds_final, "_umbral_", umbral_str, ".csv")
name

write.csv(predictSample, file.path("salidas", name), row.names = FALSE)


# ============================================================
# 16. DIAGNÓSTICOS
# ============================================================

# ------------------------------------------------------------
# 16.1. Importancia de variables
# ------------------------------------------------------------

# Modelo auxiliar entrenado con todo train (filas de peso > 0) y las variables de `cols_union`, solo
# para la importancia (la predicción de test sale de los modelos por fold). Gain = aporte a la
# reducción de la pérdida cuando la variable se usa en un corte.
filas_aux <- which(w > 0)
modelo_imp <- lgb.train(
  params = armar_params(params_final),
  data = crear_dataset(x_train[, cols_union, drop = FALSE], filas_aux, w, cats),
  nrounds = nrounds_final, verbose = -1L
)
importancia <- as_tibble(as.data.frame(lgb.importance(modelo_imp, percentage = TRUE)))
importancia |> slice_head(n = 25)

# Variables presentes en los 5 folds (selección estable) y en solo algunos.
frecuencia_vars <- table(unlist(fin$cols_folds))
sort(frecuencia_vars, decreasing = TRUE)

# ------------------------------------------------------------
# 16.2. Matriz de confusión del OOF (población de evaluación): corte 0.5 frente al umbral elegido
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
# 16.3. Bogotá frente al resto (OOF completo) y composición de test
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

# ------------------------------------------------------------
# 16.4. Errores del modelo: F1 por dominio y perfil de los errores
# ------------------------------------------------------------

# F1 por dominio: un dominio con F1 muy bajo respecto del resto sugiere señal local que faltan
# variables por capturar (por ejemplo, costo de vida o mercado laboral propio).
f1_dominio <- oof_eval |>
  mutate(Dominio = as.character(db_train$Dominio[rowIndex]),
         pred = Yes >= umbral, y = obs == "Yes") |>
  group_by(Dominio) |>
  summarise(n = n(), prevalencia = mean(y),
            recall = sum(pred & y) / sum(y),
            precision = sum(pred & y) / pmax(sum(pred), 1),
            F1 = 2 * sum(pred & y) / (sum(pred) + sum(y)),
            .groups = "drop") |>
  arrange(F1)
f1_dominio

# Perfil de las 10 variables más importantes según el tipo de acierto o error. Si los falsos
# negativos (pobres no detectados) se parecen a los verdaderos negativos en las variables clave,
# la información que los distingue no está en el conjunto de variables actual.
vars_perfil <- head(importancia$Feature, 10)
perfil <- as_tibble(x_train[idx_eval, vars_perfil, drop = FALSE]) |>
  mutate(grupo = case_when(
    oof_eval$Yes >= umbral &  y_eval ~ "TP",
    oof_eval$Yes >= umbral & !y_eval ~ "FP",
    oof_eval$Yes <  umbral &  y_eval ~ "FN",
    TRUE                              ~ "TN")) |>
  group_by(grupo) |>
  summarise(n = n(), across(all_of(vars_perfil), ~ mean(.x, na.rm = TRUE)), .groups = "drop")
as.data.frame(perfil)


# ============================================================
# 17. GUARDADO DEL MEJOR MODELO
# ============================================================

# La predicción de Kaggle es el promedio de los modelos de los 5 folds y las N_SEMILLAS semillas,
# guardados en modelos/LGBM_folds/. El umbral aplica a ese promedio. Cada fold usa sus propias
# variables (cols_folds): sin ellas no se puede reproducir la predicción.
dir.create("modelos", showWarnings = FALSE)

# Modelo auxiliar con todo train (el de la importancia, 16.1): sirve para inspección, no reproduce el envío.
lgb.save(modelo_imp, file.path("modelos", "LGBM_modelo_imp.txt"))

saveRDS(list(
  config          = mejor_cfg,                                # variables y peso de Bogotá ganadores
  K               = K_final,                                  # NULL = todas las variables
  cols_folds      = fin$cols_folds,                           # variables de cada fold
  ranking         = ranking,                                  # ranking de la primera etapa por fold
  params          = params_final,                             # hiperparámetros elegidos
  nrounds         = nrounds_final,
  semillas        = semillas,
  umbral          = umbral,
  columnas        = colnames(x_train),                        # orden exacto de las columnas del modelo
  cats            = cats,                                     # categóricas nativas (fe_cat)
  dv              = conjuntos[[mejor_cfg$variables]]$dv,      # codificador de dummies (NULL en fe_cat)
  keep            = conjuntos[[mejor_cfg$variables]]$keep,    # columnas no constantes
  ref_train       = ref_train,                                # medianas por Dominio (3.3)
  niveles_dominio = niveles_dominio,
  niveles_depto   = niveles_depto,
  vars_cero       = vars_cero
), file.path("modelos", "LGBM_pipeline.rds"))

# Reutilización del pipeline guardado (ejemplo):
#   pipe <- readRDS(file.path("modelos", "LGBM_pipeline.rds"))
#   fs <- list.files(file.path("modelos", "LGBM_folds"), full.names = TRUE)   # LGBM_fold{k}_s{s}.txt
#   prob <- rowMeans(sapply(fs, function(f) {
#     k <- as.integer(sub("LGBM_fold([0-9]+)_s.*", "\\1", basename(f)))
#     predict(lgb.load(filename = f), x_nuevo[, pipe$cols_folds[[k]], drop = FALSE])
#   }))
#   clase <- as.integer(prob >= pipe$umbral)
#
# x_nuevo (matriz numérica) se construye igual que en train: a_factores(), a_cero(),
# crear_features(db, pipe$ref_train); luego predict(pipe$dv, ...) (o data.matrix() en fe_cat),
# [, pipe$keep] y colnames(x_nuevo) <- pipe$columnas.




# ============================================================
# 18. DESCRIPCIÓN DE RESULTADOS
# ============================================================

# Dos prints y una tabla del MEJOR modelo. Todo se calcula sobre el OOF de los hogares fuera de Bogotá
# (la población que se parece a test) y en el umbral elegido:
#   print 1: hiperparámetros y resultados de la validación cruzada (F1 y umbral);
#   print 2: detalle (matriz de confusión, métricas, class weight, proporción de pobres, variables);
#   tabla  : salidas/Cesar_LGBM_mejor_modelo.csv, una fila con toda esa información en columnas.
# Hay dos F1: el del umbral elegido fuera de cada fold (el de referencia) y el del umbral
# elegido en todo el OOF (optimista). No es un CV anidado: K, el peso de Bogotá y los hiperparámetros
# se eligieron con los mismos folds.

# ------------------------------------------------------------
# 18.1. Insumos y funciones de métricas
# ------------------------------------------------------------

p_eval    <- oof_eval$Yes                 # probabilidad OOF de la población de evaluación
y_eval_l  <- as.logical(y_eval)           # TRUE = pobre
fold_eval <- fold_id[idx_eval]            # pliegue de cada hogar evaluado

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

matriz_conf <- matrix(c(cm_res$TP, cm_res$FN, cm_res$FP, cm_res$TN), 2, 2,
                      dimnames = list(Predicho = c("Pobre", "No pobre"), Observado = c("Pobre", "No pobre")))

# Hiperparámetros del modelo final (los de params_final más árboles, semillas y K).
hp <- c(as.list(params_final[PARAMS_LGB]),
        list(nrounds = nrounds_final, semillas = N_SEMILLAS, K = if (is.null(K_final)) "todas" else K_final))
fmt_hp <- function(v) if (is.numeric(v)) format(signif(v, 4)) else as.character(v)
hp_txt <- paste(names(hp), vapply(hp, fmt_hp, character(1)), sep = " = ")

# Variables del modelo: las de las columnas finales (con selección por fold, la unión de los 5 folds).
vars_modelo_final <- if (is.null(K_final)) colnames(x_train) else unique(unlist(fin$cols_folds))
n_vars_fold <- lengths(fin$cols_folds)

# Class weight: scale_pos_weight distinto de 1 pondera la clase pobre. Además, el peso de Bogotá pondera filas.
usa_cw <- params_final$scale_pos_weight != 1
modo_txt <- if (USAR_PARAMETROS_GANADORES) "parámetros ganadores (sin búsqueda)" else "búsqueda completa"

# ------------------------------------------------------------
# 18.2. Primer print: hiperparámetros y validación cruzada
# ------------------------------------------------------------

linea_igual <- strrep("=", 78)
imprimir_resumen <- function() {
  cat(c(linea_igual,
        "DESCRIPCIÓN DE RESULTADOS (1/2): MEJOR MODELO, HIPERPARÁMETROS Y VALIDACIÓN CRUZADA",
        linea_igual,
        sprintf("Modelo: LightGBM (clasificación) | conjunto de variables: %s | ejecución: %s",
                mejor_cfg$variables, modo_txt),
        "Hiperparámetros:",
        strwrap(paste(hp_txt, collapse = " | "), width = 90, indent = 2, exdent = 2),
        sprintf("  (parada temprana sobre average precision; bolsa de %d semillas; predicción de test = promedio de %d modelos)",
                N_SEMILLAS, length(folds) * N_SEMILLAS),
        "",
        sprintf("Validación cruzada (%d pliegues; OOF de %d hogares fuera de Bogotá, %.2f%% pobres):",
                length(folds), length(y_eval_l), 100 * mean(y_eval_l)),
        sprintf("  Umbral elegido: %.3f (F1-óptimo OOF suavizado; bootstrap p25-p75: %.3f - %.3f)",
                umbral, resumen_umbral$boot_p25, resumen_umbral$boot_p75),
        sprintf("  F1, umbral elegido fuera del fold (cifra a reportar): %.4f", honesto$F1),
        sprintf("  F1, umbral elegido en todo el OOF (optimista):        %.4f", cm_res$F1),
        sprintf("  F1 por pliegue en el umbral final: media %.4f | DE %.4f | mín %.4f | máx %.4f",
                mean(f1_pliegues), sd(f1_pliegues), min(f1_pliegues), max(f1_pliegues)),
        sprintf("  AUC %.4f | PR-AUC (precisión promedio) %.4f", auc_oof, ap_oof),
        linea_igual), sep = "\n")
}
imprimir_resumen()

# ------------------------------------------------------------
# 18.3. Segundo print: detalle del mejor modelo
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
        sprintf("  scale_pos_weight = %s -> %s", fmt_hp(params_final$scale_pos_weight),
                if (usa_cw) "SÍ se ponderó la clase pobre" else "NO se ponderó la clase pobre (peso 1)"),
        sprintf("  Peso de las filas de Bogotá en el entrenamiento: %s (0 = Bogotá excluida)", format(mejor_cfg$peso_bogota)),
        "",
        "Proporción de pobres:",
        sprintf("  Train (todos los hogares):          %.2f%%", 100 * mean(y01)),
        sprintf("  Población de evaluación (sin Bogotá): %.2f%%", 100 * mean(y_eval_l)),
        sprintf("  Predicha pobre en el OOF:           %.2f%%", 100 * mean(p_eval >= umbral)),
        sprintf("  Predicha pobre en test:             %.2f%%", 100 * mean(prob_pobre >= umbral)),
        "",
        sprintf("Variables: %d en el modelo final (por fold: %s)", length(vars_modelo_final),
                paste(n_vars_fold, collapse = ", ")),
        strwrap(paste(vars_modelo_final, collapse = ", "), width = 90, indent = 2, exdent = 2),
        linea_igual), sep = "\n")
}
imprimir_detalle()

# ------------------------------------------------------------
# 18.4. Tabla del mejor modelo (una fila, todo en columnas)
# ------------------------------------------------------------

tabla_mejor <- bind_cols(
  tibble(modelo = "LightGBM", autor = AUTOR, ejecucion = modo_txt, conjunto_variables = mejor_cfg$variables),
  as_tibble(setNames(hp, paste0("hp_", names(hp)))),
  tibble(cv_pliegues = length(folds), cv_hogares_evaluados = length(y_eval_l),
         umbral = umbral, umbral_boot_p25 = resumen_umbral$boot_p25, umbral_boot_p75 = resumen_umbral$boot_p75,
         F1_umbral_fuera_del_fold = honesto$F1, F1_umbral_todo_oof = cm_res$F1,
         F1_pliegue_media = mean(f1_pliegues), F1_pliegue_de = sd(f1_pliegues),
         AUC = auc_oof, PR_AUC = ap_oof),
  select(cm_res, TP, FP, FN, TN, accuracy, balanced_accuracy, precision, recall, specificity, FPR, FNR, FDR, NPV, MCC),
  tibble(class_weight = usa_cw, scale_pos_weight = params_final$scale_pos_weight,
         peso_bogota = mejor_cfg$peso_bogota,
         prop_pobres_train = mean(y01), prop_pobres_evaluacion = mean(y_eval_l),
         prop_pred_pobre_oof = mean(p_eval >= umbral), prop_pred_pobre_test = mean(prob_pobre >= umbral),
         n_variables = length(vars_modelo_final), variables = paste(vars_modelo_final, collapse = "; "),
         archivo_envio = name))
write.csv(tabla_mejor, file.path("salidas", paste0(AUTOR, "_LGBM_mejor_modelo.csv")), row.names = FALSE)

# Los dos prints también quedan en un archivo de texto.
writeLines(capture.output(imprimir_resumen(), imprimir_detalle()), file.path("salidas", paste0(AUTOR, "_LGBM_descripcion.txt")))
cat("Tabla del mejor modelo: salidas/", AUTOR, "_LGBM_mejor_modelo.csv\n", sep = "")
