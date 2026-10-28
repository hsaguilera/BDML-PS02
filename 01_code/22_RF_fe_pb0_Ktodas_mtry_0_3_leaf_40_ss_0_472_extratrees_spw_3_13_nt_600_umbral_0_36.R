# ==============================================================================
# 22_RF_fe_pb0_Ktodas_mtry_0_3_leaf_40_ss_0_472_extratrees_spw_3_13_nt_600_umbral_0_36.R
# Autor: Cesar
# Modelo: Random Forest probabilístico (ranger), con preprocesamiento aprendido por pliegue.
#
# Lee 00_data/raw/*.csv, construye la base de hogares, entrena, elige el umbral de decisión y genera el
# archivo de envío. No depende de otros scripts ni del runner (00); detecta la raíz del proyecto.
#
# El nombre del archivo recoge el modelo ganador: conjunto fe, peso de Bogotá 0, todas las variables,
# mtry 0.3, hoja 40, sample.fraction 0.472, extratrees, scale_pos_weight 3.13, 600 árboles y umbral 0.36.
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

# Configuración ganadora (del nombre del archivo y de la descripción del modelo): fe, Bogotá con peso 0,
# todas las variables (K = NA), mtry 30 % de las columnas, nodo mínimo 40, min.bucket 1,
# sample.fraction 0.472 sin reemplazo, sin límite de profundidad, extratrees con 1 corte aleatorio y
# scale_pos_weight 3.13. Con MODO "equilibrado" (parte 2): 600 árboles y 2 semillas por pliegue.
GANADOR <- list(
  variables = "fe", peso_bogota = 0, K = NA_integer_, umbral_ref = 0.36,
  params = data.frame(mtry_frac = 0.3, min_node_size = 40L, min_bucket = 1L, sample_fraction = 0.472,
                      replace = FALSE, max_depth = 0L, splitrule = "extratrees",
                      num_random_splits = 1L, scale_pos_weight = 3.13))

# ============================================================
# PAQUETES Y RAÍZ DEL PROYECTO
# ============================================================
# pacman::p_load carga cada paquete y lo instala si falta. No se carga el meta-paquete tidyverse
# (arrastra ragg): el script solo usa sus componentes.
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  readr, dplyr, tidyr, purrr, tibble, stringr,   # datos
  ranger,                                        # random forest
  caret,                                         # createFolds
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
# MODELO: RANDOM FOREST
# ============================================================================
# Random Forest probabilístico (ranger).
# Dependencias: R >= 4.1, ranger y caret (cargados con pacman::p_load).
#
# ENTRADAS: db_train y db_test.
# SALIDAS: salidas/RF/<ejecucion>/ y modelos/RF/<ejecucion>/. Cada ejecución crea su propia carpeta
# (no sobrescribe corridas anteriores). CSV de envío: id,pobre, en orden de test.
#
# DEFINICIONES
# 1. X, y y pesos se entregan alineados de forma explícita: entrenar_rf recibe y como argumento.
# 2. Referencias por Dominio, niveles categóricos, constantes, imputación y ranking se aprenden solo
#    con las filas de entrenamiento de cada fold, con peso > 0.
# 3. Preprocesamiento en caché; interfaz x/y; sin OOB en la búsqueda.
# 4. Búsqueda conjunta de K e hiperparámetros sobre los mejores screenings. Incluye referencias
#    fijas, hojas pequeñas y grandes, min.bucket, gini/extratrees y muestreo con y sin reemplazo.
# 5. Pipeline de inferencia autocontenido por fold, con manifiesto de modelos, predicciones OOF,
#    diagnósticos y sessionInfo guardados.
#
# MÉTRICAS: CV de desarrollo, no evaluación externa. Screening, K e hiperparámetros
# se eligen con estos folds. El corte ajustado en los otros folds es un diagnóstico,
# tampoco elimina el sesgo de selección ni la dependencia entre modelos OOF.
# Un F1 de generalización independiente requeriría un holdout intacto o CV anidada.
# Bootstrap: estabilidad descriptiva del corte, condicional al OOF; no es un IC del pipeline.
# Documentación ranger: https://imbs-hl.github.io/ranger/reference/ranger.html
# ============================================================================

# 1. CONFIGURACIÓN -------------------------------------------------------------
MODO <- "equilibrado"            # "rapido", "equilibrado", "exhaustivo"
SEMILLA <- 2026L
N_FOLDS <- 5L                     # mismos folds (caret, semilla 2026) que los demás modelos
VARIABLES_SCREENING <- c("base", "fe")
PESO_BOGOTA <- c(1, 0.5, 0)
K_CANDIDATOS <- c(8L, 12L, 20L, 30L, 50L)
N_CONFIGS <- 2L                   # conservar dos configs, no descartar por un único piloto
IMPORTANCIA_RANKING <- "impurity_corrected"

presupuestos <- list(
  rapido = list(busqueda=8L, top=2L, screening=80L, ranking=100L,
                arboles_busqueda=100L, arboles_final=300L, semillas=1L, boot=100L),
  equilibrado = list(busqueda=30L, top=5L, screening=120L, ranking=150L,
                    arboles_busqueda=180L, arboles_final=600L, semillas=2L, boot=200L),
  exhaustivo = list(busqueda=70L, top=8L, screening=200L, ranking=300L,
                   arboles_busqueda=300L, arboles_final=900L, semillas=3L, boot=500L)
)
stopifnot(MODO %in% names(presupuestos))
stopifnot(N_FOLDS>=2L, N_CONFIGS>=1L, length(VARIABLES_SCREENING)>0L,
          all(VARIABLES_SCREENING %in% c("base","fe")),
          length(PESO_BOGOTA)>0L, all(is.finite(PESO_BOGOTA)),
          all(PESO_BOGOTA>=0), all(K_CANDIDATOS>=1L))
B <- presupuestos[[MODO]]
n_cores <- parallel::detectCores(logical=FALSE)
N_THREADS <- if (is.na(n_cores)) 1L else max(1L, n_cores - 1L)
# Evitar dos niveles de paralelismo: ranger usa los hilos, los folds son secuenciales.

if (getRversion() < "4.1.0") stop("Se requiere R >= 4.1.")
faltantes <- c("ranger", "caret")[!vapply(c("ranger", "caret"),
                                        requireNamespace, logical(1), quietly=TRUE)]
if (length(faltantes)) stop("Instale primero: install.packages(c(",
                          paste(sprintf('"%s"', faltantes), collapse=", "), "))")
if (!"min.bucket" %in% names(formals(ranger::ranger)))
  stop("Actualice ranger: esta versión no ofrece min.bucket.")
set.seed(SEMILLA)

# 2. DATOS Y VALIDACIONES ------------------------------------------------------
# db_train (con la respuesta Pobre) y db_test. id, Dominio y Depto son texto.
requeridas <- c("id", "Dominio", "Depto")
stopifnot(all(c(requeridas, "Pobre") %in% names(db_train)),
          all(requeridas %in% names(db_test)), nrow(db_test)>0L,
          !anyDuplicated(names(db_train)), !anyDuplicated(names(db_test)),
          !anyDuplicated(db_train$id), !anyDuplicated(db_test$id),
          !anyNA(db_train$id), !anyNA(db_test$id),
          !anyNA(db_train$Dominio), !anyNA(db_test$Dominio),
          setequal(setdiff(names(db_train), "Pobre"), names(db_test)))
if (length(intersect(db_train$id, db_test$id))) stop("Hay IDs compartidos entre train y test.")
y01 <- suppressWarnings(as.integer(as.character(db_train$Pobre)))
stopifnot(!anyNA(y01), all(y01 %in% 0:1), length(unique(y01))==2L,
          min(table(y01)) >= N_FOLDS)
y_train <- factor(y01, levels=c(1,0), labels=c("Yes","No"))
es_bogota <- toupper(trimws(as.character(db_train$Dominio))) == "BOGOTA"
bogota_test <- any(toupper(trimws(as.character(db_test$Dominio))) == "BOGOTA")
idx_eval <- if (any(es_bogota) && !bogota_test) which(!es_bogota) else seq_along(y01)
stopifnot(length(unique(y01[idx_eval]))==2L)
if (!any(es_bogota)) PESO_BOGOTA <- 1
message("Población OOF: ", length(idx_eval), "/", length(y01),
        if (length(idx_eval)<length(y01)) " hogares, sin Bogotá." else " hogares, todos.")

set.seed(SEMILLA)
folds <- caret::createFolds(y_train, k=N_FOLDS, returnTrain=TRUE)
folds_val <- lapply(folds, function(tr) setdiff(seq_along(y01), tr))
fold_id <- integer(length(y01))
for (k in seq_along(folds)) fold_id[folds_val[[k]]] <- k
stopifnot(all(fold_id > 0L), !anyDuplicated(unlist(folds_val)),
          all(vapply(folds_val, function(v) length(intersect(v,idx_eval))>0L, logical(1))))

# Carpeta única: la inferencia leerá el manifiesto, nunca list.files() con modelos viejos.
run_id <- basename(tempfile(pattern=paste0(AUTOR, "_", format(Sys.time(), "%Y%m%d_%H%M%S"), "_")))
OUT <- file.path("salidas", "RF", run_id)
MODEL <- file.path("modelos", "RF", run_id)
for (d in c(OUT, MODEL)) dir.create(d, recursive=TRUE, showWarnings=FALSE)
guardar_csv <- function(x, nombre) write.csv(x, file.path(OUT,nombre), row.names=FALSE, na="")
vars_cero <- c("P5100", "P5130", "P5140", "horas_prom_ocupados")
vars_no_predictoras <- c("id", "peso", "Pobre")

# 3. INGENIERÍA DE VARIABLES ---------------------------------------------------
# Conserva las fórmulas de la versión adjunta; las referencias son locales al fold.
calcular_ref <- function(d) {
  r <- list()
  if (all(c("valor_arriendo", "Dominio") %in% names(d)))
    r$med_log_arr_dom <- tapply(log1p(pmax(d$valor_arriendo,0)), d$Dominio, median, na.rm=TRUE)
  if (all(c("educ_jefe", "Dominio") %in% names(d)))
    r$med_educ_dom <- tapply(d$educ_jefe, d$Dominio, mean, na.rm=TRUE)
  if ("educ_jefe" %in% names(d) && length(unique(na.omit(d$educ_jefe)))<=12L)
    r$niveles_educ <- sort(unique(na.omit(d$educ_jefe)))
  r
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

  # --- Interacciones (mismas fórmulas; referencias y codificación locales al fold) ---
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
  d <- d[, setdiff(names(d), a_log), drop = FALSE]

  attr(d, "omitidas") <- omitidas
  d
}


# 4. PREPROCESADOR ENTRENADO POR FOLD ------------------------------------------
preparar_base <- function(d, ceros) {
  # Bandera ANTES de reemplazar. No convierte NA generales en ceros estructurales.
  for (v in intersect(ceros,names(d))) {
    flag <- paste0(v, "__faltante_estructural")
    if (flag %in% names(d)) stop("Colisión de nombre generado: ", flag)
    d[[flag]] <- as.integer(!is.finite(d[[v]]))
    d[[v]][!is.finite(d[[v]])] <- 0
  }
  d
}
crear_matriz <- function(d, esquema) {
  piezas <- lapply(names(esquema), function(v) {
    a <- esquema[[v]]; z <- d[[v]]
    if (a$tipo=="numero") {
      z <- as.numeric(z); z[!is.finite(z)] <- NA_real_
      m <- cbind(z, as.numeric(is.na(z)))
      colnames(m) <- c(v, paste0(v,"__NA"))
    } else {
      z <- as.character(z)
      # Categoría nueva: ceros en dummies conocidas. NA: dummy explícita.
      m <- matrix(0, nrow(d), length(a$niveles)+1L)
      for (j in seq_along(a$niveles)) m[,j] <- as.numeric(!is.na(z) & z==a$niveles[j])
      m[,ncol(m)] <- as.numeric(is.na(z))
      etiquetas <- if (length(a$niveles)) paste0(v,"__",a$niveles) else character()
      colnames(m) <- c(etiquetas, paste0(v,"__NA"))
    }
    m
  })
  m <- do.call(cbind,piezas)
  colnames(m) <- make.names(colnames(m),unique=TRUE)
  storage.mode(m) <- "double"
  m
}
imputar <- function(m, med) {
  for (j in seq_len(ncol(m))) {
    bad <- !is.finite(m[,j])
    if (any(bad)) m[bad,j] <- med[j]
  }
  m
}
fit_prepro <- function(d, variables) {
  originales <- setdiff(names(d), vars_no_predictoras)
  d <- preparar_base(d[,originales,drop=FALSE], vars_cero)
  ref <- if (variables=="fe") calcular_ref(d) else NULL
  if (variables=="fe") d <- crear_features(d,ref)
  esquema <- lapply(d,function(z) {
    if (is.numeric(z) || is.logical(z)) list(tipo="numero")
    else list(tipo="categoria",niveles=sort(unique(as.character(z[!is.na(z)]))))
  })
  m <- crear_matriz(d,esquema)
  med <- vapply(seq_len(ncol(m)), function(j) {
    z <- m[,j]; z <- z[is.finite(z)]
    if (length(z)) median(z) else 0
  }, numeric(1))
  names(med) <- colnames(m)
  m <- imputar(m,med)
  keep <- vapply(seq_len(ncol(m)), function(j) any(m[,j]!=m[1,j]), logical(1))
  if (!any(keep)) stop("No quedaron predictores no constantes en el fold.")
  list(x=m[,keep,drop=FALSE], pre=list(variables=variables, originales=originales,
    ceros=vars_cero, ref=ref, esquema=esquema, medianas=med,
    columnas=colnames(m)[keep], omitidas=attr(d,"omitidas")))
}
transform_prepro <- function(d, pre) {
  if (!all(pre$originales %in% names(d))) stop("Faltan columnas requeridas para inferencia.")
  d <- preparar_base(d[,pre$originales,drop=FALSE],pre$ceros)
  if (pre$variables=="fe") d <- crear_features(d,pre$ref)
  m <- crear_matriz(d,pre$esquema)
  stopifnot(identical(colnames(m),names(pre$medianas)))
  imputar(m,pre$medianas)[,pre$columnas,drop=FALSE]
}

# 5. MÉTRICAS Y UMBRAL ---------------------------------------------------------
us <- seq(0.01,0.99,by=0.005)
f1_curva <- function(p,y,umbrales=us) {
  y <- as.logical(y)
  stopifnot(length(p)==length(y), all(is.finite(p)), !anyNA(y))
  b <- findInterval(p,umbrales)
  tp <- rev(cumsum(rev(tabulate(b[y]+1L,nbins=length(umbrales)+1L))))[-1L]
  pp <- rev(cumsum(rev(tabulate(b+1L,nbins=length(umbrales)+1L))))[-1L]
  den <- pp+sum(y)
  ifelse(den>0,2*tp/pmax(den,1),0)
}
suavizar <- function(f) {
  g <- as.numeric(stats::filter(f, rep(1/7,7), sides=2))
  g[is.na(g)] <- f[is.na(g)]; g
}
elegir_umbral <- function(p,y) {
  s <- suavizar(f1_curva(p,y))
  candidatos <- which(s>=max(s)-1e-12)
  us[candidatos[ceiling(length(candidatos)/2)]]
}
metricas <- function(p,y,u) {
  z <- p>=u; y <- as.logical(y)
  tp <- sum(z&y); fp <- sum(z&!y); fn <- sum(!z&y); tn <- sum(!z&!y)
  c(TP=tp, FP=fp, FN=fn, TN=tn,
    F1=if (2*tp+fp+fn>0) 2*tp/(2*tp+fp+fn) else 0,
    Precision=if (tp+fp>0) tp/(tp+fp) else 0,
    Recall=if (tp+fn>0) tp/(tp+fn) else NA_real_, Accuracy=mean(z==y))
}
auc_ap <- function(p,y) {
  y <- as.logical(y); n1 <- sum(y); n0 <- sum(!y)
  auc <- if (n1>0 && n0>0) (sum(rank(p)[y])-n1*(n1+1)/2)/(n1*n0) else NA_real_
  # Average Precision por grupos de empates; no se etiqueta como PR-AUC trapezoidal.
  o <- order(p,decreasing=TRUE); yy <- y[o]; pp <- p[o]
  fin <- c(which(diff(pp)!=0),length(pp))
  tp <- cumsum(yy)[fin]
  ap <- if (n1>0) sum(diff(c(0,tp))/n1 * tp/fin) else NA_real_
  c(AUC=auc, AP=ap, Brier=mean((p-as.numeric(y))^2))
}
evaluar_oof <- function(p) {
  stopifnot(length(p)==length(y01),all(is.finite(p)),all(p>=0&p<=1))
  pe <- p[idx_eval]; y <- y01[idx_eval]; fid <- fold_id[idx_eval]
  u <- elegir_umbral(pe,y)
  # Diagnóstico de corte cruzado: no es CV anidada, ni métrica independiente.
  z <- logical(length(y)); uf <- numeric(N_FOLDS)
  for (k in seq_len(N_FOLDS)) {
    otros <- fid!=k
    uf[k] <- elegir_umbral(pe[otros],y[otros])
    z[!otros] <- pe[!otros]>=uf[k]
  }
  f_cruzado <- metricas(as.numeric(z),y,0.5)["F1"]
  a <- auc_ap(pe,y)
  data.frame(F1_corte_cruzado=unname(f_cruzado),
    F1_oof=unname(metricas(pe,y,u)["F1"]), F1_max=max(f1_curva(pe,y)),
    umbral=u, AUC=unname(a["AUC"]), AP=unname(a["AP"]), Brier=unname(a["Brier"]))
}

# 6. BOSQUES: X, Y Y PESOS ALINEADOS EXPLÍCITAMENTE -----------------------------
entrenar_rf <- function(x,y,w,p,num_trees,seed,importance="none") {
  stopifnot(nrow(x)==length(y),length(y)==length(w),!anyNA(y),
            all(y %in% 0:1),length(unique(y))==2L,
            all(is.finite(x)),all(is.finite(w)),all(w>0))
  # min.node.size = tamaño mínimo para intentar dividir (NO mínimo de hoja).
  # min.bucket = mínimo de observaciones en una hoja terminal.
  mtry <- if (p$mtry_frac<=0) floor(sqrt(ncol(x))) else floor(p$mtry_frac*ncol(x))
  ranger::ranger(x=x, y=factor(y,levels=c(0,1),labels=c("No","Yes")),
    num.trees=as.integer(num_trees), mtry=max(1L,min(ncol(x),mtry)),
    min.node.size=as.integer(p$min_node_size), min.bucket=as.integer(p$min_bucket),
    sample.fraction=p$sample_fraction, replace=as.logical(p$replace),
    max.depth=as.integer(p$max_depth), splitrule=as.character(p$splitrule),
    num.random.splits=as.integer(p$num_random_splits), probability=TRUE,
    case.weights=w, class.weights=c(1,p$scale_pos_weight),
    importance=importance, oob.error=FALSE, write.forest=TRUE,
    num.threads=N_THREADS, seed=as.integer(seed), verbose=FALSE)
}
predecir_rf <- function(m,x) {
  as.numeric(predict(m,data=x,num.threads=N_THREADS)$predictions[,"Yes"])
}
params_base <- data.frame(mtry_frac=0.3,min_node_size=100L,min_bucket=1L,
  sample_fraction=0.5,replace=TRUE,max_depth=0L,splitrule="gini",
  num_random_splits=1L,scale_pos_weight=1)
PARAMS <- names(params_base)
params_clasico <- params_base
params_clasico$mtry_frac <- 0     # convención: 0 -> sqrt(p)
params_clasico$min_node_size <- 10L
params_clasico$sample_fraction <- 1

# 7. CACHÉ DE PREPROCESAMIENTO Y RANKING ---------------------------------------
# Con los parámetros ganadores hay una sola configuración (la ganadora); si no, la rejilla del screening.
configs <- if (USAR_PARAMETROS_GANADORES) {
  data.frame(variables=GANADOR$variables,peso_bogota=GANADOR$peso_bogota,stringsAsFactors=FALSE)
} else {
  expand.grid(variables=VARIABLES_SCREENING,peso_bogota=PESO_BOGOTA,stringsAsFactors=FALSE)
}
configs$config_id <- seq_len(nrow(configs))
pesos_filas <- function(pb) ifelse(es_bogota,pb,1)
# Caché en memoria: guarda los datos preprocesados de los 5 folds de una configuración. Al pedir otra
# configuración se recalculan (el preprocesamiento es determinista).
cache_mem <- new.env(parent=emptyenv())
cache_mem$id <- NA_integer_; cache_mem$datos <- NULL
preparar_config <- function(id) {
  cache_mem$datos <- NULL
  cache_mem$id <- NA_integer_
  gc(verbose=FALSE)
  cfg <- configs[configs$config_id==id,,drop=FALSE]
  w <- pesos_filas(cfg$peso_bogota)
  datos_cfg <- vector("list", length(folds))
  for (k in seq_along(folds)) {
    tr <- folds[[k]]; tr <- tr[w[tr]>0]; va <- folds_val[[k]]
    if (length(unique(y01[tr]))<2L) stop("Un fold activo tiene una sola clase; revise pesos/geografía.")
    fit <- fit_prepro(db_train[tr,,drop=FALSE],cfg$variables)
    datos_cfg[[k]] <- list(tr=tr,va=va,y=y01[tr],w=w[tr],xtr=fit$x,
              xva=transform_prepro(db_train[va,,drop=FALSE],fit$pre),pre=fit$pre)
  }
  cache_mem$datos <- datos_cfg
  cache_mem$id <- as.integer(id)
}
cargar_fold <- function(id,k) {
  if (!identical(cache_mem$id,as.integer(id))) preparar_config(id)
  cache_mem$datos[[k]]
}
rankings <- list()
ranking_config <- function(id) {
  lapply(seq_along(folds),function(k) {
    d <- cargar_fold(id,k)
    m <- entrenar_rf(d$xtr,d$y,d$w,params_clasico,B$ranking,
                    SEMILLA+1000L+k,IMPORTANCIA_RANKING)
    imp <- m$variable.importance
    imp[!is.finite(imp)] <- -Inf
    names(sort(imp,decreasing=TRUE))
  })
}
# Todos los candidatos usan las mismas semillas por fold: comparación menos ruidosa.
evaluar_candidato <- function(id,p,K=NA_integer_,arboles=B$arboles_busqueda,
                              semillas=SEMILLA,final=FALSE) {
  oof <- rep(NA_real_,length(y01))
  pt <- if (final) numeric(nrow(db_test)) else NULL
  manifest <- list(); prepro <- list(); columnas <- list()
  for (k in seq_along(folds)) {
    d <- cargar_fold(id,k)
    cols <- if (is.na(K)) colnames(d$xtr) else head(rankings[[as.character(id)]][[k]],K)
    if (!length(cols)) stop("Ranking no disponible para config ",id)
    xtr <- d$xtr[,cols,drop=FALSE]; xva <- d$xva[,cols,drop=FALSE]
    if (final) xte <- transform_prepro(db_test,d$pre)[,cols,drop=FALSE]
    pv <- numeric(length(d$va))
    for (s in semillas) {
      seed_modelo <- as.integer(s+10000L*k)
      m <- entrenar_rf(xtr,d$y,d$w,p,arboles,seed_modelo)
      pv <- pv+predecir_rf(m,xva)/length(semillas)
      if (final) {
        pt <- pt+predecir_rf(m,xte)/(length(folds)*length(semillas))
        archivo <- sprintf("RF_fold%d_s%d.rds",k,s)
        saveRDS(m,file.path(MODEL,archivo))
        manifest[[length(manifest)+1L]] <- data.frame(fold=k,semilla=s,
                                                    semilla_modelo=seed_modelo,archivo=archivo)
      }
    }
    oof[d$va] <- pv
    if (final) {prepro[[k]] <- d$pre; columnas[[k]] <- cols}
  }
  list(oof=oof,test=pt,metricas=evaluar_oof(oof),prepro=prepro,columnas=columnas,
       manifest=if (final) do.call(rbind,manifest) else NULL)
}

# 8. SCREENING ---------------------------------------------------------------
# USAR_PARAMETROS_GANADORES = FALSE: screening (7) + búsqueda conjunta de K e hiperparámetros (8).
# TRUE: se salta todo eso y se usa la configuración ganadora (rama `else` al final de la sección 9).
if (!USAR_PARAMETROS_GANADORES) {

screen <- vector("list",nrow(configs))
for (i in seq_len(nrow(configs))) {
  message(sprintf("Screening %d/%d | %s | peso Bogotá %.2f",i,nrow(configs),
                  configs$variables[i],configs$peso_bogota[i]))
  preparar_config(i)
  t <- proc.time()[3]
  ev <- evaluar_candidato(i,params_base,arboles=B$screening)$metricas
  screen[[i]] <- cbind(configs[i,,drop=FALSE],ev,minutos=(proc.time()[3]-t)/60)
  guardar_csv(do.call(rbind,screen[seq_len(i)]),"RF_screening.csv")
}
screen <- do.call(rbind,screen)
screen <- screen[order(-screen$F1_corte_cruzado,-screen$AP),]
finalistas_cfg <- head(screen$config_id,min(N_CONFIGS,nrow(screen)))
print(screen,row.names=FALSE)

# 9. BÚSQUEDA CONJUNTA K + HIPERPARÁMETROS -------------------------------------
muestrear_params <- function(n,ratio) {
  data.frame(mtry_frac=sample(c(0,0.1,0.2,0.3,0.5,0.7,1),n,replace=TRUE),
    min_node_size=sample(c(5L,10L,20L,40L,80L,150L,300L),n,replace=TRUE),
    min_bucket=sample(c(1L,3L,5L,10L,20L),n,replace=TRUE),
    sample_fraction=runif(n,0.4,1),replace=sample(c(TRUE,FALSE),n,replace=TRUE),
    max_depth=sample(c(0L,8L,12L,18L,24L),n,replace=TRUE),
    splitrule=sample(c("gini","extratrees"),n,replace=TRUE,prob=c(0.8,0.2)),
    num_random_splits=sample(c(1L,3L,5L),n,replace=TRUE),
    scale_pos_weight=exp(runif(n,0,log(max(1,ratio)))))
}
registros <- list(); pilotos <- list(); contador <- 0L
n_por_cfg <- ceiling(B$busqueda/length(finalistas_cfg))
for (id in finalistas_cfg) {
  message("Ranking por fold: configuración ",id)
  rankings[[as.character(id)]] <- ranking_config(id)
  pmin_fold <- min(vapply(rankings[[as.character(id)]],length,integer(1)))
  ks <- c(unique(K_CANDIDATOS[K_CANDIDATOS<pmin_fold]),NA_integer_)
  pil <- list()
  for (j in seq_along(ks)) {
    ev <- evaluar_candidato(id,params_base,K=ks[j])$metricas
    pil[[j]] <- cbind(config_id=id,K=ks[j],params_base,ev)
  }
  pilot <- do.call(rbind,pil)
  pilotos[[as.character(id)]] <- pilot
  guardar_csv(do.call(rbind,pilotos),"RF_seleccionK.csv")
  # Los tres K mejor evaluados + todas. Se reajusta K durante la búsqueda.
  orden <- order(-pilot$F1_corte_cruzado,ifelse(is.na(pilot$K),Inf,pilot$K))
  k_pool <- unique(c(head(pilot$K[orden],3),NA_integer_))
  w <- pesos_filas(configs$peso_bogota[configs$config_id==id])
  activo <- w>0
  ratio <- sum(y01[activo]==0)/sum(y01[activo]==1)
  set.seed(SEMILLA+id)
  candidatos <- muestrear_params(n_por_cfg,ratio)
  candidatos$K <- sample(k_pool,nrow(candidatos),replace=TRUE)
  candidatos <- rbind(cbind(params_clasico,K=NA_integer_),candidatos)
  # Recuperar los pilotos: no repetir esos ajustes durante la búsqueda.
  for (j in seq_len(nrow(pilot))) {
    contador <- contador+1L
    registros[[contador]] <- cbind(candidate_id=contador,pilot[j,,drop=FALSE])
  }
  for (j in seq_len(nrow(candidatos))) {
    message(sprintf("Búsqueda config %d | %d/%d | K=%s",id,j,nrow(candidatos),
                    ifelse(is.na(candidatos$K[j]),"todas",candidatos$K[j])))
    p <- candidatos[j,PARAMS,drop=FALSE]
    ev <- evaluar_candidato(id,p,K=candidatos$K[j])$metricas
    contador <- contador+1L
    registros[[contador]] <- cbind(candidate_id=contador,config_id=id,K=candidatos$K[j],p,ev)
    guardar_csv(do.call(rbind,registros),"RF_busqueda.csv")
  }
}
busqueda <- do.call(rbind,registros)
busqueda <- busqueda[order(-busqueda$F1_corte_cruzado,-busqueda$AP),]
# Reentrenar finalistas y el mejor piloto de cada config con el presupuesto final.
ids_piloto <- vapply(finalistas_cfg,function(id) {
  pp <- pilotos[[as.character(id)]]
  kbest <- pp$K[which.max(pp$F1_corte_cruzado)]
  match_k <- if (is.na(kbest)) is.na(busqueda$K) else !is.na(busqueda$K)&busqueda$K==kbest
  matches <- which(busqueda$config_id==id & match_k &
    busqueda$mtry_frac==params_base$mtry_frac &
    busqueda$min_node_size==params_base$min_node_size &
    busqueda$sample_fraction==params_base$sample_fraction)
  busqueda$candidate_id[matches[1]]
},integer(1))
ids_top <- unique(c(head(busqueda$candidate_id,B$top),ids_piloto))
top <- busqueda[match(ids_top,busqueda$candidate_id),]
reeval <- list()
for (i in seq_len(nrow(top))) {
  message("Finalista ",i,"/",nrow(top)," | ",B$arboles_final," árboles")
  r <- top[i,,drop=FALSE]
  ev <- evaluar_candidato(r$config_id,r[,PARAMS,drop=FALSE],r$K,
                         arboles=B$arboles_final)$metricas
  reeval[[i]] <- cbind(r[,c("candidate_id","config_id","K",PARAMS)],ev)
  guardar_csv(do.call(rbind,reeval),"RF_finalistas.csv")
}
finalistas <- do.call(rbind,reeval)
# Tolerancia explícita de parsimonia, NO intervalo estadístico ni regla 1-SE.
tolerancia <- 0.001
ok <- finalistas$F1_corte_cruzado >= max(finalistas$F1_corte_cruzado)-tolerancia
mejores <- finalistas[ok,,drop=FALSE]
mejores <- mejores[order(ifelse(is.na(mejores$K),Inf,mejores$K),
                         -mejores$F1_corte_cruzado,-mejores$AP),]
ganador <- mejores[1,,drop=FALSE]

} else {
  # Parámetros ganadores (sin búsqueda): una fila con la misma estructura que `ganador` del bloque anterior.
  message("Parámetros ganadores: conjunto ", GANADOR$variables, ", peso Bogotá ", GANADOR$peso_bogota,
          ", ", B$arboles_final, " árboles, ", B$semillas, " semillas")
  ganador <- cbind(data.frame(candidate_id=1L,config_id=1L,K=GANADOR$K),GANADOR$params)
}
print(ganador,row.names=FALSE)

# 10. ENSAMBLE FINAL ------------------------------------------------------------
semillas <- SEMILLA+100L*(seq_len(B$semillas)-1L)
fin <- evaluar_candidato(ganador$config_id,ganador[,PARAMS,drop=FALSE],ganador$K,
                         arboles=B$arboles_final,semillas=semillas,final=TRUE)
umbral <- fin$metricas$umbral
if (USAR_PARAMETROS_GANADORES) message(sprintf("Umbral obtenido %.3f | umbral de referencia (nombre del archivo) %.3f", umbral, GANADOR$umbral_ref))
stopifnot(all(is.finite(fin$test)),all(fin$test>=0 & fin$test<=1))
# OOF promedia semillas en UN fold; test promedia folds y semillas. La dispersión
# puede cambiar: no se garantiza que el corte OOF sea óptimo para el ensamble de test.
predictSample <- data.frame(id=db_test$id,pobre=as.integer(fin$test>=umbral))
stopifnot(identical(predictSample$id,db_test$id),!anyNA(predictSample))
guardar_csv(predictSample,"RF_submission.csv")
guardar_csv(data.frame(id=db_test$id,prob_pobre=fin$test),"RF_test_probabilidades.csv")
guardar_csv(data.frame(id=db_train$id,fold=fold_id,Pobre=y01,prob_pobre=fin$oof,
                       en_evaluacion=seq_along(y01)%in%idx_eval),"RF_oof.csv")

# 11. ESTABILIDAD DEL UMBRAL Y DIAGNÓSTICOS -------------------------------------
pe <- fin$oof[idx_eval]; ye <- y01[idx_eval]
set.seed(SEMILLA+777L)
# Bootstrap estratificado: conserva la prevalencia observada y ambas clases.
clases <- split(seq_along(ye),ye)
u_boot <- replicate(B$boot,{
  ii <- unlist(lapply(clases,function(z) z[sample.int(length(z),length(z),replace=TRUE)]),
               use.names=FALSE)
  elegir_umbral(pe[ii],ye[ii])
})
resumen <- cbind(fin$metricas,F1_050=unname(metricas(pe,ye,0.5)["F1"]),
                boot_p25=unname(quantile(u_boot,0.25)),boot_mediana=median(u_boot),
                boot_p75=unname(quantile(u_boot,0.75)))
guardar_csv(resumen,"RF_resumen.csv")
guardar_csv(data.frame(umbral=us,F1=f1_curva(pe,ye),F1_suave=suavizar(f1_curva(pe,ye))),
            "RF_curva_umbral.csv")
guardar_csv(data.frame(replica=seq_along(u_boot),umbral=u_boot),"RF_bootstrap_umbral.csv")
por_fold <- do.call(rbind,lapply(seq_len(N_FOLDS),function(k) {
  ii <- fold_id[idx_eval]==k
  uk <- elegir_umbral(pe[!ii],ye[!ii])
  cbind(data.frame(fold=k,n=sum(ii),umbral_otros_folds=uk),
        as.data.frame(as.list(metricas(pe[ii],ye[ii],uk))))
}))
guardar_csv(por_fold,"RF_metricas_fold.csv")
por_dom <- do.call(rbind,lapply(split(idx_eval,db_train$Dominio[idx_eval]),function(ii) {
  cbind(data.frame(Dominio=db_train$Dominio[ii[1]],n=length(ii),
                   prevalencia=mean(y01[ii]),prob_media=mean(fin$oof[ii])),
        as.data.frame(as.list(metricas(fin$oof[ii],y01[ii],umbral))))
}))
guardar_csv(por_dom,"RF_metricas_dominio.csv")
frecuencia <- sort(table(unlist(fin$columnas)),decreasing=TRUE)
guardar_csv(data.frame(variable=names(frecuencia),folds=as.integer(frecuencia)),
            "RF_frecuencia_variables.csv")
# El ranking por fold solo existe si se corrió la búsqueda (con los parámetros ganadores no hay ranking).
if (length(rankings) > 0) {
  ranking_tabla <- do.call(rbind,lapply(seq_len(N_FOLDS),function(k) {
    rr <- rankings[[as.character(ganador$config_id)]][[k]]
    data.frame(fold=k,posicion=seq_along(rr),variable=rr)
  }))
  guardar_csv(ranking_tabla,"RF_ranking_por_fold.csv")
}
confusiones <- rbind(c(umbral=0.5,metricas(pe,ye,0.5)),
                    c(umbral=umbral,metricas(pe,ye,umbral)))
guardar_csv(as.data.frame(confusiones),"RF_confusion.csv")

# 12. INFERENCIA REPRODUCIBLE SIN VOLVER A ENTRENAR -----------------------------
# Guardar sólo las funciones necesarias en un entorno autónomo (sin db_train,
# cachés ni otras variables globales). Se cargan con readRDS en otra sesión.
funciones <- list(crear_features=crear_features,preparar_base=preparar_base,
                  crear_matriz=crear_matriz,imputar=imputar,
                  transform_prepro=transform_prepro)
entorno <- new.env(parent=baseenv())
for (nom in names(funciones)) {
  f <- funciones[[nom]]; environment(f) <- entorno; assign(nom,f,envir=entorno)
}
inferir <- function(nuevos,preprocesadores,columnas,manifest,model_dir,num_threads=1L) {
  if (!requireNamespace("ranger",quietly=TRUE)) stop("Instale ranger.")
  if (!nrow(nuevos)) return(numeric())
  out <- numeric(nrow(nuevos))
  for (k in seq_along(preprocesadores)) {
    x <- transform_prepro(nuevos,preprocesadores[[k]])[,columnas[[k]],drop=FALSE]
    archivos <- manifest$archivo[manifest$fold==k]
    for (archivo in archivos) {
      ruta <- file.path(model_dir,archivo)
      if (!file.exists(ruta)) stop("Falta modelo del manifiesto: ",ruta)
      m <- readRDS(ruta)
      pr <- stats::predict(m,data=x,num.threads=num_threads)$predictions[,"Yes"]
      out <- out+as.numeric(pr)/nrow(manifest)
    }
  }
  out
}
environment(inferir) <- entorno
pipeline <- list(version="RF-1.0",config=configs[configs$config_id==ganador$config_id,],
  ganador=ganador,umbral=umbral,semillas=semillas,arboles=B$arboles_final,
  preprocesadores=fin$prepro,columnas=fin$columnas,manifest=fin$manifest,
  predict_proba=inferir,metricas_desarrollo=resumen,
  nota="CV de desarrollo con selección adaptativa; no evaluación externa.",
  versiones=vapply(c("ranger","caret"),function(p) as.character(utils::packageVersion(p)),character(1)))
saveRDS(pipeline,file.path(MODEL,"RF_pipeline.rds"))
saveRDS(list(folds=folds,fold_id=fold_id,ids=db_train$id),file.path(MODEL,"RF_folds.rds"))
guardar_csv(fin$manifest,"RF_manifest_modelos.csv")
# Comprobación de serialización/inferencia: pequeña muestra, sin nuevo entrenamiento.
pipe_check <- readRDS(file.path(MODEL,"RF_pipeline.rds"))
ii <- seq_len(min(50L,nrow(db_test)))
check <- pipe_check$predict_proba(db_test[ii,,drop=FALSE],pipe_check$preprocesadores,
    pipe_check$columnas,pipe_check$manifest,MODEL,N_THREADS)
stopifnot(isTRUE(all.equal(check,fin$test[ii],tolerance=1e-10)))

# Uso del pipeline en una sesión nueva (ejemplo); `nuevos` es una base de hogares con las mismas
# columnas que db_test:
# pipe <- readRDS("modelos/RF/<ejecucion>/RF_pipeline.rds")
# p <- pipe$predict_proba(nuevos, pipe$preprocesadores, pipe$columnas,
#                         pipe$manifest, "modelos/RF/<ejecucion>")
# submission <- data.frame(id=nuevos$id,pobre=as.integer(p>=pipe$umbral))
# El pipeline no contiene los bosques: requiere la carpeta de modelos completa.

writeLines(capture.output(sessionInfo()),file.path(OUT,"RF_sessionInfo.txt"))

# 13. DESCRIPCIÓN DE RESULTADOS --------------------------------------------------
# Dos prints y una tabla del MEJOR modelo. Todo se calcula sobre el OOF de los hogares fuera de Bogotá
# (la población que se parece a test) y en el umbral elegido:
#   print 1: hiperparámetros y resultados de la validación cruzada (F1 y umbral);
#   print 2: detalle (matriz de confusión, métricas, class weight, proporción de pobres, variables);
#   tabla  : salidas/Cesar_RF_mejor_modelo.csv, una fila con toda esa información en columnas.
# Son métricas de desarrollo, no validación externa: la configuración se eligió con los mismos folds.
# `F1 de corte cruzado` elige el umbral de cada fold con los otros 4 (el de referencia);
# `F1 con corte global` lo elige en todo el OOF (optimista).

# 12.1 Insumos y funciones de métricas ------------------------------------------
p_eval    <- pe                           # probabilidad OOF de la población de evaluación
y_eval_l  <- ye == 1                      # TRUE = pobre
fold_eval <- fold_id[idx_eval]            # pliegue de cada hogar evaluado

# AUC y precisión promedio (área bajo precisión-cobertura) en numérico, con manejo de empates.
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

# Hiperparámetros del modelo final (los de `ganador` más árboles, semillas y K).
hp <- c(as.list(ganador[, PARAMS, drop = FALSE]),
        list(arboles = B$arboles_final, semillas = B$semillas, K = if (is.na(ganador$K)) "todas" else ganador$K))
fmt_hp <- function(v) if (is.numeric(v)) format(signif(v, 4)) else as.character(v)
hp_txt <- paste(names(hp), vapply(hp, fmt_hp, character(1)), sep = " = ")

# Variables del modelo: la unión de las columnas de los 5 folds (cada fold descarta constantes y, si hay K, selecciona).
vars_modelo_final <- unique(unlist(fin$columnas))
n_vars_fold <- lengths(fin$columnas)

# Class weight: class.weights = c(1, scale_pos_weight) pondera la clase pobre. Además, el peso de Bogotá
# entra como peso de fila (case.weights; 0 = Bogotá excluida del entrenamiento).
cfg_ganadora <- configs[configs$config_id == ganador$config_id, , drop = FALSE]
usa_cw <- ganador$scale_pos_weight != 1
modo_txt <- if (USAR_PARAMETROS_GANADORES) "parámetros ganadores (sin búsqueda)" else "búsqueda completa"
f1_honesto <- resumen$F1_corte_cruzado

# 12.2 Primer print: hiperparámetros y validación cruzada -------------------------
linea_igual <- strrep("=", 78)
imprimir_resumen <- function() {
  cat(c(linea_igual,
        "DESCRIPCIÓN DE RESULTADOS (1/2): MEJOR MODELO, HIPERPARÁMETROS Y VALIDACIÓN CRUZADA",
        linea_igual,
        sprintf("Modelo: Random Forest (ranger, bosque de probabilidad) | conjunto de variables: %s | ejecución: %s",
                cfg_ganadora$variables, modo_txt),
        "Hiperparámetros:",
        strwrap(paste(hp_txt, collapse = " | "), width = 90, indent = 2, exdent = 2),
        sprintf("  (predicción de test = promedio de %d bosques: %d folds x %d semillas)",
                nrow(fin$manifest), N_FOLDS, B$semillas),
        "",
        sprintf("Validación cruzada (%d pliegues; OOF de %d hogares fuera de Bogotá, %.2f%% pobres):",
                N_FOLDS, length(y_eval_l), 100 * mean(y_eval_l)),
        sprintf("  Umbral elegido: %.3f (F1-óptimo OOF suavizado; bootstrap p25-p75: %.3f - %.3f)",
                umbral, resumen$boot_p25, resumen$boot_p75),
        sprintf("  F1, corte cruzado entre folds (cifra a reportar): %.4f", f1_honesto),
        sprintf("  F1, corte global en todo el OOF (optimista):      %.4f", cm_res$F1),
        sprintf("  F1 por pliegue en el umbral final: media %.4f | DE %.4f | mín %.4f | máx %.4f",
                mean(f1_pliegues), sd(f1_pliegues), min(f1_pliegues), max(f1_pliegues)),
        sprintf("  AUC %.4f | PR-AUC (precisión promedio) %.4f | Brier %.4f", auc_oof, ap_oof, resumen$Brier),
        linea_igual), sep = "\n")
}
imprimir_resumen()

# 12.3 Segundo print: detalle del mejor modelo ------------------------------------
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
        sprintf("  scale_pos_weight = %s -> %s", fmt_hp(ganador$scale_pos_weight),
                if (usa_cw) "SÍ se ponderó la clase pobre (class.weights = c(1, scale_pos_weight))"
                else "NO se ponderó la clase pobre (peso 1)"),
        sprintf("  Peso de las filas de Bogotá en el entrenamiento: %s (0 = Bogotá excluida)", format(cfg_ganadora$peso_bogota)),
        "",
        "Proporción de pobres:",
        sprintf("  Train (todos los hogares):          %.2f%%", 100 * mean(y01)),
        sprintf("  Población de evaluación (sin Bogotá): %.2f%%", 100 * mean(y_eval_l)),
        sprintf("  Predicha pobre en el OOF:           %.2f%%", 100 * mean(p_eval >= umbral)),
        sprintf("  Predicha pobre en test:             %.2f%%", 100 * mean(fin$test >= umbral)),
        "",
        sprintf("Variables: %d en el modelo final (por fold: %s)", length(vars_modelo_final),
                paste(n_vars_fold, collapse = ", ")),
        strwrap(paste(vars_modelo_final, collapse = ", "), width = 90, indent = 2, exdent = 2),
        linea_igual), sep = "\n")
}
imprimir_detalle()

# 12.4 Tabla del mejor modelo (una fila, todo en columnas) ------------------------
tabla_mejor <- bind_cols(
  tibble(modelo = "Random Forest", autor = AUTOR, ejecucion = modo_txt, conjunto_variables = cfg_ganadora$variables),
  as_tibble(setNames(hp, paste0("hp_", names(hp)))),
  tibble(cv_pliegues = N_FOLDS, cv_hogares_evaluados = length(y_eval_l),
         umbral = umbral, umbral_boot_p25 = resumen$boot_p25, umbral_boot_p75 = resumen$boot_p75,
         F1_umbral_fuera_del_fold = f1_honesto, F1_umbral_todo_oof = cm_res$F1,
         F1_pliegue_media = mean(f1_pliegues), F1_pliegue_de = sd(f1_pliegues),
         AUC = auc_oof, PR_AUC = ap_oof),
  select(cm_res, TP, FP, FN, TN, accuracy, balanced_accuracy, precision, recall, specificity, FPR, FNR, FDR, NPV, MCC),
  tibble(class_weight = usa_cw, scale_pos_weight = ganador$scale_pos_weight,
         peso_bogota = cfg_ganadora$peso_bogota,
         prop_pobres_train = mean(y01), prop_pobres_evaluacion = mean(y_eval_l),
         prop_pred_pobre_oof = mean(p_eval >= umbral), prop_pred_pobre_test = mean(fin$test >= umbral),
         n_variables = length(vars_modelo_final), variables = paste(vars_modelo_final, collapse = "; "),
         archivo_envio = file.path(OUT, "RF_submission.csv")))
write.csv(tabla_mejor, file.path("salidas", paste0(AUTOR, "_RF_mejor_modelo.csv")), row.names = FALSE)

# Los dos prints también quedan en un archivo de texto junto al envío.
writeLines(capture.output(imprimir_resumen(), imprimir_detalle()), file.path(OUT, "RF_descripcion.txt"))
cache_mem$datos <- NULL
cat("\nProceso terminado.\n")
cat("Tabla del mejor modelo: salidas/", AUTOR, "_RF_mejor_modelo.csv\n", sep = "")
cat("Envío: ", file.path(OUT, "RF_submission.csv"), "\n",
    "Modelo: ", file.path(MODEL, "RF_pipeline.rds"), "\n", sep = "")
