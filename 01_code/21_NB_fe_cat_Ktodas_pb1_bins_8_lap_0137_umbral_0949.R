# ==============================================================================
# 21_NB_fe_cat_Ktodas_pb1_bins_8_lap_0137_umbral_0949.R
# Autor: Cesar
# Modelo: Naive Bayes categórico, con discretización y selección de variables en el pliegue.
#
# Lee 00_data/raw/*.csv, construye la base de hogares, entrena, elige el umbral de decisión y genera el
# archivo de envío. No depende de otros scripts ni del runner (00); detecta la raíz del proyecto.
#
# El nombre del archivo recoge el modelo ganador: conjunto fe, modo categórico, todas las variables,
# peso de Bogotá 1, 8 bins, suavizado de Laplace 0.137 y umbral 0.949.
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

# Configuración ganadora (del nombre del archivo): conjunto fe, modo categórico, todas las variables
# que pasan el filtro (K = NULL), peso de Bogotá 1, 8 bins, Laplace 0.137 y filtro |r| <= 0.9.
GANADOR <- list(variables = "fe", modo = "cat", peso_bogota = 1, K = NULL,
                n_bins = 8L, laplace = 0.137, cor_max = 0.9, umbral_ref = 0.949)

# ============================================================
# PAQUETES Y RAÍZ DEL PROYECTO
# ============================================================
# pacman::p_load carga cada paquete y lo instala si falta. No se carga el meta-paquete tidyverse
# (arrastra ragg): el script solo usa sus componentes.
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  readr, dplyr, tidyr, purrr, tibble, stringr,   # datos y modelo
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
# MODELO: NAIVE BAYES
# ============================================================
#
# Naive Bayes con discretización y selección de variables dentro del pliegue. Usa los mismos folds
# (semilla 2026) y la misma población de evaluación (sin Bogotá) que los demás modelos.
#
# Definiciones, según los tres supuestos del método:
#
#   1. Normalidad. Las variables de la base son asimétricas, acotadas o con masa en cero (arriendos,
#      proporciones, conteos), por lo que un Naive Bayes gaussiano ajusta mal. El modo por defecto es
#      categórico: cada variable se discretiza en cuantiles calculados dentro del fold y el modelo
#      estima P(bin | clase) con suavizado de Laplace, sin suponer forma funcional. El modo "gauss"
#      (Naive Bayes clásico) queda disponible para comparar.
#   2. Independencia. Las variables construidas se solapan (log_renta_pc y log_renta_pc_lp, n_menores
#      y prop_menores_15), y Naive Bayes multiplica esa evidencia repetida y satura las probabilidades.
#      Un filtro de correlación dentro del fold descarta, entre dos variables con |r| por encima de
#      COR_MAX, la de menor señal univariada.
#   3. Calibración. El producto de verosimilitudes empuja las probabilidades a 0 o 1, por lo que una
#      rejilla uniforme de umbrales (0.05, 0.055, ...) cae casi siempre en una zona vacía. La rejilla
#      se construye con los cuantiles de las probabilidades del OOF, donde la curva de F1 tiene
#      resolución; equivale a elegir qué fracción de hogares se marca como pobre.
#
# Particularidades de implementación:
#   - El modelo se implementa con R base (sección 6.3): no usa e1071, klaR ni naivebayes. Son conteos
#     por bin y clase, con log-probabilidades, en una sola pasada.
#   - El ranking de la primera etapa usa señal univariada (|AUC - 0.5| calculada solo con las filas
#     de entrenamiento del fold), criterio natural para un modelo que evalúa cada variable por separado.
#   - Hiperparámetros: n_bins (número de cuantiles), laplace (suavizado), COR_MAX (filtro de
#     correlación) y K (número de variables), buscados de forma aleatoria.
#   - El peso de Bogotá entra como peso de cada fila en los conteos (0 la excluye del entrenamiento).
#   - Es determinista: no hay bolsa de semillas.
#
# Entradas: db_train y db_test.
# Salidas (carpeta salidas/):
#   NB_<variables>_<modo>_K<K>_pb<peso>_bins_*_lap_*_umbral_*.csv   envío a Kaggle
#   NB_screening.csv, NB_seleccionK.csv, NB_busqueda.csv
#   NB_descripcion.txt                                              descripción de resultados
# Salidas (carpeta modelos/):
#   NB_pipeline.rds   modelos de los 5 folds (tablas de probabilidades), variables, cortes y umbral
#
# Dependencias: caret y MLmetrics (cargados con pacman::p_load).
#
# Recursos: cada ajuste es una pasada de conteos y el costo lo domina N_BUSQUEDA; una corrida completa
# toma minutos.
#
# Referencia: en datos tabulares con variables correlacionadas, Naive Bayes suele quedar por debajo de
# boosting y bosques; sirve para dimensionar el costo del supuesto de independencia.


# ============================================================
# 1. PARÁMETROS DEL MODELO
# ============================================================

# ------------------------------------------------------------
# 1.1. Configuración de la búsqueda
# ------------------------------------------------------------

# Screening (sección 8): conjuntos de variables, modo del modelo y peso de las filas de Bogotá.
#   modo "cat"   = variables discretizadas en cuantiles (recomendado)
#   modo "gauss" = Naive Bayes gaussiano clásico (una normal por variable y clase)
VARIABLES_SCREENING <- c("base", "fe")
MODO_SCREENING      <- c("cat", "gauss")
PESO_BOGOTA         <- c(1, 0.5, 0)   # 0 = Bogotá no entra al entrenamiento

# Selección de variables dentro del pliegue (sección 9). La opción "todas" siempre se evalúa.
# En Naive Bayes, menos variables suele ser mejor: cada una añade un factor al producto.
K_CANDIDATOS <- c(5, 8, 12, 20, 30, 50)

# Búsqueda aleatoria (sección 10).
N_BUSQUEDA <- 40   # combinaciones muestreadas (cada ajuste es barato)
TOP_K      <- 5    # finalistas que se muestran en detalle

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

# Igual que en los otros scripts. En el modo "cat" los NA que queden reciben su propio bin (son
# informativos); en "gauss" se imputan con la mediana del fold.
vars_cero <- intersect(c("P5100", "P5130", "P5140", "horas_prom_ocupados"), names(db_train))

a_cero <- function(.db) .db |> mutate(across(all_of(vars_cero), ~ replace_na(.x, 0)))
db_train <- a_cero(db_train)
db_test  <- a_cero(db_test)

colSums(is.na(db_train))[colSums(is.na(db_train)) > 0]

# No son predictores: id (llave) y peso (factor de expansión del diseño muestral).
vars_no_predictoras <- c("id", "peso")

# ------------------------------------------------------------
# 3.3. Variables nuevas (mismas que Elastic Net, XGBoost, LightGBM y Random Forest)
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

  # --- Interacciones (se conservan para que "fe" sea idéntico al de los demás modelos) ---
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
  # En el modo "cat" los logaritmos no cambian nada (la discretización por cuantiles es invariante
  # a transformaciones monótonas); se conservan para que "fe" sea el mismo conjunto que en los
  # demás scripts y porque sí importan en el modo "gauss".
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

# Dummies (una por nivel, sin fullRank). Los NA se conservan (na.pass): en el modo "cat" forman su
# propio bin. Se descartan columnas constantes y se sanean los nombres.
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
  list(train = xt, test = xs, dv = dv, keep = keep)
}

y_train <- db_train$Pobre
y01     <- as.integer(y_train == "Yes")

# Con los parámetros ganadores solo se arma el conjunto ganador (no los dos del screening).
variables_usadas <- if (USAR_PARAMETROS_GANADORES) GANADOR$variables else VARIABLES_SCREENING
conjuntos <- list()
if ("base" %in% variables_usadas) conjuntos$base <- armar_matrices(db_train,    db_test)
if ("fe"   %in% variables_usadas) conjuntos$fe   <- armar_matrices(db_train_fe, db_test_fe)

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


# ============================================================
# 5. VALIDACIÓN CRUZADA
# ============================================================

# Mismos folds que los otros scripts (semilla 2026, estratificados por Pobre).
set.seed(2026)
folds <- createFolds(y_train, k = 5, returnTrain = TRUE)

# Índices de validación de cada fold.
folds_val <- unname(lapply(folds, function(tr) setdiff(seq_along(y01), tr)))

# Fold de validación al que pertenece cada hogar.
fold_id <- integer(length(y01))
for (k in seq_along(folds_val)) fold_id[folds_val[[k]]] <- k


# ============================================================
# 6. FUNCIONES: F1 POR UMBRAL, NAIVE BAYES, SELECCIÓN Y OOF
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

# Rejilla de umbrales adaptada a la distribución de p. Naive Bayes acumula probabilidades cerca de
# 0 y de 1, así que una rejilla uniforme desperdicia casi todos sus puntos. Se usan cuantiles de p
# (solo la distribución de las probabilidades, no las etiquetas) entre el percentil 50 y el 99.5,
# que es el rango donde puede caer el corte: por debajo del 50 ya se marcaría a más de la mitad de
# los hogares como pobres.
rejilla_umbrales <- function(p, n = 181) {
  us <- unique(quantile(p, seq(0.50, 0.995, length.out = n), names = FALSE, na.rm = TRUE))
  sort(us)
}

# Media móvil de 7 puntos; en los bordes se deja el valor original.
suavizar <- function(f) {
  fs <- as.numeric(stats::filter(f, rep(1 / 7, 7), sides = 2))
  fs[is.na(fs)] <- f[is.na(fs)]
  fs
}

# F1 del OOF agrupado en su mejor umbral, sobre la población comparable con test.
evaluar_oof <- function(p) {
  y <- y01[idx_eval] == 1
  pe <- p[idx_eval]
  us <- rejilla_umbrales(pe)
  f <- f1_curva(pe, y, us)
  tibble(F1_oof = max(f), umbral = us[which.max(f)],
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
  us  <- rejilla_umbrales(pe)
  tp <- fp <- fn <- 0
  umbrales <- numeric(length(folds))
  for (k in seq_along(folds)) {
    otros <- fid != k
    u <- us[which.max(suavizar(f1_curva(pe[otros], y[otros], us)))]
    umbrales[k] <- u
    pred <- pe[!otros] >= u
    yy   <- y[!otros]
    tp <- tp + sum(pred & yy); fp <- fp + sum(pred & !yy); fn <- fn + sum(!pred & yy)
  }
  list(F1 = 2 * tp / (2 * tp + fp + fn), umbrales = umbrales)
}

# ------------------------------------------------------------
# 6.2. Discretización en cuantiles (cortes calculados dentro del fold)
# ------------------------------------------------------------

# Cortes por columna: cuantiles de las filas de entrenamiento del fold. Las columnas con pocos
# valores distintos (dummies, conteos pequeños) no se cortan: cada valor es su propio bin.
calcular_cortes <- function(x, n_bins) {
  lapply(seq_len(ncol(x)), function(j) {
    v <- x[, j]
    v <- v[is.finite(v)]
    u <- unique(v)
    if (length(u) <= n_bins) return(sort(u))              # valores discretos: se usan tal cual
    br <- unique(quantile(v, seq(0, 1, length.out = n_bins + 1)[-c(1, n_bins + 1)],
                          names = FALSE))
    if (length(br) == 0) sort(u)[1] else br               # columna casi constante
  })
}

# Convierte la matriz a códigos de bin (1..B). El código 1 se reserva para los NA, que así forman
# su propio bin (el dato faltante es informativo: P5100 es NA cuando no hay crédito de vivienda).
aplicar_cortes <- function(x, cortes, discreta) {
  out <- matrix(1L, nrow(x), ncol(x), dimnames = list(NULL, colnames(x)))
  for (j in seq_len(ncol(x))) {
    v <- x[, j]
    b <- if (discreta[j]) match(v, cortes[[j]]) else findInterval(v, cortes[[j]]) + 1L
    b[is.na(b)] <- 0L                                     # NA y valores no vistos -> bin 1
    out[, j] <- b + 1L
  }
  out
}

# ------------------------------------------------------------
# 6.3. Naive Bayes (implementación propia)
# ------------------------------------------------------------

# Modo "cat": P(bin | clase) por conteos ponderados con suavizado de Laplace.
#   log P(clase | x) = log P(clase) + sum_j log P(bin_j | clase)
# Modo "gauss": una normal por variable y clase (el Naive Bayes de libro de texto).
#   Se incluye para comparar; con variables asimétricas suele ser peor que "cat".
# `w` son los pesos de las filas (el peso de Bogotá); entran en los conteos y en las medias.
entrenar_nb <- function(x, y, w, modo = "cat", laplace = 1, n_bins = 20, med = NULL) {
  pos <- y == 1
  pri <- c(No = sum(w[!pos]), Yes = sum(w[pos]))
  log_prior <- log(pri / sum(pri))

  if (modo == "gauss") {
    mu <- s2 <- matrix(0, 2, ncol(x), dimnames = list(c("No", "Yes"), colnames(x)))
    for (i in 1:2) {
      sel <- if (i == 1) !pos else pos
      ww  <- w[sel] / sum(w[sel])
      mu[i, ]  <- colSums(x[sel, , drop = FALSE] * ww)
      s2[i, ]  <- colSums((x[sel, , drop = FALSE] - rep(mu[i, ], each = sum(sel)))^2 * ww)
    }
    # Varianza mínima (como var_smoothing de scikit-learn): evita densidades infinitas.
    s2 <- pmax(s2, 1e-9 * matrix(apply(x, 2, var, na.rm = TRUE), 2, ncol(x), byrow = TRUE) + 1e-12)
    return(list(modo = "gauss", log_prior = log_prior, mu = mu, s2 = s2,
                med = med, columnas = colnames(x)))
  }

  # Modo "cat": x ya viene como códigos de bin.
  B <- max(x)
  logp <- vector("list", ncol(x))
  for (j in seq_len(ncol(x))) {
    bj <- x[, j]
    cnt <- rbind(No  = tapply(w[!pos], factor(bj[!pos], levels = 1:B), sum),
                 Yes = tapply(w[pos],  factor(bj[pos],  levels = 1:B), sum))
    cnt[is.na(cnt)] <- 0
    cnt <- cnt + laplace                                   # suavizado de Laplace
    logp[[j]] <- log(cnt / rowSums(cnt))
  }
  list(modo = "cat", log_prior = log_prior, logp = logp, B = B,
       n_bins = n_bins, laplace = laplace, columnas = colnames(x))
}

# Probabilidad de la clase pobre. Se trabaja en logaritmos y se normaliza con la diferencia entre
# clases, que evita desbordes numéricos cuando el producto de verosimilitudes es muy pequeño.
predecir_nb <- function(m, x) {
  if (m$modo == "gauss") {
    lp <- matrix(m$log_prior, nrow(x), 2, byrow = TRUE)
    for (i in 1:2) {
      z <- (x - rep(m$mu[i, ], each = nrow(x)))^2 / rep(m$s2[i, ], each = nrow(x))
      lp[, i] <- lp[, i] - 0.5 * rowSums(z + log(2 * pi * rep(m$s2[i, ], each = nrow(x))))
    }
  } else {
    lp <- matrix(m$log_prior, nrow(x), 2, byrow = TRUE)
    for (j in seq_len(ncol(x))) {
      tab <- m$logp[[j]]
      lp[, 1] <- lp[, 1] + tab[1, x[, j]]
      lp[, 2] <- lp[, 2] + tab[2, x[, j]]
    }
  }
  1 / (1 + exp(pmin(pmax(lp[, 1] - lp[, 2], -700), 700)))   # P(Yes) = sigmoide(logYes - logNo)
}

# ------------------------------------------------------------
# 6.4. Ranking y filtro de correlación por fold (primera etapa)
# ------------------------------------------------------------

# AUC univariado rápido, por rangos: mide cuánta señal trae cada variable por separado. Es el
# criterio natural para Naive Bayes, que evalúa cada variable de forma independiente.
auc_rapido <- function(v, y, w) {
  ok <- is.finite(v)
  if (sum(ok) < 10 || length(unique(v[ok])) < 2) return(0.5)
  r <- rank(v[ok]); yy <- y[ok] == 1
  n1 <- sum(yy); n0 <- sum(!yy)
  if (n1 == 0 || n0 == 0) return(0.5)
  (sum(r[yy]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

# Para cada fold: ordena las variables por |AUC - 0.5| (solo con sus filas de entrenamiento) y
# aplica un filtro de correlación. Naive Bayes supone independencia, así que dos variables con
# |r| > cor_max cuentan la misma evidencia dos veces; se conserva la de más señal.
ranking_por_fold <- function(x, w, cor_max) {
  lapply(seq_along(folds), function(k) {
    tr <- folds[[k]]
    tr <- tr[w[tr] > 0]
    xk <- x[tr, , drop = FALSE]
    s <- abs(vapply(seq_len(ncol(xk)), \(j) auc_rapido(xk[, j], y01[tr], w[tr]), numeric(1)) - 0.5)
    ord <- order(s, decreasing = TRUE)
    if (cor_max >= 1) return(colnames(xk)[ord])
    sel <- integer(0)
    for (j in ord) {
      if (length(sel) == 0) { sel <- j; next }
      r <- suppressWarnings(abs(cor(xk[, j], xk[, sel, drop = FALSE],
                                    use = "pairwise.complete.obs")))
      r[is.na(r)] <- 0
      if (max(r) <= cor_max) sel <- c(sel, j)
    }
    colnames(xk)[sel]
  })
}

# ------------------------------------------------------------
# 6.5. OOF (y predicción sobre test)
# ------------------------------------------------------------

# Por fold: toma las top-K variables del ranking de ese fold (si K es NULL, todas las que pasaron
# el filtro), calcula cortes o medianas con sus filas de entrenamiento, entrena y predice el 20 %
# restante. Si se entrega x_test, promedia las predicciones de los 5 modelos sobre test.
oof_nb <- function(x, p, w, ranking, K = NULL, x_test = NULL, guardar = FALSE) {
  oof <- numeric(nrow(x))
  pred_test <- if (!is.null(x_test)) numeric(nrow(x_test)) else NULL
  cols_folds <- vector("list", length(folds))
  modelos    <- vector("list", length(folds))

  for (k in seq_along(folds)) {
    tr <- folds[[k]]
    va <- setdiff(seq_along(y01), tr)
    tr <- tr[w[tr] > 0]
    cols <- if (is.null(K)) ranking[[k]] else head(ranking[[k]], K)
    cols_folds[[k]] <- cols

    x_tr <- x[tr, cols, drop = FALSE]
    x_va <- x[va, cols, drop = FALSE]
    x_te <- if (!is.null(x_test)) x_test[, cols, drop = FALSE] else NULL

    if (p$modo == "cat") {
      n_bins <- as.integer(p$n_bins)
      cortes <- calcular_cortes(x_tr, n_bins)
      discreta <- vapply(seq_along(cols), \(j) length(unique(x_tr[is.finite(x_tr[, j]), j])) <= n_bins,
                         logical(1))
      b_tr <- aplicar_cortes(x_tr, cortes, discreta)
      m <- entrenar_nb(b_tr, y01[tr], w[tr], "cat", p$laplace, n_bins)
      m$cortes <- cortes; m$discreta <- discreta
      oof[va] <- predecir_nb(m, aplicar_cortes(x_va, cortes, discreta))
      if (!is.null(x_te)) pred_test <- pred_test + predecir_nb(m, aplicar_cortes(x_te, cortes, discreta)) / length(folds)
    } else {
      med <- apply(x_tr, 2, median, na.rm = TRUE)          # gauss: imputación por mediana del fold
      imp <- function(mm) { na <- is.na(mm); if (any(na)) { mm[na] <- med[col(mm)[na]]; mm[is.na(mm)] <- 0 }; mm }
      m <- entrenar_nb(imp(x_tr), y01[tr], w[tr], "gauss", med = med)
      oof[va] <- predecir_nb(m, imp(x_va))
      if (!is.null(x_te)) pred_test <- pred_test + predecir_nb(m, imp(x_te)) / length(folds)
    }
    if (guardar) modelos[[k]] <- m
  }
  list(oof = oof, test = pred_test, cols_folds = cols_folds,
       modelos = if (guardar) modelos else NULL)
}


# ============================================================
# 7. ESPACIO DE BÚSQUEDA DE HIPERPARÁMETROS
# ============================================================

# n_bins: número de cuantiles de la discretización. Pocos bins suavizan y pierden detalle; muchos
#   bins afinan la forma de la distribución pero dejan conteos pequeños en cada celda.
# laplace: suavizado de los conteos. Importa más cuanto más bins haya.
# cor_max: tope de correlación entre variables seleccionadas (1 = sin filtro).
PARAMS_NB <- c("modo", "n_bins", "laplace", "cor_max")

# Valores de partida: discretización media, suavizado 1 y filtro de correlación moderado.
params_base <- tibble(modo = "cat", n_bins = 20L, laplace = 1, cor_max = 0.8)

muestrear_params <- function(n, modo) {
  tibble(
    modo    = modo,
    n_bins  = as.integer(round(10^runif(n, log10(5), log10(60)))),
    laplace = 10^runif(n, -2, 1),
    cor_max = sample(c(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1), n, replace = TRUE)
  )
}


# ============================================================
# 8. SCREENING: VARIABLES x MODO x PESO DE BOGOTÁ (parámetros de partida)
# ============================================================

dir.create("salidas", showWarnings = FALSE)

# Peso de cada fila: Bogotá recibe `pb`; el resto, 1. La evaluación siempre es sin Bogotá.
pesos_filas <- function(pb) ifelse(es_bogota, pb, 1)

# USAR_PARAMETROS_GANADORES = FALSE: screening + selección de K (9) + búsqueda aleatoria (10).
# TRUE: se omite lo anterior y se usa la configuración ganadora (rama `else` al final de la sección 10).
if (!USAR_PARAMETROS_GANADORES) {

configs <- expand.grid(variables = VARIABLES_SCREENING, modo = MODO_SCREENING,
                       peso_bogota = PESO_BOGOTA, stringsAsFactors = FALSE)
resultados <- vector("list", nrow(configs))

for (i in seq_len(nrow(configs))) {
  cfg <- configs[i, ]
  message(sprintf("[%d/%d] variables = %s | modo = %s | peso Bogotá = %s", i, nrow(configs),
                  cfg$variables, cfg$modo, cfg$peso_bogota))
  t0 <- Sys.time()
  x <- conjuntos[[cfg$variables]]$train
  w <- pesos_filas(cfg$peso_bogota)
  p <- params_base |> mutate(modo = cfg$modo)

  rk <- ranking_por_fold(x, w, p$cor_max)
  p_oof <- oof_nb(x, p, w, rk)$oof
  resultados[[i]] <- bind_cols(
    tibble(variables = cfg$variables, modo = cfg$modo, peso_bogota = cfg$peso_bogota,
           n_cols = ncol(x), n_tras_filtro = length(rk[[1]])),
    evaluar_oof(p_oof),
    tibble(minutos = as.numeric(difftime(Sys.time(), t0, units = "mins")))
  )
  rm(x, p_oof, rk); gc(verbose = FALSE)
}

screening <- bind_rows(resultados) |> arrange(desc(F1_oof))
write.csv(screening, file.path("salidas", "NB_screening.csv"), row.names = FALSE)

# F1_oof: F1 del OOF agrupado, en el umbral óptimo, sobre la población comparable con test.
# Con una sola corrida, diferencias menores a 0.002 son ruido.
print(screening |> mutate(across(where(is.numeric), ~ round(.x, 5))), width = Inf)

mejor_cfg <- screening[1, ]
x_train <- conjuntos[[mejor_cfg$variables]]$train
x_test  <- conjuntos[[mejor_cfg$variables]]$test
w       <- pesos_filas(mejor_cfg$peso_bogota)
stopifnot(identical(colnames(x_train), colnames(x_test)))


# ============================================================
# 9. SELECCIÓN DE VARIABLES DENTRO DEL PLIEGUE (top-K por señal univariada)
# ============================================================

# Primera etapa: ranking por |AUC - 0.5| y filtro de correlación, calculados en cada fold solo con
# su 80 % de train. Segunda etapa: cada fold entrena con sus top-K variables; el OOF y la predicción
# de test usan las columnas de ese fold.
message("Ranking y filtro de correlación por fold (primera etapa)")
params_cfg <- params_base |> mutate(modo = mejor_cfg$modo)
ranking <- ranking_por_fold(x_train, w, params_cfg$cor_max)

K_validos <- K_CANDIDATOS[K_CANDIDATOS < length(ranking[[1]])]
sel <- vector("list", length(K_validos) + 1)
for (i in seq_along(K_validos)) {
  K <- K_validos[i]
  message(sprintf("Selección K = %d (%d/%d)", K, i, length(K_validos)))
  p_oof <- oof_nb(x_train, params_cfg, w, ranking, K)$oof
  sel[[i]] <- bind_cols(
    tibble(K = K, n_union = length(unique(unlist(lapply(ranking, head, K))))),
    evaluar_oof(p_oof))
}
sel[[length(sel)]] <- bind_cols(
  tibble(K = NA_real_, n_union = length(unique(unlist(ranking)))),
  mejor_cfg |> select(F1_oof, umbral, PRAUC, AUC)
)
seleccionK <- bind_rows(sel)
write.csv(seleccionK, file.path("salidas", "NB_seleccionK.csv"), row.names = FALSE)
print(seleccionK |> mutate(across(where(is.numeric), ~ round(.x, 5))), width = Inf)

# Parsimonia: entre las opciones a menos de 0.001 del mejor F1, se toma la de menos variables.
ok <- seleccionK$F1_oof >= max(seleccionK$F1_oof) - 0.001
sel_fila <- seleccionK[ok, ][which.min(seleccionK$n_union[ok]), ]
K_final <- if (is.na(sel_fila$K)) NULL else sel_fila$K
message("K elegido: ", if (is.null(K_final)) "todas las que pasan el filtro" else K_final)


# ============================================================
# 10. BÚSQUEDA ALEATORIA DE HIPERPARÁMETROS
# ============================================================

# Cada candidato puede cambiar cor_max, así que se recalcula el ranking de ese candidato (es barato:
# el AUC univariado se reutiliza dentro de ranking_por_fold). Todo dentro del fold.
set.seed(2026)
candidatos <- muestrear_params(N_BUSQUEDA, mejor_cfg$modo)

busqueda <- vector("list", N_BUSQUEDA)
for (i in seq_len(N_BUSQUEDA)) {
  message(sprintf("Búsqueda %d/%d", i, N_BUSQUEDA))
  rk <- ranking_por_fold(x_train, w, candidatos$cor_max[i])
  K_i <- if (is.null(K_final)) NULL else min(K_final, length(rk[[1]]))
  ev <- evaluar_oof(oof_nb(x_train, candidatos[i, ], w, rk, K_i)$oof)
  busqueda[[i]] <- bind_cols(candidatos[i, ],
                             tibble(n_tras_filtro = length(rk[[1]])),
                             ev |> select(F1_oof, umbral, PRAUC, AUC))
}
busqueda <- bind_rows(busqueda) |> arrange(desc(F1_oof))
write.csv(busqueda, file.path("salidas", "NB_busqueda.csv"), row.names = FALSE)
print(busqueda |> slice_head(n = TOP_K) |> mutate(across(where(is.numeric), ~ round(.x, 5))),
      width = Inf)

# Si el mejor candidato no supera al de partida por más de 0.001, se conservan los de partida:
# evita premiar el ruido de la búsqueda.
if (busqueda$F1_oof[1] > sel_fila$F1_oof + 0.001) {
  params_final <- busqueda[1, ]
} else {
  message("La búsqueda no superó a los parámetros de partida: se conservan.")
  params_final <- params_cfg
}
as.data.frame(params_final[PARAMS_NB])

} else {
  # Parámetros ganadores (sin búsqueda). Define lo mismo que las secciones 8 a 10.
  message("Parámetros ganadores: conjunto ", GANADOR$variables, ", modo ", GANADOR$modo,
          ", peso Bogotá ", GANADOR$peso_bogota, ", ", GANADOR$n_bins, " bins, Laplace ", GANADOR$laplace,
          ", |r| <= ", GANADOR$cor_max)
  mejor_cfg <- tibble(variables = GANADOR$variables, modo = GANADOR$modo, peso_bogota = GANADOR$peso_bogota)
  x_train <- conjuntos[[mejor_cfg$variables]]$train
  x_test  <- conjuntos[[mejor_cfg$variables]]$test
  w       <- pesos_filas(mejor_cfg$peso_bogota)
  stopifnot(identical(colnames(x_train), colnames(x_test)))

  K_final      <- GANADOR$K                        # NULL = todas las que pasan el filtro
  params_final <- tibble(modo = GANADOR$modo, n_bins = GANADOR$n_bins, laplace = GANADOR$laplace,
                         cor_max = GANADOR$cor_max)
}


# ============================================================
# 11. OOF FINAL Y PREDICCIÓN SOBRE TEST
# ============================================================

# Se repite el OOF con la configuración elegida y se promedian los 5 modelos sobre test. Naive Bayes
# es determinista: no hay bolsa de semillas. Las probabilidades de test y las del OOF vienen del
# mismo tipo de modelo, así que el umbral elegido con el OOF aplica a ambas.
ranking_final <- ranking_por_fold(x_train, w, params_final$cor_max)
K_usado <- if (is.null(K_final)) NULL else min(K_final, length(ranking_final[[1]]))

fin <- oof_nb(x_train, params_final, w, ranking_final, K_usado, x_test = x_test, guardar = TRUE)

oof <- tibble(rowIndex = seq_along(y01), obs = y_train, Yes = fin$oof)
prob_pobre <- fin$test
stopifnot(length(prob_pobre) == nrow(db_test), !anyNA(prob_pobre))


# ============================================================
# 12. UMBRAL DE DECISIÓN
# ============================================================

#   - Población: solo hogares no-Bogotá del OOF si Bogotá no está en test (idx_eval).
#   - Rejilla por cuantiles: las probabilidades de Naive Bayes se acumulan cerca de 0 y 1, así que la
#     rejilla se construye con los cuantiles de p (sección 6.1) y no con pasos uniformes.
#   - Honestidad: f1_umbral_fuera() elige el umbral de cada fold con los otros 4. Esa es la cifra a
#     reportar; el F1 en el umbral elegido sobre todo el OOF es algo optimista.
oof_eval <- oof[oof$rowIndex %in% idx_eval, ]
y_eval   <- oof_eval$obs == "Yes"

us_fino  <- rejilla_umbrales(oof_eval$Yes)
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
  F1_en_0.50 = max(f1_curva(oof_eval$Yes, y_eval, c(0.5, 1))[1], 0),
  F1_en_umbral = f1_u[which.min(abs(us_fino - umbral_opt))],
  F1_umbral_fuera_del_fold = honesto$F1,
  frac_marcada = mean(oof_eval$Yes >= umbral_opt)        # fracción de hogares marcados como pobres
)
resumen_umbral
honesto$umbrales      # umbral de cada fold, elegido con los otros 4


# ============================================================
# 13. CONVERSIÓN DE PROBABILIDAD A CLASE
# ============================================================

umbral <- umbral_opt
if (USAR_PARAMETROS_GANADORES) message(sprintf("Umbral obtenido %.4f | umbral de referencia (nombre del archivo) %.3f", umbral, GANADOR$umbral_ref))
clase_pobre <- ifelse(prob_pobre >= umbral, "Yes", "No")

# Tasa de hogares predichos como pobres en test frente al OOF comparable: si difieren mucho,
# puede deberse a la composición por dominio (sección 16.4).
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

name <- paste0(AUTOR, "_NB_", mejor_cfg$variables, "_", params_final$modo,
               "_K", if (is.null(K_usado)) "todas" else K_usado,
               "_pb", gsub(".", "", as.character(mejor_cfg$peso_bogota), fixed = TRUE),
               "_bins_", if (params_final$modo == "cat") params_final$n_bins else 0,
               "_lap_", fmt(params_final$laplace),
               "_umbral_", umbral_str, ".csv")
name

write.csv(predictSample, file.path("salidas", name), row.names = FALSE)


# ============================================================
# 16. DIAGNÓSTICOS
# ============================================================

# ------------------------------------------------------------
# 16.1. Importancia de variables: señal univariada y discriminación por bin
# ------------------------------------------------------------

# En Naive Bayes no hay importancia por ganancia: cada variable aporta su log-razón de
# verosimilitudes. Se reportan dos cosas por variable: el AUC univariado (señal cruda) y el rango de
# log P(bin|pobre) - log P(bin|no pobre), que mide cuánto puede mover la predicción.
vars_modelo <- fin$cols_folds[[1]]
auc_vars <- vapply(vars_modelo, \(v) auc_rapido(x_train[, v], y01, w), numeric(1))

m1 <- fin$modelos[[1]]
rango_lr <- if (m1$modo == "cat") {
  vapply(seq_along(m1$columnas), function(j) {
    d <- m1$logp[[j]][2, ] - m1$logp[[j]][1, ]
    diff(range(d[is.finite(d)]))
  }, numeric(1))
} else rep(NA_real_, length(m1$columnas))

importancia <- tibble(Feature = m1$columnas,
                      AUC = unname(auc_vars[m1$columnas]),
                      senal = abs(unname(auc_vars[m1$columnas]) - 0.5),
                      rango_logratio = rango_lr) |>
  arrange(desc(senal))
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
# 16.3. Calibración de las probabilidades
# ------------------------------------------------------------

# Probabilidad media predicha frente a frecuencia real de pobreza, por decil de probabilidad. Si las
# columnas se separan mucho, el modelo ordena bien pero sus probabilidades no son interpretables como
# tales: por eso el umbral se elige por cuantiles y no se fija en 0.5.
calibracion <- tibble(p = oof_eval$Yes, y = as.integer(y_eval)) |>
  mutate(decil = ntile(p, 10)) |>
  group_by(decil) |>
  summarise(n = n(), prob_media = mean(p), frecuencia_real = mean(y), .groups = "drop")
calibracion

# Concentración de las probabilidades en los extremos (síntoma del supuesto de independencia).
c(frac_p_menor_001 = mean(oof_eval$Yes < 0.01),
  frac_p_mayor_099 = mean(oof_eval$Yes > 0.99))

# ------------------------------------------------------------
# 16.4. Bogotá frente al resto (OOF completo) y composición de test
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
# 16.5. Errores del modelo: F1 por dominio
# ------------------------------------------------------------

# Un dominio con F1 muy bajo respecto del resto sugiere señal local que faltan variables por capturar.
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


# ============================================================
# 17. GUARDADO DEL MEJOR MODELO
# ============================================================

# La predicción de Kaggle es el promedio de los 5 modelos de los folds. El umbral aplica a ese
# promedio. Cada fold usa sus propias variables (cols_folds) y sus propios cortes de discretización,
# que viajan dentro de cada modelo. Todo cabe en un solo .rds: un modelo Naive Bayes son tablas de
# probabilidades, no estructuras pesadas.
dir.create("modelos", showWarnings = FALSE)

saveRDS(list(
  config          = mejor_cfg,                                # variables, modo y peso de Bogotá
  K               = K_usado,                                  # NULL = todas las que pasan el filtro
  cols_folds      = fin$cols_folds,                           # variables de cada fold
  modelos         = fin$modelos,                              # 5 modelos (tablas + cortes)
  params          = params_final,                             # hiperparámetros elegidos
  umbral          = umbral,
  columnas        = colnames(x_train),                        # orden exacto de las columnas
  dv              = conjuntos[[mejor_cfg$variables]]$dv,      # codificador de dummies
  keep            = conjuntos[[mejor_cfg$variables]]$keep,    # columnas no constantes
  ref_train       = ref_train,                                # medianas por Dominio (3.3)
  niveles_dominio = niveles_dominio,
  niveles_depto   = niveles_depto,
  vars_cero       = vars_cero
), file.path("modelos", "NB_pipeline.rds"))

# Reutilización del pipeline guardado (requiere aplicar_cortes() y predecir_nb(), secciones 6.2 y 6.3):
#   pipe <- readRDS(file.path("modelos", "NB_pipeline.rds"))
#   prob <- rowMeans(sapply(seq_along(pipe$modelos), function(k) {
#     m <- pipe$modelos[[k]]
#     xk <- x_nuevo[, pipe$cols_folds[[k]], drop = FALSE]
#     if (m$modo == "cat") predecir_nb(m, aplicar_cortes(xk, m$cortes, m$discreta))
#     else { na <- is.na(xk); xk[na] <- m$med[col(xk)[na]]; predecir_nb(m, xk) }
#   }))
#   clase <- as.integer(prob >= pipe$umbral)
#
# x_nuevo (matriz numérica) se construye igual que en train: a_factores(), a_cero(),
# crear_features(db, pipe$ref_train); luego predict(pipe$dv, ...), [, pipe$keep] y
# colnames(x_nuevo) <- pipe$columnas.




# ============================================================
# 18. DESCRIPCIÓN DE RESULTADOS
# ============================================================

# Dos prints y una tabla del MEJOR modelo. Todo se calcula sobre el OOF de los hogares fuera de Bogotá
# (la población que se parece a test) y en el umbral elegido:
#   print 1: hiperparámetros y resultados de la validación cruzada (F1 y umbral);
#   print 2: detalle (matriz de confusión, métricas, class weight, proporción de pobres, variables);
#   tabla  : salidas/Cesar_NB_mejor_modelo.csv, una fila con toda esa información en columnas.
# Hay dos F1: el del umbral elegido fuera de cada fold (el de referencia) y el del umbral
# elegido en todo el OOF (optimista). No es un CV anidado: K, el modo, el peso de Bogotá y los
# hiperparámetros se eligieron con los mismos folds.

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

# Hiperparámetros del modelo final (los de params_final más K).
hp <- c(as.list(params_final[PARAMS_NB]), list(K = if (is.null(K_usado)) "todas" else K_usado))
fmt_hp <- function(v) if (is.numeric(v)) format(signif(v, 4)) else as.character(v)
hp_txt <- paste(names(hp), vapply(hp, fmt_hp, character(1)), sep = " = ")

# Variables del modelo: la unión de las de los 5 folds (cada fold filtra por correlación y por K).
vars_modelo_final <- unique(unlist(fin$cols_folds))
n_vars_fold <- lengths(fin$cols_folds)

# Class weight: Naive Bayes no pondera clases; la a priori sale de las frecuencias de train (ponderadas por
# el peso de Bogotá). Solo el peso de Bogotá pondera filas.
usa_cw <- FALSE
modo_txt <- if (USAR_PARAMETROS_GANADORES) "parámetros ganadores (sin búsqueda)" else "búsqueda completa"
tipo_nb <- if (params_final$modo == "cat") "categórico (variables discretizadas en cuantiles)" else "gaussiano"

# ------------------------------------------------------------
# 18.2. Primer print: hiperparámetros y validación cruzada
# ------------------------------------------------------------

linea_igual <- strrep("=", 78)
imprimir_resumen <- function() {
  cat(c(linea_igual,
        "DESCRIPCIÓN DE RESULTADOS (1/2): MEJOR MODELO, HIPERPARÁMETROS Y VALIDACIÓN CRUZADA",
        linea_igual,
        sprintf("Modelo: Naive Bayes %s | conjunto de variables: %s | ejecución: %s",
                tipo_nb, mejor_cfg$variables, modo_txt),
        "Hiperparámetros:",
        strwrap(paste(hp_txt, collapse = " | "), width = 90, indent = 2, exdent = 2),
        sprintf("  (suavizado de Laplace %s; filtro de correlación |r| <= %s; predicción de test = promedio de los %d modelos de los folds)",
                format(signif(params_final$laplace, 3)), format(signif(params_final$cor_max, 3)), length(folds)),
        "",
        sprintf("Validación cruzada (%d pliegues; OOF de %d hogares fuera de Bogotá, %.2f%% pobres):",
                length(folds), length(y_eval_l), 100 * mean(y_eval_l)),
        sprintf("  Umbral elegido: %.4f sobre cuantiles de la probabilidad (marca al %.1f%% de los hogares; bootstrap p25-p75: %.4f - %.4f)",
                umbral, 100 * resumen_umbral$frac_marcada, resumen_umbral$boot_p25, resumen_umbral$boot_p75),
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
        sprintf("Matriz de confusión (OOF sin Bogotá, umbral %.4f; filas = predicho, columnas = observado):", umbral)),
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
        "  NO: Naive Bayes no usa pesos de clase; la probabilidad a priori sale de las frecuencias de train.",
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
  tibble(modelo = "Naive Bayes", autor = AUTOR, ejecucion = modo_txt, conjunto_variables = mejor_cfg$variables),
  as_tibble(setNames(hp, paste0("hp_", names(hp)))),
  tibble(cv_pliegues = length(folds), cv_hogares_evaluados = length(y_eval_l),
         umbral = umbral, umbral_boot_p25 = resumen_umbral$boot_p25, umbral_boot_p75 = resumen_umbral$boot_p75,
         F1_umbral_fuera_del_fold = honesto$F1, F1_umbral_todo_oof = cm_res$F1,
         F1_pliegue_media = mean(f1_pliegues), F1_pliegue_de = sd(f1_pliegues),
         AUC = auc_oof, PR_AUC = ap_oof),
  select(cm_res, TP, FP, FN, TN, accuracy, balanced_accuracy, precision, recall, specificity, FPR, FNR, FDR, NPV, MCC),
  tibble(class_weight = usa_cw, scale_pos_weight = NA_real_,
         peso_bogota = mejor_cfg$peso_bogota,
         prop_pobres_train = mean(y01), prop_pobres_evaluacion = mean(y_eval_l),
         prop_pred_pobre_oof = mean(p_eval >= umbral), prop_pred_pobre_test = mean(prob_pobre >= umbral),
         n_variables = length(vars_modelo_final), variables = paste(vars_modelo_final, collapse = "; "),
         archivo_envio = name))
write.csv(tabla_mejor, file.path("salidas", paste0(AUTOR, "_NB_mejor_modelo.csv")), row.names = FALSE)

# Los dos prints también quedan en un archivo de texto.
writeLines(capture.output(imprimir_resumen(), imprimir_detalle()), file.path("salidas", paste0(AUTOR, "_NB_descripcion.txt")))
cat("Tabla del mejor modelo: salidas/", AUTOR, "_NB_mejor_modelo.csv\n", sep = "")
cat("Archivo de predicciones: salidas/", name, "\n", sep = "")
