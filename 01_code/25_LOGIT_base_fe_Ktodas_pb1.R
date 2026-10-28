# ==============================================================================
# 25_LOGIT_base_fe_Ktodas_pb1.R
# Autor: Cesar
# Modelo: Regresión logística sin penalización (glm binomial vía caret), optimizando el F1.
#
# Lee 00_data/raw/*.csv, construye la base de hogares, entrena, elige el umbral de decisión y genera el
# archivo de envío. No depende de otros scripts ni del runner (00); detecta la raíz del proyecto.
#
# Definiciones (variante "base", solo clasificación):
#   - Usa únicamente variables de la base de hogares, sin ingreso predicho.
#   - Folds estratificados por Pobre con createFolds (semilla 2026).
#   - El umbral se calcula con el OOF; 0.35 es solo un valor de referencia.
#   - glm no tiene hiperparámetros ni búsqueda: el script corre una única configuración (todas las
#     variables, peso de Bogotá 1), por lo que no incluye USAR_PARAMETROS_GANADORES.
# ==============================================================================

# ============================================================
# PARÁMETROS DE EJECUCIÓN
# ============================================================
AUTOR <- "Cesar"
UMBRAL_REF <- 0.35   # umbral de referencia (solo comparación; el umbral se recalcula con el OOF)

# ============================================================
# PAQUETES Y RAÍZ DEL PROYECTO
# ============================================================
# pacman::p_load carga cada paquete y lo instala si falta. No se carga el meta-paquete tidyverse
# (arrastra ragg): el script solo usa sus componentes.
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(
  readr, dplyr, tidyr, purrr, tibble, stringr,   # datos y modelo
  caret,                                         # dummyVars, folds, glm, findLinearCombos y F_meas
  MLmetrics,                                     # lo usa caret::prSummary por dentro
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
# MODELO: REGRESIÓN LOGÍSTICA
# ============================================================

# Prefijo de todos los archivos de salida de este modelo (en salidas/) y etiqueta de la variante.
PREF <- paste0(AUTOR, "_LOGIT_")
variante <- "base"


# ============================================================
# 1. BASES DE TRAIN Y TEST
# ============================================================

# Train y test deben tener las mismas columnas; train solo agrega la respuesta.
stopifnot(setequal(setdiff(names(db_train), names(db_test)), "Pobre"),
          length(setdiff(names(db_test), names(db_train))) == 0)

# Un hogar por fila: id no puede repetirse.
stopifnot(!anyDuplicated(db_train$id), !anyDuplicated(db_test$id))


# ============================================================
# 2. PREPARACIÓN DE LAS BASES PARA LA REGRESIÓN LOGÍSTICA
# ============================================================

# ------------------------------------------------------------
# 2.1. Variable objetivo y factores
# ------------------------------------------------------------

# 1 = pobre; 0 = no pobre. El primer nivel es la clase positiva.
db_train <- db_train |>
  mutate(Pobre = factor(Pobre, levels = c(1, 0), labels = c("Yes", "No")))

# Los niveles de la geografía se fijan solo con train. Bogotá (Dominio BOGOTA, Depto 11) está en
# train y no en test: queda como nivel sin observaciones en test.
niveles_dominio <- sort(unique(db_train$Dominio))
niveles_depto   <- sort(unique(db_train$Depto))
stopifnot(all(db_test$Dominio %in% niveles_dominio), all(db_test$Depto %in% niveles_depto))

a_factores <- function(.db) {
  .db |> mutate(Dominio = factor(Dominio, levels = niveles_dominio),
                Depto   = factor(Depto,   levels = niveles_depto))
}
db_train <- a_factores(db_train)
db_test  <- a_factores(db_test)

# ------------------------------------------------------------
# 2.2. Faltantes estructurales: cero más bandera
# ------------------------------------------------------------

# P5100, P5130 y P5140 son NA por diseño según la tenencia, y horas_prom_ocupados es NA cuando el
# hogar no tiene ocupados. Se dejan en 0: las banderas (tiene_credito_vivienda, tenencia_*,
# valor_arriendo_faltante y sin_ocupados) ya existen. Es una constante: igual en train y test.
vars_cero <- c("P5100", "P5130", "P5140", "horas_prom_ocupados")

a_cero <- function(.db) .db |> mutate(across(all_of(vars_cero), ~ replace_na(.x, 0)))
db_train <- a_cero(db_train)
db_test  <- a_cero(db_test)

# Faltantes que quedan: se imputan con la mediana dentro de cada fold (sección 6).
faltantes_restantes <- colSums(is.na(db_train))[colSums(is.na(db_train)) > 0]
print(faltantes_restantes)

# ------------------------------------------------------------
# 2.3. Variables con colas pesadas: logaritmo
# ------------------------------------------------------------

# Las variables en pesos y sus razones tienen valores extremos (P5140 llega a 300 millones con
# mediana de 380 mil; carga_arriendo_lp llega a 556 con mediana 0). Estandarizadas, esos casos
# quedan a 200 desviaciones y un logit sin penalización no converge (en una prueba, AUC de 0.63
# frente a 0.93 con el logaritmo). log1p comprime la cola y conserva el orden y los ceros.
# Es una transformación fija, igual en train y test.
vars_log <- c("P5100", "P5130", "P5140", "valor_arriendo", "renta_pc", "renta_pc_lp",
              "carga_arriendo_lp", "cuota_pc")

a_log <- function(.db) .db |> mutate(across(any_of(vars_log), ~ log1p(pmax(.x, 0))))
db_train <- a_log(db_train)
db_test  <- a_log(db_test)

# ------------------------------------------------------------
# 2.4. Matrices de predictores (todas las variables)
# ------------------------------------------------------------

# No son predictoras: id (llave) y peso (factor de expansión). No hay columna `fold` en esta variante.
vars_no_predictoras <- c("id", "peso")

x_train_df <- db_train |> select(-all_of(c(vars_no_predictoras, "Pobre")))
x_test_df  <- db_test  |> select(-all_of(vars_no_predictoras))
stopifnot(identical(names(x_train_df), names(x_test_df)))

# Dummies para Dominio y Depto, ajustadas solo con train. Los NA se conservan (na.pass).
dv <- dummyVars(~ ., data = x_train_df, fullRank = TRUE)
x_train <- predict(dv, newdata = x_train_df, na.action = na.pass)
x_test  <- predict(dv, newdata = x_test_df,  na.action = na.pass)

# ------------------------------------------------------------
# 2.5. Columnas linealmente dependientes
# ------------------------------------------------------------

# Un logit sin penalización no estima coeficientes de columnas que son combinación lineal de otras
# (p. ej. las 6 dummies de tenencia suman 1, y P5090 es una combinación de ellas). Se detectan con
# los predictores estandarizados (las columnas en pesos tienen escalas enormes) y sin mirar Pobre.
x_imp <- apply(x_train, 2, function(v) replace(v, is.na(v), median(v, na.rm = TRUE)))
con_varianza <- apply(x_imp, 2, sd) > 0

combos <- findLinearCombos(scale(x_imp[, con_varianza, drop = FALSE]))
quitar <- c(names(which(!con_varianza)),
            colnames(x_imp[, con_varianza, drop = FALSE])[combos$remove])

if (length(quitar) > 0) {
  message("Columnas retiradas por constantes o dependencia lineal: ", paste(quitar, collapse = ", "))
  x_train <- x_train[, !colnames(x_train) %in% quitar, drop = FALSE]
  x_test  <- x_test[,  colnames(x_train), drop = FALSE]
}
stopifnot(identical(colnames(x_train), colnames(x_test)))

y_train <- db_train$Pobre

print(dim(x_train))


# ============================================================
# 3. REGLA DE REFERENCIA
# ============================================================

# Estadísticas de referencia de la variable objetivo:
# número de hogares, proporción de pobres y el accuracy de clasificar todo como "No".
db_referencia <- tibble(
  Hogares = length(y_train),
  proporcion_pobres = mean(y_train == "Yes"),
  Accuracy = mean(y_train == "No")
)

# Mostrar resultados de referencia.
print(db_referencia)


# ============================================================
# 4. VALIDACIÓN CRUZADA
# ============================================================

# 5 folds estratificados por Pobre (semilla 2026), los mismos que usan los demás modelos con esta semilla.
# returnTrain = TRUE: cada elemento contiene los índices de entrenamiento (todas las filas fuera del fold k).
set.seed(2026)
folds <- createFolds(y_train, k = 5, returnTrain = TRUE)

stopifnot(length(folds) == 5, all(lengths(folds) > 0))


# ============================================================
# 5. CONFIGURACIÓN DE CARET
# ============================================================

# Configuración del procedimiento de validación cruzada.
ctrl <- trainControl(

  # Validación cruzada.
  method = "cv",

  # Número de folds.
  number = length(folds),

  # Índices definidos previamente.
  index = folds,

  # Solicitar probabilidades de clase.
  classProbs = TRUE,

  # prSummary entrega F (F1 de la clase positiva), precisión, recall y PR-AUC.
  # Con ~20 % de hogares pobres, el accuracy premiaría predecir siempre "No".
  summaryFunction = prSummary,

  # Conservar las predicciones fuera de muestra (OOF) del modelo final.
  savePredictions = "final"
)

# Restablecer la semilla antes del entrenamiento.
set.seed(2026)


# ============================================================
# 6. ENTRENAMIENTO DEL MODELO LOGIT
# ============================================================

message("Entrenando regresión logística (glm, 5 folds) ...")

# Entrenar una regresión logística (glm binomial). No hay hiperparámetros: la grilla es una fila.
tiempo <- system.time(
  logit <- train(

    # Interfaz x/y: x son todos los predictores (dummies incluidas) e y es Pobre.
    x = x_train,
    y = y_train,

    # Métrica de evaluación: F1 de la clase positiva.
    metric = "F",
    maximize = TRUE,

    # Esquema de validación cruzada.
    trControl = ctrl,

    # glm no tiene hiperparámetros.
    tuneGrid = data.frame(parameter = "none"),

    # Modelo utilizado por caret.
    method = "glm",

    # Preprocesamiento calculado con el fold de entrenamiento y aplicado al de validación
    # (y a test al predecir): quita columnas constantes, imputa los NA con la mediana y estandariza
    # (los coeficientes son más estables cuando las escalas son comparables).
    preProcess = c("zv", "medianImpute", "center", "scale"),

    # Familia binomial para clasificación.
    family = "binomial"
  )
)[["elapsed"]]

# F1 de validación (con el corte de 0.5 de caret).
print(logit$results)
F1_corte_05 <- max(logit$results$F, na.rm = TRUE)


# ============================================================
# 7. INSPECCIÓN DE PREDICCIONES DE VALIDACIÓN CRUZADA
# ============================================================

# Primeras predicciones almacenadas por caret.
print(head(logit$pred))


# ============================================================
# 8. INSPECCIÓN DE FALSOS NEGATIVOS
# ============================================================

# Observaciones cuya clase real es pobre ("Yes") pero fueron clasificadas como no pobres ("No")
# con el corte de 0.5.
print(head(select(filter(logit$pred, obs == "Yes", pred == "No"),
                  rowIndex, Resample, obs, pred, Yes, No)))


# ============================================================
# 9. PREDICCIÓN DE PROBABILIDADES SOBRE TEST
# ============================================================

# Probabilidad estimada de que cada hogar de test sea pobre ("Yes").
prob_pobre <- predict(logit, newdata = x_test, type = "prob")[["Yes"]]


# ============================================================
# 10. ELECCIÓN DEL UMBRAL CON LAS PREDICCIONES OOF
# ============================================================

# Con clases desbalanceadas el corte de 0.5 deja muchos pobres sin detectar. El umbral que
# maximiza F1 se busca solo con las predicciones fuera de muestra (OOF) de train: nunca con test
# ni con el puntaje de Kaggle.
pred_oof <- logit$pred |> arrange(rowIndex)
stopifnot(nrow(pred_oof) == nrow(db_train), !anyDuplicated(pred_oof$rowIndex))

umbrales <- round(seq(0.05, 0.95, by = 0.01), 2)

f1_por_umbral <- sapply(umbrales, function(u) {
  pred <- factor(ifelse(pred_oof$Yes >= u, "Yes", "No"), levels = c("Yes", "No"))
  f1 <- F_meas(data = pred, reference = pred_oof$obs, relevant = "Yes")
  # Si ningún hogar queda predicho como pobre, F1 no está definido: se reporta 0.
  if (is.na(f1)) 0 else f1
})

# Ante empates se toma el umbral más bajo.
umbral_opt <- umbrales[which.max(f1_por_umbral)]

# F1 OOF con el corte de 0.5 frente al corte óptimo.
print(c(F1_umbral_0.5 = f1_por_umbral[umbrales == 0.5],
        F1_umbral_opt = max(f1_por_umbral),
        umbral_opt = umbral_opt))

# Curva de F1 por umbral (permite verificar que el óptimo no sea un pico aislado).
write.csv(tibble(umbral = umbrales, F1 = f1_por_umbral),
          file.path("salidas", paste0(PREF, "curva_umbral_f1.csv")), row.names = FALSE)


# ============================================================
# 11. CONVERSIÓN DE PROBABILIDAD A CLASE
# ============================================================

# Se usa el umbral óptimo de la sección 10.
umbral <- umbral_opt

# Convertir las probabilidades predichas en etiquetas.
clase_pobre <- ifelse(prob_pobre >= umbral, "Yes", "No")


# ============================================================
# 12. PREPARACIÓN DE PREDICCIONES PARA KAGGLE
# ============================================================

# Conservar el identificador del hogar y la clase predicha con el umbral elegido.
predictSample <- db_test |>
  mutate(pobre_lab = clase_pobre) |>
  select(id, pobre_lab)

# Inspeccionar las primeras predicciones.
print(head(predictSample))


# ============================================================
# 13. RECODIFICACIÓN FINAL DE LA VARIABLE PREDICHA
# ============================================================

# Yes -> 1, No -> 0, y solo las columnas requeridas para el archivo de envío.
predictSample <- predictSample |>
  mutate(pobre = if_else(pobre_lab == "Yes", 1L, 0L)) |>
  select(id, pobre)

# Validar el archivo antes de guardarlo: un hogar por fila, sin NA y con 0/1.
stopifnot(nrow(predictSample) == nrow(db_test),
          !anyDuplicated(predictSample$id),
          !anyNA(predictSample),
          all(predictSample$pobre %in% c(0L, 1L)),
          setequal(predictSample$id, db_test$id))


# ============================================================
# 14. CONSTRUCCIÓN DEL NOMBRE DEL ARCHIVO
# ============================================================

# El nombre incorpora la variante (aquí "base", sin ingreso predicho) y el umbral. Los puntos se
# eliminan de los valores numéricos, como en el script de referencia.
umbral_str <- gsub(".", "", sprintf("%.2f", umbral), fixed = TRUE)

name <- paste0(PREF, variante, "_umbral_", umbral_str, ".csv")


# ============================================================
# 15. INSPECCIÓN DEL ARCHIVO FINAL
# ============================================================

print(head(predictSample))
print(name)


# ============================================================
# 16. EXPORTACIÓN DEL ARCHIVO
# ============================================================

# Guardar las predicciones en formato CSV sin incluir los nombres de fila de R.
write.csv(predictSample, file.path("salidas", name), row.names = FALSE)


# ============================================================
# 17. DIAGNÓSTICO: DISTANCIA A LA LÍNEA DE POBREZA
# ============================================================

# Análisis POSTERIOR al modelo: el ingreso observado (solo existe en train) no entra a ningún
# ajuste ni a la elección del umbral. Muestra qué tan cerca de la línea están los errores:
# si los falsos negativos y positivos tienen ingreso casi igual a Lp, son hogares que ningún
# modelo con estas variables podría separar. El ingreso per cápita de la unidad de gasto (Ingpcug)
# se lee de train_hogares.csv.
aux <- leer_raw("train_hogares.csv", c("id", "Ingpcug"))
ingreso <- aux$Ingpcug[match(db_train$id, aux$id)]
stopifnot(!anyNA(ingreso))

diag_oof <- pred_oof |>
  mutate(ratio_ingreso_lp = ingreso[rowIndex] / db_train$Lp[rowIndex],
         pred_umbral = factor(ifelse(Yes >= umbral, "Yes", "No"), levels = c("Yes", "No")),
         tipo = case_when(obs == "Yes" & pred_umbral == "Yes" ~ "TP (pobre detectado)",
                          obs == "Yes" & pred_umbral == "No"  ~ "FN (pobre no detectado)",
                          obs == "No"  & pred_umbral == "Yes" ~ "FP (no pobre marcado pobre)",
                          TRUE                                ~ "TN (no pobre)"))

# Ingreso per cápita sobre Lp: 1 = justo en la línea. Entre 0.8 y 1.2 las etiquetas son casi azarosas.
diagnostico_linea <- diag_oof |>
  group_by(tipo) |>
  summarize(n = n(),
            mediana_ratio = median(ratio_ingreso_lp),
            p25 = quantile(ratio_ingreso_lp, 0.25),
            p75 = quantile(ratio_ingreso_lp, 0.75),
            prop_cerca_linea = mean(ratio_ingreso_lp >= 0.8 & ratio_ingreso_lp <= 1.2),
            .groups = "drop")
print(diagnostico_linea)
write.csv(diagnostico_linea, file.path("salidas", paste0(PREF, "diagnostico_distancia_linea.csv")), row.names = FALSE)


# ============================================================
# 18. DESCRIPCIÓN DE RESULTADOS
# ============================================================

# Dos prints y una tabla del MEJOR modelo. Todo se calcula sobre el OOF de TODOS los hogares de train
# (este modelo no excluye Bogotá del OOF ni del umbral) y en el umbral elegido; la tabla agrega el F1 del
# OOF sin Bogotá, la población que se parece a test:
#   print 1: hiperparámetros y resultados de la validación cruzada (F1 y umbral);
#   print 2: detalle (matriz de confusión, métricas, class weight, proporción de pobres, variables);
#   tabla  : salidas/Cesar_LOGIT_mejor_modelo.csv, una fila con toda esa información en columnas.
# Hay dos F1: el del umbral elegido fuera de cada fold (el de referencia) y el del umbral
# elegido en todo el OOF (optimista).

# ------------------------------------------------------------
# 18.1. Insumos y funciones de métricas
# ------------------------------------------------------------

p_eval   <- pred_oof$Yes                 # probabilidad OOF de cada hogar (ordenada por rowIndex)
y_eval_l <- pred_oof$obs == "Yes"        # TRUE = pobre
# Pliegue en que cada hogar queda fuera de entrenamiento (folds guarda los índices de entrenamiento).
fold_id <- integer(nrow(db_train))
for (k in seq_along(folds)) fold_id[setdiff(seq_len(nrow(db_train)), folds[[k]])] <- k
fold_eval <- fold_id[pred_oof$rowIndex]
es_bogota <- as.character(db_train$Dominio[pred_oof$rowIndex]) == "BOGOTA"

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

cm_res <- metricas_umbral(p_eval, y_eval_l, umbral_opt)
f1_pliegues <- vapply(sort(unique(fold_eval)), function(k)
  metricas_umbral(p_eval[fold_eval == k], y_eval_l[fold_eval == k], umbral_opt)$F1, numeric(1))
auc_oof <- auc_num(p_eval, y_eval_l)
ap_oof  <- ap_num(p_eval, y_eval_l)
f1_sin_bogota <- metricas_umbral(p_eval[!es_bogota], y_eval_l[!es_bogota], umbral_opt)$F1

# F1 "honesto": el umbral de cada pliegue se elige con el OOF de los OTROS pliegues y se aplica a ese
# pliegue; el F1 se calcula sobre todas las predicciones así obtenidas (misma regla que el umbral final).
pred_honesta <- logical(length(p_eval))
for (k in sort(unique(fold_eval))) {
  dentro <- fold_eval == k
  f1_k <- vapply(umbrales, function(u) {
    v <- metricas_umbral(p_eval[!dentro], y_eval_l[!dentro], u)$F1
    if (is.na(v)) 0 else v
  }, numeric(1))
  pred_honesta[dentro] <- p_eval[dentro] >= umbrales[which.max(f1_k)]
}
f1_honesto <- 2 * sum(pred_honesta & y_eval_l) / (sum(pred_honesta) + sum(y_eval_l))

matriz_conf <- matrix(c(cm_res$TP, cm_res$FN, cm_res$FP, cm_res$TN), 2, 2,
                      dimnames = list(Predicho = c("Pobre", "No pobre"), Observado = c("Pobre", "No pobre")))

# Variables del modelo: columnas de la matriz de predictores tras quitar las dependientes. glm no
# penaliza, así que todas tienen coeficiente (los más grandes, en valor absoluto, sobre variables estandarizadas).
vars_modelo_final <- colnames(x_train)
coefs_glm <- coef(logit$finalModel)
coefs_glm <- coefs_glm[setdiff(names(coefs_glm), "(Intercept)")]
n_coef_na <- sum(is.na(coefs_glm))      # coeficientes no estimados (dependencias que se escaparon)
top_coefs <- head(sort(abs(coefs_glm), decreasing = TRUE), 10)

# Class weight: este modelo no pondera ninguna clase (caret::train sin weights).
usa_cw <- FALSE

# ------------------------------------------------------------
# 18.2. Primer print: hiperparámetros y validación cruzada
# ------------------------------------------------------------

linea_igual <- strrep("=", 78)
imprimir_resumen <- function() {
  cat(c(linea_igual,
        "DESCRIPCIÓN DE RESULTADOS (1/2): MEJOR MODELO, HIPERPARÁMETROS Y VALIDACIÓN CRUZADA",
        linea_igual,
        sprintf("Modelo: regresión logística sin penalización (glm binomial vía caret) | variante: %s", variante),
        "Hiperparámetros: ninguno (glm no tiene). Preprocesamiento por fold: zv, imputación por mediana, centrado y escala.",
        sprintf("  Variables con log1p: %s", paste(intersect(vars_log, names(x_train_df)), collapse = ", ")),
        "",
        sprintf("Validación cruzada (%d pliegues; OOF de %d hogares de train, %.2f%% pobres):",
                length(folds), length(y_eval_l), 100 * mean(y_eval_l)),
        sprintf("  Umbral elegido: %.2f (F1-óptimo en el OOF; referencia: %.2f)",
                umbral_opt, UMBRAL_REF),
        sprintf("  F1, umbral elegido fuera del fold (cifra a reportar): %.4f", f1_honesto),
        sprintf("  F1, umbral elegido en todo el OOF (optimista):        %.4f", cm_res$F1),
        sprintf("  F1 con el corte 0.5 de caret:                         %.4f", F1_corte_05),
        sprintf("  F1 por pliegue en el umbral final: media %.4f | DE %.4f | mín %.4f | máx %.4f",
                mean(f1_pliegues), sd(f1_pliegues), min(f1_pliegues), max(f1_pliegues)),
        sprintf("  F1 en el umbral final, solo hogares fuera de Bogotá: %.4f", f1_sin_bogota),
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
        sprintf("Matriz de confusión (OOF de train, umbral %.2f; filas = predicho, columnas = observado):", umbral_opt)),
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
        "  NO se ponderó la clase pobre (caret::train sin weights); el desbalance se maneja con el umbral",
        "",
        "Proporción de pobres:",
        sprintf("  Train (todos los hogares):        %.2f%%", 100 * mean(y_eval_l)),
        sprintf("  Predicha pobre en el OOF:         %.2f%%", 100 * mean(p_eval >= umbral_opt)),
        sprintf("  Predicha pobre en test:           %.2f%%", 100 * mean(prob_pobre >= umbral_opt)),
        "",
        sprintf("Variables: %d columnas de predictores (con dummies) tras retirar %d constantes o dependientes",
                length(vars_modelo_final), length(quitar)),
        strwrap(paste(vars_modelo_final, collapse = ", "), width = 90, indent = 2, exdent = 2),
        if (length(quitar) > 0) c("Columnas retiradas:", strwrap(paste(quitar, collapse = ", "), width = 90, indent = 2, exdent = 2)),
        sprintf("Coeficientes no estimados (NA) en el glm: %d", n_coef_na),
        "Mayores coeficientes en valor absoluto (variables estandarizadas):",
        sprintf("  %-40s %.4f", names(top_coefs), top_coefs),
        linea_igual), sep = "\n")
}
imprimir_detalle()

# ------------------------------------------------------------
# 18.4. Tabla del mejor modelo (una fila, todo en columnas)
# ------------------------------------------------------------

tabla_mejor <- bind_cols(
  tibble(modelo = "LogitGLM", autor = AUTOR, variante = variante, hiperparametros = "ninguno (glm)"),
  tibble(cv_pliegues = length(folds), cv_hogares_evaluados = length(y_eval_l),
         umbral = umbral_opt, umbral_ref = UMBRAL_REF,
         F1_umbral_fuera_del_fold = f1_honesto, F1_umbral_todo_oof = cm_res$F1,
         F1_corte_05_caret = F1_corte_05, F1_sin_bogota = f1_sin_bogota,
         F1_pliegue_media = mean(f1_pliegues), F1_pliegue_de = sd(f1_pliegues),
         AUC = auc_oof, PR_AUC = ap_oof, tiempo_seg = round(tiempo, 1)),
  select(cm_res, TP, FP, FN, TN, accuracy, balanced_accuracy, precision, recall, specificity, FPR, FNR, FDR, NPV, MCC),
  tibble(class_weight = usa_cw,
         prop_pobres_train = mean(y_eval_l),
         prop_pred_pobre_oof = mean(p_eval >= umbral_opt), prop_pred_pobre_test = mean(prob_pobre >= umbral_opt),
         n_variables = length(vars_modelo_final), variables = paste(vars_modelo_final, collapse = "; "),
         n_columnas_retiradas = length(quitar), columnas_retiradas = paste(quitar, collapse = "; "),
         n_coef_na = n_coef_na,
         archivo_envio = name))
write.csv(tabla_mejor, file.path("salidas", paste0(AUTOR, "_LOGIT_mejor_modelo.csv")), row.names = FALSE)

# Los dos prints también quedan en un archivo de texto.
writeLines(capture.output(imprimir_resumen(), imprimir_detalle()),
           file.path("salidas", paste0(AUTOR, "_LOGIT_descripcion.txt")))
cat("Tabla del mejor modelo: salidas/", AUTOR, "_LOGIT_mejor_modelo.csv\n", sep = "")
