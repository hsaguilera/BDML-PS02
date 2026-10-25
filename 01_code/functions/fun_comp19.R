# ==============================================================================
# fun_comp19.R  --  Funciones de la complementaria del 19 de septiembre
# Réplica del ejemplo "Predicción de la Pobreza" (Rojas Aguilar y
# Sarmiento-Barbieri, 2026-09-19). El cuerpo de las funciones es el del
# ejemplo; solo se cambió el nombre (sufijo _comp19) para no chocar con las
# funciones propias de fun_data_prep.R, y se agregó fun_hogares_comp19(), que
# reúne las dos líneas casi idénticas que el ejemplo deja como ejercicio.
# ==============================================================================

#' Preprocesar base de datos de personas (ejemplo de la complementaria)
#'
#' @description \code{fun_preprocess_personas_comp19()} retorna una base de datos
#' con nombres descriptivos y etiquetas en sus variables categóricas.
#'
#' @param .db \code{tibble}. Base de datos descargada de Kaggle a nivel de personas.
#'
#' @returns \code{tibble}. Incluye el id del hogar, \code{id}, el id de la persona
#' dentro de cada hogar, \code{id_persona}, el género, \code{bin_male}, si es la
#' cabeza del hogar, \code{bin_head}, si está ocupada, \code{bin_occup}, si es
#' inactiva, \code{bin_inactivo}, si está en edad de trabajar,
#' \code{bin_edad_trabajar}, su nivel educativo, \code{cat_educ}, si es menor de
#' edad, \code{bin_minor}, o 'muy' mayor, \code{bin_old}.
fun_preprocess_personas_comp19 <- function(.db) {
  text_vars <- c('id', 'id_persona' = 'Orden', 'num_age' = 'P6040', 'cat_gender' = 'P6020',
                 'bin_employed' = 'Oc', 'bin_inactivo' = 'Ina',
                 'cat_educ' = 'P6210', 'cat_relationship' = 'P6050')

  db_result <- .db |>
    select(all_of(text_vars)) |>
    mutate(bin_male = ifelse(test = cat_gender == 1, yes = 1, no = 0),
           bin_head = ifelse(test = cat_relationship == 1, yes = 1, no = 0),
           bin_occup = ifelse(test = is.na(bin_employed), yes = 0, no = 1),
           bin_inactivo = ifelse(test = is.na(bin_inactivo), yes = 0, no = 1),
           bin_minor = ifelse(test = num_age < 18, yes = 1, no = 0),
           bin_old = ifelse(test = num_age > 65, yes = 1, no = 0),
           bin_edad_trabajar = as.integer(between(num_age, 18, 65)),
           cat_educ = case_when(is.na(cat_educ) ~ 1,   # Si el valor no está reportado, imputamos "Ninguno"
                                cat_educ == 9 ~ 1,     # Si el encuestado no sabe, imputamos "Ninguno"
                                TRUE ~ cat_educ),      # De lo contrario, dejamos el valor
           # Si no informamos que está ordenado, no podremos escoger el máximo
           cat_educ = factor(x = cat_educ,
                             levels = 1:6,
                             ordered = TRUE,
                             labels = c("Ninguno", "Preescolar", "Primaria",
                                        "Secundaria", "Media", "Universitaria"))) |>
    select(id, id_persona, bin_male, bin_head, bin_occup, bin_inactivo,
           bin_edad_trabajar, cat_educ, bin_minor, bin_old)

  return(db_result)
}

#' Colapsar a nivel hogar las observaciones de personas (ejemplo de la complementaria)
#'
#' @description \code{fun_colapse_personas_comp19()} retorna una base de datos con
#' variables a nivel de hogar a partir de observaciones de personas.
#'
#' @param .db \code{tibble}. Base de personas preprocesada por
#' \code{fun_preprocess_personas_comp19()}.
#'
#' @returns \code{tibble}. Incluye el id del hogar, \code{id}, si el jefe es hombre,
#' \code{bin_head_male}, la tasa de dependencia, \code{prop_dependencia}, la tasa
#' de inactividad, \code{prop_inactividad}, y el máximo nivel educativo del hogar,
#' \code{cat_maxEduc}. Las tasas indefinidas (denominador cero) se fijan en 0.
fun_colapse_personas_comp19 <- function(.db) {
  db_result <- .db |>
    group_by(id) |>
    summarize(bin_head_male = max(bin_head * bin_male, na.rm = TRUE),
              num_minors = sum(bin_minor, na.rm = TRUE),
              num_old = sum(bin_old, na.rm = TRUE),
              num_inactivos = sum(bin_inactivo * bin_edad_trabajar, na.rm = TRUE),
              num_edad_trabajar = sum(bin_edad_trabajar, na.rm = TRUE),
              num_edad_dependencia = sum(!bin_edad_trabajar, na.rm = TRUE),
              cat_maxEduc = max(cat_educ, na.rm = TRUE)) |>
    mutate(prop_dependencia = (num_minors + num_old) / na_if(num_edad_dependencia, 0),
           prop_inactividad = num_inactivos / na_if(num_edad_trabajar, 0),
           prop_dependencia = replace_na(prop_dependencia, 0),
           prop_inactividad = replace_na(prop_inactividad, 0)) |>
    ungroup()

  return(db_result)
}

#' Variables de la base de hogares (ejemplo de la complementaria)
#'
#' @description Crea \code{bin_rent} (1 si el hogar arrienda, P5090 == 3), conserva
#' la ubicación (\code{Dominio}) y, solo en train, la variable objetivo
#' \code{Pobre} como factor con niveles c("Yes", "No") (el primero es la clase
#' positiva para \code{twoClassSummary()}).
#'
#' @param .db \code{data.frame}. train_hogares o test_hogares de Kaggle.
#' @param train \code{logical}. TRUE si la base trae \code{Pobre}.
fun_hogares_comp19 <- function(.db, train = TRUE) {
  out <- .db |>
    mutate(bin_rent = ifelse(P5090 == 3, 1, 0))
  if (train) {
    out <- out |>
      select(id, 'cat_ubicacion' = Dominio, bin_rent, Pobre) |>
      mutate(cat_ubicacion = factor(cat_ubicacion),
             # 1 = pobre; 0 = no pobre. El primer nivel es la clase positiva.
             Pobre = factor(Pobre, levels = c(1, 0), labels = c("Yes", "No")))
  } else {
    out <- out |>
      select(id, 'cat_ubicacion' = Dominio, bin_rent) |>
      mutate(cat_ubicacion = factor(cat_ubicacion))
  }
  out
}

#' Entrenar el Elastic Net de la complementaria (caret + glmnet)
#'
#' @description Replica el bloque "Calcular la métrica de desempeño durante la
#' validación cruzada": 5 particiones estratificadas con \code{createFolds()}
#' (semilla 2026), \code{trainControl()} con \code{twoClassSummary} y la grilla
#' alpha = seq(0, 1, 0.05), lambda = 10^seq(log10(100), log10(0.001), length.out = 20).
#' Se selecciona por accuracy, como en el ejemplo.
#'
#' @param db_train \code{data.frame}. Base final de entrenamiento sin \code{id}.
#' @param folds \code{list}. Índices de entrenamiento por partición. Si es NULL
#' se crean con \code{createFolds()} sobre toda la base, como en el ejemplo.
#' @param folds_out \code{list}. Índices de validación por partición (opcional).
#' Permite validar sin ciertos hogares (por ejemplo, sin Bogotá).
#' @returns Objeto \code{train} de caret.
fun_train_en_comp19 <- function(db_train, folds = NULL, folds_out = NULL) {
  set.seed(2026)
  if (is.null(folds)) folds <- createFolds(db_train$Pobre, k = 5, returnTrain = TRUE)

  # Nota de memoria: con savePredictions = "final" caret conserva en memoria las
  # predicciones de TODA la grilla (21 alphas x 20 lambdas x 5 particiones) hasta
  # el final, y eso agota ~8 GB de RAM. Por eso la grilla se recorre sin guardar
  # predicciones (mismas particiones, mismo modelo) y luego se vuelve a ajustar
  # solo la combinación ganadora con savePredictions = "final", como en el ejemplo.
  make_ctrl <- function(save_pred) {
    trainControl(
      method = "cv",
      number = 5,
      index = folds,
      indexOut = folds_out,
      classProbs = TRUE,
      summaryFunction = twoClassSummary,
      savePredictions = save_pred
    )
  }

  set.seed(2026)
  tune_grid <- expand.grid(
    alpha = seq(from = 0, to = 1, by = 0.05),
    lambda = 10^seq(from = log10(100), to = log10(0.001), length.out = 20)
  )

  # Con twoClassSummary() caret no calcula "Accuracy" (solo ROC, Sens y Spec):
  # al pedir metric = "Accuracy" avisa y selecciona por ROC. Se deja igual que
  # en el ejemplo y se suprime ese aviso para que no ensucie el registro.
  fit <- function(grid, save_pred) {
    suppressWarnings(train(
      Pobre ~ .,
      data = db_train,
      metric = "Accuracy",
      maximize = TRUE,
      trControl = make_ctrl(save_pred),
      tuneGrid = grid,
      method = "glmnet",
      family = "binomial",
      standardize = TRUE
    ))
  }

  grid_fit <- fit(tune_grid, "none")
  set.seed(2026)
  final_fit <- fit(grid_fit$bestTune, "final")
  final_fit$results_grid <- grid_fit$results   # métricas de toda la grilla
  final_fit
}

#' Guardar el envío a Kaggle con el formato del ejemplo
#'
#' @param model Objeto \code{train} (glmnet) ya ajustado.
#' @param db_test \code{data.frame}. Base de test con \code{id}.
#' @param dir \code{character}. Carpeta de salida.
#' @param prefix \code{character}. Prefijo del archivo: <autor>_<modelo> (p. ej. "harold_elastic_net").
#' @param suffix \code{character}. Sufijo opcional para distinguir variantes.
#' @returns Invisible: lista con la ruta del CSV y la tabla de envío (id, pobre).
fun_save_submission_comp19 <- function(model, db_test, dir, prefix = "elastic_net", suffix = "") {
  predictSample <- db_test |>
    mutate(pobre_lab = predict(model, newdata = db_test, type = "raw")) |>
    select(id, pobre_lab)

  predictSample <- predictSample |>
    mutate(pobre = if_else(pobre_lab == "Yes", 1L, 0L)) |>
    select(id, pobre)

  # Reemplazamos los puntos por rayas al piso en los valores numéricos de los hiperparámetros.
  lambda_str <- gsub(pattern = "\\.", replacement = "_",
                     x = as.character(round(model$bestTune$lambda, 4)))
  alpha_str <- gsub("\\.", "_", as.character(model$bestTune$alpha))

  name <- paste0(prefix, "_lambda_", lambda_str, "_alpha_", alpha_str, suffix, ".csv")
  path <- file.path(dir, name)
  write.csv(predictSample, path, row.names = FALSE)
  invisible(list(path = path, sample = predictSample))
}
