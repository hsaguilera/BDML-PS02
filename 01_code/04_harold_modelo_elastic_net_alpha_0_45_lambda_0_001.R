# ==============================================================================
# 04_harold_modelo_elastic_net_alpha_0_45_lambda_0_001.R
# Objetivo: replicar, sin cambios de fondo, el modelo de ejemplo de la
#           complementaria del 19 de septiembre (logit con Elastic Net en caret,
#           5 particiones estratificadas, selección por accuracy) y generar el
#           envío a Kaggle con el formato del ejemplo.
# Variantes (misma tubería, solo cambia el tratamiento de Bogotá, que existe en
# train pero no en test):
#   A. "replica"        : exactamente el ejemplo, con todos los hogares de train.
#   B. "eval_sin_bogota"  : se entrena con Bogotá, pero la validación cruzada solo
#                         evalúa hogares fuera de Bogotá (indexOut).
#   C. "sin_bogota"      : Bogotá se excluye de train (y por tanto de la validación).
# Entradas : 00_data/raw/{train,test}_{hogares,personas}.csv
# Salidas  : 02_outputs/predictions/harold_elastic_net_lambda_*_alpha_*[sufijo].csv
#            02_outputs/tables/harold_elastic_net_cv_results.tex
# ==============================================================================

# Autor del script: se usa como prefijo de los archivos de salida (<autor>_<modelo>_...).
if (!exists("DIR_PROC")) source(file.path(if (requireNamespace("here", quietly = TRUE)) here::here() else getwd(), "01_code", "00_setup.R"), encoding = "UTF-8")   # corre solo: carga el entorno común
AUTOR <- "harold"

suppressPackageStartupMessages({library(data.table); library(tidyverse); library(caret); library(glmnet); library(xtable)})

# --- 1. Lectura (solo las columnas que usa el ejemplo) ---------------------------
cols_per <- c("id", "Orden", "P6040", "P6020", "Oc", "Ina", "P6210", "P6050")
db_train_hogares  <- as.data.frame(fread(file.path(DIR_RAW, "train_hogares.csv"),  select = c("id", "Dominio", "P5090", "Pobre")))
db_test_hogares   <- as.data.frame(fread(file.path(DIR_RAW, "test_hogares.csv"),   select = c("id", "Dominio", "P5090")))
db_train_personas <- as.data.frame(fread(file.path(DIR_RAW, "train_personas.csv"), select = cols_per))
db_test_personas  <- as.data.frame(fread(file.path(DIR_RAW, "test_personas.csv"),  select = cols_per))

# --- 2. Pre-procesamiento: la misma tubería para train y test -----------------------
db_train_personas <- fun_preprocess_personas_comp19(db_train_personas)
db_test_personas  <- fun_preprocess_personas_comp19(db_test_personas)

db_train_personas_nivel_hogar <- fun_colapse_personas_comp19(db_train_personas)
db_test_personas_nivel_hogar  <- fun_colapse_personas_comp19(db_test_personas)

db_train_hogares <- fun_hogares_comp19(db_train_hogares, train = TRUE)
db_test_hogares  <- fun_hogares_comp19(db_test_hogares,  train = FALSE)

# El id no es necesario para el entrenamiento, pero sí para el test, pues es la
# forma que tiene Kaggle de identificar a qué observación pertenece cada predicción.
db_train <- db_train_hogares |>
  left_join(db_train_personas_nivel_hogar, by = "id") |>
  select(-id)

db_test <- db_test_hogares |> left_join(db_test_personas_nivel_hogar, by = "id")

stopifnot(nrow(db_train) == 164960, nrow(db_test) == 66168,
          !anyNA(db_train), !anyNA(select(db_test, -id)))

# --- 3. Cálculo de servilleta: ¿qué accuracy da "todos no pobres"? ------------------
db_referencia <- db_train |>
  summarize(
    Hogares = n(),
    proporcion_pobres = mean(Pobre == "Yes"),
    Accuracy = mean(Pobre == "No")
  )
print(db_referencia)

# --- 4. Entrenamiento y envío ---------------------------------------------------------
bog <- db_train$cat_ubicacion == "BOGOTA"
stopifnot(sum(bog) > 0, !any(db_test$cat_ubicacion == "BOGOTA"))

resumen <- list()
run_variant <- function(label, db_tr, folds = NULL, folds_out = NULL, suffix = "") {
  message("\n>>> Elastic Net (", label, ")  n = ", nrow(db_tr))
  t0 <- Sys.time()
  m <- fun_train_en_comp19(db_tr, folds = folds, folds_out = folds_out)
  sub <- fun_save_submission_comp19(m, db_test, DIR_PRED, prefix = paste0(AUTOR, "_elastic_net"), suffix = suffix)

  best <- m$results
  oof  <- m$pred            # predicciones out-of-fold de la combinación ganadora (regla 0,5)
  mm   <- conf_metrics(as.integer(oof$obs == "Yes"), as.integer(oof$pred == "Yes"))
  resumen[[label]] <<- tibble(
    Variante = label, alpha = m$bestTune$alpha, lambda = signif(m$bestTune$lambda, 3),
    Accuracy = mean(oof$obs == oof$pred), ROC = best$ROC,
    Precision = mm$precision, Cobertura = mm$recall, F1 = mm$f1, Umbral = 0.5,
    Pobres_pred_test = mean(sub$sample$pobre),
    Archivo = basename(sub$path))
  message("    listo en ", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1),
          " min -> ", basename(sub$path))
  invisible(m)
}

# A. Réplica exacta del ejemplo
m_A <- run_variant("replica", db_train)

# B. Entrena con Bogotá y valida sin Bogotá: pliegues estratificados sobre los
#    hogares fuera de Bogotá; los de Bogotá se agregan siempre al entrenamiento.
set.seed(2026)
idx_nb   <- which(!bog)
f_nb     <- createFolds(db_train$Pobre[idx_nb], k = 5, returnTrain = TRUE)
folds_B  <- lapply(f_nb, function(i) c(idx_nb[i], which(bog)))
out_B    <- lapply(f_nb, function(i) setdiff(idx_nb, idx_nb[i]))
m_B <- run_variant("eval_sin_bogota", db_train, folds = folds_B, folds_out = out_B,
                   suffix = "_eval_sin_bogota")

# C. Excluye Bogotá de train
db_train_C <- db_train[!bog, ] |> mutate(cat_ubicacion = droplevels(cat_ubicacion))
m_C <- run_variant("sin_bogota", db_train_C, suffix = "_sin_bogota")

# --- 5. Resumen ----------------------------------------------------------------------
tab <- bind_rows(resumen)
print(as.data.frame(tab), digits = 4)
print(xtable(select(tab, -Archivo), digits = 4,
             caption = "Elastic Net de la complementaria: variantes de Bogot\\'a (validaci\\'on cruzada de 5 particiones, umbral 0,5)",
             label = paste0("tab:", AUTOR, "_elastic_net_cv_results")),
      file = file.path(DIR_TAB, "harold_elastic_net_cv_results.tex"), include.rownames = FALSE,
      booktabs = TRUE)
