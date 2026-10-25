# ==============================================================================
# 05_harold_feature_engineering.R
# Objetivo: construir la base de modelado con ingeniería de variables
#           (00_data/processed/hogares_fe.rds) en tres pasos:
#   1. variables de estructura de la regla de pobreza, vivienda, composición,
#      capital humano, inserción laboral, privaciones e interacciones;
#   2. ingreso imputado fuera de muestra: un modelo de LightGBM a nivel de persona
#      (Python) predice el ingreso individual con las variables que SÍ existen en
#      test; se agrega al hogar;
#   3. pliegues estratificados compartidos (`fold`) por todos los modelos.
# Entradas : 00_data/processed/hogares_modelo.rds, 00_data/raw/*_personas.csv
# Salidas  : 00_data/processed/hogares_fe.rds, personas_ingreso_{in,out}.csv,
#            02_outputs/tables/harold_person_income_quality.tex
# ==============================================================================
AUTOR <- "harold"
suppressPackageStartupMessages({library(data.table); library(tidyverse); library(xtable)})

PY_BIN   <- Sys.getenv("PYTHON_BIN", "python3")
RERUN_PY <- Sys.getenv("RERUN_PY", "0") == "1"      # 1 = reentrenar el modelo de persona
f_in  <- file.path(DIR_PROC, "personas_ingreso_in.csv")
f_out <- file.path(DIR_PROC, "personas_ingreso_out.csv")

# --- 1. Hogares + pliegues compartidos --------------------------------------------
hog <- readRDS(file.path(DIR_PROC, "hogares_modelo.rds"))
hog$fold <- 0L
tr <- hog$sample == "train"
hog$fold[tr] <- make_folds(hog$poor[tr], k = N_FOLDS, seed = SEED)   # estratificados por pobreza
hog$bogota <- as.integer(hog$Dominio == "BOGOTA")
stopifnot(all(hog$fold[tr] %in% 1:N_FOLDS), all(hog$fold[!tr] == 0))

# --- 2. Tabla de personas para el modelo auxiliar de ingreso -----------------------
if (RERUN_PY || !file.exists(f_out)) {
  cols_test <- names(fread(file.path(DIR_RAW, "test_personas.csv"), nrows = 1))
  tp <- fread(file.path(DIR_RAW, "train_personas.csv"), select = c(cols_test, "Ingtot"))
  ep <- fread(file.path(DIR_RAW, "test_personas.csv"))
  per <- bind_rows(mutate(as_tibble(tp), sample = "train"), mutate(as_tibble(ep), sample = "test"))
  tab <- fun_person_income_table(per, hog)
  fwrite(tab, f_in)
  message("Personas de la UG: ", nrow(tab))
  rm(tp, ep, per, tab); invisible(gc())

  status <- system2(PY_BIN, c("-I", "01_code/python/harold_person_income.py", f_in, f_out))
  stopifnot(status == 0)
}
pred <- fread(f_out) |> as_tibble()

# --- 3. Variables de ingeniería ----------------------------------------------------
fe <- hog |>
  fun_add_engineered() |>
  fun_add_income_hat(pred)

fe_cols <- grep("^fe_", names(fe), value = TRUE)
# Infinitos/NaN por divisiones: se reemplazan por 0 (denominadores protegidos con pmax)
for (j in fe_cols) fe[[j]][!is.finite(fe[[j]])] <- 0
stopifnot(!anyNA(fe[fe_cols]))
saveRDS(fe, file.path(DIR_PROC, "hogares_fe.rds"))
message("hogares_fe.rds: ", nrow(fe), " hogares, ", length(fe_cols), " variables fe_")

# --- 4. Calidad del ingreso imputado (fuera de muestra, solo train) ------------------
trn <- fe |> filter(sample == "train")
q <- tibble(
  `Corr. con Ingtotug (nivel)`     = cor(trn$fe_inc_hat_sum, trn$eda_Ingtotug),
  `Corr. con Ingtotug (log1p)`     = cor(log1p(trn$fe_inc_hat_sum), log1p(trn$eda_Ingtotug)),
  `Corr. razón vs. razón verdadera` = cor(log(pmax(trn$fe_ratio_hat, 0.02)),
                                           log(pmax(trn$eda_Ingtotugarr / trn$req_income, 0.02))),
  `AUC solo con la razón imputada` = as.numeric(pROC::auc(trn$poor, -trn$fe_ratio_hat, quiet = TRUE)),
  `AUC solo con hacinamiento`     = as.numeric(pROC::auc(trn$poor, trn$crowding, quiet = TRUE)))
print(t(q))
print(xtable(t(q), digits = 3, caption = "Calidad del ingreso imputado fuera de muestra (train)"),
      file = file.path(DIR_TAB, paste0(AUTOR, "_person_income_quality.tex")))
