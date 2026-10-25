# ==============================================================================
# 15_harold_same_predictors.R
# Los siete algoritmos exigidos por el taller (LPM, Logit, Elastic Net, CART, Random Forest,
# Boosting y Naive Bayes) con EXACTAMENTE las mismas 40 variables (las del Random Forest, script 08),
# los mismos 5 pliegues y el mismo criterio de umbral: así las diferencias reflejan el algoritmo y no
# los predictores ("level playing field"). LPM, Logit, CART, Naive Bayes y Random Forest ya se
# entrenaron así (scripts 08 y 11); aquí se agregan Elastic Net y LightGBM con esas 40 variables.
# Los hiperparámetros del LightGBM son los del script 09 (elegidos con las 149 variables).
# Entradas : hogares_fe.rds, hogares_fe_grupos.rds, tables/harold_random_forest_fe_vars.csv, harold_*_scores.csv
# Salidas  : tables/harold_same_predictors_cv.{csv,tex}, processed/harold_{elastic_net,lightgbm}_40vars_scores.csv
# ==============================================================================
AUTOR <- "harold"
suppressPackageStartupMessages({library(data.table); library(tidyverse); library(glmnet); library(xtable)})
PY_BIN <- Sys.getenv("PYTHON_BIN", "python3")

fe     <- readRDS(file.path(DIR_PROC, "hogares_fe.rds"))
groups <- readRDS(file.path(DIR_PROC, "hogares_fe_grupos.rds"))
vars   <- read_csv(file.path(DIR_TAB, paste0(AUTOR, "_random_forest_fe_vars.csv")), show_col_types = FALSE)$variable
trn <- fe |> filter(sample == "train"); tst <- fe |> filter(sample == "test")
y <- trn$poor; fold <- trn$fold; bog <- trn$bogota

# --- Elastic Net (sin splines, igual que el Logit): alpha por validación cruzada con los pliegues compartidos -----------
M <- fun_matrix_en(fe, setdiff(vars, c("Dominio", "tenure")), spline_vars = character())
set.seed(SEED)
cvs <- lapply(c(0.2, 0.5, 1), function(a)
  cv.glmnet(M$X_train, y, family = "binomial", alpha = a, foldid = fold, nlambda = 40, lambda.min.ratio = 1e-5, keep = TRUE))
dev <- map_dbl(cvs, \(f) min(f$cvm)); ib <- which.min(dev)
fit <- cvs[[ib]]; alpha <- c(0.2, 0.5, 1)[ib]
oof <- plogis(fit$fit.preval[, which(fit$lambda == fit$lambda.min)])
p_te <- as.numeric(predict(fit, M$X_test, s = "lambda.min", type = "response"))
message(sprintf("Elastic Net 40 vars: alpha = %.1f, lambda = %.2e", alpha, fit$lambda.min))
fun_save_scores(fe, oof, p_te, "elastic_net_40vars")

# --- LightGBM con las mismas 40 variables (hiperparámetros fijos del script 09) -------------------------------------------
f_in <- file.path(DIR_PROC, "lgbm40_in.csv")
fun_export_fe_csv(fe, groups |> filter(variable %in% vars), f_in)
f_sc <- file.path(DIR_PROC, paste0(AUTOR, "_lightgbm_40vars_scores.csv"))
status <- system2(PY_BIN, c("-I", "01_code/python/harold_lightgbm_fixed.py", f_in,
                            file.path(DIR_PROC, paste0(AUTOR, "_lightgbm_fe_params.json")), f_sc))
stopifnot(status == 0); file.remove(f_in, sub("\\.csv$", "_vars.csv", f_in))

# --- Tabla: los siete algoritmos con las mismas variables ---------------------------------------------------------------
reg <- tribble(
  ~algoritmo,            ~archivo,                 ~hiper,
  "LPM",                 "lpm_fe",                 "MCO sobre 0/1",
  "Logit",               "logit_fe",               "Máxima verosimilitud",
  "Elastic Net",         "elastic_net_40vars",     sprintf("alpha = %.1f; lambda = %.1e", alpha, fit$lambda.min),
  "CART",                "cart_fe",                "cp = 0,001; minsplit = 50",
  "Random Forest",       "random_forest_fe",       "mtry = 14; nodo mínimo = 150; 500 árboles",
  "Boosting (LightGBM)", "lightgbm_40vars",        "15 hojas; 954 rondas; tasa 0,03",
  "Naive Bayes",         "naive_bayes_fe",         "Deciles; Laplace = 1")
res <- pmap_dfr(reg, function(algoritmo, archivo, hiper) {
  sc <- fread(file.path(DIR_PROC, paste0(AUTOR, "_", archivo, "_scores.csv"))) |> as_tibble()
  s  <- sc$score[match(trn$id, sc$id)]
  grilla <- if (archivo == "naive_bayes_fe") seq(0.10, 0.998, by = 0.002) else seq(0.10, 0.80, by = 0.01)
  ev <- fun_eval_oof(y, s, fold, bog, thr = grilla)$resumen
  yhat <- as.integer(s >= ev$umbral); cm <- conf_metrics(y[bog == 0], yhat[bog == 0])
  tibble(algoritmo = algoritmo, n_vars = length(vars), hiperparametros = hiper, umbral = ev$umbral, f1 = ev$f1,
         f1_sd_pliegues = ev$f1_sd_pliegues, precision = cm$precision, recall = cm$recall,
         filtracion = cm$leakage, subcobertura = cm$undercoverage, auc = ev$auc)
}) |> arrange(desc(f1))
write_csv(res, file.path(DIR_TAB, paste0(AUTOR, "_same_predictors_cv.csv")))
print(res, width = Inf)
print(xtable(res |> select(Algoritmo = algoritmo, F1 = f1, Precisión = precision, Cobertura = recall, AUC = auc),
             digits = c(0, 0, 4, 3, 3, 4), caption = "Siete algoritmos con las mismas 40 variables (5 pliegues, sin Bogotá)"),
      include.rownames = FALSE, file = file.path(DIR_TAB, paste0(AUTOR, "_same_predictors_cv.tex")))
