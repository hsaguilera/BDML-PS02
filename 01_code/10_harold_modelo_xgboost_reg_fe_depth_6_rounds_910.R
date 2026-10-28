# ==============================================================================
# 10_harold_modelo_xgboost_reg_fe_depth_6_rounds_910.R
# Modelo 4 (externo a la clase): XGBoost de REGRESIÓN con umbral ("regress-then-threshold").
# Predice z = log(ingreso per cápita / línea de pobreza) (el ingreso solo existe en train) y
# clasifica pobre si z_hat < tau; tau maximiza F1 en OOF. Variables: A + C + D (sin el grupo B,
# que en la ablación no aportó a los modelos de árboles). Entrenamiento en Python
# (01_code/python/harold_modelo_xgboost_reg_fe_depth_6_rounds_910.py); este script evalúa y escribe el envío.
# Entradas : 00_data/processed/hogares_fe.rds, hogares_fe_grupos.rds
# Salidas  : predictions/harold_xgboost_reg_fe_*.csv, tables/harold_xgboost_reg_fe_{search,cv}.{csv,tex},
#            figures/harold_xgboost_reg_fe_*.png
# ==============================================================================
AUTOR <- "harold"
suppressPackageStartupMessages({library(data.table); library(tidyverse); library(xtable)})
PY_BIN <- Sys.getenv("PYTHON_BIN", "python3")

fe     <- readRDS(file.path(DIR_PROC, "hogares_fe.rds"))
groups <- readRDS(file.path(DIR_PROC, "hogares_fe_grupos.rds"))
f_in   <- file.path(DIR_PROC, "xgb_in.csv")
f_sc   <- file.path(DIR_PROC, paste0(AUTOR, "_xgboost_reg_fe_scores.csv"))
f_se   <- file.path(DIR_TAB,  paste0(AUTOR, "_xgboost_reg_fe_search.csv"))
f_pa   <- file.path(DIR_PROC, paste0(AUTOR, "_xgboost_reg_fe_params.json"))

fun_export_fe_csv(fe, groups, f_in, grupos = c("A", "C", "D"))
status <- system2(PY_BIN, c("-I", "01_code/python/harold_modelo_xgboost_reg_fe_depth_6_rounds_910.py", f_in, f_sc, f_se, f_pa))
stopifnot(status == 0)
file.remove(f_in, sub("\\.csv$", "_vars.csv", f_in))

sc  <- fread(f_sc) |> as_tibble()
trn <- fe |> filter(sample == "train")
oof <- sc$score[match(trn$id, sc$id)]                    # s = -z_hat
ev  <- fun_eval_oof(trn$poor, oof, trn$fold, trn$bogota, thr = seq(-0.8, 0.8, by = 0.01))
print(ev$resumen)
prm <- jsonlite::fromJSON(f_pa)

te <- sc |> filter(sample == "test")
nm <- sprintf("%s_xgboost_reg_fe_depth_%d_rounds_%d.csv", AUTOR, prm$params$max_depth, prm$rondas)
fun_write_submission_fe(te$id, as.integer(te$score >= ev$resumen$umbral), nm)

write_csv(bind_cols(ev$resumen, tibble(rondas = prm$rondas, learning_rate = prm$learning_rate)),
          file.path(DIR_TAB, paste0(AUTOR, "_xgboost_reg_fe_cv.csv")))
srch <- read_csv(f_se, show_col_types = FALSE)
print(xtable(srch |> select(config, max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda, objective, rondas_media, f1, auc),
             digits = c(0, 0, 0, 0, 2, 2, 1, 0, 0, 4, 4), caption = "XGBoost (regresión): búsqueda aleatoria de hiperparámetros (5 pliegues, sin Bogotá)"),
      include.rownames = FALSE, file = file.path(DIR_TAB, paste0(AUTOR, "_xgboost_reg_fe_search.tex")))

p <- ggplot(ev$curva, aes(-thr, f1)) + geom_line(colour = "#d95f0e", linewidth = 1) +
  geom_vline(xintercept = -ev$resumen$umbral, linetype = 2) +
  labs(x = expression(tau ~ "(clasifica pobre si " * hat(z) < tau * ")"), y = "F1 (clase pobre)",
       title = "XGBoost de regresión: F1 según el umbral sobre log(ingreso/línea)",
       subtitle = sprintf("tau óptimo = %.2f (hogar pobre si z_hat < tau), F1 = %.3f", -ev$resumen$umbral, ev$resumen$f1)) +
  theme_pobreza()
save_fig(p, paste0(AUTOR, "_xgboost_reg_fe_threshold.png"), w = 7, h = 4.5)
saveRDS(list(eval = ev, params = prm), file.path(DIR_PROC, paste0(AUTOR, "_xgboost_reg_fe_resumen.rds")))
