# ==============================================================================
# 09_harold_modelo_lightgbm_fe_leaves_15_rounds_954.R
# Modelo 3 (externo a la clase): LightGBM -- boosting de árboles con histogramas.
# Usa TODAS las variables (grupos A+B+C+D). Imbalance: `scale_pos_weight` hace parte de la
# búsqueda de hiperparámetros y el umbral de decisión se elige para maximizar F1.
# El entrenamiento se hace en Python (01_code/python/harold_modelo_lightgbm_fe_leaves_15_rounds_954.py); este script arma la
# entrada, evalúa con las mismas funciones que los modelos en R y escribe el envío.
# Entradas : 00_data/processed/hogares_fe.rds, hogares_fe_grupos.rds
# Salidas  : predictions/harold_lightgbm_fe_*.csv, tables/harold_lightgbm_fe_{search,cv}.{csv,tex},
#            figures/harold_lightgbm_fe_*.png
# ==============================================================================
if (!exists("DIR_PROC")) source(file.path(if (requireNamespace("here", quietly = TRUE)) here::here() else getwd(), "01_code", "00_setup.R"), encoding = "UTF-8")   # corre solo: carga el entorno común
AUTOR <- "harold"
suppressPackageStartupMessages({library(data.table); library(tidyverse); library(xtable)})
PY_BIN <- Sys.getenv("PYTHON_BIN", "python3")

fe     <- readRDS(file.path(DIR_PROC, "hogares_fe.rds"))
groups <- readRDS(file.path(DIR_PROC, "hogares_fe_grupos.rds"))
f_in   <- file.path(DIR_PROC, "lgbm_in.csv")
f_sc   <- file.path(DIR_PROC, paste0(AUTOR, "_lightgbm_fe_scores.csv"))
f_se   <- file.path(DIR_TAB,  paste0(AUTOR, "_lightgbm_fe_search.csv"))
f_pa   <- file.path(DIR_PROC, paste0(AUTOR, "_lightgbm_fe_params.json"))

fun_export_fe_csv(fe, groups, f_in)
status <- system2(PY_BIN, c("-I", "01_code/python/harold_modelo_lightgbm_fe_leaves_15_rounds_954.py", f_in, f_sc, f_se, f_pa))
stopifnot(status == 0)
file.remove(f_in, sub("\\.csv$", "_vars.csv", f_in))

# --- Evaluación con las mismas funciones que los demás modelos --------------------------------------
sc  <- fread(f_sc) |> as_tibble()
trn <- fe |> filter(sample == "train")
oof <- sc$score[match(trn$id, sc$id)]
ev  <- fun_eval_oof(trn$poor, oof, trn$fold, trn$bogota)
print(ev$resumen)
prm <- jsonlite::fromJSON(f_pa)

te  <- sc |> filter(sample == "test")
nm  <- sprintf("%s_lightgbm_fe_leaves_%d_rounds_%d.csv", AUTOR, prm$params$num_leaves, prm$rondas)
fun_write_submission_fe(te$id, as.integer(te$score >= ev$resumen$umbral), nm)

write_csv(bind_cols(ev$resumen, tibble(rondas = prm$rondas, learning_rate = prm$learning_rate)),
          file.path(DIR_TAB, paste0(AUTOR, "_lightgbm_fe_cv.csv")))
srch <- read_csv(f_se, show_col_types = FALSE)
print(xtable(srch |> select(config, num_leaves, min_child_samples, feature_fraction, lambda_l2, scale_pos_weight, rondas_media, f1, auc),
             digits = c(0, 0, 0, 0, 2, 1, 1, 0, 4, 4), caption = "LightGBM: búsqueda aleatoria de hiperparámetros (5 pliegues, sin Bogotá)"),
      include.rownames = FALSE, file = file.path(DIR_TAB, paste0(AUTOR, "_lightgbm_fe_search.tex")))

p <- ggplot(ev$curva, aes(thr, f1)) + geom_line(colour = "#2c7fb8", linewidth = 1) +
  geom_vline(xintercept = ev$resumen$umbral, linetype = 2) +
  labs(x = "Umbral de probabilidad", y = "F1 (clase pobre)", title = "LightGBM: F1 según el umbral (OOF, sin Bogotá)",
       subtitle = sprintf("Umbral óptimo = %.2f, F1 = %.3f", ev$resumen$umbral, ev$resumen$f1)) + theme_pobreza()
save_fig(p, paste0(AUTOR, "_lightgbm_fe_threshold.png"), w = 7, h = 4.5)
saveRDS(list(eval = ev, params = prm), file.path(DIR_PROC, paste0(AUTOR, "_lightgbm_fe_resumen.rds")))
