# ==============================================================================
# 14_harold_lightgbm_importance.R
# Interpretación del mejor modelo (LightGBM): qué variables mueven la probabilidad de ser
# pobre y cómo (efecto SHAP promedio por tramos de cada variable).
# El modelo se reentrena en Python (01_code/python/harold_lightgbm_shap.py) con los
# hiperparámetros ya elegidos en el script 09; es solo para interpretar, no cambia los envíos.
# Entradas : 00_data/processed/hogares_fe.rds, hogares_fe_grupos.rds, harold_lightgbm_fe_params.json
# Salidas  : tables/harold_lightgbm_fe_importance.{csv,tex}, figures/harold_lightgbm_fe_{importance,effects}.png
# ==============================================================================
AUTOR <- "harold"
suppressPackageStartupMessages({library(data.table); library(tidyverse); library(xtable)})
PY_BIN <- Sys.getenv("PYTHON_BIN", "python3")

fe     <- readRDS(file.path(DIR_PROC, "hogares_fe.rds"))
groups <- readRDS(file.path(DIR_PROC, "hogares_fe_grupos.rds"))
f_in   <- file.path(DIR_PROC, "lgbm_shap_in.csv")
f_pa   <- file.path(DIR_PROC, paste0(AUTOR, "_lightgbm_fe_params.json"))
f_imp  <- file.path(DIR_TAB,  paste0(AUTOR, "_lightgbm_fe_importance.csv"))
f_shap <- file.path(DIR_PROC, paste0(AUTOR, "_lightgbm_fe_shap_sample.csv"))

fun_export_fe_csv(fe, groups, f_in)
status <- system2(PY_BIN, c("-I", "01_code/python/harold_lightgbm_shap.py", f_in, f_pa, f_imp, f_shap))
stopifnot(status == 0)
file.remove(f_in, sub("\\.csv$", "_vars.csv", f_in))

imp <- read_csv(f_imp, show_col_types = FALSE) |> left_join(select(groups, variable, grupo), by = "variable")
write_csv(imp, f_imp)
print(xtable(imp |> slice_head(n = 20) |> select(Variable = variable, Grupo = grupo, `SHAP medio abs.` = shap_media_abs, Ganancia = ganancia),
             digits = c(0, 0, 0, 4, 4), caption = "LightGBM: 20 variables más importantes (valor SHAP absoluto medio, escala log-odds)"),
      include.rownames = FALSE, file = file.path(DIR_TAB, paste0(AUTOR, "_lightgbm_fe_importance.tex")))

# --- Figura 1: importancia (top 20) coloreada por grupo de variables --------------------------------
nombres_grupo <- c(A = "A. Base", B = "B. Razones e interacciones", C = "C. Regla de pobreza / arriendo", D = "D. Ingreso imputado")
p1 <- imp |> slice_head(n = 20) |> mutate(variable = fct_reorder(variable, shap_media_abs), grupo = recode(grupo, !!!nombres_grupo)) |>
  ggplot(aes(shap_media_abs, variable, fill = grupo)) + geom_col() +
  labs(x = "Valor SHAP absoluto medio (log-odds)", y = NULL, fill = "Grupo de variables",
       title = "LightGBM: las variables de ingreso imputado dominan",
       subtitle = "20 variables más importantes, muestra de 15.000 hogares de train") + theme_pobreza() +
  guides(fill = guide_legend(nrow = 2))
save_fig(p1, paste0(AUTOR, "_lightgbm_fe_importance.png"), w = 8, h = 6)

# --- Figura 2: efecto SHAP medio por tramos de las 6 variables principales ------------------------------
sh <- read_csv(f_shap, show_col_types = FALSE) |> filter(!is.na(valor)) |>
  mutate(variable = factor(variable, levels = unique(variable)))
efecto <- sh |> group_by(variable) |>
  mutate(tramo = if (n_distinct(valor) > 10) cut(valor, unique(quantile(valor, seq(0, 1, 0.05))), include.lowest = TRUE) else factor(valor)) |>
  group_by(variable, tramo) |> summarise(x = median(valor), shap = mean(shap), .groups = "drop")
p2 <- ggplot(efecto, aes(x, shap)) + geom_hline(yintercept = 0, colour = "grey70") + geom_line(colour = "#2c7fb8") + geom_point(colour = "#2c7fb8", size = 1) +
  facet_wrap(~variable, scales = "free_x", ncol = 3) +
  labs(x = "Valor de la variable (mediana del tramo)", y = "Efecto SHAP medio (log-odds de ser pobre)",
       title = "LightGBM: efecto de las variables principales sobre la probabilidad de pobreza",
       subtitle = "Promedio por vigintiles; sobre 0 aumenta la probabilidad de ser pobre") + theme_pobreza()
save_fig(p2, paste0(AUTOR, "_lightgbm_fe_effects.png"), w = 10, h = 6)
