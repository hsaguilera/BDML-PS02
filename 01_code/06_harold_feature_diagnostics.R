# ==============================================================================
# 06_harold_feature_diagnostics.R
# Objetivo: evaluar la importancia REAL de las variables antes de modelar:
#   (i) relación univariada lineal y no lineal con la pobreza;
#   (ii) importancia de un LightGBM con todas las variables (ganancia y |SHAP|);
#   (iii) ablación por grupos: A base | B ratios e interacciones | C estructura de la
#         regla de pobreza (arriendo imputado) | D ingreso imputado por persona.
# Los cálculos pesados se hacen en Python (LightGBM, SHAP por contribuciones);
# aquí se arma la entrada, se llama al script y se hacen tablas y figuras.
# Entradas : 00_data/processed/hogares_fe.rds
# Salidas  : 02_outputs/tables/harold_fe_{univariate,importance,ablation}.csv/.tex
#            02_outputs/figures/harold_fe_{univariate,importance,ablation}.png
# ==============================================================================
AUTOR <- "harold"
suppressPackageStartupMessages({library(data.table); library(tidyverse); library(xtable)})
PY_BIN <- Sys.getenv("PYTHON_BIN", "python3")

fe <- readRDS(file.path(DIR_PROC, "hogares_fe.rds"))

# --- Grupos de variables -------------------------------------------------------------
excluir <- c("id", "Clase", "Depto", "sample", "poor", "fold", "bogota", "Fex_c", "Fex_dpto",
             grep("^eda_", names(fe), value = TRUE))
base_vars <- setdiff(names(readRDS(file.path(DIR_PROC, "hogares_modelo.rds"))), excluir)
fe_vars   <- grep("^fe_", names(fe), value = TRUE)
grp_C <- c("fe_arr_imp", "fe_req_labor", "fe_arr_share", "fe_rent_covers", "fe_log_req_labor", "fe_lp_li_ratio")
grp_D <- fe_vars[grepl("^fe_(inc_hat|ratio_hat|log_ratio_hat|gap_hat|earners_hat)", fe_vars)]
grp_B <- setdiff(fe_vars, c(grp_C, grp_D))
groups <- bind_rows(tibble(variable = base_vars, grupo = "A"), tibble(variable = grp_B, grupo = "B"),
                    tibble(variable = grp_C, grupo = "C"), tibble(variable = grp_D, grupo = "D"))
stopifnot(!anyDuplicated(groups$variable), all(groups$variable %in% names(fe)))
saveRDS(groups, file.path(DIR_PROC, "hogares_fe_grupos.rds"))

f_csv <- file.path(DIR_PROC, "hogares_fe_diag.csv")
fwrite(fe[c("id", "sample", "poor", "fold", "bogota", groups$variable)], f_csv)
fwrite(groups, file.path(DIR_PROC, "hogares_fe_grupos.csv"))

# --- Python ---------------------------------------------------------------------------
status <- system2(PY_BIN, c("-I", "01_code/python/harold_feature_diagnostics.py", f_csv,
                            file.path(DIR_PROC, "hogares_fe_grupos.csv"), DIR_TAB))
stopifnot(status == 0)
file.remove(f_csv)

# --- Tablas y figuras --------------------------------------------------------------------
uni <- fread(file.path(DIR_TAB, "harold_fe_univariate.csv"))
imp <- fread(file.path(DIR_TAB, "harold_fe_importance.csv"))
abl <- fread(file.path(DIR_TAB, "harold_fe_ablation.csv"))
pal_grp <- c(A = "#7f7f7f", B = "#2c7fb8", C = "#d95f0e", D = "#238b45")
lab_grp <- c(A = "A. Base", B = "B. Ratios e interacciones", C = "C. Estructura de la regla", D = "D. Ingreso imputado")

p1 <- uni |> slice_max(auc_binned, n = 25) |>
  mutate(variable = fct_reorder(variable, auc_binned)) |>
  ggplot(aes(y = variable)) +
  geom_segment(aes(x = auc_lineal, xend = auc_binned, yend = variable), colour = "grey70") +
  geom_point(aes(x = auc_lineal, shape = "Lineal (monótono)"), size = 2.4, colour = "grey30") +
  geom_point(aes(x = auc_binned, colour = grupo, shape = "No lineal (20 cuantiles)"), size = 2.8) +
  scale_colour_manual(values = pal_grp, labels = lab_grp, name = "Grupo") +
  scale_shape_manual(values = c(1, 16), name = NULL) +
  labs(x = "AUC univariado frente a la pobreza", y = NULL,
       title = "Poder predictivo de cada variable por sí sola", subtitle = "25 variables con mayor AUC no lineal (train)") +
  theme_pobreza()
save_fig(p1, paste0(AUTOR, "_fe_univariate.png"), w = 8.5, h = 7)

p2 <- imp |> slice_max(shap_abs, n = 25) |>
  mutate(variable = fct_reorder(variable, shap_abs)) |>
  ggplot(aes(shap_abs, variable, fill = grupo)) + geom_col() +
  scale_fill_manual(values = pal_grp, labels = lab_grp, name = "Grupo") +
  labs(x = "|SHAP| medio (log-odds)", y = NULL, title = "Importancia en LightGBM con todas las variables",
       subtitle = "Pliegue 1 de validación; 20.000 hogares") +
  theme_pobreza()
save_fig(p2, paste0(AUTOR, "_fe_importance.png"), w = 8.5, h = 7)

p3 <- abl |> mutate(conjunto = fct_reorder(conjunto, f1)) |>
  ggplot(aes(f1, conjunto)) + geom_col(fill = "#2c7fb8") +
  geom_text(aes(label = sprintf("F1 = %.3f | AUC = %.3f", f1, auc)), hjust = 1.05, colour = "white", size = 3.4) +
  coord_cartesian(xlim = c(0.4, 0.8)) +
  labs(x = "F1 de la clase pobre (validación cruzada, sin Bogotá)", y = NULL,
       title = "Aporte de cada grupo de variables", subtitle = "LightGBM con hiperparámetros fijos; umbral óptimo en OOF") +
  theme_pobreza()
save_fig(p3, paste0(AUTOR, "_fe_ablation.png"), w = 8.5, h = 4.5)

print(xtable(abl |> select(conjunto, n_vars, umbral, f1, auc, pr_auc), digits = 3,
             caption = "Ablación por grupos de variables (LightGBM, validación cruzada de 5 pliegues, sin Bogotá)"),
      include.rownames = FALSE, file = file.path(DIR_TAB, paste0(AUTOR, "_fe_ablation.tex")))
print(abl)
