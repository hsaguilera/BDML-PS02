# ==============================================================================
# 08_harold_random_forest_fe.R
# Modelo 2 (visto en clase): Random Forest (ranger, bosque de probabilidad) con ingeniería de
# variables y SELECCIÓN de variables: las 40 con mayor |SHAP| medio del diagnóstico del script 06
# (hay muchas variables muy correlacionadas y `mtry` se diluye si se incluyen todas).
# Variaciones frente al RF visto en clase: variables compuestas, ingreso imputado, selección,
# tamaño mínimo de nodo, `mtry`, pesos de clase (manejo de desbalanceo) y umbral para F1.
# Hiperparámetros: 7 configuraciones (mtry, min.node.size, class.weights), 250 árboles en la
# búsqueda y 500 en el ajuste final; 5 pliegues compartidos; test = promedio de los 5 modelos.
# Entradas : 00_data/processed/hogares_fe.rds, tables/harold_fe_importance.csv
# Salidas  : predictions/harold_random_forest_fe_*.csv, tables/harold_random_forest_fe_{search,cv,vars}.{csv,tex},
#            figures/harold_random_forest_fe_*.png
# ==============================================================================
AUTOR <- "harold"
suppressPackageStartupMessages({library(ranger); library(tidyverse); library(xtable); library(data.table)})

fe  <- readRDS(file.path(DIR_PROC, "hogares_fe.rds"))
imp <- read_csv(file.path(DIR_TAB, paste0(AUTOR, "_fe_importance.csv")), show_col_types = FALSE)
K_VARS <- 40
vars <- imp |> slice_max(shap_abs, n = K_VARS) |> pull(variable)
vars <- c(setdiff(vars, c("Dominio", "tenure")), "Dominio_code")
fe$Dominio_code <- as.integer(factor(fe$Dominio))      # ordinal sobre el dominio; los árboles lo dividen por umbrales
trn <- fe |> filter(sample == "train"); tst <- fe |> filter(sample == "test")
y <- factor(trn$poor, levels = c(0, 1), labels = c("No", "Si"))
write_csv(imp |> filter(variable %in% vars | variable %in% c("Dominio", "tenure")) |> slice_max(shap_abs, n = K_VARS),
          file.path(DIR_TAB, paste0(AUTOR, "_random_forest_fe_vars.csv")))

cfgs <- tribble(~config, ~mtry, ~min_node, ~cw,
                1, 8, 20, 1,   2, 8, 60, 1,   3, 14, 20, 1,   4, 14, 60, 1,
                5, 14, 150, 1, 6, 22, 60, 1,  7, 14, 60, 2.5)
fit_cv <- function(cfg, num_trees, test = FALSE) {
  oof <- numeric(nrow(trn)); pte <- numeric(nrow(tst))
  for (k in 1:N_FOLDS) {
    ik <- trn$fold == k
    m <- ranger(x = trn[!ik, vars], y = y[!ik], num.trees = num_trees, mtry = cfg$mtry, min.node.size = cfg$min_node,
                probability = TRUE, class.weights = c(1, cfg$cw), num.threads = 2, seed = SEED + k, respect.unordered.factors = "order")
    oof[ik] <- predict(m, trn[ik, vars], num.threads = 2)$predictions[, "Si"]
    if (test) pte <- pte + predict(m, tst[vars], num.threads = 2)$predictions[, "Si"] / N_FOLDS
    rm(m); invisible(gc())
  }
  list(oof = oof, test = pte)
}

search <- list()
for (i in seq_len(nrow(cfgs))) {
  t0 <- Sys.time(); cfg <- cfgs[i, ]
  r <- fit_cv(cfg, 250)
  # con class.weights != 1 las probabilidades se desplazan: el umbral se busca en una grilla más amplia
  ev <- fun_eval_oof(trn$poor, r$oof, trn$fold, trn$bogota, thr = seq(0.05, 0.95, by = 0.01))
  search[[i]] <- bind_cols(cfg, ev$resumen |> select(umbral, f1, auc, logloss), minutos = as.numeric(difftime(Sys.time(), t0, units = "mins")))
  message(sprintf("config %d | mtry=%d node=%d cw=%.1f | F1=%.4f AUC=%.4f | %.1f min", i, cfg$mtry, cfg$min_node, cfg$cw,
                  ev$resumen$f1, ev$resumen$auc, search[[i]]$minutos))
}
S <- bind_rows(search); best <- S[which.max(S$f1), ]
write_csv(S, file.path(DIR_TAB, paste0(AUTOR, "_random_forest_fe_search.csv")))

fin <- fit_cv(best, 500, test = TRUE)
ev  <- fun_eval_oof(trn$poor, fin$oof, trn$fold, trn$bogota, thr = seq(0.05, 0.95, by = 0.01))
print(ev$resumen)
nm <- sprintf("%s_random_forest_fe_mtry_%d_node_%d.csv", AUTOR, best$mtry, best$min_node)
fun_write_submission_fe(tst$id, as.integer(fin$test >= ev$resumen$umbral), nm)
fun_save_scores(fe, fin$oof, fin$test, "random_forest_fe")
write_csv(bind_cols(ev$resumen, best |> select(mtry, min_node, cw)), file.path(DIR_TAB, paste0(AUTOR, "_random_forest_fe_cv.csv")))
print(xtable(S |> select(config, mtry, min_node, cw, umbral, f1, auc, minutos), digits = c(0, 0, 0, 0, 1, 2, 4, 4, 1),
             caption = "Random Forest: configuraciones evaluadas (250 árboles, 5 pliegues, sin Bogotá)"),
      include.rownames = FALSE, file = file.path(DIR_TAB, paste0(AUTOR, "_random_forest_fe_search.tex")))

# --- Importancia por permutación del ajuste final (un modelo en todo train, 300 árboles) -----------------
m_imp <- ranger(x = trn[vars], y = y, num.trees = 300, mtry = best$mtry, min.node.size = best$min_node, probability = TRUE,
                class.weights = c(1, best$cw), importance = "impurity", num.threads = 2, seed = SEED)
vi <- tibble(variable = names(m_imp$variable.importance), importancia = m_imp$variable.importance)
p <- vi |> slice_max(importancia, n = 25) |> mutate(variable = fct_reorder(variable, importancia)) |>
  ggplot(aes(importancia, variable)) + geom_col(fill = "#238b45") +
  labs(x = "Disminución media de impureza (Gini)", y = NULL, title = "Random Forest: importancia de variables",
       subtitle = sprintf("mtry = %d, min.node.size = %d", best$mtry, best$min_node)) + theme_pobreza()
save_fig(p, paste0(AUTOR, "_random_forest_fe_importance.png"), w = 8, h = 6.5)
saveRDS(list(eval = ev, best = best, search = S), file.path(DIR_PROC, paste0(AUTOR, "_random_forest_fe_resumen.rds")))
