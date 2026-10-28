# ==============================================================================
# 13_harold_analisis_comparacion_modelos.R
# Comparación de TODOS los modelos de Harold con el mismo protocolo de validación
# (5 pliegues compartidos, hogares fuera de Bogotá, umbral que maximiza F1 en OOF) y
# comparación de sus predicciones sobre test.
#
# No entrena nada: lee los puntajes fuera de muestra (OOF) y de test que guardó cada
# script de modelo en 00_data/processed/*_scores.csv y las predicciones finales en
# 02_outputs/predictions/.
# Entradas : 00_data/processed/hogares_fe.rds, harold_*_scores.csv, harold_robust_logit_resumen.rds,
#            02_outputs/predictions/harold_*.csv
# Salidas  : tables/harold_comparison_{metrics,agreement}.{csv,tex},
#            figures/harold_comparison_{f1,precision_recall,errors,agreement}.png
# ==============================================================================
if (!exists("DIR_PROC")) source(file.path(if (requireNamespace("here", quietly = TRUE)) here::here() else getwd(), "01_code", "00_setup.R"), encoding = "UTF-8")   # corre solo: carga el entorno común
AUTOR <- "harold"
suppressPackageStartupMessages({library(data.table); library(tidyverse); library(xtable)})

fe  <- readRDS(file.path(DIR_PROC, "hogares_fe.rds"))
trn <- fe |> filter(sample == "train")
y   <- trn$poor
nb  <- trn$bogota == 0

# --- Registro de modelos --------------------------------------------------------
# `signo` = -1 cuando el puntaje guardado es -z_hat (XGBoost de regresión: pobre si z_hat < tau).
# `thr` = NA: el umbral se recalcula con fun_eval_oof(); si no, se usa el umbral robusto de script 12.
rob <- readRDS(file.path(DIR_PROC, "harold_robust_logit_resumen.rds"))
modelos <- tribble(
  ~archivo,                  ~modelo,                          ~familia,        ~enviado, ~thr,           ~signo,
  "lpm_fe",                  "LPM (40 vars)",                  "Lineal",        FALSE,    NA_real_,       1,
  "logit_fe",                "Logit (40 vars)",                "Lineal",        FALSE,    NA_real_,       1,
  "elastic_net_fe",          "Elastic Net (alpha 0,7)",        "Lineal",        TRUE,     NA_real_,       1,
  "logit_robust_fe",         "Logit robusto (59 vars)",        "Lineal",        TRUE,     rob$thr,        1,
  "probit_robust_fe",        "Probit robusto (59 vars)",       "Lineal",        TRUE,     rob$thr_probit, 1,
  "naive_bayes_fe",          "Naive Bayes (deciles)",          "Probabilístico", FALSE,   NA_real_,       1,
  "cart_fe",                 "CART",                           "Árboles",       FALSE,    NA_real_,       1,
  "random_forest_fe",        "Random Forest",                  "Árboles",       TRUE,     NA_real_,       1,
  "lightgbm_fe",             "LightGBM",                       "Boosting",      TRUE,     NA_real_,       1,
  "xgboost_reg_fe",          "XGBoost (regresión + umbral)",   "Boosting",      TRUE,     NA_real_,      -1
)

oof_de <- function(archivo) {
  sc <- fread(file.path(DIR_PROC, paste0(AUTOR, "_", archivo, "_scores.csv"))) |> as_tibble()
  sc$score[match(trn$id, sc$id)]
}

# --- Métricas por modelo ----------------------------------------------------------
res <- vector("list", nrow(modelos)); pr <- vector("list", nrow(modelos))
for (i in seq_len(nrow(modelos))) {
  m <- modelos[i, ]
  s <- oof_de(m$archivo)                                     # puntaje: mayor = más probable pobre
  grilla <- if (m$signo == 1) seq(0.10, 0.80, by = 0.01) else seq(-0.8, 0.8, by = 0.01)
  if (m$archivo == "naive_bayes_fe") grilla <- seq(0.10, 0.998, by = 0.002)
  ev  <- fun_eval_oof(y, s, trn$fold, trn$bogota, thr = grilla)
  thr <- if (is.na(m$thr)) ev$resumen$umbral else m$thr
  yhat <- as.integer(s >= thr)
  cm  <- conf_metrics(y[nb], yhat[nb])
  f1k <- vapply(1:5, function(k) f1_score(y[nb & trn$fold == k], yhat[nb & trn$fold == k]), numeric(1))
  res[[i]] <- tibble(modelo = m$modelo, familia = m$familia, enviado_kaggle = m$enviado,
                     f1 = cm$f1, f1_sd_pliegues = sd(f1k), precision = cm$precision, recall = cm$recall,
                     filtracion = cm$leakage, subcobertura = cm$undercoverage,
                     auc = fun_auc(y[nb], s[nb]), pct_pred_pobre = mean(yhat[nb]))
  # Curva precisión-recall (un punto cada 1 % de hogares ordenados por puntaje)
  o  <- order(-s[nb]); yo <- y[nb][o]; k <- round(seq(0.01, 1, by = 0.01) * length(yo))
  tp <- cumsum(yo)[k]
  pr[[i]] <- tibble(modelo = m$modelo, recall = tp / sum(yo), precision = tp / k)
}
met <- bind_rows(res) |> arrange(desc(f1))
write_csv(met, file.path(DIR_TAB, paste0(AUTOR, "_comparison_metrics.csv")))
print(met, n = Inf, width = Inf)

tab <- met |> transmute(Modelo = modelo, `F1` = f1, `DE pliegues` = f1_sd_pliegues, Precisión = precision,
                        Cobertura = recall, `Filtración` = filtracion, Subcobertura = subcobertura, AUC = auc)
print(xtable(tab, digits = c(0, 0, 4, 4, 3, 3, 3, 3, 4),
             caption = "Comparación de modelos (OOF, 5 pliegues, sin Bogotá; umbral que maximiza F1)"),
      include.rownames = FALSE, file = file.path(DIR_TAB, paste0(AUTOR, "_comparison_metrics.tex")))

# --- Figura 1: F1 con el desvío entre pliegues ------------------------------------------------
orden <- rev(met$modelo)
p1 <- met |> mutate(modelo = factor(modelo, levels = orden)) |>
  ggplot(aes(f1, modelo, colour = familia)) +
  geom_errorbarh(aes(xmin = f1 - f1_sd_pliegues, xmax = f1 + f1_sd_pliegues), height = 0.25, linewidth = 0.6) +
  geom_point(aes(shape = enviado_kaggle), size = 3) +
  scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 1), labels = c(`TRUE` = "Sí", `FALSE` = "No"), name = "Enviado a Kaggle") +
  labs(x = "F1 de la clase pobre (media y ± 1 DE entre pliegues)", y = NULL, colour = "Familia",
       title = "Los ocho mejores modelos quedan a menos de un punto de F1 entre sí",
       subtitle = "Validación cruzada de 5 pliegues, hogares fuera de Bogotá") + theme_pobreza() +
  theme(legend.box = "vertical", legend.margin = margin(0, 0, 0, 0))
save_fig(p1, paste0(AUTOR, "_comparison_f1.png"), w = 8.5, h = 5.5)

# --- Figura 2: curvas precisión-cobertura ----------------------------------------------------------
p2 <- bind_rows(pr) |> mutate(modelo = factor(modelo, levels = met$modelo)) |>
  filter(modelo %in% c("LightGBM", "Random Forest", "Elastic Net (alpha 0,7)", "Logit robusto (59 vars)", "CART", "Naive Bayes (deciles)")) |>
  ggplot(aes(recall, precision, colour = modelo)) + geom_line(linewidth = 0.9) +
  coord_cartesian(xlim = c(0.4, 1), ylim = c(0.4, 1)) +
  labs(x = "Cobertura (recall)", y = "Precisión", colour = NULL,
       title = "Curvas precisión-cobertura (OOF)", subtitle = "Los modelos de pocas variables y los de boosting casi coinciden") +
  theme_pobreza() + guides(colour = guide_legend(nrow = 2))
save_fig(p2, paste0(AUTOR, "_comparison_precision_recall.png"), w = 7.5, h = 5)

# --- Figura 3: filtración vs subcobertura (errores de política) ------------------------------------
p3 <- met |> ggplot(aes(subcobertura, filtracion, colour = familia, label = modelo)) +
  geom_point(size = 3) + ggrepel::geom_text_repel(size = 3, show.legend = FALSE, max.overlaps = 20) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1)) + scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(x = "Subcobertura: % de pobres excluidos", y = "Filtración: % de beneficiarios no pobres", colour = "Familia",
       title = "Errores de política de cada clasificador", subtitle = "Umbral que maximiza F1; OOF sin Bogotá") + theme_pobreza()
save_fig(p3, paste0(AUTOR, "_comparison_errors.png"), w = 8, h = 5.5)

# --- Acuerdo entre las predicciones sobre test ----------------------------------------------------
archivos <- list.files(DIR_PRED, pattern = paste0("^", AUTOR, "_.*\\.csv$"), full.names = TRUE)
P <- map(archivos, \(f) fread(f) |> as_tibble() |> arrange(id) |> pull(pobre)) |>
  set_names(sub("\\.csv$", "", sub(paste0("^", AUTOR, "_"), "", basename(archivos))))
agr <- crossing(a = names(P), b = names(P)) |>
  mutate(acuerdo = map2_dbl(a, b, \(x, z) mean(P[[x]] == P[[z]])),
         jaccard = map2_dbl(a, b, \(x, z) sum(P[[x]] == 1 & P[[z]] == 1) / sum(P[[x]] == 1 | P[[z]] == 1)))
write_csv(agr, file.path(DIR_TAB, paste0(AUTOR, "_comparison_agreement.csv")))
resumen_pred <- tibble(prediccion = names(P), pct_pobre = map_dbl(P, mean))
write_csv(resumen_pred, file.path(DIR_TAB, paste0(AUTOR, "_comparison_test_share.csv")))

p4 <- agr |> mutate(across(c(a, b), \(v) factor(v, levels = sort(unique(v))))) |>
  ggplot(aes(a, b, fill = jaccard)) + geom_tile(colour = "white") +
  geom_text(aes(label = sprintf("%.2f", jaccard)), size = 2.4) +
  scale_fill_gradient(low = "#fee8c8", high = "#e34a33", limits = c(0.6, 1), oob = scales::squish) +
  labs(x = NULL, y = NULL, fill = "Jaccard", title = "Acuerdo entre las predicciones de test",
       subtitle = "Índice de Jaccard sobre los hogares clasificados como pobres") +
  theme_pobreza() + theme(axis.text.x = element_text(angle = 60, hjust = 1))
save_fig(p4, paste0(AUTOR, "_comparison_agreement.png"), w = 10, h = 8)
