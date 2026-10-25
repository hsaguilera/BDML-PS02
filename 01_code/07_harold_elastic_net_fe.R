# ==============================================================================
# 07_harold_elastic_net_fe.R
# Modelo 1 (visto en clase): Elastic Net (logit penalizado) con ingeniería de variables.
# Variaciones respecto a la réplica de la complementaria (script 04):
#   * variables: base + ratios/interacciones + estructura de la regla + ingreso imputado
#   * pre-procesamiento: log1p a variables asimétricas, winsorización, splines naturales
#     en las variables más informativas, dummies de dominio y tenencia, estandarización
#   * validación: 5 pliegues estratificados compartidos; umbral que maximiza F1
# Hiperparámetros: alpha en {0,1; 0,4; 0,7; 1}; lambda por camino de glmnet (60 valores, razón mínima 1e-4).
# Entradas : 00_data/processed/hogares_fe.rds, hogares_fe_grupos.rds
# Salidas  : predictions/harold_elastic_net_fe_*.csv, tables/harold_elastic_net_fe_cv.{csv,tex},
#            figures/harold_elastic_net_fe_*.png, processed/harold_elastic_net_fe_scores.csv
# ==============================================================================
AUTOR <- "harold"
suppressPackageStartupMessages({library(glmnet); library(tidyverse); library(xtable); library(splines)})

fe     <- readRDS(file.path(DIR_PROC, "hogares_fe.rds"))
groups <- readRDS(file.path(DIR_PROC, "hogares_fe_grupos.rds"))
# Se excluyen las duplicadas lineales exactas (log y nivel del mismo cociente) para mejorar el
# condicionamiento: se conserva el log del cociente y el cociente condicional.
vars <- setdiff(groups$variable, c("fe_gap_hat", "fe_gap_hat_cond", "fe_log_ratio_hat", "Dominio", "tenure",
                                   "rent_missing", "log_rent_pc", "log_req_income", "fe_log_req_labor"))
spl  <- c("fe_log_ratio_hat_cond", "fe_ratio_hat_cond", "fe_arr_share", "crowding", "head_age", "Npersug",
          "fe_inc_hat_pc", "dep_ratio", "educ_mean", "earners_pc")
M <- fun_matrix_en(fe, vars, spline_vars = spl, df = 5)
message("Columnas en la matriz de diseño: ", ncol(M$X_train))

trn  <- fe |> filter(sample == "train")
y    <- trn$poor; fold <- trn$fold; bog <- trn$bogota
alphas <- c(0.1, 0.4, 0.7, 1)

# --- Validación cruzada: para cada alpha, camino de lambda común y OOF para todos los lambda -----------
f_cache <- file.path(DIR_PROC, paste0(AUTOR, "_elastic_net_fe_cv_cache.rds"))   # permite reanudar si el proceso se interrumpe
if (file.exists(f_cache)) { cc <- readRDS(f_cache); res <- cc$res; oofs <- cc$oofs; lam_list <- cc$lam_list } else { res <- list(); oofs <- list(); lam_list <- list() }
for (a in setdiff(alphas, as.numeric(names(res)))) {
  t0 <- Sys.time()
  g0  <- glmnet(M$X_train, y, family = "binomial", alpha = a, nlambda = 60, lambda.min.ratio = 1e-4, standardize = FALSE)
  lam <- g0$lambda
  P   <- matrix(NA_real_, nrow(M$X_train), length(lam))
  for (k in 1:N_FOLDS) {
    ik <- fold == k
    gk <- glmnet(M$X_train[!ik, ], y[!ik], family = "binomial", alpha = a, lambda = lam, standardize = FALSE)
    P[ik, ] <- predict(gk, M$X_train[ik, ], type = "response")
  }
  nb <- bog == 0
  auc <- apply(P[nb, ], 2, function(p) fun_auc(y[nb], p))
  ll  <- apply(P[nb, ], 2, function(p) -mean(ifelse(y[nb] == 1, log(pmax(p, 1e-12)), log(pmax(1 - p, 1e-12)))))
  # F1 óptimo solo para los 8 lambdas con menor log-loss (el resto no compite)
  cand <- order(ll)[1:8]
  f1c  <- vapply(cand, function(j) max(fun_f1_curve(y[nb], P[nb, j])$f1), numeric(1))
  j    <- cand[which.max(f1c)]
  res[[as.character(a)]] <- tibble(alpha = a, lambda = lam[j], f1 = max(f1c), auc = auc[j], logloss = ll[j],
                                   n_lambda = length(lam), idx = j)
  oofs[[as.character(a)]] <- P[, j]; lam_list[[as.character(a)]] <- lam
  message(sprintf("alpha=%.1f | lambda=%.5f | F1=%.4f | AUC=%.4f | %.1f min", a, lam[j], max(f1c), auc[j],
                  as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  saveRDS(list(res = res, oofs = oofs, lam_list = lam_list), f_cache)
  rm(P); invisible(gc())
}
cv <- bind_rows(res)
best <- cv[which.max(cv$f1), ]
oof  <- oofs[[as.character(best$alpha)]]
ev   <- fun_eval_oof(y, oof, fold, bog)
print(ev$resumen)

# --- Ajuste final con todo el train y predicción de test -------------------------------------------
fit  <- glmnet(M$X_train, y, family = "binomial", alpha = best$alpha, lambda = lam_list[[as.character(best$alpha)]], standardize = FALSE)
p_te <- as.numeric(predict(fit, M$X_test, s = best$lambda, type = "response"))
test_ids <- fe$id[fe$sample == "test"]
nm <- paste0(AUTOR, "_elastic_net_fe_alpha_", gsub("\\.", "_", best$alpha), "_lambda_", gsub("\\.", "_", format(round(best$lambda, 5), scientific = FALSE)), ".csv")
fun_write_submission_fe(test_ids, as.integer(p_te >= ev$resumen$umbral), nm)
fun_save_scores(fe, oof, p_te, "elastic_net_fe")

# --- Coeficientes no nulos y tablas ---------------------------------------------------------------------
cf <- coef(fit, s = best$lambda); cf <- tibble(variable = rownames(cf), coef = as.numeric(cf)) |> filter(coef != 0, variable != "(Intercept)")
message("Coeficientes no nulos: ", nrow(cf), " de ", length(M$cols))
write_csv(cf |> arrange(desc(abs(coef))), file.path(DIR_TAB, paste0(AUTOR, "_elastic_net_fe_coefs.csv")))
out <- cv |> select(alpha, lambda, f1, auc, logloss) |> mutate(elegido = alpha == best$alpha)
write_csv(bind_cols(out, ev$resumen[rep(1, nrow(out)), c("umbral")]) , file.path(DIR_TAB, paste0(AUTOR, "_elastic_net_fe_cv.csv")))
print(xtable(out |> select(alpha, lambda, f1, auc, logloss), digits = c(0, 1, 5, 4, 4, 4),
             caption = "Elastic Net con ingeniería de variables: mejor lambda por alpha (validación cruzada, sin Bogotá)"),
      include.rownames = FALSE, file = file.path(DIR_TAB, paste0(AUTOR, "_elastic_net_fe_cv.tex")))

p <- cf |> slice_max(abs(coef), n = 25) |> mutate(variable = fct_reorder(variable, coef)) |>
  ggplot(aes(coef, variable, fill = coef > 0)) + geom_col(show.legend = FALSE) +
  scale_fill_manual(values = c("#2c7fb8", "#d95f0e")) +
  labs(x = "Coeficiente estandarizado (log-odds de ser pobre)", y = NULL,
       title = "Elastic Net: 25 coeficientes más grandes", subtitle = sprintf("alpha = %.1f, lambda = %.4f", best$alpha, best$lambda)) +
  theme_pobreza()
save_fig(p, paste0(AUTOR, "_elastic_net_fe_coefs.png"), w = 8, h = 6.5)
saveRDS(list(cv = cv, best = best, eval = ev, n_cols = length(M$cols), n_nonzero = nrow(cf)),
        file.path(DIR_PROC, paste0(AUTOR, "_elastic_net_fe_resumen.rds")))
