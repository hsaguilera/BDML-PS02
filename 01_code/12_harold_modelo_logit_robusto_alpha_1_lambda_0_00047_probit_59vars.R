# ==============================================================================
# 12_harold_modelo_logit_robusto_alpha_1_lambda_0_00047_probit_59vars.R
# Logit y probit REGULARIZADOS y diseñados para generalizar (no para ganar en train):
#   1. Se descartan variables casi duplicadas (|correlación| >= 0,95) conservando la que más
#      se asocia con la pobreza (AUC univariado no lineal del script 06).
#   2. Selección por estabilidad (Meinshausen y Bühlmann, 2010): 40 submuestras de la mitad de
#      train, lasso en cada una; solo se conservan las variables elegidas en >= 60 % de ellas.
#   3. Elastic Net logit con alpha y lambda elegidos por validación cruzada REPETIDA (2 x 5
#      pliegues) y regla de una desviación estándar (se prefiere el modelo más simple que no es
#      peor que el mejor dentro del error estándar).
#   4. Probit (y logit sin penalización) sobre las mismas variables estables.
#   5. Pruebas de estrés de generalización: brecha dentro vs. fuera de muestra, validación
#      agrupada por dominio (regiones no vistas), calibración y ruido de muestreo de un
#      subconjunto del tamaño de la tabla pública de Kaggle. Se compara con LightGBM.
# Entradas : hogares_fe.rds, hogares_fe_grupos.rds, tables/harold_fe_univariate.csv,
#            processed/harold_lightgbm_fe_{scores.csv,params.json}
# Salidas  : predictions/harold_{logit_robust,probit_robust}_fe_*.csv, tables/harold_robust_*.{csv,tex},
#            figures/harold_robust_*.png
# ==============================================================================
if (!exists("DIR_PROC")) source(file.path(if (requireNamespace("here", quietly = TRUE)) here::here() else getwd(), "01_code", "00_setup.R"), encoding = "UTF-8")   # corre solo: carga el entorno común
AUTOR <- "harold"
suppressPackageStartupMessages({library(glmnet); library(tidyverse); library(xtable); library(splines); library(data.table)})
PY_BIN <- Sys.getenv("PYTHON_BIN", "python3")

fe     <- readRDS(file.path(DIR_PROC, "hogares_fe.rds"))
groups <- readRDS(file.path(DIR_PROC, "hogares_fe_grupos.rds"))
uni    <- read_csv(file.path(DIR_TAB, paste0(AUTOR, "_fe_univariate.csv")), show_col_types = FALSE)
trn <- fe |> filter(sample == "train"); tst <- fe |> filter(sample == "test")
y <- trn$poor; fold <- trn$fold; bog <- trn$bogota; nb <- bog == 0
test_ids <- tst$id

# --- 1. Variables candidatas y eliminación de duplicados ------------------------------------------------
excl <- c("fe_gap_hat", "fe_gap_hat_cond", "fe_log_ratio_hat", "Dominio", "tenure", "rent_missing",
          "log_rent_pc", "log_req_income", "fe_log_req_labor")
pool <- setdiff(groups$variable, excl)
M0   <- fun_matrix_en(fe, pool)                                   # log1p + winsorización + estandarización
prio <- uni$auc_binned[match(pool, uni$variable)]; names(prio) <- pool
R    <- abs(cor(M0$X_train[, pool]))
keep <- character()
for (v in pool[order(-prio)]) if (length(keep) == 0 || max(R[v, keep]) < 0.95) keep <- c(keep, v)
message("Variables candidatas: ", length(pool), " -> sin duplicados (|r| < 0,95): ", length(keep))
dupl <- tibble(variable = setdiff(pool, keep),
               parecida_a = vapply(setdiff(pool, keep), function(v) keep[which.max(R[v, keep])], character(1)))
write_csv(dupl, file.path(DIR_TAB, paste0(AUTOR, "_robust_dropped_duplicates.csv")))

# Splines (4 g.l.) solo en 6 variables continuas con relación claramente no lineal y de distinta naturaleza
spl  <- intersect(c("fe_ratio_hat_cond", "fe_inc_hat_pc", "fe_arr_share", "crowding", "head_age", "rent_burden"), keep)
M    <- fun_matrix_en(fe, keep, spline_vars = spl, df = 4)
Xtr  <- M$X_train; Xte <- M$X_test
base_of <- function(cn) sub("_ns[0-9]+$", "", sub("^(Dominio|tenure)_.*$", "\\1", cn))
message("Columnas de diseño: ", ncol(Xtr), " (splines en: ", paste(spl, collapse = ", "), ")")

# --- 2. Selección por estabilidad --------------------------------------------------------------------------
f_cache <- file.path(DIR_PROC, paste0(AUTOR, "_robust_stability_cache.rds"))   # permite reanudar si el proceso se interrumpe
if (file.exists(f_cache)) { sel <- readRDS(f_cache) } else {
cv0 <- cv.glmnet(Xtr, y, family = "binomial", alpha = 1, nfolds = N_FOLDS, foldid = fold, type.measure = "deviance", standardize = FALSE)
  lam_stab <- cv0$lambda.1se
  B <- 40; set.seed(SEED)
  sel <- matrix(0, B, ncol(Xtr), dimnames = list(NULL, colnames(Xtr)))
  for (b in seq_len(B)) {
    idx <- unlist(lapply(0:1, function(cl) sample(which(y == cl), floor(sum(y == cl) / 2))))
    g <- glmnet(Xtr[idx, ], y[idx], family = "binomial", alpha = 1, lambda = lam_stab, standardize = FALSE)
    sel[b, ] <- as.numeric(as.matrix(coef(g))[-1, 1] != 0)
  }
  saveRDS(sel, f_cache)
}
freq_col <- colMeans(sel)
freq_var <- tapply(freq_col, base_of(names(freq_col)), max)
estables <- names(freq_var)[freq_var >= 0.6]
cols_est <- names(freq_col)[base_of(names(freq_col)) %in% estables]
message("Variables estables (>= 60 % de las submuestras): ", length(estables), " de ", length(freq_var), " | columnas: ", length(cols_est))
write_csv(tibble(variable = names(freq_var), frecuencia = as.numeric(freq_var), estable = as.vector(freq_var >= 0.6)) |> arrange(desc(frecuencia)),
          file.path(DIR_TAB, paste0(AUTOR, "_robust_stability.csv")))
Xs <- Xtr[, cols_est]; Xs_te <- Xte[, cols_est]

# --- 3. Elastic Net con validación repetida y regla de 1 error estándar -----------------------------------------
alphas <- c(0.1, 0.5, 1); reps <- list(fold, make_folds(y, k = N_FOLDS, seed = SEED + 1))
dev_fun <- function(yy, p) -2 * mean(ifelse(yy == 1, log(pmax(p, 1e-12)), log(pmax(1 - p, 1e-12))))
res <- list(); lam_path <- list()
for (a in alphas) {
  lam <- glmnet(Xs, y, family = "binomial", alpha = a, nlambda = 50, lambda.min.ratio = 1e-4, standardize = FALSE)$lambda
  lam_path[[as.character(a)]] <- lam
  D <- NULL
  for (r in seq_along(reps)) for (k in 1:N_FOLDS) {
    ik <- reps[[r]] == k
    g  <- glmnet(Xs[!ik, ], y[!ik], family = "binomial", alpha = a, lambda = lam, standardize = FALSE)
    P  <- predict(g, Xs[ik & nb, ], type = "response")
    D  <- rbind(D, apply(P, 2, function(p) dev_fun(y[ik & nb], p)))
  }
  m <- colMeans(D); se <- apply(D, 2, sd) / sqrt(nrow(D)); j0 <- which.min(m)
  j1 <- min(which(m <= m[j0] + se[j0]))                    # lambda más grande dentro de 1 error estándar
  res[[as.character(a)]] <- tibble(alpha = a, lambda_min = lam[j0], lambda_1se = lam[j1], dev_min = m[j0], dev_1se = m[j1], se = se[j0])
  message(sprintf("alpha=%.1f | lambda_min=%.5f lambda_1se=%.5f | devianza %.4f -> %.4f", a, lam[j0], lam[j1], m[j0], m[j1]))
}
cvt <- bind_rows(res); write_csv(cvt, file.path(DIR_TAB, paste0(AUTOR, "_robust_logit_cv.csv")))
ib <- which.min(cvt$dev_min); A <- cvt$alpha[ib]; LAM <- cvt$lambda_1se[ib]
message("Elegido: alpha = ", A, ", lambda (1 SE) = ", signif(LAM, 3))

oof_pen <- numeric(nrow(Xs))
for (k in 1:N_FOLDS) { ik <- fold == k
  g <- glmnet(Xs[!ik, ], y[!ik], family = "binomial", alpha = A, lambda = LAM, standardize = FALSE)
  oof_pen[ik] <- as.numeric(predict(g, Xs[ik, ], type = "response")) }
fit_pen <- glmnet(Xs, y, family = "binomial", alpha = A, lambda = LAM, standardize = FALSE)
p_te_pen <- as.numeric(predict(fit_pen, Xs_te, type = "response"))
p_in_pen <- as.numeric(predict(fit_pen, Xs, type = "response"))

# Umbral robusto: centro de la meseta donde el F1 queda a menos de 0,001 del máximo
thr_robusto <- function(yy, p) { cv <- fun_f1_curve(yy, p, seq(0.10, 0.80, by = 0.005)); ok <- cv$thr[cv$f1 >= max(cv$f1) - 0.001]; list(thr = median(ok), curva = cv, rango = range(ok)) }
tr_pen <- thr_robusto(y[nb], oof_pen[nb])
ev_pen <- fun_eval_oof(y, oof_pen, fold, bog)
f1_a <- function(p, t) f1_score(y[nb], as.integer(p[nb] >= t))
message(sprintf("Logit penalizado: F1 OOF = %.4f (umbral %.3f; meseta %.3f-%.3f)", f1_a(oof_pen, tr_pen$thr), tr_pen$thr, tr_pen$rango[1], tr_pen$rango[2]))
nm1 <- sprintf("%s_logit_robust_fe_alpha_%s_lambda_%s.csv", AUTOR, gsub("\\.", "_", A), gsub("\\.", "_", format(round(LAM, 5), scientific = FALSE)))
fun_write_submission_fe(test_ids, as.integer(p_te_pen >= tr_pen$thr), nm1)
fun_save_scores(fe, oof_pen, p_te_pen, "logit_robust_fe")

# --- 4. Probit y logit sin penalización sobre las mismas variables ----------------------------------------------------
glm_oof <- function(link) {
  oo <- numeric(nrow(Xs)); X1 <- cbind(1, Xs)
  ajustar <- function(X, yy) { f <- suppressWarnings(glm.fit(X, yy, family = binomial(link = link), control = list(maxit = 100))); b <- f$coefficients; b[is.na(b)] <- 0; b }
  inv <- if (link == "probit") pnorm else plogis
  for (k in 1:N_FOLDS) { ik <- fold == k; oo[ik] <- inv(as.numeric(X1[ik, ] %*% ajustar(X1[!ik, ], y[!ik]))) }
  b <- ajustar(X1, y)
  list(oof = oo, test = inv(as.numeric(cbind(1, Xs_te) %*% b)), insample = inv(as.numeric(X1 %*% b)))
}
pro <- glm_oof("probit"); lgt <- glm_oof("logit")
tr_pro <- thr_robusto(y[nb], pro$oof[nb])
nm2 <- sprintf("%s_probit_robust_fe_%dvars.csv", AUTOR, length(estables))
fun_write_submission_fe(test_ids, as.integer(pro$test >= tr_pro$thr), nm2)
fun_save_scores(fe, pro$oof, pro$test, "probit_robust_fe")

# --- 5. Pruebas de estrés ---------------------------------------------------------------------------------------------------
# (a) Brecha dentro vs. fuera de muestra
brecha <- tibble(
  modelo = c("Logit penalizado (1 SE)", "Logit sin penalizar", "Probit"),
  f1_dentro = c(f1_a(p_in_pen, tr_pen$thr), f1_a(lgt$insample, thr_robusto(y[nb], lgt$oof[nb])$thr), f1_a(pro$insample, tr_pro$thr)),
  f1_fuera  = c(f1_a(oof_pen, tr_pen$thr), f1_a(lgt$oof, thr_robusto(y[nb], lgt$oof[nb])$thr), f1_a(pro$oof, tr_pro$thr)),
  auc_fuera = c(fun_auc(y[nb], oof_pen[nb]), fun_auc(y[nb], lgt$oof[nb]), fun_auc(y[nb], pro$oof[nb])),
  logloss_fuera = c(dev_fun(y[nb], oof_pen[nb]) / 2, dev_fun(y[nb], lgt$oof[nb]) / 2, dev_fun(y[nb], pro$oof[nb]) / 2)) |>
  mutate(brecha_f1 = f1_dentro - f1_fuera)

# LightGBM (referencia): sus predicciones dentro de muestra se obtienen con el modelo final de Python solo para la brecha
sc_l <- fread(file.path(DIR_PROC, paste0(AUTOR, "_lightgbm_fe_scores.csv"))) |> as_tibble()
oof_l <- sc_l$score[match(trn$id, sc_l$id)]
thr_l <- fun_eval_oof(y, oof_l, fold, bog)$resumen$umbral

# (b) Validación agrupada por dominio: regiones completas fuera de muestra
dom <- trn |> count(Dominio, sort = TRUE) |> mutate(gfold = NA_integer_)
carga <- rep(0, N_FOLDS)
for (i in seq_len(nrow(dom))) { k <- which.min(carga); dom$gfold[i] <- k; carga[k] <- carga[k] + dom$n[i] }
gf <- dom$gfold[match(trn$Dominio, dom$Dominio)]
oof_g <- numeric(nrow(Xs))
for (k in 1:N_FOLDS) { ik <- gf == k
  g <- glmnet(Xs[!ik, ], y[!ik], family = "binomial", alpha = A, lambda = LAM, standardize = FALSE)
  oof_g[ik] <- as.numeric(predict(g, Xs[ik, ], type = "response")) }
groups_all <- groups
f_in <- file.path(DIR_PROC, "gcv_in.csv"); f_out <- file.path(DIR_PROC, "gcv_out.csv")
fun_export_fe_csv(fe, groups_all, f_in, extra = list(gfold = c(gf, rep(0L, nrow(tst)))[order(c(which(fe$sample == "train"), which(fe$sample == "test")))]))
st <- system2(PY_BIN, c("-I", "01_code/python/harold_analisis_cv_agrupada_lightgbm_leaves_15_rounds_954.py", f_in, file.path(DIR_PROC, paste0(AUTOR, "_lightgbm_fe_params.json")), f_out))
stopifnot(st == 0); file.remove(f_in, sub("\\.csv$", "_vars.csv", f_in))
oof_lg <- fread(f_out)$score[match(trn$id, fread(f_out)$id)]; file.remove(f_out)

agrupado <- tibble(
  modelo = c("Logit penalizado (1 SE)", "LightGBM"),
  f1_pliegues_aleatorios = c(f1_a(oof_pen, tr_pen$thr), f1_a(oof_l, thr_l)),
  f1_regiones_nuevas     = c(f1_a(oof_g, tr_pen$thr), f1_a(oof_lg, thr_l)),
  auc_regiones_nuevas    = c(fun_auc(y[nb], oof_g[nb]), fun_auc(y[nb], oof_lg[nb]))) |> mutate(caida = f1_pliegues_aleatorios - f1_regiones_nuevas)
f1_gf <- trn |> mutate(pen = oof_g, lg = oof_lg, gf = gf) |> filter(bogota == 0) |>
  group_by(gfold = gf) |> summarise(n = n(), `Logit penalizado` = f1_score(poor, as.integer(pen >= tr_pen$thr)), LightGBM = f1_score(poor, as.integer(lg >= thr_l)), .groups = "drop")

# (c) Ruido de muestreo de un subconjunto del tamaño de la tabla pública (20 % de 66.168 hogares)
set.seed(SEED); n_pub <- round(0.2 * nrow(tst)); idx_nb <- which(nb)
dif <- replicate(500, { i <- sample(idx_nb, n_pub); c(pen = f1_score(y[i], as.integer(oof_pen[i] >= tr_pen$thr)), lgb = f1_score(y[i], as.integer(oof_l[i] >= thr_l))) })
ruido <- tibble(modelo = c("Logit penalizado (1 SE)", "LightGBM"), f1_medio = rowMeans(dif), desv_est = apply(dif, 1, sd),
                p2.5 = apply(dif, 1, quantile, 0.025), p97.5 = apply(dif, 1, quantile, 0.975))
prob_lgb_gana <- mean(dif["lgb", ] > dif["pen", ])

# (d) Calibración (decil de probabilidad predicha vs. frecuencia observada)
cal <- tibble(p = oof_pen[nb], y = y[nb]) |> mutate(dec = ntile(p, 10)) |> group_by(dec) |> summarise(predicha = mean(p), observada = mean(y), .groups = "drop")
pend <- coef(glm(y[nb] ~ qlogis(pmin(pmax(oof_pen[nb], 1e-6), 1 - 1e-6)), family = binomial()))

write_csv(brecha, file.path(DIR_TAB, paste0(AUTOR, "_robust_gap.csv")))
write_csv(agrupado, file.path(DIR_TAB, paste0(AUTOR, "_robust_grouped_cv.csv")))
write_csv(f1_gf, file.path(DIR_TAB, paste0(AUTOR, "_robust_grouped_cv_by_fold.csv")))
write_csv(ruido, file.path(DIR_TAB, paste0(AUTOR, "_robust_public_noise.csv")))
write_csv(cal, file.path(DIR_TAB, paste0(AUTOR, "_robust_calibration.csv")))
print(brecha); print(agrupado); print(f1_gf); print(ruido); message("P(LightGBM > logit en un 20 % aleatorio) = ", prob_lgb_gana)
message("Calibración: intercepto = ", round(pend[1], 3), "; pendiente = ", round(pend[2], 3))
print(xtable(brecha, digits = 4, caption = "Brecha dentro vs. fuera de muestra"), include.rownames = FALSE, file = file.path(DIR_TAB, paste0(AUTOR, "_robust_gap.tex")))

# --- Figuras --------------------------------------------------------------------------------------------------------------------
p1 <- tibble(variable = names(freq_var), frecuencia = as.numeric(freq_var)) |> slice_max(frecuencia, n = 30, with_ties = FALSE) |>
  mutate(variable = fct_reorder(variable, frecuencia), estable = frecuencia >= 0.6) |>
  ggplot(aes(frecuencia, variable, fill = estable)) + geom_col() + geom_vline(xintercept = 0.6, linetype = 2) +
  scale_fill_manual(values = c("grey70", "#2c7fb8"), labels = c("No", "Sí"), name = "Estable") +
  labs(x = "Frecuencia de selección en 40 submuestras (lasso)", y = NULL, title = "Selección por estabilidad", subtitle = "Variables elegidas en al menos el 60 % de las submuestras") + theme_pobreza()
save_fig(p1, paste0(AUTOR, "_robust_stability.png"), w = 8, h = 7)
p2 <- ggplot(cal, aes(predicha, observada)) + geom_abline(linetype = 2) + geom_line(colour = "#2c7fb8") + geom_point(size = 2.5, colour = "#2c7fb8") +
  labs(x = "Probabilidad predicha (promedio del decil)", y = "Frecuencia observada de pobreza", title = "Calibración fuera de muestra del logit penalizado", subtitle = sprintf("Pendiente de calibración = %.2f", pend[2])) + theme_pobreza()
save_fig(p2, paste0(AUTOR, "_robust_calibration.png"), w = 6, h = 5)
p3 <- tr_pen$curva |> ggplot(aes(thr, f1)) + annotate("rect", xmin = tr_pen$rango[1], xmax = tr_pen$rango[2], ymin = -Inf, ymax = Inf, alpha = 0.2, fill = "#2c7fb8") +
  geom_line(linewidth = 1) + geom_vline(xintercept = tr_pen$thr, linetype = 2) + coord_cartesian(ylim = c(0.6, 0.8)) +
  labs(x = "Umbral de probabilidad", y = "F1 fuera de muestra", title = "F1 según el umbral: meseta y umbral elegido", subtitle = "Banda: F1 a menos de 0,001 del máximo") + theme_pobreza()
save_fig(p3, paste0(AUTOR, "_robust_threshold.png"), w = 7, h = 4.5)
p4 <- f1_gf |> pivot_longer(c(`Logit penalizado`, LightGBM), names_to = "modelo", values_to = "f1") |>
  ggplot(aes(factor(gfold), f1, fill = modelo)) + geom_col(position = "dodge") + coord_cartesian(ylim = c(0.6, 0.85)) +
  scale_fill_manual(values = c("#2c7fb8", "#d95f0e"), name = NULL) +
  labs(x = "Pliegue de regiones (dominios completos fuera de muestra)", y = "F1", title = "Validación agrupada por dominio", subtitle = "Cada pliegue deja fuera ciudades completas") + theme_pobreza()
save_fig(p4, paste0(AUTOR, "_robust_grouped_cv.png"), w = 7.5, h = 4.8)
saveRDS(list(A = A, LAM = LAM, estables = estables, thr = tr_pen$thr, thr_probit = tr_pro$thr, brecha = brecha, agrupado = agrupado, ruido = ruido, prob_lgb_gana = prob_lgb_gana, pend = pend),
        file.path(DIR_PROC, paste0(AUTOR, "_robust_logit_resumen.rds")))
