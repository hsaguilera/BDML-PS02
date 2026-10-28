# ==============================================================================
# 11_harold_modelo_lpm_logit_cart_cp_0_001_naive_bayes_deciles.R
# Modelos 5 a 8 (algoritmos exigidos por el taller): Modelo de Probabilidad Lineal (LPM),
# Logit, CART y Naive Bayes, con las MISMAS 40 variables del Random Forest (script 08) para
# que la comparación sea de algoritmos y no de predictores ("level playing field").
# Misma validación (5 pliegues compartidos), mismo criterio (F1 con umbral óptimo en OOF,
# sin Bogotá). Los modelos finales se reajustan con todo train; el umbral viene del OOF.
# Entradas : 00_data/processed/hogares_fe.rds, tables/harold_random_forest_fe_vars.csv
# Salidas  : predictions/harold_no_enviado_{lpm,logit,cart,naive_bayes}_fe_*.csv (no se enviaron a Kaggle),
#            tables/harold_classical_fe_cv.{csv,tex}, processed/harold_*_fe_scores.csv
# ==============================================================================
AUTOR <- "harold"
suppressPackageStartupMessages({library(tidyverse); library(rpart); library(e1071); library(xtable); library(splines)})

fe   <- readRDS(file.path(DIR_PROC, "hogares_fe.rds"))
vars <- read_csv(file.path(DIR_TAB, paste0(AUTOR, "_random_forest_fe_vars.csv")), show_col_types = FALSE)$variable
fe$Dominio_code <- as.integer(factor(fe$Dominio))
vars_raw <- c(setdiff(vars, c("Dominio", "tenure")), "Dominio_code")      # para árboles y Naive Bayes
trn <- fe |> filter(sample == "train"); tst <- fe |> filter(sample == "test")
y <- trn$poor; fold <- trn$fold; bog <- trn$bogota
test_ids <- tst$id
thr_grid <- c(seq(0.05, 0.95, by = 0.01), seq(0.952, 0.998, by = 0.002))   # Naive Bayes da probabilidades extremas
resumen <- list()
registrar <- function(nombre, oof, p_te, archivo, hiper) {
  ev <- fun_eval_oof(y, oof, fold, bog, thr = thr_grid)
  fun_write_submission_fe(test_ids, as.integer(p_te >= ev$resumen$umbral), archivo)
  fun_save_scores(fe, oof, p_te, paste0(nombre, "_fe"))
  resumen[[nombre]] <<- bind_cols(modelo = nombre, ev$resumen |> select(umbral, f1, f1_sd_pliegues, precision, recall, auc, pct_pred_pobre),
                                  hiperparametros = hiper)
  message(sprintf("%-12s F1 = %.4f | AUC = %.4f | umbral = %.2f", nombre, ev$resumen$f1, ev$resumen$auc, ev$resumen$umbral))
}

# --- LPM y Logit: matriz de diseño estandarizada (log1p, winsorización, dummies de dominio/tenencia) --------
vars_num <- setdiff(vars, c("Dominio", "tenure"))
M  <- fun_matrix_en(fe, vars_num, spline_vars = character())
Xtr <- cbind(1, M$X_train); Xte <- cbind(1, M$X_test)
message("LPM/Logit: ", ncol(M$X_train), " columnas de diseño")

fit_lpm <- function(X, yy) { f <- lm.fit(X, yy); b <- f$coefficients; b[is.na(b)] <- 0; b }
oof <- numeric(nrow(Xtr))
for (k in 1:N_FOLDS) { ik <- fold == k; oof[ik] <- as.numeric(Xtr[ik, ] %*% fit_lpm(Xtr[!ik, ], y[!ik])) }
b <- fit_lpm(Xtr, y)
registrar("lpm", oof, as.numeric(Xte %*% b), sprintf("%s_no_enviado_lpm_fe_%dvars.csv", AUTOR, length(vars)), "MCO sobre 0/1; sin penalización")

fit_logit <- function(X, yy) { f <- glm.fit(X, yy, family = binomial(), control = list(maxit = 50)); b <- f$coefficients; b[is.na(b)] <- 0; b }
oof <- numeric(nrow(Xtr))
for (k in 1:N_FOLDS) { ik <- fold == k; oof[ik] <- plogis(as.numeric(Xtr[ik, ] %*% fit_logit(Xtr[!ik, ], y[!ik]))) }
b <- fit_logit(Xtr, y)
registrar("logit", oof, plogis(as.numeric(Xte %*% b)), sprintf("%s_no_enviado_logit_fe_%dvars.csv", AUTOR, length(vars)), "Máxima verosimilitud; sin penalización")
rm(Xtr, Xte, M); invisible(gc())

# --- CART: malla de cp y minsplit, selección por F1 en OOF ---------------------------------------------------------
yf <- factor(y, levels = c(0, 1), labels = c("No", "Si"))
d_tr <- cbind(trn[vars_raw], y = yf); d_te <- tst[vars_raw]
mallas <- expand.grid(cp = c(0.0003, 0.001, 0.003), minsplit = c(50, 200))
cart_res <- vector("list", nrow(mallas)); cart_oof <- vector("list", nrow(mallas))
for (i in seq_len(nrow(mallas))) {
  oo <- numeric(nrow(d_tr))
  for (k in 1:N_FOLDS) {
    ik <- fold == k
    m <- rpart(y ~ ., data = d_tr[!ik, ], method = "class", control = rpart.control(cp = mallas$cp[i], minsplit = mallas$minsplit[i], maxdepth = 12, xval = 0))
    oo[ik] <- predict(m, d_tr[ik, ], type = "prob")[, "Si"]
  }
  ev <- fun_eval_oof(y, oo, fold, bog, thr = thr_grid)
  cart_res[[i]] <- tibble(cp = mallas$cp[i], minsplit = mallas$minsplit[i], f1 = ev$resumen$f1, auc = ev$resumen$auc); cart_oof[[i]] <- oo
  message(sprintf("CART cp=%.4f minsplit=%d | F1=%.4f AUC=%.4f", mallas$cp[i], mallas$minsplit[i], ev$resumen$f1, ev$resumen$auc))
}
cart_tab <- bind_rows(cart_res); ib <- which.max(cart_tab$f1)
write_csv(cart_tab, file.path(DIR_TAB, paste0(AUTOR, "_cart_fe_search.csv")))
m <- rpart(y ~ ., data = d_tr, method = "class", control = rpart.control(cp = mallas$cp[ib], minsplit = mallas$minsplit[ib], maxdepth = 12, xval = 0))
registrar("cart", cart_oof[[ib]], predict(m, d_te, type = "prob")[, "Si"],
          sprintf("%s_no_enviado_cart_fe_cp_%s.csv", AUTOR, gsub("\\.", "_", format(mallas$cp[ib], scientific = FALSE))),
          sprintf("cp = %s; minsplit = %d; profundidad máxima 12", mallas$cp[ib], mallas$minsplit[ib]))

# --- Naive Bayes: variables discretizadas en deciles (cortes aprendidos con train), suavizado de Laplace -------------
cortes <- lapply(vars_raw, function(v) { q <- unique(quantile(trn[[v]], seq(0, 1, 0.1))); if (length(q) < 3) q <- c(-Inf, Inf); q[1] <- -Inf; q[length(q)] <- Inf; q })
names(cortes) <- vars_raw
disc <- function(df) as.data.frame(lapply(vars_raw, function(v) cut(df[[v]], cortes[[v]], include.lowest = TRUE)) |> setNames(vars_raw))
nb_tr <- disc(trn); nb_te <- disc(tst)
oof <- numeric(nrow(nb_tr))
for (k in 1:N_FOLDS) {
  ik <- fold == k
  m <- naiveBayes(x = nb_tr[!ik, ], y = yf[!ik], laplace = 1)
  oof[ik] <- predict(m, nb_tr[ik, ], type = "raw")[, "Si"]
}
m <- naiveBayes(x = nb_tr, y = yf, laplace = 1)
registrar("naive_bayes", oof, predict(m, nb_te, type = "raw")[, "Si"], sprintf("%s_no_enviado_naive_bayes_fe_deciles.csv", AUTOR), "Variables en deciles; Laplace = 1")

# --- Tabla resumen ---------------------------------------------------------------------------------------------------
tab <- bind_rows(resumen)
write_csv(tab, file.path(DIR_TAB, paste0(AUTOR, "_classical_fe_cv.csv")))
print(xtable(tab |> select(modelo, umbral, f1, precision, recall, auc), digits = c(0, 0, 2, 4, 4, 4, 4),
             caption = "LPM, Logit, CART y Naive Bayes con las 40 variables del Random Forest (5 pliegues, sin Bogotá)"),
      include.rownames = FALSE, file = file.path(DIR_TAB, paste0(AUTOR, "_classical_fe_cv.tex")))
print(tab)
