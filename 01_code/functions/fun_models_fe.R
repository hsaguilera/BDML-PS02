# ==============================================================================
# fun_models_fe.R  --  Utilidades comunes de los modelos con ingeniería de variables
# (evaluación fuera de muestra, umbral óptimo para F1, armado y validación del envío)
# ==============================================================================

#' AUC por rangos (rápido, sin dependencias)
#' @param y vector 0/1. @param p puntaje (mayor = más probable ser pobre).
fun_auc <- function(y, p) {
  r <- rank(p, ties.method = "average"); n1 <- as.numeric(sum(y == 1)); n0 <- as.numeric(sum(y == 0))  # numérico: evita desbordamiento entero
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

#' F1 para una grilla de umbrales (vectorizado con ordenamiento)
#' @returns tibble con umbral y F1; sirve para elegir el umbral óptimo sobre puntajes.
fun_f1_curve <- function(y, p, thr = seq(0.10, 0.80, by = 0.01)) {
  tp <- vapply(thr, function(t) sum(y == 1 & p >= t), numeric(1))
  fp <- vapply(thr, function(t) sum(y == 0 & p >= t), numeric(1))
  fn <- sum(y == 1) - tp
  tibble::tibble(thr = thr, f1 = ifelse(tp == 0, 0, 2 * tp / (2 * tp + fp + fn)))
}

#' Evaluación fuera de muestra de un puntaje (probabilidad) por validación cruzada
#'
#' @description El umbral se elige UNA vez sobre todas las predicciones fuera de
#' muestra de hogares que no están en Bogotá (Bogotá no existe en test) y se
#' reporta el F1 con ese umbral, también por pliegue, y el F1 incluyendo Bogotá.
#' @param y 0/1, @param p puntaje, @param fold pliegue (1..K), @param bogota 0/1.
#' @param thr grilla de umbrales (para regresores en otra escala, pasar su grilla).
#' @returns lista con `resumen` (una fila), `por_pliegue` y `curva`.
fun_eval_oof <- function(y, p, fold, bogota, thr = seq(0.10, 0.80, by = 0.01)) {
  nb <- bogota == 0
  curva <- fun_f1_curve(y[nb], p[nb], thr)
  t_opt <- curva$thr[which.max(curva$f1)]
  yhat  <- as.integer(p >= t_opt)
  cm    <- conf_metrics(y[nb], yhat[nb])
  fold_id <- fold   # copia: dentro de mutate(), `fold` sería la columna nueva y no el vector de pliegues de los hogares
  por_pliegue <- tibble::tibble(fold = sort(unique(fold_id))) |>
    dplyr::mutate(f1 = vapply(fold, function(k) f1_score(y[nb & fold_id == k], yhat[nb & fold_id == k]), numeric(1)))
  ll <- -mean(ifelse(y[nb] == 1, log(pmax(p[nb], 1e-12)), log(pmax(1 - p[nb], 1e-12))))
  resumen <- tibble::tibble(
    umbral = t_opt, f1 = max(curva$f1), f1_sd_pliegues = sd(por_pliegue$f1),
    precision = cm$precision, recall = cm$recall, auc = fun_auc(y[nb], p[nb]),
    logloss = if (all(p >= 0 & p <= 1)) ll else NA_real_,
    f1_con_bogota = f1_score(y, yhat), pct_pred_pobre = mean(yhat[nb]), pct_pobre_real = mean(y[nb]))
  list(resumen = resumen, por_pliegue = por_pliegue, curva = curva)
}

#' Armar y validar el archivo de envío a Kaggle
#'
#' @param id identificadores del test (en el orden de predicción), @param pobre 0/1.
#' @param name nombre del archivo (sin carpeta). Se valida contra sample_submission.csv.
fun_write_submission_fe <- function(id, pobre, name, dir = DIR_PRED) {
  ss <- data.table::fread(file.path(DIR_RAW, "sample_submission.csv"))
  sub <- tibble::tibble(id = id, pobre = as.integer(pobre))
  stopifnot(nrow(sub) == nrow(ss), setequal(sub$id, ss$id), !anyNA(sub$pobre), all(sub$pobre %in% 0:1),
            identical(names(sub), names(ss)))
  sub <- sub[match(ss$id, sub$id), ]                     # mismo orden que sample_submission
  path <- file.path(dir, name)
  utils::write.csv(sub, path, row.names = FALSE)
  message("Envío guardado: ", path, " | % predicho pobre: ", round(100 * mean(sub$pobre), 1))
  invisible(path)
}

#' Guardar predicciones fuera de muestra (OOF) y de test de un modelo (para apilar después)
fun_save_scores <- function(fe, oof, test_score, model_tag, dir = DIR_PROC) {
  out <- dplyr::bind_rows(
    tibble::tibble(id = fe$id[fe$sample == "train"], sample = "train", fold = fe$fold[fe$sample == "train"], score = oof),
    tibble::tibble(id = fe$id[fe$sample == "test"],  sample = "test",  fold = 0L, score = test_score))
  data.table::fwrite(out, file.path(dir, paste0(AUTOR, "_", model_tag, "_scores.csv")))
  invisible(out)
}

#' Matriz de diseño para modelos lineales (Elastic Net)
#'
#' @description Transformaciones aprendidas SOLO con train y aplicadas igual a test:
#' (1) log1p a las variables no negativas muy asimétricas (asimetría > 2);
#' (2) winsorización a los percentiles 0,5 y 99,5 de las no binarias;
#' (3) base de splines naturales (`df` grados) para las variables de `spline_vars`,
#'     que permiten relaciones no lineales con la pobreza;
#' (4) dummies de `Dominio` y `tenure`; (5) estandarización con media y desvío de train.
#' @param fe base con `sample`; @param vars variables numéricas; @param spline_vars subconjunto con splines.
#' @returns lista con `X_train`, `X_test` (matrices) y los nombres de las columnas.
fun_matrix_en <- function(fe, vars, spline_vars = character(), df = 5) {
  tr <- fe$sample == "train"
  Z  <- as.data.frame(fe[vars])
  skew <- function(x) { x <- x[is.finite(x)]; mean((x - mean(x))^3) / sd(x)^3 }
  for (v in vars) {
    x <- Z[[v]]; binario <- all(x[tr] %in% c(0, 1))
    if (binario) next
    if (min(x[tr]) >= 0 && skew(x[tr]) > 2) { x <- log1p(x); Z[[v]] <- x }
    q <- stats::quantile(x[tr], c(0.005, 0.995)); Z[[v]] <- pmin(pmax(x, q[1]), q[2])
  }
  parts <- list(as.matrix(Z))
  for (v in spline_vars) {
    kn <- stats::quantile(Z[[v]][tr], seq(0, 1, length.out = df + 1)[-c(1, df + 1)])
    kn <- unique(kn)
    B <- splines::ns(Z[[v]], knots = kn, Boundary.knots = range(Z[[v]][tr]))
    colnames(B) <- paste0(v, "_ns", seq_len(ncol(B)))
    parts[[length(parts) + 1]] <- unclass(B)[, , drop = TRUE]
  }
  cat_vars <- intersect(c("Dominio", "tenure"), names(fe))
  for (cv in cat_vars) {
    f <- factor(fe[[cv]], levels = levels(factor(fe[[cv]][tr])))
    D <- stats::model.matrix(~ f - 1); colnames(D) <- paste0(cv, "_", levels(f))
    parts[[length(parts) + 1]] <- D
  }
  X <- do.call(cbind, parts)
  mu <- colMeans(X[tr, ]); sdv <- apply(X[tr, ], 2, sd); sdv[sdv == 0] <- 1
  X <- sweep(sweep(X, 2, mu, "-"), 2, sdv, "/")
  list(X_train = X[tr, ], X_test = X[!tr, ], cols = colnames(X))
}

#' Exportar la base de modelado a CSV para los modelos en Python
#'
#' @description Escribe `id`, `sample`, `fold`, `bogota`, `poor`, el objetivo continuo `z`
#' (log del ingreso per cápita sobre la línea de pobreza, recortado en [0,05; 20]; solo train)
#' y las variables de los grupos indicados. Devuelve la ruta; quien llama borra el archivo.
fun_export_fe_csv <- function(fe, groups, path, grupos = c("A", "B", "C", "D"), extra = NULL) {
  vars <- groups$variable[groups$grupo %in% grupos]
  out <- fe[c("id", "sample", "fold", "bogota", "poor", vars)]
  out$z <- log(pmin(pmax(fe$eda_Ingpcug / fe$Lp, 0.05), 20))
  if (!is.null(extra)) for (nm in names(extra)) out[[nm]] <- extra[[nm]]
  data.table::fwrite(out, path)
  data.table::fwrite(groups[groups$grupo %in% grupos, ], sub("\\.csv$", "_vars.csv", path))
  invisible(path)
}
