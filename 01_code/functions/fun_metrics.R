# Funciones de métricas y utilidades para clasificación de pobreza ---------------

# F1 de la clase positiva (pobre = 1)
f1_score <- function(y, yhat) {
  tp <- sum(y == 1 & yhat == 1)
  fp <- sum(y == 0 & yhat == 1)
  fn <- sum(y == 1 & yhat == 0)
  if (tp == 0) return(0)
  prec <- tp / (tp + fp); rec <- tp / (tp + fn)
  2 * prec * rec / (prec + rec)
}

# Precisión, recall (cobertura) y tasas de error de política
conf_metrics <- function(y, yhat) {
  tp <- sum(y == 1 & yhat == 1); fp <- sum(y == 0 & yhat == 1)
  fn <- sum(y == 1 & yhat == 0); tn <- sum(y == 0 & yhat == 0)
  tibble::tibble(
    precision = tp / (tp + fp), recall = tp / (tp + fn), f1 = f1_score(y, yhat),
    leakage = fp / (tp + fp),            # % de beneficiarios que no son pobres
    undercoverage = fn / (tp + fn),      # % de pobres excluidos
    tp = tp, fp = fp, fn = fn, tn = tn)
}

# Umbral que maximiza F1 sobre probabilidades (en datos de validación / OOF)
best_threshold <- function(y, p, grid = seq(0.10, 0.80, by = 0.01)) {
  f1s <- vapply(grid, function(t) f1_score(y, as.integer(p >= t)), numeric(1))
  list(threshold = grid[which.max(f1s)], f1 = max(f1s), curve = tibble::tibble(t = grid, f1 = f1s))
}

# Pliegues estratificados reproducibles
make_folds <- function(y, k = 5, seed = 2026) {
  set.seed(seed)
  fold <- integer(length(y))
  for (cl in unique(y)) {
    idx <- which(y == cl)
    fold[idx] <- sample(rep_len(seq_len(k), length(idx)))
  }
  fold
}
