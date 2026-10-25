# ==============================================================================
# 01_build_household_dataset.R
# Objetivo: limpiar, unir hogares y personas y construir variables a nivel de
#           hogar. Produce 00_data/processed/hogares_modelo.rds
# Reglas (complementaria del 19 de septiembre):
#   1) solo se usan variables que existen en TEST;
#   2) el mismo pre-procesamiento se aplica a train y test (se apilan y se
#      procesan juntos con las mismas funciones);
#   3) las variables de ingreso solo existen en train: se guardan aparte con
#      prefijo `eda_` para exploración y nunca como predictores.
# ==============================================================================

suppressPackageStartupMessages({library(data.table); library(tidyverse)})

# --- 1. Lectura -----------------------------------------------------------------
cols_test <- names(fread(file.path(DIR_RAW, "test_personas.csv"), nrows = 1))
th <- fread(file.path(DIR_RAW, "train_hogares.csv"))
eh <- fread(file.path(DIR_RAW, "test_hogares.csv"))
tp <- fread(file.path(DIR_RAW, "train_personas.csv"), select = cols_test)   # solo columnas presentes en test
ep <- fread(file.path(DIR_RAW, "test_personas.csv"))

hog <- bind_rows(mutate(as_tibble(th), sample = "train"), mutate(as_tibble(eh), sample = "test"))
per <- bind_rows(mutate(as_tibble(tp), sample = "train"), mutate(as_tibble(ep), sample = "test"))

# --- 2. Verificaciones de integridad ----------------------------------------------
stopifnot(
  !anyDuplicated(hog$id),                                   # un registro por hogar
  all(per$id %in% hog$id), all(hog$id %in% per$id),         # todos los hogares tienen personas y viceversa
  length(intersect(th$id, eh$id)) == 0,                     # sin solapamiento train/test
  all(th$Pobre == as.integer(th$Ingpcug < th$Lp))           # Pobre = 1(ingreso per cápita < línea)
)

# --- 3. Personas -> hogar ----------------------------------------------------------
per_prep <- fun_preprocess_personas(per)
per_hog  <- fun_collapse_personas(per_prep)

# La unidad de gasto (UG) del DANE excluye a empleados domésticos, pensionistas y
# trabajadores (P6050 = 6, 7, 8): el tamaño de la UG calculado así debe coincidir con Npersug.
chk <- hog |> select(id, Npersug) |> inner_join(per_hog |> select(id, n_members), by = "id")
stopifnot(all(chk$Npersug == chk$n_members))

# --- 4. Unión y variables del hogar ------------------------------------------------------
d <- hog |>
  fun_preprocess_hogares() |>
  left_join(per_hog, by = "id") |>
  fun_add_ratios() |>
  as.data.table()

# --- 5. Imputación simple (medianas del TRAIN; los factores quedan sin NA) -------------
eda_cols <- c("Ingtotug", "Ingtotugarr", "Ingpcug", "Indigente", "Npobres", "Nindigentes")
num_cols <- setdiff(names(d)[sapply(d, is.numeric)], c("Pobre", eda_cols))
for (j in num_cols) {
  if (anyNA(d[[j]])) {
    med <- median(d[sample == "train"][[j]], na.rm = TRUE)
    set(d, which(is.na(d[[j]])), j, med)
  }
}
d[, `:=`(Dominio = factor(Dominio), Depto = factor(Depto), tenure = fct_na_value_to_level(tenure, "otra"))]

# --- 6. Guardado ------------------------------------------------------------------------
setnames(d, eda_cols, paste0("eda_", eda_cols))
setnames(d, "Pobre", "poor")
saveRDS(as_tibble(d), file.path(DIR_PROC, "hogares_modelo.rds"))
message("Hogares: ", nrow(d), " | train: ", sum(d$sample == "train"), " | test: ", sum(d$sample == "test"),
        " | columnas: ", ncol(d))
