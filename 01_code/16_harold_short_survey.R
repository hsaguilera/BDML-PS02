# ==============================================================================
# 16_harold_short_survey.R
# ¿Se puede focalizar con una encuesta corta y barata? LightGBM (hiperparámetros del script 09) con
# conjuntos de variables cada vez más fáciles de recolectar y verificar en campo:
#   S1  vivienda, composición del hogar, edades y educación (sin preguntas de empleo ni de ingresos);
#   S2  S1 + condición de actividad (ocupado, desocupado, inactivo) y afiliación a salud;
#   Completo  las 149 variables, que incluyen el ingreso por persona imputado con ~60 preguntas laborales.
# Mismos 5 pliegues, mismo criterio de umbral. Entradas: hogares_fe.rds, hogares_fe_grupos.rds, params del script 09.
# Salidas : tables/harold_short_survey_{vars,cv}.csv, processed/harold_lightgbm_{s1,s2}_scores.csv
# ==============================================================================
AUTOR <- "harold"
suppressPackageStartupMessages({library(data.table); library(tidyverse)})
PY_BIN <- Sys.getenv("PYTHON_BIN", "python3")
fe     <- readRDS(file.path(DIR_PROC, "hogares_fe.rds"))
groups <- readRDS(file.path(DIR_PROC, "hogares_fe_grupos.rds"))
trn <- fe |> filter(sample == "train"); y <- trn$poor

# --- Conjuntos de variables ----------------------------------------------------------------------------------------------
s1 <- c("Dominio", "tenure", "urban", "Lp", "req_income", "P5000", "P5010", "P5090", "Nper", "Npersug", "n_members", "n_outside_ug",
        "crowding", "rooms_pc", "share_female", "n_child_0_5", "n_child_6_17", "n_adult", "n_senior", "age_mean", "age_max",
        "educ_mean", "educ_max", "n_superior", "n_no_educ", "head_female", "head_age", "head_educ", "dep_ratio", "kids_pc",
        "fe_persons_room", "fe_bedrooms_pc", "fe_crowd_sq", "fe_crowd_gt3", "fe_one_room", "fe_owner", "fe_precarious_tenure",
        "fe_hh_size_sq", "fe_adult_equiv", "fe_adult_equiv_ratio", "fe_young_kid_share", "fe_senior_share", "fe_single_adult",
        "fe_single_parent_f", "fe_only_seniors", "fe_age_gap_head", "fe_educ_gap_head", "fe_educ_max_mean", "fe_share_superior",
        "fe_share_no_educ", "fe_head_educ_low", "fe_educ_x_adults", "fe_rural_x_crowd", "fe_dep_x_educ")
s2 <- c(s1, "n_employed", "n_unemployed", "n_inactive", "head_employed", "head_unemployed", "head_inactive", "earners_pc", "no_earner",
        "share_insured", "n_subsid_regime", "n_contrib_regime", "head_insured", "fe_workers_per_dep", "fe_unemp_share",
        "fe_inactive_share", "fe_no_head_earner", "fe_head_only_earner", "fe_rural_x_noearn", "fe_size_x_noearn",
        "fe_crowd_x_noearn", "fe_earners_sq")
conjuntos <- list(S1 = s1, S2 = s2)
stopifnot(all(unlist(conjuntos) %in% groups$variable))
write_csv(bind_rows(imap(conjuntos, \(v, n) tibble(conjunto = n, variable = v))), file.path(DIR_TAB, paste0(AUTOR, "_short_survey_vars.csv")))

# --- Entrenamiento con hiperparámetros fijos ---------------------------------------------------------------------------
f_pa <- file.path(DIR_PROC, paste0(AUTOR, "_lightgbm_fe_params.json"))
res <- map_dfr(names(conjuntos), function(n) {
  f_in <- file.path(DIR_PROC, "lgbm_short_in.csv")
  fun_export_fe_csv(fe, groups |> filter(variable %in% conjuntos[[n]]), f_in)
  f_sc <- file.path(DIR_PROC, paste0(AUTOR, "_lightgbm_", tolower(n), "_scores.csv"))
  stopifnot(system2(PY_BIN, c("-I", "01_code/python/harold_lightgbm_fixed.py", f_in, f_pa, f_sc)) == 0)
  file.remove(f_in, sub("\\.csv$", "_vars.csv", f_in))
  sc <- fread(f_sc) |> as_tibble(); s <- sc$score[match(trn$id, sc$id)]
  ev <- fun_eval_oof(y, s, trn$fold, trn$bogota)$resumen
  nb <- trn$bogota == 0; cm <- conf_metrics(y[nb], as.integer(s[nb] >= ev$umbral))
  tibble(conjunto = n, n_vars = length(conjuntos[[n]]), umbral = ev$umbral, f1 = ev$f1, f1_sd_pliegues = ev$f1_sd_pliegues,
         precision = cm$precision, recall = cm$recall, filtracion = cm$leakage, subcobertura = cm$undercoverage, auc = ev$auc)
})
# Referencia: modelo completo (script 09) y ablación (script 06)
ref <- read_csv(file.path(DIR_TAB, paste0(AUTOR, "_lightgbm_fe_cv.csv")), show_col_types = FALSE)
sc  <- fread(file.path(DIR_PROC, paste0(AUTOR, "_lightgbm_fe_scores.csv"))) |> as_tibble(); s <- sc$score[match(trn$id, sc$id)]
nb  <- trn$bogota == 0; cm <- conf_metrics(y[nb], as.integer(s[nb] >= ref$umbral))
full <- tibble(conjunto = "Completo", n_vars = ref$rondas * 0 + nrow(groups), umbral = ref$umbral, f1 = ref$f1, f1_sd_pliegues = ref$f1_sd_pliegues,
               precision = cm$precision, recall = cm$recall, filtracion = cm$leakage, subcobertura = cm$undercoverage,
               auc = fun_auc(y[nb], s[nb]))
res <- bind_rows(res, full)
write_csv(res, file.path(DIR_TAB, paste0(AUTOR, "_short_survey_cv.csv")))
print(res, width = Inf)
