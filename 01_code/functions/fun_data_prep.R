# ==============================================================================
# fun_data_prep.R  --  Funciones de pre-procesamiento (personas -> hogar)
# Se aplican de la misma forma a train y test (se llaman sobre la base apilada).
# ==============================================================================

#' Máximo seguro: devuelve NA (y no -Inf) cuando todos los valores son NA
#' @param x vector numérico.
safe_max <- function(x) if (all(is.na(x))) NA_real_ else max(x, na.rm = TRUE)

#' Preprocesar la base de personas
#'
#' @description Recodifica las variables de la GEIH (1 = sí, 2 = no; 9 = "no sabe"
#' pasa a faltante) y crea indicadores a nivel de persona.
#' La unidad de gasto (UG) del DANE excluye a empleados domésticos, pensionistas
#' y trabajadores (P6050 en 6, 7 y 8) y sus parientes; el ingreso per cápita y la
#' pobreza se calculan solo sobre la UG, por lo que se marca con `bin_in_ug`.
#' @param .db tibble de personas (train y test apilados) con la columna `Orden`.
#' @returns tibble con una fila por persona: `id`, `id_persona`, `bin_in_ug` y los
#' indicadores/recodificaciones usados luego para agregar al hogar.
fun_preprocess_personas <- function(.db) {
  yes <- function(x) as.integer(x %in% 1)
  .db |>
    mutate(across(c(P6090, P6100, P6210, P6240, P6430, P6870, P6920),
                  ~ if_else(.x == 9, NA_real_, as.numeric(.x)))) |>
    transmute(
      id, sample, id_persona = Orden,
      bin_in_ug   = as.integer(!P6050 %in% c(6, 7, 8)),
      bin_domestic = as.integer(P6050 == 6),
      bin_head    = as.integer(P6050 == 1),
      bin_female  = as.integer(P6020 == 2),
      num_age     = as.numeric(P6040),
      # años de educación aproximados a partir del nivel más alto aprobado
      num_educ    = case_when(P6210 %in% c(1, 2) ~ 0, P6210 == 3 ~ 5, P6210 == 4 ~ 9,
                              P6210 == 5 ~ 11, P6210 == 6 ~ 15, TRUE ~ NA_real_),
      bin_superior = as.integer(P6210 == 6 & !is.na(P6210)),
      bin_no_educ  = as.integer(P6210 %in% c(1, 2) & P6040 >= 15),
      bin_insured = yes(P6090),
      bin_regime_contrib = as.integer(P6100 == 1 & !is.na(P6100)),
      bin_regime_subsid  = as.integer(P6100 == 3 & !is.na(P6100)),
      bin_employed   = yes(Oc), bin_unemployed = yes(Des), bin_inactive = yes(Ina),
      num_hours   = coalesce(as.numeric(P6800), 0),
      bin_formal  = yes(P6920),                                  # cotiza a pensión
      bin_selfemp = as.integer(P6430 %in% c(4, 5)),              # cuenta propia o patrón
      bin_wage_emp = as.integer(P6430 %in% c(1, 2, 3, 8)),
      bin_unpaid  = as.integer(P6430 %in% c(6, 7)),
      bin_big_firm = as.integer(P6870 >= 5 & !is.na(P6870)),
      bin_second_job = yes(P7040), bin_more_hours = yes(P7090),
      bin_pension = yes(P7495), bin_other_inc = yes(P7505),
      bin_remit_nat = yes(P7510s1), bin_remit_int = yes(P7510s2),
      bin_aid_inst  = yes(P7510s3), bin_interest = yes(P7510s5),
      num_subsidy = yes(P6585s1) + yes(P6585s2) + yes(P6585s3) + yes(P6585s4),
      num_in_kind = yes(P6590) + yes(P6600) + yes(P6610) + yes(P6620),
      num_job_tenure = as.numeric(P6426),
      bin_child_0_5   = as.integer(P6040 <= 5),
      bin_child_6_17  = as.integer(P6040 >= 6 & P6040 <= 17),
      bin_adult_18_64 = as.integer(P6040 >= 18 & P6040 <= 64),
      bin_senior_65   = as.integer(P6040 >= 65)
    )
}

#' Colapsar las personas a nivel de hogar
#'
#' @description Agrega, sobre los miembros de la unidad de gasto, conteos y
#' proporciones por hogar; añade las características del jefe y cuántas personas
#' del hogar quedan fuera de la UG (señal de servicio doméstico).
#' @param .db tibble devuelto por `fun_preprocess_personas()`.
#' @returns tibble con una fila por hogar (`id`).
fun_collapse_personas <- function(.db) {
  ug <- .db |> filter(bin_in_ug == 1)

  agg <- ug |>
    group_by(id) |>
    summarise(
      n_members = n(), share_female = mean(bin_female),
      n_child_0_5 = sum(bin_child_0_5), n_child_6_17 = sum(bin_child_6_17),
      n_adult = sum(bin_adult_18_64), n_senior = sum(bin_senior_65),
      age_mean = mean(num_age), age_max = max(num_age),
      educ_mean = mean(num_educ[num_age >= 15], na.rm = TRUE),
      educ_max = safe_max(num_educ), n_superior = sum(bin_superior), n_no_educ = sum(bin_no_educ),
      share_insured = mean(bin_insured), n_subsid_regime = sum(bin_regime_subsid),
      n_contrib_regime = sum(bin_regime_contrib),
      n_employed = sum(bin_employed), n_unemployed = sum(bin_unemployed), n_inactive = sum(bin_inactive),
      hours_total = sum(num_hours), n_formal = sum(bin_formal), n_selfemp = sum(bin_selfemp),
      n_wage_emp = sum(bin_wage_emp), n_unpaid = sum(bin_unpaid), n_big_firm = sum(bin_big_firm),
      n_second_job = sum(bin_second_job), n_more_hours = sum(bin_more_hours),
      n_pension = sum(bin_pension), n_other_inc = sum(bin_other_inc),
      remit_nat = max(bin_remit_nat), remit_int = max(bin_remit_int),
      aid_inst = max(bin_aid_inst), interest = max(bin_interest),
      n_subsidy = sum(num_subsidy), n_in_kind = sum(num_in_kind),
      job_tenure_max = safe_max(num_job_tenure),
      .groups = "drop")

  head_ug <- ug |>
    filter(bin_head == 1) |>
    group_by(id) |> slice(1) |> ungroup() |>
    transmute(id, head_female = bin_female, head_age = num_age, head_educ = num_educ,
              head_employed = bin_employed, head_inactive = bin_inactive,
              head_unemployed = bin_unemployed, head_formal = bin_formal,
              head_selfemp = bin_selfemp, head_insured = bin_insured,
              head_pension = bin_pension, head_hours = num_hours)

  outside <- .db |>
    group_by(id) |>
    summarise(n_outside_ug = sum(1L - bin_in_ug), n_domestic = sum(bin_domestic), .groups = "drop")

  agg |> left_join(head_ug, by = "id") |> left_join(outside, by = "id")
}

#' Preprocesar la base de hogares
#'
#' @description Variables de vivienda y ubicación a nivel de hogar: tenencia,
#' zona, arriendo (pagado o imputado), hacinamiento y cuartos por persona.
#' @param .db tibble de hogares (train y test apilados).
#' @returns tibble de hogares con las variables de vivienda adicionales.
fun_preprocess_hogares <- function(.db) {
  .db |>
    mutate(
      rent_est     = coalesce(as.numeric(P5130), as.numeric(P5140)),
      rent_missing = as.integer(is.na(P5130) & is.na(P5140)),
      tenure       = factor(P5090, levels = 1:6,
                            labels = c("propia_pagada", "propia_pagando", "arriendo",
                                       "usufructo", "ocupante", "otra")),
      urban        = as.integer(Clase == 1),
      crowding     = Nper / pmax(P5010, 1),          # personas por dormitorio
      rooms_pc     = P5000 / Nper,
      req_income   = Lp * Npersug)                    # ingreso que cruza la línea para la UG
}

#' Razones y tasas del hogar
#'
#' @description Construye las razones per cápita sobre la unidad de gasto. Cuando
#' el denominador es cero la tasa no está definida: se asigna un valor auxiliar y
#' se crea un indicador (`bin_*`) para que el modelo pueda distinguir ese caso.
#' @param .db tibble con las variables de `fun_preprocess_hogares()` y de
#' `fun_collapse_personas()`.
#' @returns tibble con las razones y los indicadores de tasa indefinida.
fun_add_ratios <- function(.db) {
  .db |>
    mutate(
      rent_est       = coalesce(rent_est, 0),
      rent_pc        = rent_est / Npersug,
      rent_burden    = rent_est / req_income,                 # arriendo / presupuesto de la línea
      bin_no_adults  = as.integer(n_adult == 0),               # dep_ratio no definida de forma natural
      dep_ratio      = (n_child_0_5 + n_child_6_17 + n_senior) / pmax(n_adult, 1),
      earners_pc     = n_employed / Npersug,
      hours_pc       = hours_total / Npersug,
      no_earner      = as.integer(n_employed == 0),            # formal_share no definida si es 1
      formal_share   = if_else(n_employed > 0, n_formal / n_employed, 0),
      kids_pc        = (n_child_0_5 + n_child_6_17) / Npersug,
      log_rent_pc    = log1p(rent_pc),
      log_req_income = log(req_income))
}
