# ==============================================================================
# fun_features_fe.R  --  Ingeniería de variables (proxy means testing)
# Todas las funciones usan solo variables que existen en TEST y se aplican igual
# a train y test. Las variables de ingreso de train solo se usan como OBJETIVO de
# un modelo auxiliar (ingreso imputado fuera de muestra), nunca como predictores.
# ==============================================================================

#' Arriendo imputado que el DANE suma al ingreso de la unidad de gasto
#'
#' @description Ingtotugarr = Ingtotug + arriendo imputado. En train, esa
#' diferencia es igual a `P5130` (lo que pagaría de arriendo) para propietarios y
#' usufructuarios (`P5090` = 1 o 4) con `P5130` entre 17.500 y 3.010.000, y a
#' max(P5130 - P5100, 0) para propietarios que aún pagan (`P5090` = 2). En los
#' demás casos es 0. La regla reproduce el valor exacto en el 95,7 % de los
#' hogares de train y el error absoluto medio es de unos $9.500.
#' @param P5090,P5130,P5100 tenencia, arriendo estimado y cuota de amortización.
#' @returns vector numérico con el arriendo imputado estimado (>= 0).
fun_arriendo_imputado <- function(P5090, P5130, P5100) {
  est <- dplyr::case_when(
    P5090 %in% c(1, 4) & P5130 >= 17500 & P5130 < 3010000 ~ as.numeric(P5130),
    P5090 == 2 ~ pmax(as.numeric(P5130) - as.numeric(P5100), 0),
    TRUE ~ 0)
  dplyr::coalesce(est, 0)
}

#' Tabla de personas de la unidad de gasto para el modelo auxiliar de ingreso
#'
#' @description Una fila por persona de la UG con todas las variables de persona
#' que existen en test (códigos originales de la GEIH, sin recodificar, para que
#' el modelo de árboles los use tal cual) y el contexto del hogar (`Lp`, `P5090`).
#' El objetivo `Ingtot` solo existe en train (faltante para menores de 10 años).
#' @param per tibble de personas apiladas (train y test) con `sample`.
#' @param hog tibble de hogares apilados con `id`, `Lp`, `fold`.
#' @returns tibble listo para exportar a Python.
fun_person_income_table <- function(per, hog) {
  vars_test <- c("P6020", "P6040", "P6050", "P6090", "P6100", "P6210", "P6210s1", "P6240",
                 "Oficio", "P6426", "P6430", "P6510", "P6545", "P6580",
                 "P6585s1", "P6585s2", "P6585s3", "P6585s4", "P6590", "P6600", "P6610", "P6620",
                 "P6630s1", "P6630s2", "P6630s3", "P6630s4", "P6630s6",
                 "P6800", "P6870", "P6920", "P7040", "P7045", "P7050", "P7090", "P7110", "P7120",
                 "P7150", "P7160", "P7310", "P7350", "P7422", "P7472", "P7495", "P7500s2", "P7500s3",
                 "P7505", "P7510s1", "P7510s2", "P7510s3", "P7510s5", "P7510s6", "P7510s7",
                 "Pet", "Oc", "Des", "Ina")
  per |>
    dplyr::filter(!P6050 %in% c(6, 7, 8)) |>
    dplyr::select(id, Orden, sample, Clase, Dominio, Depto, dplyr::all_of(vars_test),
                  dplyr::any_of("Ingtot")) |>
    dplyr::left_join(dplyr::select(hog, id, fold, Lp, P5090), by = "id") |>
    dplyr::mutate(Dominio = as.character(Dominio))
}

#' Variables de hogar de ingeniería (no usan el ingreso de train)
#'
#' @description Añade a la base de hogares: la estructura de la regla de pobreza
#' (arriendo imputado, ingreso laboral requerido), hacinamiento y vivienda,
#' composición del hogar, capital humano, inserción laboral, índices de privación
#' e interacciones.
#' @param .db tibble con las columnas de `fun_preprocess_hogares()`,
#' `fun_collapse_personas()` y `fun_add_ratios()`.
#' @returns tibble con las variables nuevas (prefijos `fe_`).
fun_add_engineered <- function(.db) {
  .db |>
    dplyr::mutate(
      # --- A. Estructura de la regla de pobreza -----------------------------------
      # Pobre = 1(Ingtotug + arriendo imputado < Lp * Npersug): el arriendo imputado
      # es conocido, así que solo falta el ingreso de las personas.
      fe_arr_imp        = fun_arriendo_imputado(P5090, P5130, P5100),
      fe_req_labor      = pmax(req_income - fe_arr_imp, 0),     # ingreso de personas que necesita el hogar
      fe_arr_share      = fe_arr_imp / req_income,             # fracción de la línea cubierta por vivienda propia
      fe_rent_covers    = as.integer(fe_arr_imp >= req_income),  # no pobre con certeza si es 1
      fe_log_req_labor  = log1p(fe_req_labor),
      fe_lp_li_ratio    = Lp / Li,                              # distancia entre línea de pobreza e indigencia
      # --- B. Hacinamiento y vivienda ---------------------------------------------
      fe_persons_room   = Nper / pmax(P5000, 1),                # personas por cuarto
      fe_bedrooms_pc    = P5010 / Nper,
      fe_crowd_sq       = crowding^2,
      fe_crowd_gt3      = as.integer(crowding > 3),
      fe_one_room       = as.integer(P5000 <= 1),
      fe_rent_to_line   = rent_est / Lp,                        # arriendo en múltiplos de la línea per cápita
      fe_owner          = as.integer(P5090 %in% c(1, 2)),
      fe_precarious_tenure = as.integer(P5090 %in% c(5, 6)),
      # --- C. Composición ---------------------------------------------------------
      fe_hh_size_sq     = Npersug^2,
      fe_adult_equiv    = 1 + 0.7 * pmax(n_adult + n_senior - 1, 0) + 0.5 * (n_child_0_5 + n_child_6_17),  # escala OECD modificada
      fe_adult_equiv_ratio = fe_adult_equiv / Npersug,
      fe_young_kid_share = n_child_0_5 / Npersug,
      fe_senior_share   = n_senior / Npersug,
      fe_single_adult   = as.integer(n_adult + n_senior == 1),
      fe_single_parent_f = as.integer(n_adult + n_senior == 1 & head_female == 1 & (n_child_0_5 + n_child_6_17) > 0),
      fe_only_seniors   = as.integer(n_adult == 0 & n_senior > 0),
      fe_age_gap_head   = head_age - age_mean,
      # --- D. Capital humano ------------------------------------------------------
      fe_educ_gap_head  = head_educ - educ_mean,
      fe_educ_max_mean  = educ_max - educ_mean,
      fe_share_superior = n_superior / Npersug,
      fe_share_no_educ  = n_no_educ / Npersug,
      fe_head_educ_low  = as.integer(head_educ <= 5),
      fe_educ_x_adults  = educ_mean * (n_adult / Npersug),     # educación ponderada por adultos en edad de trabajar
      # --- E. Inserción laboral ---------------------------------------------------
      fe_workers_per_dep = n_employed / pmax(Npersug - n_employed, 1),   # ocupados por cada no ocupado
      fe_unemp_share    = n_unemployed / pmax(n_employed + n_unemployed, 1),
      fe_inactive_share = n_inactive / Npersug,
      fe_selfemp_share  = n_selfemp / pmax(n_employed, 1),
      fe_wageemp_share  = n_wage_emp / pmax(n_employed, 1),
      fe_unpaid_share   = n_unpaid / pmax(n_employed, 1),
      fe_bigfirm_share  = n_big_firm / pmax(n_employed, 1),
      fe_hours_per_worker = hours_total / pmax(n_employed, 1),
      fe_underemployed  = as.integer(n_more_hours > 0),
      fe_multi_job      = as.integer(n_second_job > 0),
      fe_head_only_earner = as.integer(head_employed == 1 & n_employed == 1),
      fe_no_head_earner = as.integer(head_employed == 0 & n_employed > 0),
      fe_pension_no_earner = as.integer(n_pension > 0 & n_employed == 0),     # jubilados sin ocupados: baja pobreza
      fe_pension_pc     = n_pension / Npersug,
      fe_transfers      = n_subsidy + remit_nat + remit_int + aid_inst + interest + n_other_inc,
      fe_transfers_pc   = fe_transfers / Npersug,
      fe_remit_any      = as.integer(remit_nat + remit_int > 0),
      # --- F. Índices de privación (conteo de señales de pobreza) -----------------
      fe_deprivation    = fe_crowd_gt3 + no_earner + fe_head_educ_low + as.integer(formal_share == 0) +
                          as.integer(share_insured < 1) + as.integer(n_subsid_regime > 0) +
                          as.integer(rent_burden > 0.8) + fe_one_room + as.integer(dep_ratio >= 2),
      fe_security       = as.integer(n_pension > 0) + as.integer(head_formal == 1) +
                          as.integer(n_superior > 0) + fe_owner + as.integer(n_big_firm > 0) +
                          as.integer(remit_int > 0),
      # --- G. Interacciones y no linealidades -------------------------------------
      fe_crowd_x_noearn = crowding * (1 - pmin(earners_pc * 2, 1)),
      fe_dep_x_educ     = dep_ratio * (15 - head_educ),
      fe_size_x_noearn  = Npersug * no_earner,
      fe_rural_x_crowd  = (1 - urban) * crowding,
      fe_rural_x_noearn = (1 - urban) * no_earner,
      fe_kids_x_single  = kids_pc * fe_single_adult,
      fe_earners_sq     = earners_pc^2,
      fe_log_hours_pc   = log1p(hours_pc),
      fe_rentburden_x_rent = rent_burden * as.integer(P5090 == 3)
    )
}

#' Agregar el ingreso imputado fuera de muestra al hogar
#'
#' @description A partir de las predicciones por persona (modelo auxiliar de
#' ingreso en Python) calcula, por hogar, el ingreso imputado de la UG y su
#' distancia a la línea de pobreza una vez descontado el arriendo imputado.
#' @param .db tibble de hogares con las variables de `fun_add_engineered()`.
#' @param pred tibble por persona con `id`, `p_pos`, `mu_log`, `mean_raw`,
#' `bin_head` (1 si es el jefe).
#' @returns tibble con las variables `fe_inc_*`.
fun_add_income_hat <- function(.db, pred) {
  agg <- pred |>
    dplyr::mutate(inc_cond = p_pos * exp(mu_log)) |>
    dplyr::group_by(id) |>
    dplyr::summarise(
      fe_inc_hat_sum   = sum(mean_raw),
      fe_inc_hat_cond  = sum(inc_cond),
      fe_inc_hat_max   = max(mean_raw),
      fe_inc_hat_head  = sum(mean_raw * bin_head),
      fe_earners_hat   = sum(p_pos),
      fe_earners_hat_max = max(p_pos),
      .groups = "drop")

  .db |>
    dplyr::left_join(agg, by = "id") |>
    dplyr::mutate(
      across(c(fe_inc_hat_sum, fe_inc_hat_cond, fe_inc_hat_max, fe_inc_hat_head,
               fe_earners_hat, fe_earners_hat_max), ~ dplyr::coalesce(.x, 0)),
      # Cociente ingreso imputado + arriendo imputado sobre la línea de la UG
      fe_ratio_hat      = (fe_inc_hat_sum + fe_arr_imp) / req_income,
      fe_ratio_hat_cond = (fe_inc_hat_cond + fe_arr_imp) / req_income,
      fe_log_ratio_hat  = log(pmax(fe_ratio_hat, 0.02)),
      fe_log_ratio_hat_cond = log(pmax(fe_ratio_hat_cond, 0.02)),
      fe_gap_hat        = (fe_inc_hat_sum - fe_req_labor) / req_income,   # < 0 => ingreso imputado no cubre la línea
      fe_gap_hat_cond   = (fe_inc_hat_cond - fe_req_labor) / req_income,
      fe_inc_hat_pc     = fe_inc_hat_sum / Npersug,
      fe_inc_hat_head_share = fe_inc_hat_head / pmax(fe_inc_hat_sum, 1),
      fe_earners_hat_pc = fe_earners_hat / Npersug
    )
}
