# ==============================================================================
# 17_harold_slide_assets.R
# Figuras y tablas de las diapositivas (03_slides/). No entrena nada: lee las bases y los puntajes
# fuera de muestra (OOF) de los modelos ya entrenados y arma
#   - figuras de tamaño de diapositiva  -> 02_outputs/figures/harold_slide_*.png
#   - tablas en LaTeX (solo `tabular`)   -> 03_slides/tables/*.tex
#   - cifras citadas en el texto         -> 03_slides/numbers.tex (macros \newcommand)
# Así, ninguna cifra de las diapositivas se escribe a mano.
# Entradas : hogares_fe.rds, harold_*_scores.csv, tables/harold_*.csv
# ==============================================================================
AUTOR <- "harold"
suppressPackageStartupMessages({library(data.table); library(tidyverse)})
options(OutDec = ",")   # decimales con coma en los ejes
dir.create("03_slides/tables", recursive = TRUE, showWarnings = FALSE)

# --- Estilo común --------------------------------------------------------------------------------------
PAL <- c(azul = "#1B3A6B", naranja = "#E07B00", verde = "#2A9D8F", gris = "#8D99AE", rojo = "#C0392B", claro = "#A9BCD9")
theme_slide <- function() {
  theme_minimal(base_size = 15) +
    theme(plot.title = element_text(face = "bold", size = 16), plot.title.position = "plot",
          plot.subtitle = element_text(colour = "grey30", size = 12.5), legend.position = "bottom",
          panel.grid.minor = element_blank(), strip.text = element_text(face = "bold", size = 13),
          plot.caption = element_text(colour = "grey40", size = 10))
}
guardar <- function(p, nombre, w = 9, h = 5) ggsave(file.path(DIR_FIG, paste0(AUTOR, "_slide_", nombre, ".png")), p,
                                                     width = 0.85 * w, height = 0.85 * h, dpi = 200, device = ragg::agg_png)   # 0,85: letra más grande en la diapositiva
fmtn <- function(x) format(x, big.mark = ".", decimal.mark = ",", scientific = FALSE, trim = TRUE)
pct <- function(x, d = 1) formatC(100 * x, format = "f", digits = d, decimal.mark = ",")
num <- function(x, d = 3) formatC(x, format = "f", digits = d, decimal.mark = ",")
etiquetas <- c(fe_log_ratio_hat_cond = "Log razón ingreso imputado / línea", fe_ratio_hat_cond = "Razón ingreso imputado / línea",
  fe_gap_hat_cond = "Brecha de ingreso imputado (% de la línea)", n_subsidy = "Personas con subsidios",
  fe_arr_share = "Arriendo imputado / línea", fe_inc_hat_cond = "Ingreso imputado del hogar", fe_arr_imp = "Arriendo imputado",
  Dominio = "Dominio geográfico", fe_earners_hat_pc = "Perceptores esperados per cápita", fe_earners_hat_max = "Mayor prob. de percibir ingreso",
  fe_req_labor = "Ingreso laboral requerido", fe_ratio_hat = "Razón ingreso imputado (suma) / línea", fe_rent_to_line = "Arriendo / línea",
  fe_inc_hat_max = "Mayor ingreso individual imputado", fe_earners_hat = "Perceptores esperados", fe_wageemp_share = "% de ocupados asalariados",
  hours_pc = "Horas trabajadas per cápita", n_wage_emp = "Asalariados", fe_pension_pc = "Pensionados per cápita", n_formal = "Ocupados formales",
  fe_rent_covers = "Arriendo imputado cubre la línea", fe_inc_hat_pc = "Ingreso imputado per cápita", fe_inc_hat_sum = "Ingreso imputado (suma)",
  fe_arr_share_ns1 = "Arriendo / línea (spline 1)", head_educ = "Educación del jefe", educ_mean = "Educación media del hogar",
  crowding = "Hacinamiento", Npersug = "Tamaño del hogar", n_pension = "Pensionados", fe_earners_sq = "Perceptores per cápita al cuadrado",
  n_employed = "Ocupados", rent_pc = "Arriendo per cápita", fe_log_req_labor = "Log ingreso laboral requerido", req_income = "Línea de pobreza del hogar",
  fe_deprivation = "Índice de privaciones", earners_pc = "Perceptores per cápita", fe_unemp_share = "% desocupados")
et <- function(v) ifelse(v %in% names(etiquetas), etiquetas[v], v)

# --- Datos y puntajes -----------------------------------------------------------------------------------
fe  <- readRDS(file.path(DIR_PROC, "hogares_fe.rds"))
trn <- fe |> filter(sample == "train"); y <- trn$poor; nb <- trn$bogota == 0
oof <- function(arch) { sc <- fread(file.path(DIR_PROC, paste0(AUTOR, "_", arch, "_scores.csv"))) |> as_tibble(); sc$score[match(trn$id, sc$id)] }
s_lgb <- oof("lightgbm_fe"); cv_l <- read_csv(file.path(DIR_TAB, paste0(AUTOR, "_lightgbm_fe_cv.csv")), show_col_types = FALSE)
thr_l <- cv_l$umbral
num_macros <- list()
mac <- function(nombre, valor) num_macros[[nombre]] <<- as.character(valor)

# =========================== 1. Gradientes de pobreza (datos) =========================================
tasa <- function(df, var, etiqueta) df |> group_by(grupo = {{ var }}) |>
  summarise(n = n(), pobre = mean(poor), .groups = "drop") |> mutate(panel = etiqueta)
d1 <- trn |> filter(bogota == 0)
g1 <- bind_rows(
  d1 |> mutate(g = pmin(Npersug, 7)) |> tasa(g, "Tamaño del hogar (7 = 7 o más)") |> mutate(grupo = as.character(grupo)),
  d1 |> mutate(g = factor(head_educ, c(0, 5, 9, 11, 15), c("Ninguna", "Primaria", "Básica", "Media", "Superior"))) |> tasa(g, "Educación del jefe") |> mutate(grupo = as.character(grupo)),
  d1 |> mutate(g = cut(crowding, c(0, 1, 2, 3, Inf), c("≤ 1", "1–2", "2–3", "> 3"))) |> tasa(g, "Personas por cuarto (hacinamiento)") |> mutate(grupo = as.character(grupo)),
  d1 |> mutate(g = recode(as.character(tenure), propia_pagada = "Propia pagada", propia_pagando = "Propia pagando", arriendo = "Arriendo",
                          usufructo = "Usufructo", ocupante = "Ocupante", otra = "Otra")) |> filter(g != "Otra") |> tasa(g, "Tenencia de la vivienda") |> mutate(grupo = as.character(grupo)))
orden <- g1 |> group_by(panel) |> mutate(o = row_number()) |> ungroup()
ord_tenencia <- c("Propia pagada", "Propia pagando", "Arriendo", "Usufructo", "Ocupante")
g1 <- g1 |> mutate(panel = factor(panel, levels = c("Tamaño del hogar (7 = 7 o más)", "Educación del jefe", "Personas por cuarto (hacinamiento)", "Tenencia de la vivienda")), grupo = factor(grupo, levels = unique(c(as.character(1:7), "Ninguna", "Primaria", "Básica", "Media", "Superior", "≤ 1", "1–2", "2–3", "> 3", ord_tenencia))))
p1 <- ggplot(g1, aes(grupo, pobre)) + geom_col(fill = PAL["azul"], width = 0.7) +
  geom_hline(yintercept = mean(d1$poor), linetype = 2, colour = PAL["naranja"]) +
  geom_text(aes(label = paste0(pct(pobre, 0), " %")), vjust = -0.3, size = 3.8) + facet_wrap(~panel, scales = "free_x") +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1), expand = expansion(mult = c(0, 0.15))) +
  labs(x = NULL, y = "Hogares pobres", title = "La pobreza crece con el tamaño del hogar, el hacinamiento y la baja educación",
       subtitle = "Tasa de pobreza en train (sin Bogotá); la línea naranja es la tasa promedio", caption = "Fuente: GEIH-MESE, DANE; cálculos propios.") + theme_slide() +
  scale_x_discrete(labels = \(x) str_wrap(x, 9)) + theme(axis.text.x = element_text(size = 11))
guardar(p1, "gradients", 10, 6)
mac("PobreTrain", pct(mean(d1$poor))); mac("NTrain", fmtn(nrow(trn))); mac("NTest", fmtn(sum(fe$sample == "test")))

# =========================== 2. Razón ingreso imputado / línea por clase ===============================
d2 <- d1 |> transmute(poor = factor(poor, 0:1, c("No pobre", "Pobre")), x = pmin(pmax(fe_log_ratio_hat_cond, -2.2), 2.2))
p2 <- ggplot(d2, aes(x, fill = poor, colour = poor)) + geom_density(alpha = 0.35, linewidth = 0.8, adjust = 1.2) +
  geom_vline(xintercept = 0, linetype = 2) + annotate("text", x = 0.05, y = Inf, label = "ingreso imputado = línea", hjust = 0, vjust = 1.6, size = 4.2) +
  scale_fill_manual(values = c("No pobre" = PAL[["azul"]], "Pobre" = PAL[["naranja"]])) + scale_colour_manual(values = c("No pobre" = PAL[["azul"]], "Pobre" = PAL[["naranja"]])) +
  labs(x = "Log de la razón (ingreso imputado + arriendo imputado) / línea de pobreza del hogar", y = "Densidad", fill = NULL, colour = NULL,
       title = "El ingreso imputado separa a los pobres de los no pobres",
       subtitle = "Ingreso por persona predicho fuera de muestra con variables que existen en test") + theme_slide()
guardar(p2, "ratio_density", 9, 5)
q <- read_lines(file.path(DIR_TAB, "harold_person_income_quality.tex"))
mac("AUCRazon", sub(".*AUC solo con la razón imputada & ([0-9.]+).*", "\\1", q[grep("AUC solo con la raz", q)]) |> as.numeric() |> num())
mac("CorrIng", sub(".*Corr. con Ingtotug \\(nivel\\) & ([0-9.]+).*", "\\1", q[grep("nivel", q)]) |> as.numeric() |> num(2))
mac("AUCHacin", sub(".*AUC solo con hacinamiento & ([0-9.]+).*", "\\1", q[grep("hacinamiento", q)]) |> as.numeric() |> num())

# =========================== 3. Ablación por grupos de variables =============================================
ab <- read_csv(file.path(DIR_TAB, "harold_fe_ablation.csv"), show_col_types = FALSE) |>
  mutate(conjunto = recode(conjunto, "A base" = "A. Variables base", "A + B (ratios e interacciones)" = "A + B. Razones e interacciones",
                           "A + C (estructura de la regla)" = "A + C. Regla de pobreza y arriendo", "A + D (ingreso imputado)" = "A + D. Ingreso imputado",
                           "A + C + D" = "A + C + D", "A + B + C + D (todas)" = "A + B + C + D. Todas"),
         d = str_detect(conjunto, "D"))
ab$conjunto <- factor(ab$conjunto, levels = rev(ab$conjunto))
p3 <- ggplot(ab, aes(f1, conjunto, colour = d)) + geom_segment(aes(x = 0.74, xend = f1, yend = conjunto), linewidth = 1.2) + geom_point(size = 4.5) +
  geom_text(aes(label = num(f1, 3)), nudge_x = 0.0028, size = 4.6, show.legend = FALSE) +
  scale_colour_manual(values = c(`FALSE` = PAL[["gris"]], `TRUE` = PAL[["naranja"]]), labels = c(`FALSE` = "Sin ingreso imputado", `TRUE` = "Con ingreso imputado"), name = NULL) +
  scale_x_continuous(limits = c(0.74, 0.786)) +
  labs(x = "F1 de la clase pobre (LightGBM, hiperparámetros fijos)", y = NULL,
       title = "El ingreso imputado aporta casi todo el salto de F1", subtitle = "Ablación por grupos de variables") + theme_slide()
guardar(p3, "ablation", 9.5, 4.8)
mac("FoneA", num(ab$f1[ab$conjunto == "A. Variables base"])); mac("FoneAD", num(ab$f1[ab$conjunto == "A + D. Ingreso imputado"]))
mac("FoneAll", num(ab$f1[str_detect(ab$conjunto, "Todas")])); mac("FoneAB", num(ab$f1[str_detect(ab$conjunto, "^A \\+ B\\.")]))
mac("FoneAC", num(ab$f1[str_detect(ab$conjunto, "^A \\+ C\\.")]))
mac("FoneAblAtHalf", num(ab$f1_umbral_050[str_detect(ab$conjunto, "Todas")]))

# =========================== 4. Búsqueda de hiperparámetros (LightGBM) ======================================
bs <- read_csv(file.path(DIR_TAB, "harold_lightgbm_fe_search.csv"), show_col_types = FALSE) |>
  mutate(elegida = f1 == max(f1), etiqueta = sprintf("hojas %d · mín. %d · col. %.1f · L2 %g · pos. %.1f", num_leaves, min_child_samples, feature_fraction, lambda_l2, scale_pos_weight))
bs$etiqueta <- factor(bs$etiqueta, levels = bs$etiqueta[order(bs$f1)])
p4 <- ggplot(bs, aes(f1, etiqueta, colour = elegida)) + geom_segment(aes(x = 0.772, xend = f1, yend = etiqueta), linewidth = 0.9) + geom_point(size = 3.8) +
  scale_colour_manual(values = c(`FALSE` = PAL[["gris"]], `TRUE` = PAL[["naranja"]]), guide = "none") +
  scale_x_continuous(limits = c(0.772, 0.7785)) +
  labs(x = "F1 en validación cruzada (tasa de aprendizaje 0,08 con parada temprana)", y = NULL,
       title = "11 configuraciones aleatorias: el F1 casi no depende de los hiperparámetros",
       subtitle = "Hojas, mínimo de hogares por hoja, fracción de columnas, penalización L2 y peso de la clase pobre") + theme_slide() + theme(axis.text.y = element_text(size = 10.5))
guardar(p4, "hyper_search", 10, 5.2)
mac("FoneSearchMin", num(min(bs$f1), 4)); mac("FoneSearchMax", num(max(bs$f1), 4)); mac("FoneRef", num(bs$f1[bs$config == 0], 4))
mac("NConfigs", nrow(bs))

# =========================== 5. Importancia y efectos (LightGBM) ============================================
imp <- read_csv(file.path(DIR_TAB, "harold_lightgbm_fe_importance.csv"), show_col_types = FALSE) |> mutate(lab = et(variable))
gnom <- c(A = "Variables base", B = "Razones e interacciones", C = "Regla de pobreza y arriendo", D = "Ingreso imputado")
pi_ <- imp |> slice_head(n = 12) |> mutate(lab = fct_reorder(lab, shap_media_abs), g = gnom[grupo])
p5 <- ggplot(pi_, aes(shap_media_abs, lab, fill = g)) + geom_col(width = 0.75) +
  scale_fill_manual(values = c("Ingreso imputado" = PAL[["naranja"]], "Regla de pobreza y arriendo" = PAL[["verde"]], "Variables base" = PAL[["azul"]], "Razones e interacciones" = PAL[["gris"]]), name = NULL) +
  labs(x = "Valor SHAP absoluto medio (log-odds de ser pobre)", y = NULL, title = "Las variables de ingreso imputado concentran la importancia",
       subtitle = "12 variables más importantes del LightGBM; 15.000 hogares de train") + theme_slide() + guides(fill = guide_legend(nrow = 2))
guardar(p5, "importance", 9.5, 5.5)
sh_g <- imp |> mutate(top3 = rank(-shap_media_abs) <= 3) ; tot <- sum(imp$shap_media_abs)
mac("ShareSHAPD", pct(sum(imp$shap_media_abs[imp$grupo == "D"]) / tot, 0))
mac("ShareSHAPTopThree", pct(sum(sort(imp$shap_media_abs, decreasing = TRUE)[1:3]) / tot, 0))

sh <- read_csv(file.path(DIR_PROC, paste0(AUTOR, "_lightgbm_fe_shap_sample.csv")), show_col_types = FALSE) |>
  filter(variable %in% c("fe_log_ratio_hat_cond", "fe_gap_hat_cond", "n_subsidy", "fe_arr_share"), !is.na(valor))
ef <- sh |> group_by(variable) |> mutate(tramo = if (n_distinct(valor) > 12) cut(valor, unique(quantile(valor, seq(0, 1, 0.05))), include.lowest = TRUE) else factor(valor)) |>
  group_by(variable, tramo) |> summarise(x = median(valor), shap = mean(shap), .groups = "drop") |>
  mutate(panel = factor(et(variable), levels = et(c("fe_log_ratio_hat_cond", "fe_gap_hat_cond", "n_subsidy", "fe_arr_share"))))
p6 <- ggplot(ef, aes(x, shap)) + geom_hline(yintercept = 0, colour = "grey60") + geom_line(colour = PAL["azul"], linewidth = 1) + geom_point(colour = PAL["azul"], size = 2) +
  facet_wrap(~panel, scales = "free_x") +
  labs(x = "Valor de la variable (mediana de cada vigintil)", y = "Efecto SHAP medio (log-odds de ser pobre)",
       title = "Efecto de las variables principales sobre la probabilidad de pobreza", subtitle = "Sobre cero aumenta la probabilidad de ser pobre") + theme_slide()
guardar(p6, "effects", 10, 5.8)

# =========================== 6. Umbral y errores de política (LightGBM) ======================================
grilla <- seq(0.05, 0.80, by = 0.01)
cu <- map_dfr(grilla, function(t) { cm <- conf_metrics(y[nb], as.integer(s_lgb[nb] >= t)); tibble(umbral = t, `Subcobertura: pobres excluidos` = cm$undercoverage, `Filtración: beneficiarios no pobres` = cm$leakage, F1 = cm$f1) })
cu_l <- cu |> pivot_longer(-umbral) |> mutate(panel = if_else(name == "F1", "F1 de la clase pobre", "Errores de política"))
p7 <- ggplot(cu_l, aes(umbral, value, colour = name)) + geom_line(linewidth = 1.2) + geom_vline(xintercept = thr_l, linetype = 2) +
  facet_wrap(~panel, scales = "free_y") +
  scale_colour_manual(values = c("F1" = PAL[["azul"]], "Subcobertura: pobres excluidos" = PAL[["naranja"]], "Filtración: beneficiarios no pobres" = PAL[["verde"]]), name = NULL) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(x = "Umbral de probabilidad para clasificar un hogar como pobre", y = NULL,
       title = "Subir el umbral reduce la filtración pero deja fuera a más pobres",
       subtitle = paste0("LightGBM, fuera de muestra; línea punteada: umbral que maximiza el F1 (", num(thr_l, 2), ")")) + theme_slide() + guides(colour = guide_legend(nrow = 2))
guardar(p7, "threshold_tradeoff", 10, 5)
cm_l <- conf_metrics(y[nb], as.integer(s_lgb[nb] >= thr_l))
mac("ThrLGBM", num(thr_l, 2)); mac("FoneLGBM", num(cm_l$f1)); mac("PrecLGBM", num(cm_l$precision)); mac("RecLGBM", num(cm_l$recall))
mac("LeakLGBM", pct(cm_l$leakage)); mac("UnderLGBM", pct(cm_l$undercoverage)); mac("AUCLGBM", num(fun_auc(y[nb], s_lgb[nb])))
mac("FoneLGBMsd", num(cv_l$f1_sd_pliegues, 4)); mac("PctPredPobre", pct(cv_l$pct_pred_pobre)); mac("PctRealPobre", pct(cv_l$pct_pobre_real))
f1_050 <- f1_score(y[nb], as.integer(s_lgb[nb] >= 0.5)); mac("FoneLGBMAtHalf", num(f1_050)); mac("GainThr", num(cm_l$f1 - f1_050, 3))
mac("UnderAtHalf", pct(conf_metrics(y[nb], as.integer(s_lgb[nb] >= 0.5))$undercoverage)); mac("LeakAtHalf", pct(conf_metrics(y[nb], as.integer(s_lgb[nb] >= 0.5))$leakage))
# Umbral que lleva la subcobertura a 10 %
t10 <- cu$umbral[which.min(abs(cu$`Subcobertura: pobres excluidos` - 0.10))]; l10 <- cu$`Filtración: beneficiarios no pobres`[cu$umbral == t10]
mac("ThrUnderTen", num(t10, 2)); mac("LeakAtUnderTen", pct(l10))
# Tabla de confusión
writeLines(c("\\begin{tabular}{lrr}", "\\toprule", " & \\textbf{Predicho pobre} & \\textbf{Predicho no pobre} \\\\", "\\midrule",
             sprintf("\\textbf{Pobre} & %s (VP) & %s (FN) \\\\", fmtn(cm_l$tp), fmtn(cm_l$fn)),
             sprintf("\\textbf{No pobre} & %s (FP) & %s (VN) \\\\", fmtn(cm_l$fp), fmtn(cm_l$tn)),
             "\\bottomrule", "\\end{tabular}"), "03_slides/tables/confusion_lgbm.tex")

# =========================== 7. Errores por grupo ====================================================================
d7 <- trn |> mutate(yhat = as.integer(s_lgb >= thr_l)) |> filter(bogota == 0)
grp <- function(df, g, nombre) df |> mutate(grupo = {{ g }}) |> group_by(grupo) |>
  summarise(sub_k = sum(poor == 1 & yhat == 0), sub_n = sum(poor == 1), fil_k = sum(poor == 0 & yhat == 1), fil_n = sum(yhat == 1), .groups = "drop") |> mutate(variable = nombre)
e7 <- bind_rows(
  grp(d7, if_else(urban == 1, "Cabecera", "Resto rural"), "Zona"),
  grp(d7, if_else(head_female == 1, "Jefatura femenina", "Jefatura masculina"), "Sexo del jefe"),
  grp(d7, cut(Npersug, c(0, 2, 4, Inf), c("1–2 personas", "3–4 personas", "5 o más")), "Tamaño del hogar"),
  grp(d7, factor(case_when(tenure %in% c("propia_pagada", "propia_pagando") ~ "Propia", tenure == "arriendo" ~ "Arriendo", TRUE ~ "Usufructo u otra"), c("Propia", "Arriendo", "Usufructo u otra")), "Tenencia")) |>
  mutate(grupo = as.character(grupo), Subcobertura = sub_k / sub_n, Filtración = fil_k / fil_n,
         s_lo = wilson_ci(sub_k, sub_n)$lo, s_hi = wilson_ci(sub_k, sub_n)$hi, f_lo = wilson_ci(fil_k, fil_n)$lo, f_hi = wilson_ci(fil_k, fil_n)$hi)
e7l <- bind_rows(e7 |> transmute(variable, grupo, medida = "Subcobertura (pobres excluidos)", v = Subcobertura, lo = s_lo, hi = s_hi),
                 e7 |> transmute(variable, grupo, medida = "Filtración (beneficiarios no pobres)", v = Filtración, lo = f_lo, hi = f_hi)) |>
  mutate(grupo = factor(grupo, levels = rev(unique(grupo))), variable = factor(variable, levels = c("Zona", "Sexo del jefe", "Tamaño del hogar", "Tenencia")))
p8 <- ggplot(e7l, aes(v, grupo, colour = medida)) + geom_pointrange(aes(xmin = lo, xmax = hi), position = position_dodge(width = 0.5), size = 0.55) +
  geom_vline(data = tibble(medida = c("Subcobertura (pobres excluidos)", "Filtración (beneficiarios no pobres)"), x = c(cm_l$undercoverage, cm_l$leakage)),
             aes(xintercept = x, colour = medida), linetype = 3, show.legend = FALSE) +
  facet_wrap(~variable, scales = "free_y", ncol = 2) +
  scale_colour_manual(values = c("Subcobertura (pobres excluidos)" = PAL[["naranja"]], "Filtración (beneficiarios no pobres)" = PAL[["verde"]]), name = NULL) +
  scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(x = "Tasa de error (intervalo de Wilson al 95 %)", y = NULL, title = "Los errores no se reparten parejo entre tipos de hogar",
       subtitle = "LightGBM con el umbral del F1; líneas punteadas: promedio general") + theme_slide() + guides(colour = guide_legend(nrow = 1))
guardar(p8, "errors_by_group", 10, 5.8)
for (i in which(e7$variable != "Tamaño del hogar")) mac(paste0("Under", gsub("[^A-Za-z]", "", e7$grupo[i])), pct(e7$Subcobertura[i]))

# =========================== 8. Encuesta corta ===========================================================================
ss <- read_csv(file.path(DIR_TAB, "harold_short_survey_cv.csv"), show_col_types = FALSE) |>
  mutate(nombre = recode(conjunto, S1 = "Encuesta mínima\nvivienda, hogar y educación", S2 = "Encuesta corta\n+ empleo y salud", Completo = "Modelo completo\n+ ingreso imputado"))
ss$nombre <- factor(ss$nombre, levels = ss$nombre)
p9 <- ggplot(ss, aes(nombre, f1, fill = conjunto)) + geom_col(width = 0.65) + geom_text(aes(label = paste0("F1 = ", num(f1, 3), "\n", n_vars, " variables")), vjust = -0.25, size = 4.6) +
  scale_fill_manual(values = c(S1 = PAL[["gris"]], S2 = PAL[["claro"]], Completo = PAL[["naranja"]]), guide = "none") +
  scale_y_continuous(limits = c(0, 0.9), expand = expansion(mult = c(0, 0.02))) +
  labs(x = NULL, y = "F1 de la clase pobre", title = "Una encuesta corta cuesta más de 10 puntos de F1",
       subtitle = "LightGBM con los mismos hiperparámetros; 5 pliegues, sin Bogotá") + theme_slide() + theme(axis.text.x = element_text(size = 12))
guardar(p9, "short_survey", 9.5, 5.2)
cod <- c(S1 = "Min", S2 = "Short", Completo = "Full")
for (i in seq_len(nrow(ss))) { k <- cod[[as.character(ss$conjunto[i])]]; mac(paste0("FoneSurvey", k), num(ss$f1[i])); mac(paste0("UnderSurvey", k), pct(ss$subcobertura[i], 0)); mac(paste0("LeakSurvey", k), pct(ss$filtracion[i], 0)); mac(paste0("NVarsSurvey", k), ss$n_vars[i]) }

# =========================== 9. Siete algoritmos con las mismas 40 variables ==============================================
sp <- read_csv(file.path(DIR_TAB, "harold_same_predictors_cv.csv"), show_col_types = FALSE)
sp$hiperparametros <- gsub("([0-9])\\.([0-9])", "\\1,\\2", sp$hiperparametros)
sp$algoritmo <- factor(sp$algoritmo, levels = rev(sp$algoritmo))
p10 <- ggplot(sp, aes(f1, algoritmo)) + geom_errorbarh(aes(xmin = f1 - f1_sd_pliegues, xmax = f1 + f1_sd_pliegues), height = 0.25, colour = PAL["gris"], linewidth = 0.8) +
  geom_point(size = 4.5, colour = PAL["azul"]) + geom_text(aes(label = num(f1, 3)), nudge_y = 0.33, size = 4.4) +
  labs(x = "F1 de la clase pobre (media ± 1 DE entre pliegues)", y = NULL,
       title = "Con las mismas variables, seis de siete algoritmos quedan a 1 punto de F1",
       subtitle = "Mismas 40 variables, mismos pliegues, umbral que maximiza el F1; fuera de Bogotá") + theme_slide()
guardar(p10, "same_predictors", 9.5, 5)
for (i in seq_len(nrow(sp))) mac(paste0("SP", gsub("[^A-Za-z]", "", as.character(sp$algoritmo[i]))), num(sp$f1[i]))
mac("SPRange", num(max(sp$f1) - min(sp$f1[sp$algoritmo != "Naive Bayes"]), 3))

# =========================== 10. ¿Qué mueve el F1? ====================================================================
mv <- tribble(
  ~efecto, ~ganancia,
  "Ingreso imputado (A → A + D)", ab$f1[ab$conjunto == "A + D. Ingreso imputado"] - ab$f1[ab$conjunto == "A. Variables base"],
  "Umbral óptimo en lugar de 0,5", cm_l$f1 - f1_050,
  "Algoritmo: Logit → LightGBM (mismas variables)", sp$f1[sp$algoritmo == "Boosting (LightGBM)"] - sp$f1[sp$algoritmo == "Logit"],
  "Hiperparámetros: mejor vs. peor de 11 configuraciones", max(bs$f1) - min(bs$f1),
  "Razones e interacciones (A → A + B)", ab$f1[ab$conjunto == "A + B. Razones e interacciones"] - ab$f1[ab$conjunto == "A. Variables base"])
mv$efecto <- factor(mv$efecto, levels = rev(mv$efecto[order(-mv$ganancia)]))
p11 <- ggplot(mv, aes(ganancia, efecto)) + geom_col(fill = PAL["azul"], width = 0.65) +
  geom_text(aes(label = paste0("+", if_else(ganancia < 0.001, num(ganancia, 4), num(ganancia, 3)))), hjust = -0.15, size = 4.8) + scale_x_continuous(limits = c(0, 0.034), expand = expansion(mult = c(0, 0))) +
  labs(x = "Ganancia de F1 (puntos de F1)", y = NULL, title = "Los datos pesan más que el algoritmo",
       subtitle = "Ganancia de F1 asociada a cada decisión de modelado") + theme_slide()
guardar(p11, "what_moves_f1", 10, 4.8)
mac("GainIncome", num(mv$ganancia[1], 3)); mac("GainAlgo", num(mv$ganancia[3], 3)); mac("GainHyper", num(mv$ganancia[4], 4))

# =========================== 11. Linealidad de la variable clave =====================================================
x <- pmin(pmax(d1$fe_log_ratio_hat_cond, -2), 2); yy <- d1$poor; sl <- s_lgb[nb]
brk <- unique(quantile(x, seq(0, 1, length.out = 41))); bin <- cut(x, brk, include.lowest = TRUE)
bd <- tibble(x, yy, sl, bin) |> group_by(bin) |> summarise(x = mean(x), Observada = mean(yy), LightGBM = mean(sl), .groups = "drop")
glm_l <- glm(yy ~ x, family = binomial); glm_n <- glm(yy ~ splines::ns(x, 4), family = binomial)
bd$`Logit lineal en la variable` <- predict(glm_l, tibble(x = bd$x), type = "response"); bd$`Logit con splines` <- predict(glm_n, tibble(x = bd$x), type = "response")
p12 <- bd |> pivot_longer(c(LightGBM, `Logit lineal en la variable`, `Logit con splines`)) |> ggplot(aes(x)) +
  geom_line(aes(y = value, colour = name, linetype = name), linewidth = 1.1) + geom_point(aes(y = Observada), colour = "black", size = 2.2) +
  scale_colour_manual(values = c(LightGBM = PAL[["naranja"]], `Logit lineal en la variable` = PAL[["gris"]], `Logit con splines` = PAL[["azul"]]), name = NULL) +
  scale_linetype_manual(values = c(LightGBM = 1, `Logit lineal en la variable` = 2, `Logit con splines` = 1), name = NULL) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  labs(x = "Log de la razón ingreso imputado / línea (40 grupos de igual tamaño)", y = "Probabilidad de ser pobre", title = "La variable clave es casi logística: un logit con splines la captura",
       subtitle = "Puntos negros: tasa observada de pobreza; líneas: ajustes con solo esta variable") + theme_slide() + guides(colour = guide_legend(nrow = 1), linetype = guide_legend(nrow = 1))
guardar(p12, "nonlinearity", 9.5, 5)

# =========================== 12. Coeficientes del Elastic Net y comparación de importancias ============================
co <- read_csv(file.path(DIR_TAB, "harold_elastic_net_fe_coefs.csv"), show_col_types = FALSE) |> filter(variable != "(Intercept)") |>
  mutate(base = sub("_ns[0-9]+$", "", variable)) |> group_by(base) |> summarise(coef = coef[which.max(abs(coef))], .groups = "drop") |> arrange(desc(abs(coef))) |> slice_head(n = 10) |>
  mutate(lab = fct_reorder(et(base), abs(coef)), signo = if_else(coef < 0, "Reduce la pobreza", "Aumenta la pobreza"))
p13 <- ggplot(co, aes(abs(coef), lab, fill = signo)) + geom_col(width = 0.75) +
  scale_fill_manual(values = c("Reduce la pobreza" = PAL[["azul"]], "Aumenta la pobreza" = PAL[["naranja"]]), name = NULL) +
  labs(x = "|Coeficiente| (variables estandarizadas)", y = NULL, title = "Elastic Net: 10 mayores coeficientes", subtitle = "Coeficiente de mayor tamaño entre la variable y sus splines") + theme_slide() + guides(fill = guide_legend(nrow = 2))
guardar(p13, "en_coefs", 6.2, 5.2)
rf <- read_csv(file.path(DIR_TAB, "harold_random_forest_fe_vars.csv"), show_col_types = FALSE) |> slice_max(shap_abs, n = 10) |> mutate(lab = fct_reorder(et(variable), shap_abs))
p14 <- ggplot(rf, aes(shap_abs, lab)) + geom_col(fill = PAL["verde"], width = 0.75) +
  labs(x = "Valor SHAP absoluto medio (log-odds)", y = NULL, title = "Random Forest: 10 variables principales", subtitle = "Importancia calculada con un LightGBM auxiliar (script 06)") + theme_slide()
guardar(p14, "rf_importance", 6.2, 5.2)
lg10 <- imp |> slice_head(n = 10) |> mutate(lab = fct_reorder(lab, shap_media_abs))
p15 <- ggplot(lg10, aes(shap_media_abs, lab)) + geom_col(fill = PAL["naranja"], width = 0.75) +
  labs(x = "Valor SHAP absoluto medio (log-odds)", y = NULL, title = "LightGBM: 10 variables principales", subtitle = "Mismo cálculo SHAP sobre el modelo final") + theme_slide()
guardar(p15, "lgbm_importance10", 6.2, 5.2)

# Ranking de importancia de las variables principales en tres modelos
en_all <- read_csv(file.path(DIR_TAB, "harold_elastic_net_fe_coefs.csv"), show_col_types = FALSE) |> filter(variable != "(Intercept)") |>
  mutate(base = sub("_ns[0-9]+$", "", variable)) |> group_by(base) |> summarise(a = max(abs(coef)), .groups = "drop") |> mutate(r_en = rank(-a))
rf_all <- read_csv(file.path(DIR_TAB, "harold_random_forest_fe_vars.csv"), show_col_types = FALSE) |> mutate(r_rf = rank(-shap_abs)) |> select(variable, r_rf)
rk <- imp |> mutate(r_l = rank(-shap_media_abs)) |> select(variable, r_l) |> full_join(rf_all, by = "variable") |> full_join(select(en_all, variable = base, r_en), by = "variable")
mostrar <- unique(c(imp$variable[1:6], rf_all$variable[rf_all$r_rf <= 6], en_all$base[en_all$r_en <= 6]))
rk <- rk |> filter(variable %in% mostrar) |> arrange(pmin(r_l, r_rf, r_en, na.rm = TRUE)) |> slice_head(n = 9)
f_r <- \(x) ifelse(is.na(x), "--", as.character(x))
tab_write <- function(lines, f) writeLines(lines, file.path("03_slides/tables", f))
tab_write(c("\\begin{tabular}{lccc}", "\\toprule", "\\textbf{Variable} & \\textbf{LightGBM} & \\textbf{Random Forest} & \\textbf{Elastic Net} \\\\", "\\midrule",
            sprintf("%s & %s & %s & %s \\\\", et(rk$variable), f_r(rk$r_l), f_r(rk$r_rf), f_r(rk$r_en)), "\\bottomrule", "\\end{tabular}"), "importance_rank.tex")

# --- Selección de variables: poder predictivo de cada variable por sí sola (AUC univariado, train) ---------------------------------------
etiquetas <- c(etiquetas, rent_burden = "Carga del arriendo", rent_pc = "Arriendo per cápita", fe_inc_hat_head = "Ingreso imputado del jefe",
               fe_security = "Seguridad social (índice)", hours_pc = "Horas trabajadas per cápita", fe_log_rent_pc = "Log arriendo per cápita",
               log_rent_pc = "Log arriendo per cápita", fe_inc_hat_cond = "Ingreso imputado del hogar", n_subsid_regime = "Régimen de subsidios",
               fe_log_hours_pc = "Log horas per cápita", fe_deprivation = "Índice de privaciones")
et <- function(v) ifelse(v %in% names(etiquetas), etiquetas[v], v)
uni2 <- read_csv(file.path(DIR_TAB, "harold_fe_univariate.csv"), show_col_types = FALSE) |>
  filter(!variable %in% c("fe_ratio_hat_cond", "fe_log_ratio_hat", "fe_gap_hat", "fe_inc_hat_max")) |>   # casi duplicadas de otra ya mostrada
  slice_max(auc_binned, n = 14) |> mutate(etq = fct_reorder(et(variable), auc_binned))
lab_grp <- c(A = "A. Base", B = "B. Ratios", C = "C. Regla", D = "D. Ingreso")
pal_grp <- c(A = PAL[["gris"]], B = PAL[["azul"]], C = PAL[["naranja"]], D = PAL[["verde"]])
p_uni <- ggplot(uni2, aes(auc_binned, etq, colour = grupo)) +
  geom_vline(xintercept = 0.5, linetype = "dashed", colour = "grey60") +
  geom_segment(aes(x = 0.5, xend = auc_binned, yend = etq), linewidth = 0.8) + geom_point(size = 3.6) +
  scale_colour_manual(values = pal_grp, labels = lab_grp, name = "Grupo") +
  scale_x_continuous(limits = c(0.5, 0.96), labels = \(x) format(x, decimal.mark = ",")) +
  labs(title = "Qué variables predicen mejor la pobreza por sí solas", subtitle = "14 variables con mayor AUC no lineal (20 cuantiles), datos de entrenamiento",
       x = "AUC univariado (0,5 = sin poder)", y = NULL) + theme_slide() + guides(colour = guide_legend(nrow = 1))
guardar(p_uni, "univariate", 9, 5.4)

# --- Selección de modelo: F1 de los 10 modelos con su variabilidad entre pliegues --------------------------------------------------------
cm <- read_csv(file.path(DIR_TAB, "harold_comparison_metrics.csv"), show_col_types = FALSE) |>
  mutate(modelo = fct_reorder(modelo, f1), familia = factor(familia, c("Boosting", "Árboles", "Lineal", "Probabilístico")))
p_mod <- ggplot(cm, aes(f1, modelo, colour = familia)) +
  geom_vline(xintercept = max(cm$f1), linetype = "dashed", colour = "grey60") +
  geom_errorbar(aes(xmin = f1 - f1_sd_pliegues, xmax = f1 + f1_sd_pliegues), width = 0.25, orientation = "y", linewidth = 0.8) +
  geom_point(size = 3.6) +
  scale_colour_manual(values = c(Boosting = PAL[["verde"]], "Árboles" = PAL[["naranja"]], Lineal = PAL[["azul"]], Probabilístico = PAL[["gris"]]), name = NULL) +
  scale_x_continuous(labels = \(x) format(x, decimal.mark = ",", nsmall = 2)) +
  labs(title = "Nueve de diez modelos quedan a menos de un punto del mejor", subtitle = "F1 de la clase pobre (media ± 1 DE entre pliegues), validación cruzada sin Bogotá",
       x = "F1 de la clase pobre", y = NULL, caption = "Cada modelo en su mejor versión; la línea punteada marca el máximo.") + theme_slide()
guardar(p_mod, "model_f1", 9, 5.4)

# =========================== Tablas LaTeX =================================================================================
ltx <- function(x) gsub("%", "\\\\%", gsub("&", "\\\\&", x))
tab_write <- function(lines, f) writeLines(lines, file.path("03_slides/tables", f))
cvs <- read_csv(file.path(DIR_TAB, "harold_comparison_metrics.csv"), show_col_types = FALSE)
# (a) Resumen de lo enviado/mejor versión de cada algoritmo
mejor <- tribble(
  ~algoritmo, ~version, ~modelo_tabla, ~kg,
  "Boosting", "LightGBM, 149 variables", "LightGBM", "kgLGBM",
  "Random Forest", "ranger, 40 variables", "Random Forest", "kgRF",
  "Elastic Net", "glmnet con splines, 102 columnas", "Elastic Net (alpha 0,7)", "kgEN",
  "Logit", "Logit regularizado, 59 variables", "Logit robusto (59 vars)", "kgLogit",
  "LPM", "MCO, 40 variables", "LPM (40 vars)", "kgLPM",
  "CART", "rpart, 40 variables", "CART", "kgCART",
  "Naive Bayes", "deciles, 40 variables", "Naive Bayes (deciles)", "kgNB")
cuerpo <- pmap_chr(mejor, function(algoritmo, version, modelo_tabla, kg) {
  r <- cvs[cvs$modelo == modelo_tabla, ]
  sprintf("%s & %s & %s & %s & %s & \\%s \\\\", algoritmo, version, num(r$f1), num(r$precision), num(r$recall), kg) })
tab_write(c("\\begin{tabular}{llcccc}", "\\toprule", "\\textbf{Algoritmo} & \\textbf{Versión} & \\textbf{F1 (CV)} & \\textbf{Precisión} & \\textbf{Cobertura} & \\textbf{F1 Kaggle} \\\\", "\\midrule",
            cuerpo, "\\bottomrule", "\\end{tabular}"), "best_versions.tex")
# (b) Mismas 40 variables
cuerpo <- sprintf("%s & %s & %s & %s & %s & %s \\\\", sp$algoritmo, ltx(sp$hiperparametros), num(sp$f1), num(sp$precision), num(sp$recall), num(sp$auc))
tab_write(c("\\begin{tabular}{llccccc}", "\\toprule", "\\textbf{Algoritmo} & \\textbf{Hiperparámetros} & \\textbf{F1} & \\textbf{Precisión} & \\textbf{Cobertura} & \\textbf{AUC} \\\\", "\\midrule",
            cuerpo, "\\bottomrule", "\\end{tabular}") |> str_replace("llccccc", "llcccc"), "same_predictors.tex")
# (c) Hiperparámetros finales del LightGBM
pj <- jsonlite::fromJSON(file.path(DIR_PROC, "harold_lightgbm_fe_params.json"))
hp <- tribble(~k, ~v, "Hojas por árbol", as.character(pj$params$num_leaves), "Profundidad máxima", as.character(pj$params$max_depth), "Mínimo de hogares por hoja", as.character(pj$params$min_child_samples),
              "Fracción de columnas por árbol", num(pj$params$feature_fraction, 1), "Fracción de hogares por árbol", num(pj$params$bagging_fraction, 2),
              "Penalización L2", as.character(pj$params$lambda_l2), "Peso de la clase pobre", num(pj$params$scale_pos_weight, 1), "Tasa de aprendizaje", num(pj$learning_rate, 2),
              "Rondas (árboles)", as.character(pj$rondas), "Umbral de clasificación", num(thr_l, 2))
tab_write(c("\\begin{tabular}{lr}", "\\toprule", "\\textbf{Hiperparámetro} & \\textbf{Valor} \\\\", "\\midrule", sprintf("%s & %s \\\\", hp$k, hp$v), "\\bottomrule", "\\end{tabular}"), "lgbm_hyper.tex")
mac("RondasLGBM", pj$rondas); mac("NVarsLGBM", pj$n_vars); mac("HojasLGBM", pj$params$num_leaves)
# (d) Generalización: brecha dentro/fuera y regiones nuevas
gp <- read_csv(file.path(DIR_TAB, "harold_robust_gap.csv"), show_col_types = FALSE); gc <- read_csv(file.path(DIR_TAB, "harold_robust_grouped_cv.csv"), show_col_types = FALSE)
pn <- read_csv(file.path(DIR_TAB, "harold_robust_public_noise.csv"), show_col_types = FALSE)
tab_write(c("\\begin{tabular}{lcccc}", "\\toprule", "\\textbf{Modelo} & \\textbf{F1 dentro} & \\textbf{F1 fuera} & \\textbf{F1 regiones nuevas} & \\textbf{Caída} \\\\", "\\midrule",
            sprintf("Logit regularizado & %s & %s & %s & %s \\\\", num(gp$f1_dentro[1], 4), num(gp$f1_fuera[1], 4), num(gc$f1_regiones_nuevas[1], 4), num(gc$caida[1], 4)),
            sprintf("LightGBM & -- & %s & %s & %s \\\\", num(gc$f1_pliegues_aleatorios[2], 4), num(gc$f1_regiones_nuevas[2], 4), num(gc$caida[2], 4)),
            "\\bottomrule", "\\end{tabular}"), "generalization.tex")
mac("PublicSD", num(pn$desv_est[2], 4)); mac("DropLogitRegions", num(gc$caida[1], 4)); mac("DropLGBMRegions", num(gc$caida[2], 4))
mac("GapLogit", num(gp$brecha_f1[1], 4))
# (e) Desacuerdo LightGBM vs. Elastic Net
s_en <- oof("elastic_net_fe"); thr_e <- read_csv(file.path(DIR_TAB, "harold_elastic_net_fe_cv.csv"), show_col_types = FALSE) |> filter(elegido) |> pull(umbral)
a <- as.integer(s_lgb >= thr_l); b <- as.integer(s_en >= thr_e)
dd <- tibble(a, b, poor = y)[nb, ] |> count(a, b, name = "n") |> left_join(tibble(a, b, poor = y)[nb, ] |> group_by(a, b) |> summarise(pobre = mean(poor), .groups = "drop"), by = c("a", "b")) |>
  mutate(grupo = case_when(a == 1 & b == 1 ~ "Ambos: pobre", a == 0 & b == 0 ~ "Ambos: no pobre", a == 1 & b == 0 ~ "Solo LightGBM: pobre", TRUE ~ "Solo Elastic Net: pobre"))
dd <- dd |> arrange(match(grupo, c("Ambos: pobre", "Ambos: no pobre", "Solo LightGBM: pobre", "Solo Elastic Net: pobre")))
tab_write(c("\\begin{tabular}{lrrr}", "\\toprule", "\\textbf{Clasificación} & \\textbf{Hogares} & \\textbf{\\% del total} & \\textbf{\\% realmente pobres} \\\\", "\\midrule",
            sprintf("%s & %s & %s & %s \\\\", dd$grupo, fmtn(dd$n), ltx(paste0(pct(dd$n / sum(dd$n)), "")), ltx(pct(dd$pobre, 0))), "\\bottomrule", "\\end{tabular}"), "disagreement.tex")
desacuerdo <- dd |> filter(str_detect(grupo, "^Solo")); mac("DisagreeShare", pct(sum(desacuerdo$n) / sum(dd$n))); mac("DisagreePoor", pct(weighted.mean(desacuerdo$pobre, desacuerdo$n), 0))
# Cifras sueltas de otros modelos
cod_m <- c("LPM (40 vars)" = "LPM", "Logit (40 vars)" = "LogitFortyVars", "Elastic Net (alpha 0,7)" = "EN", "Logit robusto (59 vars)" = "LogitRob", "Probit robusto (59 vars)" = "ProbitRob",
           "Naive Bayes (deciles)" = "NB", "CART" = "CART", "Random Forest" = "RF", "LightGBM" = "LGBMAll", "XGBoost (regresión + umbral)" = "XGB")
for (i in seq_len(nrow(cvs))) { k <- cod_m[[cvs$modelo[i]]]; mac(paste0("Fone", k), num(cvs$f1[i])); mac(paste0("Prec", k), num(cvs$precision[i])); mac(paste0("Rec", k), num(cvs$recall[i])) }
rfcv <- read_csv(file.path(DIR_TAB, "harold_random_forest_fe_cv.csv"), show_col_types = FALSE)
mac("MtryRF", rfcv$mtry); mac("NodeRF", rfcv$min_node)

# --- numbers.tex -------------------------------------------------------------------------------------------------------------
writeLines(c("% Generado por 01_code/17_harold_slide_assets.R; no editar a mano.",
             unlist(imap(num_macros, \(v, k) sprintf("\\newcommand{\\n%s}{%s}", k, v)))), "03_slides/numbers.tex")
message("Figuras y tablas de las diapositivas listas.")
