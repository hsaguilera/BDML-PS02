# ==============================================================================
# 03_harold_analisis_caracterizar_train_test.R
# Objetivo: caracterizar los datos con gráficos antes de modelar.
#   (a) ¿train y test son comparables?   (b) ¿qué falta y por qué?
#   (c) ¿cómo varía la pobreza con composición, ciclo de vida, educación,
#       vivienda y empleo?   (d) ¿qué predictores se solapan entre sí?
# Guía: complementaria del 19 de septiembre (exploración y variables de hogar) y
#       Lamichhane et al. (2025): los modelos de pobreza ligan el bienestar del hogar
#       con atributos del hogar, características de sus miembros y ubicación; el
#       desempeño suele diferir entre zona urbana y rural, así que se muestra por zona.
# Salidas: 02_outputs/figures/desc_*.png ; 02_outputs/tables/desc_*.tex
# ==============================================================================
suppressPackageStartupMessages({library(tidyverse); library(data.table); library(xtable)})
theme_set(theme_pobreza())

d  <- readRDS(file.path(DIR_PROC, "hogares_modelo.rds"))
tr <- d |> filter(sample == "train")
lab_sample <- c(train = "Train", test = "Test")
col_sample <- c("Train" = "#7F7F7F", "Test" = "#F58518")
col_zona   <- c("Urbano" = "#54A24B", "Rural" = "#B279A2")
col_sexo   <- c("Hombre" = "#54A24B", "Mujer" = "#B279A2")

# Proporción de pobres por grupo con IC de Wilson al 95 %
rate_ci <- function(df, ...) {
  out <- df |> group_by(...) |> summarise(k = sum(poor), n = n(), .groups = "drop") |>
    mutate(rate = k / n)
  bind_cols(out, wilson_ci(out$k, out$n))
}

# --- 1. ¿Train y test son comparables? -------------------------------------------------
vars_num <- c(head_age = "Edad del jefe (años)", educ_mean = "Educación media (años, 15+)",
              earners_pc = "Ocupados por miembro", crowding = "Personas por dormitorio")
p1 <- d |>
  select(sample, all_of(names(vars_num))) |>
  pivot_longer(-sample) |>
  mutate(name = factor(vars_num[name], levels = vars_num), sample = lab_sample[sample]) |>
  ggplot(aes(value, colour = sample, fill = sample)) +
  geom_density(alpha = .25, linewidth = .6, adjust = 1.5) +
  facet_wrap(~ name, scales = "free") +
  scale_colour_manual(values = col_sample, name = NULL) + scale_fill_manual(values = col_sample, name = NULL) +
  labs(title = "Los predictores numéricos se distribuyen igual en train y test",
       subtitle = "Densidades de cuatro predictores clave (la mayor diferencia estandarizada entre muestras es 0,04)", x = NULL, y = "Densidad")
save_fig(p1, "desc_train_vs_test_numeric.png", 8, 5.5)

p1b <- d |> count(sample, Dominio) |> group_by(sample) |> mutate(share = n / sum(n)) |> ungroup() |>
  mutate(sample = lab_sample[sample]) |>
  ggplot(aes(reorder(Dominio, share), share, fill = sample)) +
  geom_col(position = position_dodge(.8), width = .75) + coord_flip() +
  scale_fill_manual(values = col_sample, name = NULL) + scale_y_continuous(labels = scales::percent) +
  labs(title = "Bogotá está en train pero no en test",
       subtitle = paste0("Los ", scales::comma(sum(tr$Dominio == "BOGOTA")), " hogares de Bogotá (", scales::percent(mean(tr$Dominio == "BOGOTA"), .1), " de train) no tienen contraparte en test"),
       x = NULL, y = "% de hogares de cada muestra")
save_fig(p1b, "desc_train_vs_test_domain.png", 7, 6.5)

# Diferencia estandarizada de medias (|SMD| > 0,1 sugiere desbalance entre muestras)
smd_vars <- names(d)[sapply(d, is.numeric)]
smd_vars <- setdiff(smd_vars, c("poor", grep("^eda_", names(d), value = TRUE), "Fex_c", "Fex_dpto"))
smd <- map_dfr(smd_vars, function(v) {
  a <- d[[v]][d$sample == "train"]; b <- d[[v]][d$sample == "test"]
  tibble(Variable = v, SMD = (mean(b) - mean(a)) / sqrt((var(a) + var(b)) / 2))
}) |> filter(is.finite(SMD)) |> arrange(desc(abs(SMD)))
cat("Mayor |SMD| train-test:", round(max(abs(smd$SMD)), 3), " (", smd$Variable[1], ")\n")
print(xtable(head(smd, 10), caption = "Mayores diferencias estandarizadas de medias entre test y train",
             label = "tab:desc_smd", digits = 3),
      file = file.path(DIR_TAB, "desc_covariate_shift.tex"), include.rownames = FALSE, booktabs = TRUE)

# --- 2. ¿Qué falta y por qué? -------------------------------------------------------------
cols_p <- c("P6090", "P6210", "P6240", "P6426", "P6430", "P6800", "P6870", "P6920", "P7495", "P7505")
miss_p <- bind_rows(
  fread(file.path(DIR_RAW, "train_personas.csv"), select = cols_p)[, sample := "train"],
  fread(file.path(DIR_RAW, "test_personas.csv"),  select = cols_p)[, sample := "test"]) |>
  as_tibble() |>
  pivot_longer(-sample) |> group_by(sample, name) |> summarise(miss = mean(is.na(value)), .groups = "drop")
lab_p <- c(P6090 = "Afiliación a salud", P6210 = "Nivel educativo", P6240 = "Actividad principal",
           P6426 = "Antigüedad en el empleo", P6430 = "Posición ocupacional", P6800 = "Horas trabajadas",
           P6870 = "Tamaño de la firma", P6920 = "Cotiza a pensión", P7495 = "Recibe pensión/jubilación",
           P7505 = "Otros ingresos (12 meses)")
p2 <- miss_p |> mutate(name = factor(lab_p[name], levels = rev(lab_p)), sample = lab_sample[sample]) |>
  ggplot(aes(name, miss, fill = sample)) +
  geom_col(position = position_dodge(.8), width = .75) + coord_flip() +
  scale_fill_manual(values = col_sample, name = NULL) + scale_y_continuous(labels = scales::percent) +
  labs(title = "Datos faltantes en variables de personas",
       subtitle = "Faltantes estructurales: las preguntas laborales solo aplican a ocupados\ny varias solo a mayores de 12 años",
       x = NULL, y = "% de personas con dato faltante")
save_fig(p2, "desc_missingness.png", 7.5, 5)

# --- 3. Pobreza, tamaño del hogar y niños -------------------------------------------------------
p3 <- tr |>
  mutate(tam = pmin(Npersug, 8), ninos = if_else(n_child_0_5 + n_child_6_17 > 0, "Con menores de 18", "Sin menores de 18")) |>
  rate_ci(tam, ninos) |>
  ggplot(aes(tam, rate, colour = ninos, fill = ninos)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = .2, colour = NA) + geom_line(linewidth = .8) + geom_point(size = 2) +
  scale_x_continuous(breaks = 1:8, labels = c(1:7, "8+")) + scale_y_continuous(labels = scales::percent) +
  scale_colour_manual(values = c("#4C78A8", "#E45756"), name = NULL) + scale_fill_manual(values = c("#4C78A8", "#E45756"), name = NULL) +
  labs(title = "El tamaño del hogar eleva la pobreza solo si hay menores",
       subtitle = "Tasa de pobreza e IC 95 % según personas en la unidad de gasto",
       x = "Personas en la unidad de gasto", y = "% de hogares pobres")
save_fig(p3, "desc_poverty_by_hh_size.png")

# --- 4. Ciclo de vida del jefe y sexo ---------------------------------------------------------------
p4 <- tr |>
  mutate(edad = cut(head_age, c(17, 24, 34, 44, 54, 64, 74, 120), labels = c("18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+")),
         sexo = if_else(head_female == 1, "Mujer", "Hombre")) |>
  filter(!is.na(edad)) |> rate_ci(edad, sexo) |>
  ggplot(aes(edad, rate, colour = sexo, group = sexo)) +
  geom_pointrange(aes(ymin = lo, ymax = hi), position = position_dodge(.3)) +
  geom_line(position = position_dodge(.3), linewidth = .6) +
  scale_colour_manual(values = col_sexo, name = "Sexo del jefe") + scale_y_continuous(labels = scales::percent) +
  labs(title = "Pobreza según edad y sexo del jefe del hogar",
       subtitle = "Los hogares con jefatura femenina son más pobres hasta los 64 años; después la brecha se cierra",
       x = "Edad del jefe (años)", y = "% de hogares pobres")
save_fig(p4, "desc_poverty_by_head_age_sex.png")

# --- 5. Educación del jefe y zona (urbano vs rural) -------------------------------------------------------
p5 <- tr |>
  mutate(educ = factor(head_educ, levels = c(0, 5, 9, 11, 15),
                       labels = c("Ninguna o\npreescolar", "Primaria", "Secundaria", "Media", "Superior")),
         zona = if_else(urban == 1, "Urbano", "Rural")) |>
  filter(!is.na(educ)) |> rate_ci(educ, zona) |>
  ggplot(aes(educ, rate, fill = zona)) +
  geom_col(position = position_dodge(.8), width = .75) +
  geom_errorbar(aes(ymin = lo, ymax = hi), position = position_dodge(.8), width = .25) +
  scale_fill_manual(values = col_zona, name = NULL) + scale_y_continuous(labels = scales::percent) +
  labs(title = "La educación del jefe reduce la pobreza en ambas zonas",
       subtitle = "Desde primaria, los hogares rurales son más pobres que los urbanos con la misma educación",
       x = "Máximo nivel educativo del jefe", y = "% de hogares pobres")
save_fig(p5, "desc_poverty_by_head_educ_zone.png")

# --- 6. Tenencia de la vivienda --------------------------------------------------------------------------------
lab_ten <- c(propia_pagada = "Propia, pagada", propia_pagando = "Propia, pagando", arriendo = "Arriendo",
             usufructo = "Usufructo", ocupante = "Ocupante sin título", otra = "Otra")
p6 <- tr |> rate_ci(tenure) |> mutate(tenure = factor(lab_ten[as.character(tenure)], levels = rev(lab_ten))) |>
  ggplot(aes(tenure, rate)) +
  geom_col(fill = COL_ACENTO, width = .65) + geom_errorbar(aes(ymin = lo, ymax = hi), width = .25) +
  geom_text(aes(y = 0.005, label = paste0("n = ", scales::comma(n))), hjust = 0, colour = "white", size = 3.2) +
  coord_flip() + scale_y_continuous(labels = scales::percent) +
  labs(title = "Pobreza según la tenencia de la vivienda", x = NULL, y = "% de hogares pobres")
save_fig(p6, "desc_poverty_by_tenure.png", 7, 4)

# --- 7. Interacción hacinamiento x ocupados por miembro (mapa de calor) -------------------------------------------
p7 <- tr |>
  mutate(hac = cut(crowding, c(0, 1, 2, 3, 20), labels = c("hasta 1", "1 a 2", "2 a 3", "más de 3")),
         ocu = cut(earners_pc, c(-.01, 0, .25, .5, 1.01), labels = c("0", "hasta 0,25", "0,25 a 0,5", "más de 0,5"))) |>
  rate_ci(hac, ocu) |> filter(n >= 50) |>
  ggplot(aes(ocu, hac, fill = rate)) + geom_tile(colour = "white") +
  geom_text(aes(label = scales::percent(rate, 1)), size = 3.6) +
  scale_fill_gradient(low = "#F7F7F7", high = COL_ACENTO, labels = scales::percent, name = "% pobres") +
  labs(title = "El hacinamiento pesa más cuando hay pocos ocupados",
       subtitle = "Tasa de pobreza por celda; se omiten celdas con menos de 50 hogares",
       x = "Ocupados por miembro del hogar", y = "Personas por dormitorio")
save_fig(p7, "desc_heatmap_crowding_earners.png", 7.5, 4.8)

# --- 8. Empleo y pensión: una interacción clave -------------------------------------------------------------------------
p8 <- tr |>
  mutate(empleo = if_else(n_employed > 0, "Con ocupados", "Sin ocupados"),
         pension = if_else(n_pension > 0, "Con pensión", "Sin pensión")) |>
  rate_ci(empleo, pension) |>
  ggplot(aes(empleo, rate, fill = pension)) +
  geom_col(position = position_dodge(.8), width = .7) +
  geom_errorbar(aes(ymin = lo, ymax = hi), position = position_dodge(.8), width = .2) +
  geom_text(aes(label = scales::percent(rate, .1)), position = position_dodge(.8), vjust = -1.2, size = 3.5) +
  scale_fill_manual(values = c("Con pensión" = "#4C78A8", "Sin pensión" = COL_ACENTO), name = NULL) +
  scale_y_continuous(labels = scales::percent, expand = expansion(mult = c(0, .12))) +
  labs(title = "Sin ocupados, la pensión marca la diferencia",
       subtitle = "Entre hogares sin ocupados: 6 % de pobreza con pensión y 49 % sin ella",
       x = NULL, y = "% de hogares pobres")
save_fig(p8, "desc_pension_interaction.png", 7, 4.5)

# --- 9. Unidad de gasto: servicio doméstico fuera del hogar de gasto ------------------------------------------------------
p9 <- tr |>
  mutate(fuera = case_when(n_outside_ug == 0 ~ "Ninguna", n_outside_ug == 1 ~ "1 persona", TRUE ~ "2 o más")) |>
  mutate(fuera = factor(fuera, levels = c("Ninguna", "1 persona", "2 o más"))) |>
  rate_ci(fuera) |>
  mutate(eje = paste0(fuera, "\n(n = ", scales::comma(n), ")")) |>
  ggplot(aes(fct_reorder(eje, as.integer(fuera)), rate)) + geom_col(fill = COL_NEUTRO, width = .6) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = .2) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Con personas fuera de la unidad de gasto, la pobreza baja a 4-5 %",
       subtitle = "Empleados domésticos, pensionistas y trabajadores residentes no cuentan\nen el ingreso per cápita del hogar",
       x = "Personas del hogar fuera de la unidad de gasto", y = "% de hogares pobres")
save_fig(p9, "desc_poverty_by_outside_ug.png", 7.5, 4.5)

# --- 10. Correlaciones entre predictores (multicolinealidad) ---------------------------------------------------------------------
cv <- c(Npersug = "Personas en la UG", kids_pc = "Menores por miembro", dep_ratio = "Razón de dependencia",
        earners_pc = "Ocupados por miembro", hours_pc = "Horas por miembro", formal_share = "Proporción formal",
        educ_mean = "Educación media", head_educ = "Educación del jefe", share_insured = "Afiliación a salud",
        n_pension = "Pensionados", n_subsidy = "Subsidios", crowding = "Hacinamiento",
        log_rent_pc = "Log arriendo per cápita", urban = "Zona urbana", head_female = "Jefatura femenina",
        head_age = "Edad del jefe", poor = "Pobre")
cm <- cor(tr[, names(cv)])
ord <- hclust(as.dist(1 - abs(cm)))$order
p10 <- as_tibble(cm, rownames = "a") |> mutate(a = rownames(cm)) |> pivot_longer(-a, names_to = "b", values_to = "r") |>
  mutate(a = factor(cv[a], levels = cv[rownames(cm)[ord]]), b = factor(cv[b], levels = cv[rownames(cm)[ord]])) |>
  ggplot(aes(a, b, fill = r)) + geom_tile(colour = "white") +
  geom_text(aes(label = sprintf("%.1f", r)), size = 2.6) +
  scale_fill_gradient2(low = "#4C78A8", mid = "white", high = COL_ACENTO, limits = c(-1, 1), name = "Correlación") +
  labs(title = "Correlaciones entre predictores y con la pobreza", x = NULL, y = NULL) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.grid = element_blank())
save_fig(p10, "desc_correlation_matrix.png", 9, 8)

message("Figuras de caracterización guardadas en ", DIR_FIG)
