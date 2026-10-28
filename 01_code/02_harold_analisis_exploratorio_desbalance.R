# ==============================================================================
# 02_harold_analisis_exploratorio_desbalance.R
# Objetivo: entender el resultado (pobreza), el desbalance de clases y la relación
#           de los predictores candidatos con la pobreza. Solo usa TRAIN.
# Salidas: 02_outputs/figures/eda_*.png ; 02_outputs/tables/eda_*.tex
# ==============================================================================
suppressPackageStartupMessages({library(tidyverse); library(xtable)})

d  <- readRDS(file.path(DIR_PROC, "hogares_modelo.rds"))
tr <- d %>% filter(sample == "train")

theme_set(theme_pobreza())
col_pobre <- COL_POBRE

# --- 1. Desbalance de clases -------------------------------------------------------------
cat("Tasa de pobreza en train:", round(mean(tr$poor), 4), "\n")
p1 <- tr %>% count(poor) %>% mutate(share = n / sum(n),
        etiqueta = factor(poor, labels = c("No pobre", "Pobre"))) %>%
  ggplot(aes(etiqueta, share, fill = factor(poor))) + geom_col(width = .6) +
  geom_text(aes(label = scales::percent(share, .1)), vjust = -.4) +
  scale_fill_manual(values = col_pobre, guide = "none") +
  scale_y_continuous(labels = scales::percent, limits = c(0, .9)) +
  labs(title = "Desbalance de clases en el conjunto de entrenamiento",
       x = NULL, y = "Proporción de hogares")
save_fig(p1, "eda_class_imbalance.png", 5, 4)

# --- 2. Pobreza por zona y dominio ---------------------------------------------------------
p2 <- tr %>% group_by(Dominio) %>% summarise(rate = mean(poor), n = n()) %>%
  ggplot(aes(reorder(Dominio, rate), rate)) + geom_col(fill = "#E45756") + coord_flip() +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Tasa de pobreza por dominio geográfico", x = NULL, y = "% de hogares pobres")
save_fig(p2, "eda_poverty_by_domain.png", 7, 6)

# --- 3. Ingreso per cápita vs línea de pobreza ---------------------------------------------
p3 <- tr %>% filter(eda_Ingpcug > 0) %>%
  mutate(ratio = eda_Ingpcug / Lp) %>%
  ggplot(aes(ratio, fill = factor(poor))) +
  geom_density(alpha = .6, color = NA) + geom_vline(xintercept = 1, linetype = 2) +
  scale_x_log10() + scale_fill_manual(values = col_pobre, labels = c("No pobre", "Pobre"), name = NULL) +
  labs(title = "Ingreso per cápita relativo a la línea de pobreza",
       subtitle = "El ingreso no está en el test: es la fuente de la etiqueta, no un predictor",
       x = "Ingreso per cápita / línea de pobreza (escala log)", y = "Densidad")
save_fig(p3, "eda_income_vs_poverty_line.png")

# --- 4. Correlaciones con la pobreza ---------------------------------------------------------
cand <- c("kids_pc","crowding","hours_pc","earners_pc","formal_share","dep_ratio","educ_mean","head_educ",
          "share_insured","n_pension","n_subsidy","n_employed","no_earner","urban","head_female",
          "n_subsid_regime","n_superior","rent_burden","log_rent_pc","n_child_0_5","n_senior","Nper")
cors <- tr %>% summarise(across(all_of(cand), ~ cor(.x, poor))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "cor")
p4 <- cors %>% ggplot(aes(reorder(variable, cor), cor, fill = cor > 0)) + geom_col() + coord_flip() +
  scale_fill_manual(values = c("TRUE" = "#E45756", "FALSE" = "#4C78A8"), guide = "none") +
  labs(title = "Correlación de los predictores candidatos con la pobreza", x = NULL, y = "Correlación con P(pobre)")
save_fig(p4, "eda_correlations.png", 7, 6)

# --- 5. Gradientes económicos clave -------------------------------------------------------------
grad <- function(var, breaks, lab) {
  tr %>% mutate(bin = cut(.data[[var]], breaks, include.lowest = TRUE)) %>%
    group_by(bin) %>% summarise(rate = mean(poor), n = n(), .groups = "drop") %>% mutate(var = lab)
}
g <- bind_rows(
  grad("earners_pc", c(0, .001, .25, .5, 1, 5), "Ocupados por miembro del hogar"),
  grad("crowding",   c(0, 1, 2, 3, 4, 20),       "Personas por dormitorio"),
  grad("educ_mean",  c(0, 5, 9, 11, 14, 16),     "Educación media del hogar (años, 15+)"),
  grad("dep_ratio",  c(0, .25, .5, 1, 2, 20),    "Razón de dependencia"))
p5 <- g %>% ggplot(aes(bin, rate)) + geom_col(fill = "#E45756") +
  facet_wrap(~ var, scales = "free_x") + scale_y_continuous(labels = scales::percent) +
  labs(title = "La pobreza sigue gradientes que predice la teoría económica",
       x = NULL, y = "% de hogares pobres") + theme(axis.text.x = element_text(angle = 30, hjust = 1))
save_fig(p5, "eda_gradients.png", 9, 6)

# --- 6. Tabla descriptiva por grupo (LaTeX) -----------------------------------------------------
vars <- c("Nper","kids_pc","earners_pc","hours_pc","educ_mean","head_educ","crowding","share_insured","formal_share","dep_ratio")
tab <- tr %>% group_by(poor) %>% summarise(across(all_of(vars), mean)) %>%
  pivot_longer(-poor) %>% pivot_wider(names_from = poor, values_from = value, names_prefix = "poor_") %>%
  rename(`Variable` = name, `No pobre` = poor_0, `Pobre` = poor_1)
print(xtable(tab, caption = "Medias de predictores clave por condición de pobreza (train)", label = "tab:eda_means", digits = 3),
      file = file.path(DIR_TAB, "eda_means_by_poverty.tex"), include.rownames = FALSE, booktabs = TRUE)
print(tab)
