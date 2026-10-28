# ==============================================================================
# 18_harold_diapositivas_envios_equipo.R
# Figuras, tablas y cifras de las diapositivas que usan los envíos de TODO el equipo
# y la matriz de confusión con porcentajes por fila. No entrena nada.
#   - figuras -> 02_outputs/figures/harold_slide_kaggle_envios.png, harold_slide_model_f1_compare.png,
#                harold_slide_same_predictors_compare.png
#   - tablas  -> 03_slides/tables/best_versions_equipo.tex, confusion_lgbm_pct.tex;
#                02_outputs/tables/equipo_envios_kaggle_resumen.csv (la misma tabla de resumen, en csv)
#   - cifras  -> 03_slides/numbers_equipo.tex
# Entradas : 02_outputs/tables/equipo_envios_kaggle.csv (registro de los 23 envíos: autor, algoritmo, variables y
#            puntajes público y privado leídos del tablero de Kaggle, que no se pueden recalcular; una fila por
#            envío, igual al registro del README §6), harold_comparison_metrics.csv, harold_same_predictors_cv.csv,
#            harold_lightgbm_fe_cv.csv, processed/harold_lightgbm_fe_scores.csv, hogares_fe.rds
# Nota     : el F1 de validación cruzada NO es comparable entre autores (cada uno usó su protocolo);
#            entre autores solo se compara el puntaje de Kaggle (público 20 %, privado 80 %).
# ==============================================================================
if (!exists("DIR_PROC")) source(file.path(if (requireNamespace("here", quietly = TRUE)) here::here() else getwd(), "01_code", "00_setup.R"), encoding = "UTF-8")   # corre solo: carga el entorno común
AUTOR <- "harold"
suppressPackageStartupMessages({library(tidyverse); library(data.table)})
options(OutDec = ",")
dir.create("03_slides/tables", recursive = TRUE, showWarnings = FALSE)

PAL <- c(azul = "#1B3A6B", naranja = "#E07B00", verde = "#2A9D8F", gris = "#8D99AE")
num <- function(x, d = 3) formatC(x, format = "f", digits = d, decimal.mark = ",")
pct <- function(x, d = 1) formatC(100 * x, format = "f", digits = d, decimal.mark = ",")
fmtn <- function(x) format(x, big.mark = ".", decimal.mark = ",", scientific = FALSE, trim = TRUE)
dos <- function(x) formatC(x, format = "f", digits = 2, decimal.mark = ",")
macros <- list()
mac <- function(nombre, valor) macros[[nombre]] <<- as.character(valor)

env_raw <- read_csv(file.path(DIR_TAB, "equipo_envios_kaggle.csv"), show_col_types = FALSE)
# Control: cada envío registrado debe tener su archivo de predicciones en el repositorio
falta <- env_raw$archivo[!file.exists(file.path(DIR_PRED, env_raw$archivo))]
if (length(falta)) message("Aviso: ", length(falta), " envío(s) del registro sin archivo en ", DIR_PRED, ": ", paste(falta, collapse = "; "))
stopifnot(nrow(env_raw) >= 15, all(env_raw$publico >= 0 & env_raw$publico <= 1), all(env_raw$privado >= 0 & env_raw$privado <= 1))
env <- env_raw |>
  filter(algoritmo != "Boosting (regresión)") |>   # el XGBoost de regresión no entra en la comparación
  mutate(autor = factor(autor, c("Harold", "Jhon", "César")),
         # en las diapositivas no se nombra a quien hizo cada modelo
         etiqueta = str_remove(etiqueta, "^(Harold|Jhon|César) · "),
         familia  = case_when(str_detect(algoritmo, "Boosting") ~ "Boosting", TRUE ~ algoritmo))

# --- 1. Figura: público y privado de los envíos -------------------------------------------------------------
d <- env |> filter(linea_base == 0) |>
  arrange(privado, publico) |> mutate(etiqueta = factor(etiqueta, unique(etiqueta)))
p <- ggplot(d, aes(y = etiqueta, colour = familia)) +
  geom_segment(aes(x = publico, xend = privado, yend = etiqueta), linewidth = 0.8, alpha = 0.6) +
  geom_point(aes(x = publico, shape = "Público (20 %)"), size = 3.2, fill = "white", stroke = 1) +
  geom_point(aes(x = privado, shape = "Privado (80 %)"), size = 3.2) +
  scale_shape_manual(values = c("Público (20 %)" = 21, "Privado (80 %)" = 16), name = NULL) +
  scale_colour_manual(values = c(Boosting = PAL[["azul"]], "Random Forest" = PAL[["naranja"]], "Elastic Net" = PAL[["verde"]],
                                 Logit = "#7B2D8E", LPM = PAL[["gris"]], "Naive Bayes" = "#C0392B"), name = NULL) +
  scale_x_continuous(breaks = seq(0.64, 0.80, 0.02), labels = \(x) format(x, decimal.mark = ",", nsmall = 2)) +
  labs(x = "F1 de la clase pobre en Kaggle", y = NULL,
       title = "Puntaje de Kaggle de los envíos del equipo",
       subtitle = "Círculo vacío: público (20 %); círculo lleno: privado (80 %). Sin los dos envíos base (0,44 y 0,45)") +
  guides(shape = "none") +
  theme_minimal(base_size = 15) +
  theme(plot.title = element_text(face = "bold", size = 16), plot.title.position = "plot",
        plot.subtitle = element_text(colour = "grey30", size = 12.5), legend.position = "bottom",
        panel.grid.minor = element_blank(), axis.text.y = element_text(size = 12.5))
ggsave(file.path(DIR_FIG, paste0(AUTOR, "_slide_kaggle_envios.png")), p, width = 0.85 * 9.5, height = 0.85 * 7.4, dpi = 200, device = ragg::agg_png)

# --- 2. Cifras ------------------------------------------------------------------------------------------------
mac("KgEnvios", nrow(env)); mac("KgPrivMasSetentaYSiete", sum(env$privado >= 0.77)); mac("KgPubMasSetentaYSiete", sum(env$publico >= 0.77))
mac("KgPrivMax", dos(max(env$privado))); mac("KgPubMax", dos(max(env$publico)))
mac("KgNPrivMax", sum(env$privado == max(env$privado)))
mac("KgEnviosHarold", sum(env$autor == "Harold")); mac("KgEnviosJhon", sum(env$autor == "Jhon")); mac("KgEnviosCesar", sum(env$autor == "César"))

# --- 3. Tabla resumen por algoritmo (mejor público y mejor privado entre los envíos del equipo) -------------------------------
algs <- c("Boosting", "Random Forest", "Elastic Net", "Logit", "LPM", "CART", "Naive Bayes")
cvs <- read_csv(file.path(DIR_TAB, "harold_comparison_metrics.csv"), show_col_types = FALSE)
mejor <- tribble(
  ~algoritmo, ~version, ~modelo_tabla,
  "Boosting", "LightGBM, 149 var.", "LightGBM",
  "Random Forest", "ranger, 40 var.", "Random Forest",
  "Elastic Net", "glmnet, splines", "Elastic Net (alpha 0,7)",
  "Logit", "regularizado, 59 var.", "Logit robusto (59 vars)",
  "LPM", "MCO, 40 var.", "LPM (40 vars)",
  "Naive Bayes", "deciles, 40 var.", "Naive Bayes (deciles)")
# Para cada algoritmo: n.º de envíos y el mejor puntaje público y privado entre todos los envíos del equipo
filas <- pmap_chr(mejor, function(algoritmo, version, modelo_tabla) {
  r <- cvs[cvs$modelo == modelo_tabla, ]
  e <- env |> filter(str_detect(algoritmo, fixed(!!algoritmo)))
  pub <- if (nrow(e)) dos(max(e$publico)) else "--"; pri <- if (nrow(e)) dos(max(e$privado)) else "--"
  sprintf("%s & %s & %s & %s & %s & %d & %s & %s \\\\", algoritmo, version, num(r$f1), num(r$precision), num(r$recall), nrow(e), pub, pri)
})
# Misma tabla en csv (para el README y para auditar las cifras de las diapositivas)
resumen <- pmap_dfr(mejor, function(algoritmo, version, modelo_tabla) {
  e <- env |> filter(str_detect(.data$algoritmo, fixed(!!algoritmo)))
  tibble(algoritmo = algoritmo, version_referencia = version, f1_cv = cvs$f1[cvs$modelo == modelo_tabla], envios = nrow(e),
         max_publico = if (nrow(e)) max(e$publico) else NA_real_, max_privado = if (nrow(e)) max(e$privado) else NA_real_)
})
write_csv(resumen, file.path(DIR_TAB, "equipo_envios_kaggle_resumen.csv"))
writeLines(c("\\begin{tabular}{llcccccc}", "\\toprule",
  " & & \\multicolumn{3}{c}{\\textbf{Validación cruzada}} & \\multicolumn{3}{c}{\\textbf{Kaggle}} \\\\",
  "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
  "\\textbf{Algoritmo} & \\textbf{Versión de referencia} & \\textbf{F1} & \\textbf{Prec.} & \\textbf{Cob.} & \\textbf{Envíos} & \\textbf{Máx. público} & \\textbf{Máx. privado} \\\\",
  "\\midrule", filas, "\\bottomrule", "\\end{tabular}"), "03_slides/tables/best_versions_equipo.tex")

# --- 3b. Figuras de la presentación de comparación, sin CART ni XGBoost de regresión ------------------------------------------
tema <- theme_minimal(base_size = 15) + theme(plot.title = element_text(face = "bold", size = 16), plot.title.position = "plot",
  plot.subtitle = element_text(colour = "grey30", size = 12.5), legend.position = "bottom", panel.grid.minor = element_blank())
cm <- read_csv(file.path(DIR_TAB, "harold_comparison_metrics.csv"), show_col_types = FALSE) |>
  filter(!modelo %in% c("CART", "XGBoost (regresión + umbral)")) |>
  mutate(modelo = fct_reorder(modelo, f1), familia = factor(familia, c("Boosting", "Árboles", "Lineal", "Probabilístico")))
n_cerca <- sum(max(cm$f1) - cm$f1 < 0.01)
p_mod <- ggplot(cm, aes(f1, modelo, colour = familia)) +
  geom_vline(xintercept = max(cm$f1), linetype = "dashed", colour = "grey60") +
  geom_errorbar(aes(xmin = f1 - f1_sd_pliegues, xmax = f1 + f1_sd_pliegues), width = 0.25, orientation = "y", linewidth = 0.8) +
  geom_point(size = 3.6) +
  scale_colour_manual(values = c(Boosting = PAL[["verde"]], "Árboles" = PAL[["naranja"]], Lineal = PAL[["azul"]], Probabilístico = PAL[["gris"]]), name = NULL) +
  scale_x_continuous(labels = \(x) format(x, decimal.mark = ",", nsmall = 2)) +
  labs(title = sprintf("%d de %d modelos quedan a menos de un punto del mejor", n_cerca, nrow(cm)),
       subtitle = "F1 de la clase pobre (media ± 1 DE entre pliegues), validación cruzada sin Bogotá",
       x = "F1 de la clase pobre", y = NULL, caption = "Cada modelo en su mejor versión; la línea punteada marca el máximo.") + tema
ggsave(file.path(DIR_FIG, paste0(AUTOR, "_slide_model_f1_compare.png")), p_mod, width = 9, height = 5.4, dpi = 200, device = ragg::agg_png)
mac("NModelosCerca", c("cero","uno","dos","tres","cuatro","cinco","seis","siete","ocho","nueve","diez")[n_cerca + 1]); mac("NModelosTotal", c("cero","uno","dos","tres","cuatro","cinco","seis","siete","ocho","nueve","diez")[nrow(cm) + 1])

sp <- read_csv(file.path(DIR_TAB, "harold_same_predictors_cv.csv"), show_col_types = FALSE) |> filter(algoritmo != "CART")
sp$algoritmo <- factor(sp$algoritmo, levels = rev(sp$algoritmo))
cerca_sp <- sum(max(sp$f1) - sp$f1 < 0.01)
p_sp <- ggplot(sp, aes(f1, algoritmo)) + geom_errorbarh(aes(xmin = f1 - f1_sd_pliegues, xmax = f1 + f1_sd_pliegues), height = 0.25, colour = PAL["gris"], linewidth = 0.8) +
  geom_point(size = 4.5, colour = PAL["azul"]) + geom_text(aes(label = num(f1, 3)), nudge_y = 0.33, size = 4.4) +
  labs(x = "F1 de la clase pobre (media ± 1 DE entre pliegues)", y = NULL,
       title = "Con las mismas variables, cinco de seis algoritmos quedan a 1 punto de F1",
       subtitle = "Mismas 40 variables, mismos pliegues, umbral que maximiza el F1; fuera de Bogotá") + tema
ggsave(file.path(DIR_FIG, paste0(AUTOR, "_slide_same_predictors_compare.png")), p_sp, width = 9.5, height = 5, dpi = 200, device = ragg::agg_png)
mac("SPRangeCinco", num(max(sp$f1) - min(sp$f1[sp$algoritmo != "Naive Bayes"]), 3))

# --- 4. Matriz de confusión del LightGBM con % por fila ----------------------------------------------------------------
fe  <- readRDS(file.path(DIR_PROC, "hogares_fe.rds")); trn <- fe |> filter(sample == "train")
sc  <- fread(file.path(DIR_PROC, "harold_lightgbm_fe_scores.csv")) |> as_tibble()
s   <- sc$score[match(trn$id, sc$id)]; thr <- read_csv(file.path(DIR_TAB, "harold_lightgbm_fe_cv.csv"), show_col_types = FALSE)$umbral
nb  <- trn$bogota == 0; cm <- conf_metrics(trn$poor[nb], as.integer(s[nb] >= thr))
writeLines(c("\\begin{tabular}{lcc}", "\\toprule", " & \\textbf{Predicho pobre} & \\textbf{Predicho no pobre} \\\\", "\\midrule",
  sprintf("\\textbf{Pobre} & %s (VP) & %s (FN) \\\\", fmtn(cm$tp), fmtn(cm$fn)),
  sprintf(" & {\\footnotesize %s\\,\\%%} & {\\footnotesize %s\\,\\%%} \\\\", pct(cm$tp / (cm$tp + cm$fn)), pct(cm$fn / (cm$tp + cm$fn))),
  sprintf("\\textbf{No pobre} & %s (FP) & %s (VN) \\\\", fmtn(cm$fp), fmtn(cm$tn)),
  sprintf(" & {\\footnotesize %s\\,\\%%} & {\\footnotesize %s\\,\\%%} \\\\", pct(cm$fp / (cm$fp + cm$tn)), pct(cm$tn / (cm$fp + cm$tn))),
  "\\bottomrule", "\\end{tabular}"), "03_slides/tables/confusion_lgbm_pct.tex")

# --- 5. Cifras de Jhon que se citan en el texto (tabla del equipo, protocolo propio de cada autor) ----------------------------
# F1 de validación cruzada anidada sin Bogotá de Jhon; "sin primera etapa" = las mismas 173--179 variables de la v5 sin las de la primera etapa
mac("JhFoneLGBMocho", "0,777"); mac("JhFoneRFveinte", "0,778"); mac("JhPrecRFveinte", "0,740"); mac("JhRecRFveinte", "0,820")
mac("JhFoneLGBMsin", "0,765"); mac("JhFoneXGBsin", "0,761"); mac("JhFoneRFsin", "0,721"); mac("JhFoneXGBocho", "0,777")
mac("JhFoneLGBMmil", "0,773"); mac("JhFoneCorta", "0,706"); mac("JhFoneLogitCorta", "0,689")

writeLines(c("% Generado por 01_code/18_harold_diapositivas_envios_equipo.R; no editar a mano.",
             map_chr(names(macros), \(k) sprintf("\\newcommand{\\%s}{%s}", k, macros[[k]]))), "03_slides/numbers_equipo.tex")
message("Listo: 18_harold_diapositivas_envios_equipo")
