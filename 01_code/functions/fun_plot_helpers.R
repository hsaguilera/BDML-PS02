# ==============================================================================
# fun_plot_helpers.R  --  Tema y guardado de figuras (usado por 02 y 03)
# ==============================================================================

#' Paleta del proyecto
COL_POBRE <- c("0" = "#4C78A8", "1" = "#E45756")
COL_ACENTO <- "#E45756"
COL_NEUTRO <- "#4C78A8"

#' Tema común de las figuras
theme_pobreza <- function() {
  ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(plot.title = ggplot2::element_text(face = "bold"), plot.title.position = "plot",
                   plot.subtitle = ggplot2::element_text(colour = "grey30"),
                   legend.position = "bottom")
}

#' Guardar una figura en 02_outputs/figures/ (PNG, UTF-8 con ragg)
#' @param p objeto ggplot. @param name nombre del archivo. @param w,h tamaño en pulgadas.
save_fig <- function(p, name, w = 7, h = 4.5) {
  ggplot2::ggsave(file.path(DIR_FIG, name), p, width = w, height = h, dpi = 200, device = ragg::agg_png)
}

#' Intervalo de confianza (Wilson, 95 %) para una proporción
#' @param k éxitos. @param n tamaño de muestra.
wilson_ci <- function(k, n, z = 1.96) {
  p <- k / n; den <- 1 + z^2 / n
  centre <- (p + z^2 / (2 * n)) / den
  half <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / den
  tibble::tibble(lo = centre - half, hi = centre + half)
}
