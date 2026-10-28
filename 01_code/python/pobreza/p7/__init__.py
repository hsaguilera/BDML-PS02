r"""Protocolo v7 (fase C del plan maestro): implementación del «Protocolo recomendado» del memo metodológico
(`01_code/python/docs/memo_pronostico.md`, §10 y tabla §11) para los siete algoritmos del enunciado.

Submódulos
----------
entorno     Rutas de lectura y de escritura (separadas: las pruebas escriben sólo en el scratchpad), hilos y presupuestos.
datos       `Datos7`: tablas v7 (o el respaldo S2 + categóricas), verificaciones del Paso 0, particiones repetidas
            (Paso 1), codificación con la etiqueta dentro del pliegue y el mapa variable → preguntas de la encuesta.
metricas    Métricas en el dominio objetivo 𝒯 (sin Bogotá): F1 agregado anidado, log-loss, Brier, AUC, AUC-PR, ACE,
            MCC; EE del F1 por función de influencia y bootstrap; curva F1-umbral y meseta; umbral suavizado.
comparar    t corregido de Nadeau–Bengio, prueba bayesiana correlacionada con ROPE, bootstrap pareado, Holm,
            corrección de Tibshirani–Tibshirani y conjunto de confianza CVC (Lei, 2020).
glm         Caminos lasso / red elástica: MCO con Gram (LPM) y logit por IRLS + descenso por coordenadas con reglas
            fuertes y verificación KKT (Friedman, Hastie y Tibshirani, 2010; Tibshirani et al., 2012).
motores     Los nueve motores (LPM, Logit, EN, CART, RF, LightGBM, XGBoost, CatBoost, NB) con preprocesamiento
            ajustado dentro del pliegue.
cvmotor     Motor de validación cruzada común (Paso 2) con λ_B, parada temprana sólo con 𝒯, semillas y caché en disco.
seleccion   Selección de variables por familia con la pregunta como unidad (Paso 6, §3 y §11.1).
busqueda    Sobol + TPE con poda por la mediana, confirmación R×K×S, OOB en RF, elección y sesgo de selección (Paso 7–8).
final       Modelo final, umbral trasladado, CSV verificado, registro común y señal `3x_LISTO.json` (Paso 11–12).
figuras     Figuras de publicación (español).
"""
