# Problem Set 2: Predicting Poverty

Big Data and Machine Learning para Economía Aplicada (MECA 4107, 2026-20) · Equipo 02

**Integrantes (Equipo 02):**
- Jhon Sebastián Vela Salcedo (`jhon`)
- Harold Santiago Aguilera Mulfo (`harold`)
- Cesar Augusto Vallejo Quiroga (`cesar`)

Entre paréntesis, el prefijo que cada integrante usa en los nombres de sus scripts y salidas.

## Pregunta
¿Cuál es el mejor modelo para predecir pobreza en Colombia a nivel de hogar y qué implica para la política pública?
Métrica en Kaggle: F1 de la clase "pobre".

## Replicación en un comando
1. Descargar los datos de la competencia de Kaggle y copiarlos en `00_data/raw/`
   (`train_hogares.csv`, `train_personas.csv`, `test_hogares.csv`, `test_personas.csv`, `sample_submission.csv`).
2. Desde la raíz del repositorio, en R:
```r
source("01_code/00_rundirectory.R")
```

## Estructura
```
00_data/raw/             datos originales de Kaggle (no versionados; ver 00_data/README.md)
00_data/processed/       bases intermedias y puntajes de los modelos (no versionados; los regenera el pipeline)
01_code/
  00_rundirectory.R      script maestro: paquetes, parámetros, semilla y orden de ejecución
  01_build_household_dataset.R   limpieza y unión de personas con hogares
  02_exploratory_analysis.R      desbalance de clases y relaciones con los predictores
  03_characterize_data.R         comparación train-test, faltantes y gráficos descriptivos
  04_harold_elastic_net.R        réplica del modelo de la complementaria del 19 de septiembre
  05_harold_feature_engineering.R  variables nuevas + ingreso imputado por persona (llama a Python)
  06_harold_feature_diagnostics.R  importancia de variables y ablación por grupos
  07_harold_elastic_net_fe.R     modelo 1: Elastic Net con splines
  08_harold_random_forest_fe.R   modelo 2: Random Forest (ranger) con selección de variables
  09_harold_lightgbm_fe.R        modelo 3: LightGBM (Python)
  10_harold_xgboost_reg_fe.R     modelo 4: XGBoost de regresión con umbral (Python)
  11_harold_classical_fe.R       LPM, Logit, CART y Naive Bayes con las 40 variables del Random Forest
  12_harold_robust_logit.R       logit y probit regularizados + pruebas de estrés de generalización
  13_harold_model_comparison.R   comparación de todos los modelos (métricas, errores de política, acuerdo en test)
  14_harold_lightgbm_importance.R  importancia SHAP y efectos del mejor modelo (Python)
  15_harold_same_predictors.R    Elastic Net y LightGBM con las mismas 40 variables del Random Forest (igualdad de condiciones)
  16_harold_short_survey.R       LightGBM con una encuesta corta (S1 y S2) frente al conjunto completo
  17_harold_slide_assets.R       figuras y tablas (.tex) y cifras (numbers.tex) que alimentan las diapositivas
  python/                scripts de Python que llaman los scripts 05, 06, 09, 10, 12, 14, 15 y 16 (no se ejecutan a mano)
  functions/             funciones reutilizables
02_outputs/
  figures/               gráficos (PNG)
  tables/                tablas (.tex y .csv)
  predictions/           envíos a Kaggle (.csv)
  session_info.txt       versiones de R, paquetes y Python con que se generaron las salidas
03_slides/               diapositivas en LaTeX (Beamer): best_equipo_02.tex, compare_equipo_02.tex y los PDF finales
requirements.txt         paquetes de Python
```

## Salidas
Todas se generan automáticamente al correr `00_rundirectory.R` y se guardan en `02_outputs/`: `figures/` (gráficos en PNG), `tables/` (tablas en `.tex` y `.csv`) y `predictions/` (envíos a Kaggle en `.csv`, con el modelo y los hiperparámetros en el nombre del archivo).

## Scripts
| Script | Responsabilidad | Salidas |
|---|---|---|
| `00_rundirectory.R` | Paquetes, parámetros globales (semilla 2026, 5 pliegues, rutas) y ejecución en orden de los demás scripts | — |
| `01_build_household_dataset.R` | Lee y valida las cuatro bases, procesa personas y hogares con las mismas funciones y construye las variables a nivel de hogar. Solo usa variables que existen en test | `00_data/processed/hogares_modelo.rds` |
| `02_exploratory_analysis.R` | Desbalance de clases, pobreza por dominio, ingreso vs. línea de pobreza, correlaciones y gradientes económicos | `eda_*.png`, `eda_means_by_poverty.tex` |
| `03_characterize_data.R` | Comparabilidad train-test, datos faltantes, pobreza por tamaño del hogar, ciclo de vida del jefe, educación por zona, tenencia, interacciones y correlaciones entre predictores | `desc_*.png`, `desc_covariate_shift.tex` |
| `04_harold_elastic_net.R` | Réplica del ejemplo de la complementaria del 19 de septiembre (logit con Elastic Net en `caret`, 5 particiones estratificadas, semilla 2026) con sus propias variables de hogar. Corre tres variantes de Bogotá: `replica` (como el ejemplo), `eval_sin_bogota` (entrena con Bogotá y valida sin ella) y `sin_bogota` (la excluye) | `harold_elastic_net_lambda_*_alpha_*[_variante].csv`, `harold_elastic_net_cv_results.tex` |
| `05_harold_feature_engineering.R` | Crea 72 variables (`fe_*`): estructura de la regla de pobreza (arriendo imputado), hacinamiento, composición, capital humano, inserción laboral, privaciones e interacciones. Entrena en Python (`python/harold_person_income.py`) un LightGBM por persona que imputa el ingreso individual fuera de muestra con las variables que existen en test y lo agrega al hogar. Crea los pliegues compartidos `fold` | `00_data/processed/hogares_fe.rds`, `harold_person_income_quality.tex` |
| `06_harold_feature_diagnostics.R` | Diagnóstico antes de modelar (`python/harold_feature_diagnostics.py`): AUC univariado lineal y no lineal, información mutua, importancia por ganancia y |SHAP|, y ablación por grupos de variables (A base, B ratios e interacciones, C estructura de la regla, D ingreso imputado) | `harold_fe_{univariate,importance,ablation}.csv`, `harold_fe_ablation.tex`, `harold_fe_*.png` |
| `07_harold_elastic_net_fe.R` | Modelo 1: Elastic Net (`glmnet`) con log1p, winsorización, splines naturales y dummies; malla de alpha, camino de lambda, umbral que maximiza F1 | `harold_elastic_net_fe_alpha_*_lambda_*.csv`, `harold_elastic_net_fe_{cv,coefs}.*`, `.png` |
| `08_harold_random_forest_fe.R` | Modelo 2: Random Forest (`ranger`) con las 40 variables de mayor |SHAP|; búsqueda de `mtry`, `min.node.size` y pesos de clase | `harold_random_forest_fe_mtry_*_node_*.csv`, `harold_random_forest_fe_{search,cv,vars}.*`, `.png` |
| `09_harold_lightgbm_fe.R` | Modelo 3: LightGBM de clasificación (`python/harold_lightgbm.py`); búsqueda aleatoria de hiperparámetros con parada temprana y reentrenamiento con tasa de aprendizaje baja | `harold_lightgbm_fe_leaves_*_rounds_*.csv`, `harold_lightgbm_fe_{search,cv}.*`, `.png` |
| `10_harold_xgboost_reg_fe.R` | Modelo 4: XGBoost de regresión sobre log(ingreso per cápita / línea de pobreza) con umbral para F1 (`python/harold_xgboost_reg.py`) | `harold_xgboost_reg_fe_depth_*_rounds_*.csv`, `harold_xgboost_reg_fe_{search,cv}.*`, `.png` |
| `11_harold_classical_fe.R` | Modelos 5 a 8, algoritmos exigidos por el taller: LPM, Logit, CART (malla de `cp` y `minsplit`) y Naive Bayes (variables en deciles), con las mismas 40 variables del Random Forest para comparar en igualdad de condiciones | `harold_{lpm,logit,cart,naive_bayes}_fe_*.csv`, `harold_classical_fe_cv.*`, `harold_cart_fe_search.csv` |
| `12_harold_robust_logit.R` | Logit y probit diseñados para generalizar: eliminación de variables casi duplicadas (|r| ≥ 0,95), selección por estabilidad (40 submuestras), Elastic Net con validación repetida y regla de 1 error estándar, umbral en la meseta del F1. Pruebas de estrés: brecha dentro/fuera de muestra, validación agrupada por dominio (con LightGBM en `python/harold_grouped_cv_lightgbm.py`), calibración y ruido de un 20 % del test | `harold_{logit,probit}_robust_fe_*.csv`, `harold_robust_*.csv`, `harold_robust_*.png` |
| `13_harold_model_comparison.R` | Compara los 10 modelos con el mismo protocolo (OOF, 5 pliegues, sin Bogotá): F1 con su desvío entre pliegues, precisión, cobertura, filtración (*leakage*), subcobertura y AUC; curvas precisión-cobertura; y acuerdo (Jaccard) entre las predicciones de test | `harold_comparison_*.{csv,tex,png}` |
| `14_harold_lightgbm_importance.R` | Interpreta el mejor modelo: reentrena el LightGBM con los hiperparámetros del script 09 (`python/harold_lightgbm_shap.py`), importancia SHAP y por ganancia, y efecto SHAP medio de las seis variables principales | `harold_lightgbm_fe_importance.{csv,tex,png}`, `harold_lightgbm_fe_effects.png` |
| `15_harold_same_predictors.R` | Elastic Net (`cv.glmnet`, malla de alpha, mismos pliegues) y LightGBM con las 40 variables del Random Forest, para que las diferencias de F1 reflejen el algoritmo y no el conjunto de predictores | `harold_same_predictors_cv.{csv,tex}`, `harold_{elastic_net,lightgbm}_40vars_scores.csv` |
| `16_harold_short_survey.R` | Pregunta de despliegue: LightGBM con una encuesta corta (S1: hogar y vivienda; S2: S1 más composición y educación) frente al modelo completo con ingreso imputado | `harold_short_survey_{vars,cv}.csv`, `harold_lightgbm_{s1,s2}_scores.csv` |
| `17_harold_slide_assets.R` | Genera las figuras `harold_slide_*.png`, las tablas de `03_slides/tables/` y las cifras de `03_slides/numbers.tex`; depende de las salidas de los scripts 09 a 16 | `harold_slide_*.png`, `03_slides/numbers.tex`, `03_slides/tables/*.tex` |

| Función (`01_code/functions/`) | Qué hace |
|---|---|
| `fun_data_prep.R` | `fun_preprocess_personas()`, `fun_collapse_personas()`, `fun_preprocess_hogares()`, `fun_add_ratios()`, `safe_max()` |
| `fun_plot_helpers.R` | Tema, paleta, `save_fig()` e intervalo de Wilson para proporciones |
| `fun_comp19.R` | Funciones del ejemplo de la complementaria con sufijo `_comp19`: `fun_preprocess_personas_comp19()`, `fun_colapse_personas_comp19()`, `fun_hogares_comp19()`, `fun_train_en_comp19()`, `fun_save_submission_comp19()` |
| `fun_features_fe.R` | `fun_arriendo_imputado()`, `fun_person_income_table()`, `fun_add_engineered()`, `fun_add_income_hat()` |
| `fun_models_fe.R` | `fun_auc()`, `fun_f1_curve()`, `fun_eval_oof()` (F1 con umbral óptimo, sin Bogotá), `fun_write_submission_fe()` (valida contra `sample_submission.csv`), `fun_save_scores()`, `fun_matrix_en()`, `fun_export_fe_csv()` (con columnas extra opcionales) |
| `fun_metrics.R` | `f1_score()`, `conf_metrics()` (precisión, recall, *leakage*, *undercoverage*), `best_threshold()`, `make_folds()` |

Los demás modelos y envíos a Kaggle se agregan como `18_*.R`, `19_*.R`, … y escriben en `02_outputs/predictions/`.


## Diapositivas
Los dos PDF (`best_equipo_02.pdf`, `compare_equipo_02.pdf`) se compilan desde `03_slides/` con LaTeX (pdflatex + Beamer):

```
cd 03_slides
latexmk -pdf best_equipo_02.tex compare_equipo_02.tex
```

Todas las cifras de las diapositivas se leen de `numbers.tex` (generado por el script 17), de modo que no se digitan a mano. Los puntajes públicos de Kaggle se escriben en `03_slides/kaggle_scores.tex` (macros `\kgLGBM`, `\kgRF`, etc.) y se recompila.

## Resultados
Validación cruzada de 5 pliegues compartidos, hogares fuera de Bogotá (no existe en test) y umbral que maximiza el F1 de la clase pobre (`13_harold_model_comparison.R`). Entre paréntesis, la desviación estándar del F1 entre pliegues. *Filtración* = % de los clasificados como pobres que no lo son; *subcobertura* = % de los pobres que quedan excluidos.

| Modelo | F1 | Precisión | Cobertura | Filtración | Subcobertura | AUC | Enviado a Kaggle |
|---|---|---|---|---|---|---|---|
| LightGBM | 0,777 (±0,006) | 0,746 | 0,811 | 0,254 | 0,189 | 0,958 | sí |
| XGBoost (regresión + umbral) | 0,775 (±0,006) | 0,749 | 0,803 | 0,251 | 0,197 | 0,951 | sí |
| Random Forest | 0,774 (±0,006) | 0,748 | 0,801 | 0,252 | 0,199 | 0,956 | sí |
| Elastic Net (alpha 0,7) | 0,772 (±0,004) | 0,730 | 0,819 | 0,270 | 0,181 | 0,955 | sí |
| Logit robusto (59 vars) | 0,771 (±0,005) | 0,739 | 0,806 | 0,261 | 0,194 | 0,954 | sí |
| Logit (40 vars) | 0,771 (±0,005) | 0,738 | 0,807 | 0,262 | 0,193 | 0,954 | no |
| Probit robusto (59 vars) | 0,771 (±0,004) | 0,739 | 0,807 | 0,261 | 0,193 | 0,954 | sí |
| LPM (40 vars) | 0,770 (±0,006) | 0,728 | 0,816 | 0,272 | 0,184 | 0,951 | no |
| CART | 0,769 (±0,005) | 0,755 | 0,783 | 0,245 | 0,217 | 0,871 | no |
| Naive Bayes (deciles) | 0,758 (±0,005) | 0,694 | 0,834 | 0,306 | 0,166 | 0,948 | no |

- Las variables de ingreso imputado fuera de muestra (grupo D) son las que mejoran el ajuste: el F1 de LightGBM pasa de 0,748 con las variables base a 0,776 al incorporarlas (`harold_fe_ablation.*`); el LightGBM final las usa como sus tres variables más importantes (`harold_lightgbm_fe_importance.png`).
- Los ocho mejores modelos quedan a menos de un punto de F1 entre sí, dentro del ruido entre pliegues. Entre un logit penalizado de 59 variables y un LightGBM la brecha es de 0,006 y se reduce al validar por regiones completas (`harold_robust_grouped_cv.*`), por eso se enviaron ambos.
- El Elastic Net de la complementaria (`harold_elastic_net_lambda_0_001_alpha_0_45.csv`) clasifica como pobre solo el 10 % de los hogares de test con umbral 0,5 (F1 de validación 0,445). Sirve de punto de partida, no de candidato.

Gráficos principales: `harold_comparison_f1.png`, `harold_comparison_precision_recall.png`, `harold_comparison_errors.png`, `harold_comparison_agreement.png`, `harold_lightgbm_fe_importance.png`, `harold_lightgbm_fe_effects.png`, `harold_robust_calibration.png`, `harold_robust_grouped_cv.png`.

## Decisiones metodológicas
1. **Mismo pre-procesamiento en train y test:** las bases se apilan y se procesan con las mismas funciones. Las variables de ingreso, que solo existen en train, se guardan con prefijo `eda_` y no se usan como predictores.
2. **Unidad de gasto (UG):** el DANE calcula la pobreza sobre la UG, que excluye empleados domésticos, pensionistas y trabajadores (`P6050` = 6, 7, 8). Las variables del hogar se agregan solo sobre la UG (su tamaño coincide con `Npersug` en todos los hogares) y `n_outside_ug` registra cuántas personas quedan fuera.
3. **Faltantes:** los 9 ("no sabe") pasan a faltante; los faltantes numéricos se imputan con la mediana de train; cuando una tasa no está definida (sin adultos o sin ocupados) se crea un indicador (`bin_no_adults`, `no_earner`).
4. **Educación:** años aproximados por nivel máximo aprobado (ninguno y preescolar = 0, primaria = 5, secundaria = 9, media = 11, superior = 15).
5. **Arriendo:** pagado (`P5140`) o imputado a propietarios (`P5130`); 0 con indicador `rent_missing` cuando no aplica.
6. **Bogotá solo está en train:** los 10.567 hogares de Bogotá (6,4 % de train, 9,8 % de pobreza) no tienen contraparte en test. Se documenta aquí; en `04_harold_elastic_net.R` se comparan tres tratamientos (entrenar con todo, entrenar con Bogotá y validar sin ella, excluirla).

## Trabajo en equipo
Un solo repositorio y un script por modelo. Así se reparten los modelos sin pisarse:

1. **El autor va en el nombre.** Los scripts de modelos se llaman `NN_<autor>_<modelo>.R` y sus salidas `<autor>_<modelo>_<detalle>.<ext>`, con el nombre de pila en minúsculas. Ejemplos: `15_jhon_random_forest.R`, `harold_elastic_net_lambda_0_001_alpha_0_45.csv`, `harold_elastic_net_cv_results.tex`. Los scripts `01` a `03` y `functions/` son del equipo y no llevan nombre.
2. **Numeración única.** El prefijo numérico indica el orden de ejecución y es único en todo el repositorio (`04_harold_...`, `15_jhon_...`, `16_...`). El script maestro los corre en ese orden, así que el siguiente número libre se reserva al abrir la rama.
3. **Lo compartido no se duplica.** Los mismos predictores, el mismo preprocesamiento y las mismas particiones valen para todos los algoritmos (es lo que exige la comparación entre modelos): cada modelo lee `00_data/processed/hogares_modelo.rds` y usa `make_folds()` con la semilla 2026. Las funciones comunes (métricas, umbral, formato de envío) van en `01_code/functions/`; si se cambian, o se cambia el script `01`, se avisa al equipo.
4. **Cada script es autónomo.** Define `AUTOR` al inicio, fija sus semillas, lee sus insumos de `00_data/processed/` y escribe solo en `02_outputs/`, de modo que el script maestro los ejecute sin pasos manuales. No se edita el script de otra persona sin avisarle.
5. **Git.** Una rama por tarea (`modelo/<autor>-<algoritmo>`), commits pequeños con mensaje claro y *pull request* a `main`, revisado por otra persona. Cada integrante necesita al menos 5 contribuciones sustanciales en `main`; el historial de Git es la evidencia, y los nombres de archivo permiten ver de un vistazo quién hizo qué.

### Registro de envíos a Kaggle
Cada fila es un archivo de `02_outputs/predictions/`. La columna de Kaggle se completa con el puntaje público al enviarlo.

| Archivo | Modelo | Algoritmo del taller | F1 en validación cruzada | F1 en Kaggle (público) | Enviado |
|---|---|---|---|---|---|
| `harold_elastic_net_lambda_0_001_alpha_0_45.csv` | Elastic Net de la complementaria (umbral 0,5) | Elastic Net | 0,445 | | sí |
| `harold_elastic_net_fe_alpha_0_7_lambda_0_00004.csv` | Elastic Net con ingeniería de variables | Elastic Net | 0,772 | | sí |
| `harold_logit_robust_fe_alpha_1_lambda_0_00047.csv` | Logit regularizado y robusto (59 variables) | Logit | 0,771 | | sí |
| `harold_probit_robust_fe_59vars.csv` | Probit con las mismas 59 variables | (variante del Logit) | 0,771 | | sí |
| `harold_random_forest_fe_mtry_14_node_150.csv` | Random Forest | Random Forest | 0,774 | | sí |
| `harold_lightgbm_fe_leaves_15_rounds_954.csv` | LightGBM | Boosting | 0,777 | | sí |
| `harold_xgboost_reg_fe_depth_6_rounds_910.csv` | XGBoost de regresión con umbral | Boosting (regresión) | 0,775 | | sí |
| `harold_lpm_fe_40vars.csv` | LPM | LPM | 0,770 | | no |
| `harold_logit_fe_40vars.csv` | Logit sin penalización | Logit | 0,771 | | no |
| `harold_cart_fe_cp_0_001.csv` | CART | CART | 0,769 | | no |
| `harold_naive_bayes_fe_deciles.csv` | Naive Bayes | Naive Bayes | 0,758 | | no |
| `harold_elastic_net_lambda_0_0018_alpha_0_2_eval_sin_bogota.csv` | Elastic Net de la complementaria, valida sin Bogotá | Elastic Net | 0,451 | | no (99,95 % de coincidencia con la réplica) |
| `harold_elastic_net_lambda_0_001_alpha_0_2_sin_bogota.csv` | Elastic Net de la complementaria, sin Bogotá | Elastic Net | 0,453 | | no (99,8 % de coincidencia con la réplica) |

El F1 de validación cruzada de los modelos con ingeniería de variables es el de los hogares fuera de Bogotá; el de la complementaria es el de la tabla `harold_elastic_net_cv_results.tex`. Solo se pueden enviar 4 archivos por día a Kaggle. Los envíos de Jhon y Cesar se agregan con sus prefijos.

## Convenciones
- Nombres en `snake_case` con prefijo numérico que indica el orden de ejecución.
- Una tarea por script; el código repetido va como función en `01_code/functions/`.
- Las salidas las generan los scripts, nunca se editan a mano. Sin rutas absolutas. El script maestro también guarda `02_outputs/session_info.txt` con la versión de R y de los paquetes.
- Semilla global: 2026.

## Entorno
R ≥ 4.3 (desarrollado con 4.3.3). Paquetes, instalados y cargados con `pacman::p_load()` desde `00_rundirectory.R`: tidyverse, data.table, glmnet, caret, rpart, ranger, gbm, e1071, ragg, pROC, xtable, jsonlite, ggrepel, scales y here.

**Python (scripts 05, 06, 09, 10, 12 y 14).** Python ≥ 3.10 con los paquetes de `requirements.txt` (`pip install -r requirements.txt`); el script maestro los llama con `system2()` (variables de entorno opcionales: `PYTHON_BIN`, `PY_THREADS`, `RERUN_PY=1` para reentrenar el modelo de ingreso por persona). Tiempo aproximado en 2 núcleos: scripts 05 y 06, 25 min; 07, 25 min; 08, 1,5 h; 09, 20 min; 10, 15 min; 11, 10 min; 12, 40 min; 13, 1 min; 14, 3 min. El archivo vacío `.here` marca la raíz del proyecto para el paquete `here`.

**Nota sobre `04_harold_elastic_net.R`:** el ejemplo pide `metric = "Accuracy"`, pero `twoClassSummary` no calcula accuracy; `caret` avisa y selecciona por ROC. Se replicó así. Además, `caret` guarda en memoria las predicciones de toda la grilla con `savePredictions = "final"` (unos 8 GB de RAM), por lo que la función recorre la grilla sin guardarlas y vuelve a ajustar solo la combinación ganadora; el resultado es el mismo.
