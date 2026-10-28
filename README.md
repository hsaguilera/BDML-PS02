# Problem Set 2: Predicting Poverty

Big Data and Machine Learning para Economía Aplicada (MECA 4107, 2026-20) · Equipo 02

**Integrantes (Equipo 02):**
- Jhon Sebastián Vela Salcedo
- Harold Santiago Aguilera Mulfo 
- Cesar Augusto Vallejo Quiroga 

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
| `LGBM_leaves_10_lr_0_0253_mcs_22_ff_0_442_l2_0_00326_n_8.csv` | Boosting (LightGBM) v5 (principal, 8 variables; script 26) | Boosting | 0,777 | 0,77 | sí (19 oct) |
| `LGBM_leaves_10_lr_0_0253_mcs_22_ff_0_442_l2_0_00326_n_20_pars.csv` | Boosting (LightGBM) v5 (parsimoniosa, 20 variables; script 26) | Boosting | 0,777 |  | no |
| `LGBM_leaves_10_lr_0_0253_mcs_22_ff_0_442_l2_0_00326_n_173_singen.csv` | Boosting (LightGBM) v5 (sin primera etapa, 173 variables; script 26) | Boosting | 0,765 |  | no |
| `XGB_depth_3_eta_0_0473_mcw_1_68_sub_0_561_col_0_876_n_8_bog0_5.csv` | Boosting (XGBoost) v5 (principal, 8 variables; script 27) | Boosting | 0,777 | 0,77 | sí (19 oct) |
| `XGB_depth_3_eta_0_0473_mcw_1_68_sub_0_561_col_0_876_n_173_singen_bog0_5.csv` | Boosting (XGBoost) v5 (sin primera etapa, 173 variables; script 27) | Boosting | 0,761 |  | no |
| `RF_mtry_0_292_minleaf_156_ms_0_443_depth_None_cw_none_trees_600_n_25_bog0_5.csv` | Random Forest v5 (principal, 25 variables; script 28) | Random Forest | 0,777 |  | no |
| `RF_mtry_0_292_minleaf_156_ms_0_443_depth_None_cw_none_trees_360_n_20_pars_bog0_5.csv` | Random Forest v5 (parsimoniosa, 20 variables; script 28) | Random Forest | 0,778 | 0,77 | sí (19 oct) |
| `RF_mtry_0_292_minleaf_156_ms_0_443_depth_None_cw_none_trees_360_n_179_singen_bog0_5.csv` | Random Forest v5 (sin primera etapa, 179 variables; script 28) | Random Forest | 0,721 |  | no |
| `LGBM_hojas_40_lr_0_02_mcs_15_ff_0_336_l2_0_14_n1152_lb1_s10_v7.csv` | Boosting (LightGBM) v7 (principal, 1.152 variables; script 29) | Boosting | 0,773 | 0,77 | sí (20 oct) |
| `LGBM_hojas_40_lr_0_02_mcs_15_ff_0_336_l2_0_14_n25_lb1_s10_corta_v7.csv` | Boosting (LightGBM) v7 (encuesta corta, 25 variables; script 29) | Boosting | 0,706 |  | no |
| `LOGIT_spl_T_n348_lb1_s1_v7.csv` | Logit v7 (principal, 348 variables; script 30) | Logit | 0,763 | 0,76 | sí (21 oct) |
| `LOGIT_spl_T_n25_lb1_s1_corta_v7.csv` | Logit v7 (encuesta corta, 25 variables; script 30) | Logit | 0,689 |  | no |
| `LPM_spl_T_n1152_lb0_s1_v7.csv` | Linear Probability Model v7 (principal, 1.152 variables; script 31) | Linear Probability Model | 0,750 | 0,74 | sí (23 oct) |
| `LPM_spl_T_n25_lb1_s1_corta_v7.csv` | Linear Probability Model v7 (encuesta corta, 25 variables; script 31) | Linear Probability Model | 0,669 |  | no |
| `EN_a_1_lam_0_00022_n414_lb1_s1_v7.csv` | Elastic Net v7 (principal, 414 variables; script 32) | Elastic Net | 0,763 | 0,76 | sí (23 oct) |
| `EN_a_1_lam_0_000279_n386_lb1_s1_pars_v7.csv` | Elastic Net v7 (parsimoniosa, 386 variables; script 32) | Elastic Net | 0,762 |  | no |
| `EN_a_1_lam_0_00022_n25_lb1_s1_corta_v7.csv` | Elastic Net v7 (encuesta corta, 25 variables; script 32) | Elastic Net | 0,688 |  | no |
| `RF_mtry_0_109_hoja_16_ms_0_67_crit_log_loss_arb_1000_n1152_lb1_s1000_v7.csv` | Random Forest v7 (principal, 1.152 variables; script 33) | Random Forest | 0,752 | 0,74 | sí (23 oct) |
| `RF_mtry_0_109_hoja_16_ms_0_67_crit_log_loss_arb_1000_n25_lb1_s1000_corta_v7.csv` | Random Forest v7 (encuesta corta, 25 variables; script 33) | Random Forest | 0,687 |  | no |
| `XGB_prof_7_lr_0_05_mcw_86_4_sub_0_945_col_0_381_l2_0_0027_n1152_lb1_s3_v7.csv` | Boosting (XGBoost) v7 (principal, 1.152 variables; script 34) | Boosting | 0,770 | 0,77 | sí (23 oct) |
| `XGB_prof_7_lr_0_05_mcw_86_4_sub_0_945_col_0_381_l2_0_0027_n25_lb1_s3_corta_v7.csv` | Boosting (XGBoost) v7 (encuesta corta, 25 variables; script 34) | Boosting | 0,704 |  | no |
| `CART_prof_24_hoja_132_mss_35_crit_entropy_cp_0_000104_n111_lb1_s1_v7.csv` | CART v7 (principal, 111 variables; script 35) | CART | 0,721 |  | no |
| `CART_prof_24_hoja_132_mss_35_crit_entropy_cp_0_000149_n83_lb1_s1_pars_v7.csv` | CART v7 (parsimoniosa, 83 variables; script 35) | CART | 0,718 |  | no |
| `CART_prof_24_hoja_132_mss_35_crit_entropy_cp_0_000104_n19_lb1_s1_corta_v7.csv` | CART v7 (encuesta corta, 19 variables; script 35) | CART | 0,673 |  | no |
| `NB_nb_categorico_bins_20_disc_cuantiles_alfa_9_98_n8_lb0_s1_v7.csv` | Naive Bayes v7 (principal, 8 variables; script 36) | Naive Bayes | 0,690 |  | no |
| `NB_nb_categorico_bins_20_disc_cuantiles_alfa_9_98_n25_lb1_s1_corta_v7.csv` | Naive Bayes v7 (encuesta corta, 25 variables; script 36) | Naive Bayes | 0,632 |  | no |

El F1 de validación cruzada de los modelos con ingeniería de variables es el de los hogares fuera de Bogotá; el de la complementaria es el de la tabla `harold_elastic_net_cv_results.tex`. Solo se pueden enviar 4 archivos por día a Kaggle. Los envíos de Jhon y Cesar se agregan con sus prefijos.

Los archivos de los scripts 26–36 conservan el nombre con que se generaron y subieron a Kaggle; su F1 de validación cruzada es el F1 anidado sin Bogotá (el umbral de cada pliegue se elige en los otros pliegues).

## Modelos 26–37 en Python (protocolo v7)

Once modelos y una comparación, escritos en Python y ejecutados desde R como los de Harold. Cada script lleva el
**nombre del archivo de envío tal como se subió a Kaggle** y el número sigue el orden en que se subieron (26–34);
CART y Naive Bayes v7, que no se subieron, van en 35 y 36, y la comparación en 37.

```
01_code/
  26_…_n_8.R … 36_….R        un script por modelo: fija hiperparámetros de búsqueda, presupuesto e hilos y llama a python/
  37_comparacion_modelos_v7.R  comparación de los modelos v7 (pruebas pareadas, conjunto de confianza, política)
  functions/fun_python_modelos.R  llama a Python desde R (intérprete, variables de entorno, errores)
  python/
    26_….py … 37_….py        el script de cada modelo (markdown en comentarios: qué se hace y por qué)
    datos_cadena.py          cadena de datos: la corre cada script si falta la base que necesita
    datos_00_… datos_02d_…   00 ingesta, 01 EDA, 02 variables y pliegues, 09 búsqueda, 02b primera etapa, 02d variables v7
    pobreza/                 paquete: variables, métricas, protocolo v7 (selección, búsqueda, CV, umbral, figuras) y pruebas
    docs/                    memo metodológico del protocolo v7 (memo_pronostico.md) y su bibliografía
02_outputs/
  predictions/               CSV de envío con el nombre con que se generaron (y subieron)
  tables/<script>/, figures/<script>/   tablas y figuras de cada script (26_LGBM_v5 … 37_comparacion_modelos_v7, cadena_datos)
```

**Cómo se corre.** El script maestro (`00_rundirectory.R`) también los ejecuta, en orden. Para uno solo, desde la
raíz: `source("01_code/29_LGBM_hojas_40_lr_0_02_mcs_15_ff_0_336_l2_0_14_n1152_lb1_s10_v7.R")`. Requiere Python ≥ 3.10
(desarrollado con 3.13.5) y `pip install -r requirements.txt`; `PYTHON_BIN` indica el intérprete y `PY_HILOS` los
hilos. La primera vez cada script construye la base que necesita desde `00_data/raw` (cadena `datos_cadena.py`: puede
tardar varias horas, porque la construcción de v7 evalúa cada familia de variables con LightGBM fuera de pliegue; la
etapa v5 basta para 26–28); después la reutiliza. Todo queda en `00_data/processed` (no versionado):
las bases, las predicciones fuera de pliegue (`modelos/`) y las cachés de cada búsqueda, que permiten retomar una
corrida interrumpida donde iba.

**Dos versiones de variables.**
* **v5 (scripts 26–28, envíos del 19 oct): con primera etapa.** `02b_primera_etapa_v5.py` estima el ingreso del
  hogar *fuera de pliegue* (los mismos cinco pliegues de hogares, de modo que ningún hogar aporta su propio ingreso) y
  lo usa como variable de un clasificador. El modelo final es de clasificación y sólo usa variables que existen en
  test, pero la variable más importante es justamente «ingreso esperado frente a la línea»: es una **zona gris**
  frente a la ruta de clasificación del enunciado (§3), que el equipo considera válida y documenta aquí. Cada script
  v5 deja también la versión **sin primera etapa** (`…_singen…csv`) para medir lo que aporta.
* **v7 (scripts 29–36): sin primera etapa por diseño.** `02d_variables_v7.py` construye 1.152 variables candidatas de
  64 preguntas de la encuesta (umbral observable de pobreza, ingresos declarados que también existen en test,
  capacidad de generación de ingreso, vivienda, territorio), sin nada derivado de modelos entrenados con ingresos.
  Todos los algoritmos parten de esas mismas variables, preprocesamiento y particiones; cada uno elige su subconjunto
  con el método que le corresponde (lasso en LPM y Logit, selección por estabilidad en Elastic Net, poda en CART,
  eliminación recursiva con OOB en Random Forest, SHAP en Boosting, mRMR en Naive Bayes).

**Protocolo v7** (detalle y referencias en `01_code/python/docs/memo_pronostico.md`). Bogotá no está en test: toda
métrica, umbral y comparación se calcula sólo fuera de Bogotá, y su peso en el entrenamiento ($\lambda_B$ ∈ {0, ½, 1})
es un hiperparámetro. Métrica principal: **F1 anidado sin Bogotá** (el umbral de cada pliegue se elige en los otros
pliegues), con IC 95 % por función de influencia y bootstrap. Hiperparámetros: búsqueda cuasi-aleatoria (Sobol) y
luego bayesiana (Optuna TPE con poda por la mediana), objetivo log-loss, validación cruzada repetida (hasta 3 × 5
pliegues, estratificados por Pobre × Dominio) y **re-evaluación de la configuración elegida en una partición nueva**
(corrige el optimismo de elegir y evaluar con los mismos datos). Modelo final con el 100 % de train y umbral elegido
fuera de pliegue, trasladado al test igualando la tasa de positivos. Por plazo, los modelos más lentos usaron
presupuestos menores que los del memo («express» en Elastic Net y Random Forest; «minimo» en XGBoost, con la
selección restringida a las variables candidatas de LightGBM), y el diagnóstico de Random Forest usa 150 árboles en
vez de 1.000 (sólo importancia y dependencia parcial; el modelo final y su envío usan 1.000).

**Resultados v7** (`02_outputs/tables/37_comparacion_modelos_v7/`, `resultados_modelos_v7.xlsx`). Ordenados por F1:

| Script | Modelo | F1 sin Bogotá [IC 95 %] | F1 partición nueva | log-loss | AUC | Preguntas | Kaggle |
|---|---|---|---|---|---|---|---|
| 29 | LightGBM | 0,7730 [0,7695–0,7764] | 0,7717 | 0,2113 | 0,957 | 64 | 0,77 |
| 34 | XGBoost | 0,7703 [0,7668–0,7737] | 0,7719 | 0,2133 | 0,956 | 64 | 0,77 |
| 32 | Elastic Net | 0,7628 [0,7594–0,7663] | 0,7613 | 0,2220 | 0,952 | 63 | 0,76 |
| 30 | Logit | 0,7627 [0,7592–0,7662] | 0,7612 | 0,2218 | 0,952 | 63 | 0,76 |
| 33 | Random Forest | 0,7518 [0,7482–0,7553] | 0,7512 | 0,2306 | 0,949 | 64 | 0,74 |
| 31 | LPM | 0,7498 [0,7463–0,7534] | 0,7493 | 0,2546 | 0,944 | 64 | 0,74 |
| 35 | CART | 0,7209 [0,7171–0,7247] | 0,7179 | 0,2565 | 0,935 | 57 | no se subió |
| 36 | Naive Bayes | 0,6903 [0,6863–0,6943] | 0,6897 | 0,2906 | 0,916 | 28 | no se subió |

LightGBM es el único modelo dentro del conjunto de confianza al 90 % (F1, log-loss y Brier); XGBoost es el más
cercano (−0,002 de F1: p de Holm 0,021 frente al mejor y 0,062 con todos los pares; en la partición nueva queda
levemente arriba), y del tercero en adelante la diferencia es de al menos 0,011 con p < 0,001. Con la encuesta corta
(25 variables de 19 preguntas) el mejor F1 baja a 0,706. Referencia v5 con primera etapa (no comparable en las
pruebas pareadas: otras variables): Random Forest v5 0,7778, LightGBM v5 0,7775, XGBoost v5 0,7771; es decir, la primera etapa aporta unos 0,004–0,005 de F1 sobre el mejor
v7 y en Kaggle ambos dan 0,77.

**Correspondencia entre scripts y salidas.** Las tablas y figuras conservan el prefijo del cuaderno con que se
produjeron (p. ej. `35_umbral_pr.png` en `02_outputs/figures/29_LGBM_v7/`):

| Script | Cuaderno interno | Prefijo de tablas y figuras | Carpeta de tablas y figuras | Presupuesto | Hilos | Tiempo aprox. |
|---|---|---|---|---|---|---|
| `26_LGBM_leaves_10_lr_0_0253_mcs_22_ff_0_442_l2_0_00326_n_8` | `10_lightgbm` | `10_` | `26_LGBM_v5/` | — | 6 | 22 min |
| `27_XGB_depth_3_eta_0_0473_mcw_1_68_sub_0_561_col_0_876_n_8_bog0_5` | `11_xgboost` | `11_` | `27_XGB_v5/` | — | 6 | 32 min |
| `28_RF_mtry_0_292_minleaf_156_ms_0_443_depth_None_cw_none_trees_360_n_20_pars_bog0_5` | `12_random_forest` | `12_` | `28_RF_v5/` | — | 6 | 45 min |
| `29_LGBM_hojas_40_lr_0_02_mcs_15_ff_0_336_l2_0_14_n1152_lb1_s10_v7` | `35_lightgbm` | `35_` | `29_LGBM_v7/` | reducido | 8 | 5,0 h |
| `30_LOGIT_spl_T_n348_lb1_s1_v7` | `31_logit` | `31_` | `30_LOGIT_v7/` | reducido | 3 | 5,1 h |
| `31_LPM_spl_T_n1152_lb0_s1_v7` | `30_lpm` | `30_` | `31_LPM_v7/` | reducido | 2 | 76 min |
| `32_EN_a_1_lam_0_00022_n414_lb1_s1_v7` | `32_elastic_net` | `32_` | `32_EN_v7/` | express | 3 | 3,0 h |
| `33_RF_mtry_0_109_hoja_16_ms_0_67_crit_log_loss_arb_1000_n1152_lb1_s1000_v7` | `34_random_forest` | `34_` | `33_RF_v7/` | express | 12 | ≈ 4,0 h (estimado) |
| `34_XGB_prof_7_lr_0_05_mcw_86_4_sub_0_945_col_0_381_l2_0_0027_n1152_lb1_s3_v7` | `36_xgboost` | `36_` | `34_XGB_v7/` | minimo | 12 | 7,7 h |
| `35_CART_prof_24_hoja_132_mss_35_crit_entropy_cp_0_000104_n111_lb1_s1_v7` | `33_cart` | `33_` | `35_CART_v7/` | reducido | 3 | 3,6 h |
| `36_NB_nb_categorico_bins_20_disc_cuantiles_alfa_9_98_n8_lb0_s1_v7` | `38_naive_bayes` | `38_` | `36_NB_v7/` | reducido | 2 | 22 min |
| `37_comparacion_modelos_v7` | `39_comparacion_v7` | `39_` | `37_comparacion_modelos_v7/` | — | 4 | 15 min |

Los tiempos son los de la última corrida registrada en la señal `*_LISTO.json` de cada script, en un PC de 20 hilos
lógicos y 24 GB de RAM; cuando una corrida se retomó desde la caché, la completa fue más larga (el de Random Forest v7
es una estimación, porque su última corrida se retomó casi al final). Las cifras de esta sección no se digitan a mano: salen
de `39_resumen.csv` y de las señales `*_LISTO.json` de cada script. Los seis CSV v7 subidos a Kaggle se verificaron
además con un guion independiente: 66.168 filas en el orden de `sample_submission.csv`, valores 0/1, reproducción
exacta desde los puntajes de test, F1 recalculado con dos implementaciones y ninguna variable que no exista en test.

## Convenciones
- Nombres en `snake_case` con prefijo numérico que indica el orden de ejecución.
- Una tarea por script; el código repetido va como función en `01_code/functions/`.
- Las salidas las generan los scripts, nunca se editan a mano. Sin rutas absolutas. El script maestro también guarda `02_outputs/session_info.txt` con la versión de R y de los paquetes.
- Semilla global: 2026.

## Entorno
R ≥ 4.3 (desarrollado con 4.3.3). Paquetes, instalados y cargados con `pacman::p_load()` desde `00_rundirectory.R`: tidyverse, data.table, glmnet, caret, rpart, ranger, gbm, e1071, ragg, pROC, xtable, jsonlite, ggrepel, scales y here.

**Python (scripts 05, 06, 09, 10, 12 y 14).** Python ≥ 3.10 con los paquetes de `requirements.txt` (`pip install -r requirements.txt`); el script maestro los llama con `system2()` (variables de entorno opcionales: `PYTHON_BIN`, `PY_THREADS`, `RERUN_PY=1` para reentrenar el modelo de ingreso por persona). Tiempo aproximado en 2 núcleos: scripts 05 y 06, 25 min; 07, 25 min; 08, 1,5 h; 09, 20 min; 10, 15 min; 11, 10 min; 12, 40 min; 13, 1 min; 14, 3 min. El archivo vacío `.here` marca la raíz del proyecto para el paquete `here`.

**Nota sobre `04_harold_elastic_net.R`:** el ejemplo pide `metric = "Accuracy"`, pero `twoClassSummary` no calcula accuracy; `caret` avisa y selecciona por ROC. Se replicó así. Además, `caret` guarda en memoria las predicciones de toda la grilla con `savePredictions = "final"` (unos 8 GB de RAM), por lo que la función recorre la grilla sin guardarlas y vuelve a ajustar solo la combinación ganadora; el resultado es el mismo.
