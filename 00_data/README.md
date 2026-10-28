# Datos

Los datos no se versionan: `train_personas.csv` pesa 242 MB (GitHub rechaza archivos de más de 100 MB) y las bases son
del concurso de Kaggle del curso. Las carpetas `raw/` y `processed/` solo conservan un `.gitkeep`
(ver `.gitignore`).

## `raw/` (se descargan de Kaggle: `uniandes-bdml-2026_20-ps2`)
| Archivo | Contenido | Tamaño aprox. |
|---|---|---|
| `train_hogares.csv` | Hogares de entrenamiento, con `Pobre` e ingresos | 29 MB |
| `train_personas.csv` | Personas de los hogares de entrenamiento | 242 MB |
| `test_hogares.csv` | Hogares de prueba (sin `Pobre` ni ingresos) | 9 MB |
| `test_personas.csv` | Personas de los hogares de prueba | 51 MB |
| `sample_submission.csv` | Formato del envío (`id`, `pobre`); los scripts validan cada envío contra este archivo | 2 MB |

Diccionario de variables: `ddi-documentation-spanish-608.pdf` (documento del curso; no se versiona).

## `processed/` (los genera el pipeline; se pueden borrar y regenerar)
| Archivo | Lo crea | Contenido |
|---|---|---|
| `hogares_modelo.rds` | `01_harold_datos_construir_base_hogares.R` | Base de hogares con las variables de personas agregadas |
| `personas_ingreso_in.csv`, `personas_ingreso_out.csv` | `05_harold_datos_ingenieria_variables_ingreso_imputado.R` | Entrada y salida del modelo de ingreso por persona (Python); son temporales |
| `hogares_fe.rds`, `hogares_fe_grupos.{rds,csv}` | `05` y `06` | Base final de modelado (hogares × variables, columna `fold` con los 5 pliegues compartidos) y catálogo de variables por grupo (A, B, C, D) |
| `harold_<modelo>_scores.csv` | scripts 07 a 12 | Puntajes fuera de muestra (train) y de test de cada modelo (`id`, `sample`, `fold`, `score`); los usan los scripts 13 y 14 |
| `harold_*_params.json`, `harold_*_resumen.rds`, `*_cache.rds` | scripts 07 a 12 | Hiperparámetros elegidos y cachés para retomar búsquedas largas |
