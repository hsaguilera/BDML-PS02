# Memo metodológico para el pronóstico de pobreza (PS2, BDML 2026-20) — fases C y D

*Versión 1, 19–20 de octubre de 2026. Fase A del plan v7. Referencias verificadas en `referencias_metodologia.bib`
(misma carpeta); todas las entradas llevan en `note` la URL de la fuente primaria con la que se comprobaron. No se subió
nada a Kaggle.*

**Para qué sirve este memo.** Fija, con justificación formal y referencias, cómo deben evaluarse, ajustarse,
seleccionarse y compararse los siete algoritmos permitidos (LPM, Logit, Elastic Net, CART, Random Forest, Boosting y
Naive Bayes) para la clasificación binaria de pobreza del hogar con métrica F1, y cómo tratar a Bogotá (presente en
entrenamiento y ausente del test). Cada sección trae: síntesis de la teoría, recomendación concreta para este problema,
valores por defecto, cómo verificarlo empíricamente aquí y riesgos. La §10 es el protocolo paso a paso que la fase C
puede programar directamente; la §11 es la tabla por algoritmo. Las cifras «propias» provienen de comprobaciones
pequeñas hechas para este memo sobre `00_data/processed/modelo_train.parquet` (180 variables permitidas, **sin** las de
primera etapa) con los pliegues actuales. Los scripts y resultados están en `docs/metodologia/comprobaciones/`:
`exp_bogota.py` (etapas `lambda`, `dominio socio` y `placebo`), `exp_logit.py`, `exp_curva.py`, los análisis
`anal_lambda.py` y `anal_placebo.py` (el primero lee `oof_lambda.npz`, que se regenera con la etapa `lambda`), los
`chk*.py` descriptivos, y los `res_*.json` y `log_*.txt` con las cifras citadas (§1.5–1.7, §2.1–2.2, §8.1). Ninguno
escribe en `data/`. Con las variables v7 los niveles cambiarán; las conclusiones metodológicas no deberían cambiar.

---

## 0. Resumen ejecutivo

1. **Bogotá — evaluación (sin margen de decisión).** El test es una muestra aleatoria de los 24 dominios objetivo
   ($\mathcal T$). El riesgo que importa es $R_{\mathcal T}$, y la CV ponderada por importancia con los pesos *exactos*
   ($w=\mathbb 1\{d\neq B\}/(1-\pi_B)$) equivale a medir **sólo en $\mathcal T$**. Umbral, prevalencia esperada
   (20,75 % ± 0,16 pp), métricas, comparaciones y selección: siempre sin Bogotá (§1.2).
2. **Bogotá — entrenamiento (se mide, no se presupone).** Bogotá difiere mucho en $P(X)$: $L_p$ tiene soporte disjunto
   (0 de sus 12 valores aparecen en el test) y un clasificador la separa de las demás cabeceras con AUC 0,90–0,94. En
   cambio, $P(Y\mid X)$ es casi la misma: un LightGBM que nunca vio Bogotá la ordena con AUC 0,972, frente a 0,973 si la
   ve. La teoría se ordena así:
   - La cota de Ben-David et al. (peor caso) no puede certificar ganancia.
   - Con el dominio como variable, la condicional es compartida por construcción y $dP_{\mathcal T}/dP_S\le1{,}068$. En
     ese régimen el estimador agrupado es minimax-óptimo (Ma, Pathak y Wainwright 2023).

   Los experimentos lo confirman: incluir Bogotá cambia la log-loss del objetivo en −0,0005 (LightGBM, n. s.) y −0,00003
   (Logit), y el F1 en +0,0001 a +0,0009, todo dentro de la ROPE. Un placebo con 8 ciudades como «dominio extra» da el
   mismo signo (7 de 8). **Decisión:** $\lambda_B$ es un hiperparámetro elegido por la log-loss CV en $\mathcal T$; por
   defecto, $\lambda_B=1$; en la comparación en igualdad de condiciones, $\lambda_B=1$ para todos (§1.9).
3. **Evaluación.**
   - Métrica principal: **F1 agregado anidado en $\mathcal T$**, con $R\times K\times S=3\times5\times3$ para los
     finalistas.
   - Comparaciones pareadas con el t corregido de Nadeau–Bengio y la prueba bayesiana correlacionada, con ROPE de
     0,002 en F1 y 0,0005 en log-loss.
   - Sesgo de selección: corrección de Tibshirani–Tibshirani y re-evaluación en una partición nueva.
   - Errores estándar del F1: 0,0018 en CV, **0,0062 en el tablero público** y 0,0031 en el privado. El público no
     distingue diferencias menores de ≈ 0,005, así que se elige por CV, nunca por el público (§2).
4. **Selección de variables.**
   - Siempre dentro del pliegue. La unidad de parsimonia es la **pregunta de la encuesta**, con costo $c(S)$.
   - Método por familia:
     - LPM y Logit: lasso + reajuste (post-lasso / relaxed).
     - EN: integrada, con $\lambda_{1se}$ y stability selection CPSS.
     - CART: poda 1-EE.
     - RF: RFE por bloques con permutación agrupada OOB y LOCO.
     - Boosting: SHAP agrupado + RFE + LOCO.
     - NB: mRMR + envoltorio hacia adelante y discretización.
   - Conjunto parsimonioso: el menor $c(S)$ dentro de la ROPE del mejor. Se reporta su estabilidad con Jaccard (§3).
5. **Ajuste de hiperparámetros.**
   - Objetivo: **log-loss** (regla estrictamente propia), no F1. En las búsquedas previas el rango de F1 entre
     configuraciones, 0,0012–0,0015, era del orden del ruido entre semillas, 0,0007.
   - Estrategia: búsqueda cuasi-aleatoria y luego TPE con poda por la mediana; los 5 mejores se confirman con
     $3\times5\times3$.
   - Espacios guiados por la *tunability* de Probst et al.
   - RF se ajusta con OOB.
   - Restricciones monótonas por teoría económica, adoptadas por no inferioridad (§4, §9).
6. **Umbral.**
   - El clasificador F1-óptimo es $\mathbb 1\{\eta>F^*/2\}$, derivado en §5.1. El plug-in con umbral estimado en una
     muestra honesta es consistente (Koyejo et al.; Narasimhan et al.).
   - Un solo umbral global, elegido en las OOF de $\mathcal T$ (mediana bootstrap).
   - El F1 está en una meseta: menos de 0,001 de pérdida para umbrales entre 0,35 y 0,41.
   - Diagnósticos: $\hat t\approx\hat F/2$, y la tasa predicha en test debe coincidir con la OOF (±0,003).
7. **Desbalance (20 %).** Sin pesos ni remuestreo, porque los pesos de clase equivalen a mover el umbral (Elkan) y el
   remuestreo descalibra. Se hace una corrida de sensibilidad por algoritmo para la etapa 3 del enunciado (§6).
8. **Promedio de semillas.** Es la mejora barata más grande medida: con 3 semillas, +0,002 de F1 y −0,0016 de log-loss,
   cuatro veces el efecto de Bogotá.
   - Predicción final: reajuste con el 100 % de los datos (la curva de aprendizaje medida promete ≈ +0,001 de F1
     frente a los modelos de pliegue) y, en boosting, 10 semillas; el umbral se traslada igualando la tasa de positivos.
   - Permitido sólo dentro de un mismo algoritmo. Mezclar familias (o *stacking*) no es un algoritmo de la lista (§8).
9. **OOB y .632+.**
   - OOB honesto en RF; con $n=154$ mil sus sesgos conocidos son despreciables. Se usa para ajustar RF y se confirma con
     la CV común.
   - En boosting no hay OOB honesto: `oob_improvement_` de sklearn sólo orienta el número de iteraciones.
   - El .632+ no aporta frente a la CV para CART, Logit, LPM, EN ni NB con este $n$ (§9).

---

## Notación y hechos del problema

- Hogar $i$, covariables $x_i$ (agregadas de personas a hogar), etiqueta $y_i\in\{0,1\}$ (pobre), dominio $d_i$.
  $\eta(x)=P(Y=1\mid X=x)$.
- Entrenamiento: 164.960 hogares en 25 dominios. **Dominio objetivo** $\mathcal T$: los 24 dominios distintos de Bogotá
  (154.393 hogares, prevalencia 20,72 %). **Bogotá** $\mathcal B$: 10.567 hogares (6,41 %), prevalencia 9,76 %.
- Test: 66.168 hogares en los mismos 24 dominios; tablero público ≈ 20 % (≈ 13.234) y privado ≈ 80 % (≈ 52.934).
- La razón test/entrenamiento por dominio está entre 0,412 y 0,442 en los 24 dominios; las participaciones de dominio
  del test y de $\mathcal T$ difieren como máximo en 0,15 pp (correlación 0,9994); el clasificador adversarial
  $\mathcal T$ vs test tiene AUC 0,499. **El test es una muestra aleatoria (estratificada por dominio) de la misma
  población que $\mathcal T$.**
- Prevalencia esperada del test: 20,75 % (post-estratificada; EE muestral 0,16 pp), ≈ 13.730 pobres.
- Métrica: $\mathrm{F1}=2TP/(2TP+FP+FN)$ de la clase pobre.
- ROPE (región de equivalencia práctica) usada en todo el memo: 0,002 en F1 y 0,0005 en log-loss (justificación en
  §2.4).

---

## 1. Bogotá y el cambio de distribución

### 1.1 Formalización

Sea $\pi_d$ la participación del dominio $d$ en el entrenamiento y $P_d(x,y)$ su distribución. El entrenamiento es una
muestra de la mezcla $P_S=\sum_{d}\pi_dP_d=(1-\pi_B)P_{\mathcal T}+\pi_BP_B$, con $\pi_B=0{,}0641$, y el test es una
muestra de $P_{\mathcal T}=\sum_{d\neq B}\frac{\pi_d}{1-\pi_B}P_d$ (evidencia en «Notación y hechos»). Es un problema de
**sesgo de selección sobre una variable observada** (Zadrozny 2004; Cortes et al. 2008): la probabilidad de que un
hogar de la población de entrenamiento pertenezca a la población del test es conocida y vale
$s(x)=\mathbb 1\{d(x)\neq B\}$. En la taxonomía de Moreno-Torres et al. (2012) y Quiñonero-Candela et al. (2009), es un
desplazamiento de covariables (cambia $P(X)$) con **soporte disjunto** en algunas coordenadas:

- **Dominio y línea de pobreza.** El dominio Bogotá no existe en el test, y $L_p$ (y $L_i$) es una función determinista
  de dominio × mes: Bogotá tiene 12 valores de $L_p$, **ninguno** aparece en el test, y todos los valores de $L_p$ del
  test aparecen en $\mathcal T$. Cualquier variable normalizada por $L_p$ (arriendo sobre línea, ingreso requerido,
  índice Mincer sobre requerido…) permite reconstruir $L_p$ y, con ella, identificar a Bogotá: un clasificador de
  dominio (LightGBM, 5 pliegues) que separa Bogotá de las demás cabeceras con todas las variables salvo Dominio y Depto
  alcanza **AUC = 0,944**.
- **Covariables socioeconómicas.** Aun excluyendo $L_p$, $L_i$ y todo lo normalizado por ellas, el clasificador separa
  a Bogotá con **AUC = 0,899** (arriendos nominales más altos, menos régimen subsidiado, más formalidad y educación): el
  desplazamiento de covariables es fuerte.

### 1.2 El estimando y su estimación: CV ponderada por importancia con pesos exactos

El riesgo que importa es el del dominio objetivo,
$$R_{\mathcal T}(f)=E_{P_{\mathcal T}}[\ell(f(X),Y)]=E_{P_S}\bigl[w(X)\,\ell(f(X),Y)\bigr],\qquad
w(x)=\frac{p_{\mathcal T}(x)}{p_S(x)}=\frac{\mathbb 1\{d(x)\neq B\}}{1-\pi_B}.$$
La validación cruzada ponderada por importancia (IWCV; Sugiyama, Krauledat y Müller 2007) es insesgada para
$R_{\mathcal T}$ bajo desplazamiento de covariables; aquí los pesos **no hay que estimarlos**: son exactamente 0 en
Bogotá y constantes en el resto. Por tanto **IWCV = promediar la pérdida sólo sobre los hogares de $\mathcal T$ en los
pliegues de validación**, y para el F1 (un cociente) el F1 ponderado
$2\sum w\,TP/\sum w(2TP+FP+FN)$ es el F1 calculado sólo en $\mathcal T$. El «F1 anidado sin Bogotá» del proyecto es,
entonces, el estimador IWCV correcto. Un refinamiento de robustez es post-estratificar por la composición exacta del test
con pesos por dominio $w_d=(n^{test}_d/n^{test})/(n^{\mathcal T}_d/n^{\mathcal T})$, que aquí están entre 0,962 y 1,032
(efecto despreciable; se reporta como sensibilidad). **No hay ninguna decisión que tomar sobre la evaluación**: umbral,
prevalencia, métricas, comparaciones y selección se calculan sólo con $\mathcal T$.

### 1.3 El entrenamiento: qué dice la teoría

La pregunta legítima es otra: ¿los 10.567 hogares de Bogotá, que nunca se evaluarán, **mejoran** el clasificador para
$\mathcal T$? Se parametriza con un peso relativo $\lambda_B\ge0$ para las filas de Bogotá (y 1 para las de
$\mathcal T$); equivale a minimizar $\alpha\hat\varepsilon_{\mathcal T}+(1-\alpha)\hat\varepsilon_B$ con
$\alpha=n_{\mathcal T}/(n_{\mathcal T}+\lambda_Bn_B)$: $\lambda_B=0\Leftrightarrow\alpha=1$ (sólo objetivo) y
$\lambda_B=1\Leftrightarrow\alpha=\beta=n_{\mathcal T}/n=0{,}936$ (mezcla natural).

**(a) Ponderación por importancia (Shimodaira 2000).** Si el modelo está **bien especificado**, el MV sin ponderar es
asintóticamente óptimo bajo desplazamiento de covariables y ponderar sólo añade varianza; si está **mal especificado**,
el MV sin ponderar converge al parámetro pseudo-verdadero de $P_S$ (no de $P_{\mathcal T}$), el MV ponderado por
$w=p_{\mathcal T}/p_S$ es consistente para el de $P_{\mathcal T}$, y para $n$ moderado conviene un peso intermedio elegido
con un criterio de información (Shimodaira 2000; la familia $w^{\gamma}$, $\gamma\in[0,1]$, es el AIWERM de Sugiyama,
Krauledat y Müller 2007, con $\gamma$ elegido por IWCV). Aquí $w$ es exactamente 0 en Bogotá, así que **cualquier**
$\gamma>0$ equivale a eliminar Bogotá: la ERM ponderada por importancia es $\lambda_B=0$. Pero ésa es la respuesta
*consistente*, no necesariamente la de menor riesgo en muestra finita.

**(b) Cota de adaptación de dominio con datos de ambos dominios (Ben-David et al. 2010).** Con $m$ ejemplos
etiquetados, $\beta m$ del objetivo y $(1-\beta)m$ de la fuente, y $\hat h$ el minimizador de
$\hat\varepsilon_\alpha=\alpha\hat\varepsilon_T+(1-\alpha)\hat\varepsilon_S$ en una clase de dimensión VC $d$, su Teorema 3
(p. 160) da, con probabilidad $\ge1-\delta$,
$$\varepsilon_T(\hat h)\le\varepsilon_T(h_T^*)+\underbrace{4\sqrt{\tfrac{\alpha^2}{\beta}+\tfrac{(1-\alpha)^2}{1-\beta}}\sqrt{\tfrac{2d\log(2(m+1))+2\log(8/\delta)}{m}}}_{\text{varianza}}
+\underbrace{2(1-\alpha)\Bigl[\tfrac12\hat d_{\mathcal H\Delta\mathcal H}(U_S,U_T)+4\sqrt{\tfrac{2d\log(2m')+\log(8/\delta)}{m'}}+\lambda\Bigr]}_{\text{sesgo por discrepancia}},$$
con $\lambda=\min_h[\varepsilon_S(h)+\varepsilon_T(h)]$ el error de la hipótesis conjunta ideal (mide si la función de
etiquetado es compartida) y $d_{\mathcal H\Delta\mathcal H}$ la divergencia entre las marginales de $X$ (Blitzer et al.
2008; Mansour, Mohri y Rostamizadeh 2009 la generalizan a la *discrepancia*). El lema 4 aísla el sesgo:
$|\varepsilon_\alpha(h)-\varepsilon_T(h)|\le(1-\alpha)(\tfrac12d_{\mathcal H\Delta\mathcal H}+\lambda)$. Minimizando la cota
en $\alpha$ (su §6, ec. 2), con $A$ el término entre corchetes y $D=\sqrt d/A$:
$$\alpha^*=\begin{cases}1 & \text{si } m_T\ge D^2=d/A^2,\\ \min\{1,\nu\},\ \ \nu=\beta\Bigl(1+\dfrac{m_S}{\sqrt{D^2(m_S+m_T)-m_Sm_T}}\Bigr) & \text{si } m_T<D^2,\end{cases}$$
y si $A=0$ (dominios indistinguibles para la clase), $\alpha^*=\beta$: la mezcla natural ($\lambda_B=1$). **Lectura para
Bogotá**: los datos de la fuente sólo ayudan si la complejidad del modelo es grande frente a $m_TA^2$ (es decir, si
$m_T<d/A^2$). Con $m_T=154.393$ y la discrepancia marginal **máxima** que implica el soporte disjunto en Dominio y $L_p$
($\hat d_{\mathcal H\Delta\mathcal H}=2$, luego $A\ge1$), la cota recomienda $\alpha^*=1$ (ignorar Bogotá) para toda clase con
$d<154.393$, es decir, para cualquier modelo paramétrico y para ensambles de árboles de tamaño moderado. Pero la cota
es de **peor caso**: cobra la discrepancia marginal aunque la función de etiquetado coincida, y no puede *certificar*
una ganancia. Su mensaje cualitativo es el útil: Bogotá sólo puede ayudar a modelos de complejidad efectiva alta y sólo
si $\lambda$ es pequeño; ambas cosas se miden (§1.5–1.6). Minimizar en $\alpha$ una estimación **insesgada** del riesgo
objetivo (la IWCV de §1.2) en lugar de una cota superior es el paso empírico correcto.

**(c) El caso favorable: condicional compartida y razón de verosimilitud acotada.** Si el dominio forma parte de $x$,
el supuesto de desplazamiento de covariables se cumple **por construcción**: en el soporte de $\mathcal T$,
$P_{\mathcal T}(y\mid x)=P_S(y\mid x)$, porque $P_{\mathcal T}$ es una sub-población de $P_S$. Además la razón de
verosimilitud objetivo/fuente está **uniformemente acotada**: $dP_{\mathcal T}/dP_S=\mathbb 1\{d\neq B\}/(1-\pi_B)\le1{,}068$.
En ese régimen, Ma, Pathak y Wainwright (2023) prueban, para regresión no paramétrica en un RKHS, que el estimador **sin
ponderar** (agrupado), con la regularización bien elegida, ya es minimax-óptimo para el riesgo objetivo: reponderar no
mejora la tasa (la reponderación truncada sólo hace falta cuando la razón no está acotada). Traducido: con el dominio
como variable, entrenar con Bogotá ($\lambda_B=1$) es estadísticamente legítimo y no sesga el riesgo objetivo en la
tasa; eliminarla ($\lambda_B=0$, la ERM ponderada exacta) también es consistente y cuesta a lo sumo el 6,4 % de la
muestra. **Las dos opciones son correctas; difieren en constantes pequeñas que hay que medir.**

**(d) Soporte disjunto e imposibilidad.** Ben-David, Lu, Luu y Pál (2010) prueban que sin supuestos sobre la relación
entre las funciones de etiquetado (o sin solapamiento de soportes) ningún algoritmo puede garantizar aprovechar la
fuente. Aquí el soporte es disjunto en Dominio y $L_p$, de modo que **la teoría no puede certificar a priori** que
Bogotá ayude; tampoco hace falta: hay 154 mil etiquetas del objetivo, y el valor de Bogotá se puede **medir** con
validación en el objetivo. Lo que la teoría sí dice es *de qué depende*: de que la función de etiquetado sea compartida
(término $\lambda$ de la cota pequeño) y de que la complejidad efectiva del modelo sea grande frente a $n_{\mathcal T}$.

**(e) Cuánta información «tipo objetivo» trae Bogotá.** Si se ignoran Dominio y $L_p$, la razón de densidades de cada
hogar de Bogotá puede estimarse con un clasificador de dominio (Bickel, Brückner y Scheffer 2009; alternativas que la
estiman directamente: KMM de Huang et al. 2007 y uLSIF de Kanamori, Hido y Sugiyama 2009; revisión en Sugiyama, Suzuki
y Kanamori 2012 y Sugiyama y Kawanabe 2012):
$\hat w(x)=\frac{1-\hat c(x)}{\hat c(x)}\cdot\frac{\pi_B}{1-\pi_B}$ con $\hat c(x)=\hat P(B\mid x)$. Cortes, Mansour y Mohri
(2010) muestran que la calidad del aprendizaje ponderado depende del segundo momento de los pesos, que es lo que mide
el tamaño efectivo de muestra. Con las covariables socioeconómicas, el tamaño efectivo de muestra de Bogotá, $\mathrm{ESS}=(\sum\hat w)^2/\sum\hat w^2$, es **≈ 877
hogares** (8 % de 10.567; mediana de $\hat w$ = 0,22). Es decir, Bogotá aporta el equivalente a ≈ 0,6 % de la muestra
objetivo en las regiones del espacio donde ambos dominios se solapan: la reducción máxima de varianza por
desplazamiento de covariables es del orden de $1-\sqrt{n_{\mathcal T}/(n_{\mathcal T}+877)}\approx0{,}3\,\%$ del EE.

**(f) Orden de magnitud del sesgo y de la varianza.** En un modelo paramétrico mal especificado, el parámetro
pseudo-verdadero de la mezcla difiere del de $\mathcal T$ en $\theta_S-\theta_{\mathcal T}=O(\pi_B\Delta)$, con $\Delta$ la
diferencia entre los parámetros pseudo-verdaderos de Bogotá y del objetivo; el exceso de riesgo en $\mathcal T$ es
$\tfrac12(\theta_S-\theta_{\mathcal T})^\top H(\theta_S-\theta_{\mathcal T})=O(\pi_B^2\Delta^2)$, de **segundo orden**. La
ganancia de varianza es $O(\pi_B\cdot p/n)$, también despreciable con $p\approx200\ll n$. Para modelos paramétricos
se predice, entonces, **diferencia nula**. En modelos flexibles (árboles potenciados con miles de hojas) la complejidad
efectiva es mucho mayor, las hojas en regiones poco pobladas sí se benefician de más datos, y Bogotá puede ayudar si la
función de etiquetado es compartida (y dañar si no lo es).

### 1.4 Desplazamiento de etiquetas y umbral

Con $\lambda_B=1$ la prevalencia de entrenamiento es 20,02 % y la del objetivo 20,72 % (test esperado: 20,75 %). Si el
modelo no distingue a Bogotá (p. ej. un modelo de encuesta corta sin Dominio ni $L_p$), su tasa base mezcla ambas y sus
probabilidades quedan levemente sesgadas a la baja en $\mathcal T$ (prior shift de 0,7 pp; Saerens, Latinne y
Decaestecker 2002; Lipton, Wang y Smola 2018). **No importa para el F1** siempre que el umbral se elija con las OOF de
$\mathcal T$ (§5.4): el umbral F1-óptimo absorbe cualquier desplazamiento global monótono. Sí importaría si se usara un
umbral teórico ($F^*/2$) o si se reportaran probabilidades: por eso se verifica que la media de las probabilidades en el
test sea $0{,}2075\pm0{,}003$.

### 1.5 Diagnósticos medidos

La «validación adversarial» (Pan et al. 2020) es una prueba de dos muestras basada en un clasificador (Lopez-Paz y
Oquab 2017): si un clasificador no distingue dos muestras mejor que el azar (AUC ≈ 0,5), no hay evidencia de
desplazamiento detectable por esa clase de modelos; Rabanser, Günnemann y Lipton (2019) la incluyen entre los
detectores de desplazamiento más eficaces. Se complementa con una **prueba de transferencia** que mide lo que la
adversarial no ve: si cambia $P(Y\mid X)$.

| Diagnóstico | Resultado | Lectura |
|---|---|---|
| Adversarial $\mathcal T$ vs test (cuaderno 01) | AUC 0,499 | test ≡ objetivo |
| Adversarial train completo vs test | AUC 0,531 | la única diferencia es Bogotá |
| Bogotá vs otras cabeceras, todas las variables salvo Dominio/Depto | AUC 0,944 | $L_p$ y lo normalizado por $L_p$ identifican el dominio |
| Ídem, sólo variables socioeconómicas | AUC 0,899; ESS ≈ 877 | desplazamiento de covariables fuerte |
| Valores de $L_p$ de Bogotá presentes en test | 0 de 12 | soporte disjunto |
| **Transferencia** LightGBM entrenado sólo con $\mathcal T$ → Bogotá | AUC 0,972 (con Bogotá en el entrenamiento, CV: 0,973); probabilidad media 0,092 vs tasa observada 0,0976 (con Bogotá: 0,096) | la función de etiquetado es casi la misma |
| **Transferencia** Logit entrenado sólo con $\mathcal T$ → Bogotá | AUC 0,9676 (con Bogotá: 0,9677); media 0,099 vs 0,0976 | ídem |

**Conclusión de los diagnósticos**: Bogotá difiere mucho en $P(X)$ pero muy poco en $P(Y\mid X)$: un modelo que nunca
vio Bogotá ordena a sus hogares igual de bien que uno entrenado con ellos y yerra la tasa agregada en sólo 0,1–0,6 pp.
Es el caso favorable para agrupar (desplazamiento de covariables con condicional compartida), pero con poca
información nueva (ESS pequeño).

### 1.6 Evidencia experimental: peso de Bogotá

**Diseño.** Variables: las 180 permitidas de `modelo_train.parquet` sin primera etapa. LightGBM fijo (tasa 0,05, 63
hojas, `min_child_samples` 50, `feature_fraction` 0,5, `bagging_fraction` 0,8, $\ell_2=1$, parada temprana con un
10 % interno **sólo de $\mathcal T$**), $\lambda_B\in\{0;0{,}5;1\}$, 3 semillas, los 5 pliegues actuales; métricas sólo
en $\mathcal T$. Logit con L2 débil ($C=1$), imputación por mediana con indicadores, winsorización y dummies de dominio,
$\lambda_B\in\{0,1\}$. Pruebas: diferencias por pliegue (semillas promediadas dentro del pliegue), t corregido de
Nadeau–Bengio con $J=5$.

| Modelo | $\lambda_B$ | log-loss $\mathcal T$ (1 semilla, media) | F1 anidado (1 semilla, media) | log-loss (OOF promediadas en 3 semillas) | F1 OOF óptimo (3 semillas) |
|---|---|---|---|---|---|
| LightGBM | 0 | 0,21738 | 0,76540 | 0,21579 | 0,76797 |
| LightGBM | 0,5 | 0,21701 | 0,76548 | 0,21546 | 0,76789 |
| LightGBM | 1 | 0,21689 | 0,76628 | 0,21538 | 0,76808 |
| Logit | 0 | 0,23477 | 0,74677 | (determinista) | 0,74763 |
| Logit | 1 | 0,23474 | 0,74735 | (determinista) | 0,74769 |

| Contraste pareado (vs $\lambda_B=0$) | Δ log-loss (EE corregido; p) | Δ F1 anidado por pliegue (EE corregido; p) |
|---|---|---|
| LightGBM $\lambda_B=0{,}5$ | −0,00037 (0,00018; p = 0,11) | +0,00008 (0,00065; p = 0,90) |
| LightGBM $\lambda_B=1$ | −0,00050 (0,00028; p = 0,16) | +0,00087 (0,00020; p = 0,013) |
| Logit $\lambda_B=1$ | −0,00003 (0,00009) | +0,00057 (0,00092) |

Lecturas: (i) en el Logit la diferencia es exactamente nula, como predice §1.3(f); (ii) en LightGBM, incluir Bogotá
mejora la log-loss del objetivo en ≈ 0,0004–0,0005 (dentro de la ROPE de 0,0005 y no significativa) y el F1 en
+0,0001 a +0,0009 (dentro de la ROPE de 0,002); (iii) el efecto de **promediar 3 semillas** (−0,0016 en log-loss,
+0,002 en F1) es cuatro veces mayor que el de Bogotá; (iv) la desviación entre semillas del F1 anidado de un solo ajuste
(0,0007–0,0008) es del mismo orden que el efecto de Bogotá. **Bogotá no importa para el pronóstico a la escala que la
competencia puede detectar**, y si algo, ayuda levemente a los modelos flexibles.

### 1.7 Simulación del escenario «dominio extra» (placebo) y deja-un-dominio-fuera

**Placebo.** Para replicar el escenario Bogotá → resto con dominios donde *sí* sabemos la respuesta, se toma cada una de
8 ciudades $j\in\mathcal T$ (Medellín, Cali, Barranquilla, Bucaramanga, Manizales, Pereira, Cartagena, Pasto) como
«dominio extra»: se entrena con $\mathcal T$ (sin Bogotá) **con y sin** $j$ y se evalúa en $\mathcal T\setminus\{j\}$
(los mismos pliegues; 2 semillas). La distribución de los 8 efectos es la distribución de referencia de «agregar un
dominio que no se evalúa».

| Dominio «extra» $j$ (placebo) | Δ log-loss en $\mathcal T\setminus\{j\}$ (EE corregido) | Δ F1 anidado (EE corregido) |
|---|---|---|
| Barranquilla | −0,00023 (0,00031) | +0,00120 (0,00041) |
| Bucaramanga | −0,00006 (0,00019) | −0,00040 (0,00084) |
| Cali | −0,00030 (0,00036) | +0,00059 (0,00078) |
| Cartagena | −0,00041 (0,00031) | +0,00054 (0,00089) |
| Manizales | −0,00043 (0,00021) | +0,00066 (0,00122) |
| Medellín | +0,00011 (0,00026) | −0,00022 (0,00060) |
| Pasto | −0,00011 (0,00028) | −0,00002 (0,00051) |
| Pereira | −0,00007 (0,00022) | +0,00061 (0,00050) |
| **Promedio de los 8** | **−0,00019** (EE 0,00007; 7 de 8 negativos; p = 0,025) | **+0,00037** (EE 0,00019; p = 0,09) |
| **Bogotá** (§1.6, mismo modelo) | −0,00050 (1 semilla) / −0,00041 (3 semillas) | +0,00087 / +0,00011 |

Lectura: **agregar un dominio que no se evalúa mejora levemente la log-loss del objetivo en un modelo flexible** (7 de 8
réplicas; efecto medio −0,0002 en log-loss, 40 % de su ROPE, y +0,0004 en F1, un quinto de su ROPE), y el efecto de Bogotá (con más hogares que
cualquier placebo) es del mismo signo y de magnitud comparable. Bogotá se comporta como «una ciudad más»: no hay
evidencia de transferencia negativa.

**Deja-un-dominio-fuera (LODO) como prueba de estrés.** Para los 2 modelos finalistas, entrenar en
$\mathcal T\setminus\{d\}$ y predecir $d$, para cada $d$: mide cuánto depende el modelo de haber visto el dominio
(D'Amour et al. 2022 lo llaman prueba de estrés bajo desplazamiento). Gulrajani y Lopez-Paz (2021) muestran que el
criterio de selección de modelos es parte del algoritmo y debe declararse, y que, en igualdad de condiciones, validar en
los dominios de entrenamiento superó a dejar un dominio fuera; aquí, además, el test no trae dominios nuevos (es
$\mathcal T$), así que la validación en $\mathcal T$ es la correcta y el LODO es sólo un **diagnóstico de robustez**
para la diapositiva de política (¿serviría el modelo en una ciudad no encuestada?), no un criterio de selección.

### 1.8 «Borrowing strength» entre los 24 dominios objetivo: encogimiento bayesiano empírico

El encogimiento relevante para el pronóstico no es el de Bogotá (no se predice) sino el de **efectos de grupo con pocas
observaciones** dentro de $\mathcal T$: dominio, departamento, oficio del jefe (84 niveles; 17 con menos de 100 hogares
y 6 con menos de 30), actividad, etc.

**Derivación (beta-binomial).** Si $p_j\sim\mathrm{Beta}(\mu m,(1-\mu)m)$ y $y_{ij}\mid p_j\sim\mathrm{Bernoulli}(p_j)$, la
media posterior es
$$\tilde p_j=E[p_j\mid\text{datos}]=\frac{n_j\bar y_j+m\mu}{n_j+m}=w_j\bar y_j+(1-w_j)\mu,\qquad w_j=\frac{n_j}{n_j+m},$$
el estimador de credibilidad de Bühlmann, que es la versión con fundamento bayesiano de la codificación con la etiqueta
de Micci-Barreca (2001). Por momentos, $\mathrm{Var}_{entre}(p_j)=\tau^2=\mu(1-\mu)/(m+1)$, de donde
$\hat m=\hat\mu(1-\hat\mu)/\hat\tau^2-1$. En la escala logit, con $\hat\theta_j=\mathrm{logit}(\bar y_j)$ y
$v_j\approx1/\{n_j\bar y_j(1-\bar y_j)\}$, el análogo de Efron y Morris (1975) / Fay y Herriot (1979) es
$\tilde\theta_j=\hat\mu_\theta+\frac{\tau_\theta^2}{\tau_\theta^2+v_j}(\hat\theta_j-\hat\mu_\theta)$; dentro de un modelo, el
intercepto aleatorio de un GLMM (BLUP) hace lo mismo condicionando en las demás variables (Elbers, Lanjouw y Lanjouw
2003; Molina y Rao 2010).

**Cifras.** Para Dominio: $\hat\tau=0{,}0775$ (desviación entre dominios de la tasa de pobreza) y $\hat m=26{,}3$; con
$n_j\ge4.075$, el peso de encogimiento $1-w_j$ es como máximo **0,64 %**: para el dominio, el encogimiento es irrelevante
y las dummies o la categórica nativa bastan. Para el **oficio del jefe** y otras categóricas con celdas pequeñas el
encogimiento sí importa.

**Reglas de implementación.** (i) La codificación de cada hogar no puede usar su propia etiqueta: se calcula
*cross-fitted* con un K interno (5) dentro del entrenamiento de cada pliegue externo, o con las estadísticas ordenadas
de CatBoost (Prokhorenkova et al. 2018); sin esto hay fuga de la etiqueta y la CV es optimista. (ii) $m$ se estima por
momentos dentro del pliegue. (iii) Pargent et al. (2022) encuentran que la codificación regularizada (con CV interna o
GLMM) es la mejor opción general para categóricas de alta cardinalidad. (iv) Para LightGBM, `cat_smooth` y
`min_data_per_group` cumplen el mismo papel de encogimiento.

### 1.9 Decisión sobre Bogotá (procedimiento exacto)

**Principio.** La evaluación es fija (sólo $\mathcal T$, IWCV exacta). El peso de Bogotá en el entrenamiento,
$\lambda_B$, es un **hiperparámetro científico** cuyo único criterio es el riesgo en el objetivo. Como la IWCV da, para
cada $\lambda_B$, un estimador insesgado de ese riesgo y sólo se comparan 2–3 valores (sesgo de selección despreciable),
la regla de Bayes con previa plana sobre la diferencia es elegir el $\lambda_B$ de **menor log-loss media en
$\mathcal T$**: es la contraparte empírica de minimizar en $\alpha$ la cota de Ben-David et al. (2010), con un estimador
insesgado en lugar de una cota de peor caso.

**Procedimiento exacto** (para cada algoritmo, con su configuración ya ajustada):

1. Evaluar $\lambda_B\in\{0,1\}$ (y 0,5 en RF y boosting) con $R\times K\times S=3\times5\times3$, mismas particiones y
   semillas; métricas sólo en $\mathcal T$; para $\lambda_B=0$, eliminar físicamente a Bogotá del ajuste.
2. Elegir $\hat\lambda_B=\arg\min$ log-loss media en $\mathcal T$. Reportar la diferencia con IC corregido, la prueba
   bayesiana ($P$(mejor), $P$(ROPE)) y la diferencia en F1 anidado.
3. **Valor por defecto** (si no se corre el contraste, o si la diferencia es exactamente nula): $\lambda_B=1$.
   Justificación: (i) con el dominio en $x$ la condicional es compartida por construcción y la razón de verosimilitud
   está acotada por 1,068, el régimen en que el estimador agrupado es minimax-óptimo (§1.3c); (ii) la transferencia
   muestra condicional casi invariante (§1.5); (iii) el placebo muestra que un dominio extra ayuda levemente (7 de 8
   réplicas) y Bogotá se comporta igual (§1.7); (iv) en LightGBM la estimación puntual favorece $\lambda_B=1$ y en el
   Logit es indiferente (§1.6).
4. **Comparación «en igualdad de condiciones»** entre algoritmos: todos con el mismo $\lambda_B=1$. **«Mejor de cada
   algoritmo»**: su $\hat\lambda_B$.
5. **Modelos sin Dominio ni $L_p$** (encuesta corta): repetir el paso 1–2; allí la condicional deja de ser compartida
   por construcción (§1.10).
6. **Umbral, prevalencia esperada (20,75 % ± 0,16 pp) y todas las métricas: sólo $\mathcal T$**, cualquiera que sea
   $\hat\lambda_B$.

**Lo que se espera y cómo contarlo.** A la escala de la competencia (EE del F1 privado 0,003) la elección de
$\lambda_B$ es irrelevante; el valor del análisis está en mostrar que la decisión **no es un dogma**: Bogotá no se
elimina por estar ausente del test ni se incluye por inercia; se mide su aporte al riesgo del dominio objetivo con un
estimador insesgado, y ese aporte es pequeño y no negativo. Alternativas más elaboradas —aumentar las variables con
copias específicas de Bogotá (Daumé III 2007), pesos por observación aprendidos con validación objetivo (DVRL; Yoon,
Arik y Pfister 2020) o efectos aleatorios de dominio— no se justifican con un efecto de esta magnitud.

### 1.10 Cómo verificarlo y riesgos

- *Verificación*: repetir la tabla de §1.6 con las variables v7 y con cada algoritmo ajustado; repetir el placebo para
  el modelo final; comprobar que ninguna métrica, umbral o curva use filas de Bogotá (prueba automática del Paso 14 del
  protocolo).
- *Riesgo 1*: un modelo de encuesta corta **sin** Dominio ni $L_p$ no puede aislar a Bogotá; ahí la pequeña
  diferencia de condicional (error de 0,1–0,6 pp en la tasa agregada de Bogotá, §1.5) se traslada al objetivo. Repetir el contraste
  $\lambda_B\in\{0,1\}$ para los modelos parsimoniosos.
- *Riesgo 2*: interpretar diferencias de 0,001 como reales; la decisión se toma con ROPE y prueba corregida.
- *Riesgo 3*: usar la validación adversarial con variables que codifican el dominio ($L_p$) y concluir que hay
  desplazamiento donde no lo hay (o al revés): la adversarial correcta para «train vs test» se hace con $\mathcal T$.

---

## 2. Evaluación y selección de modelos

### 2.1 Qué se estima

Sea $\mathcal A$ un procedimiento (variables + algoritmo + hiperparámetros + umbral) y $\hat f_{\mathcal D}=\mathcal A(\mathcal D)$
el clasificador que produce con los datos $\mathcal D$. Lo que decide la competencia es
$\mathrm{F1}_{priv}(\hat f_{\mathcal D})$, el F1 sobre los ≈ 52.934 hogares del tablero privado. Hay tres fuentes de
aleatoriedad: (a) la muestra de evaluación dado $\hat f$; (b) los datos de entrenamiento $\mathcal D$; (c) la
aleatoriedad interna del algoritmo (semillas, submuestreo, parada temprana). Bates, Hastie y Tibshirani (2024) muestran
que la CV **no** estima el error condicional $\mathrm{Err}_{XY}$ del modelo ajustado a *estos* datos, sino su promedio
sobre conjuntos de entrenamiento $\mathrm{Err}=E[\mathrm{Err}_{XY}]$, y que el intervalo ingenuo
$\hat{\mathrm{Err}}\pm z\,\widehat{\mathrm{sd}}/\sqrt n$ sub-cubre porque ignora la correlación entre pliegues; proponen
una CV anidada para estimar el error cuadrático medio del estimador CV y ensanchar el intervalo. Con
$n_{\mathcal T}=154.393$ la variabilidad de $\mathrm{Err}_{XY}$ entre conjuntos de entrenamiento es pequeña frente al
ruido de evaluación del tablero, de modo que la distinción importa poco para el *nivel*, pero sí para **comparar**
procedimientos cuyas diferencias son de milésimas. Wager (2020) discute por qué la CV puede ser buena para *seleccionar*
aunque estime mal el error de un ajuste concreto: lo que importa para elegir es la diferencia entre procedimientos,
que se estima con mucha más precisión que cada nivel (aquí, 0,0006 frente a 0,0018; §2.1).

**F1 no es un promedio de pérdidas individuales**: es un cociente $\mathrm{F1}=2\,TP/(2\,TP+FP+FN)$. El estimador CV
recomendado es el **F1 agregado** (sumar $TP,FP,FN$ de todos los pliegues y calcular un solo cociente), que es
consistente para el F1 poblacional; el promedio de F1 por pliegue es un promedio de cocientes con sesgo de orden $1/n_k$
(Forman y Scholz 2010). Los F1 por pliegue se usan sólo para errores estándar y pruebas pareadas.

**Error estándar analítico de F1 (función de influencia).** Con $t_i=\mathbb 1\{TP_i\}$, $f_i=\mathbb 1\{FP_i\}$,
$g_i=\mathbb 1\{FN_i\}$ y $D=E[2t+f+g]$, la linealización de $F=N/D$ con $N=2E[t]$ da
$$\hat F-F\approx\frac1n\sum_i\psi_i,\qquad \psi_i=\frac{2t_i-F\,(2t_i+f_i+g_i)}{D},\qquad
\mathrm{EE}(\hat F)\approx\mathrm{sd}(\psi)/\sqrt n .$$
Para dos clasificadores evaluados en los mismos hogares, $\mathrm{EE}(\hat F_A-\hat F_B)\approx\mathrm{sd}(\psi^A-\psi^B)/\sqrt n$.
**Cifras de este problema** (LightGBM de referencia con las variables permitidas, OOF sin Bogotá, §1.6):
$\mathrm{sd}(\psi)=0{,}714$, de donde

| Muestra de evaluación | $n$ | EE de F1 (nivel) | EE de una diferencia pareada típica ($\rho_\psi=0{,}94$) |
|---|---|---|---|
| OOF objetivo (24 dominios) | 154.393 | 0,0018 | 0,0006 |
| Test completo | 66.168 | 0,0028 | 0,0009 |
| Tablero **privado** (80 %) | ≈ 52.934 | 0,0031 | 0,0011 |
| Tablero **público** (20 %) | ≈ 13.234 | **0,0062** (bootstrap: 0,0062) | 0,0021 |

Consecuencias: (i) el tablero público no distingue modelos que difieran en menos de ≈ 0,004–0,005 de F1 aun de forma
pareada, y su nivel tiene un EE de 0,006; (ii) la CV con 154 mil hogares es unas 3,4 veces más precisa que el público en
nivel y en diferencias; (iii) seleccionar el máximo de $k$ envíos por el público produce la «maldición del ganador»: si
los $k$ tienen el mismo F1 verdadero, el máximo observado excede al verdadero en
$\approx\sigma_\Delta\,E[\max_{j\le k}Z_j]$ ($E\max$ de 15 normales ≈ 1,74), es decir, ≈ 0,004 con
$\sigma_\Delta=0{,}0021$. La lección práctica coincide con Blum y Hardt (2015) y con el meta-análisis de Roelofs et al.
(2019) sobre Kaggle: el sobreajuste adaptativo al tablero es moderado, pero el ruido del público es grande; **la elección
del envío final se hace por CV, nunca por el público**.

### 2.2 Validación cruzada: pliegues, repeticiones y estratificación

- **K = 5** (los `pliegues.csv` actuales, estratificados por Pobre × Dominio). El sesgo pesimista de entrenar con 80 %
  en lugar de 100 % depende de la pendiente de la curva de aprendizaje (Kohavi 1995; Arlot y Celisse 2010). **Medida
  aquí** (LightGBM de §1.6, 2 semillas): entrenar con 60 % de los datos en lugar de 80 % empeora la log-loss del objetivo
  en 0,0020 (0,21892 frente a 0,21692) y el F1 OOF óptimo en 0,0020 (0,7646 frente a 0,7666). Con un decaimiento
  $\propto1/n$, pasar de 80 % a 100 % ganaría ≈ 0,0012 en ambas medidas. Consecuencias: (i) la CV-5 subestima el nivel
  del modelo final en ≈ 0,001 de F1 (por debajo de la ROPE y casi igual para todos los algoritmos, así que no altera las
  comparaciones); (ii) el **predictor final** debe entrenarse con el 100 % de los datos (§8.3), no ser sólo el promedio
  de los modelos de pliegue. Más pliegues aumentan el costo sin reducir apreciablemente la varianza; el presupuesto
  rinde más en **repeticiones** y **semillas**.
- **R = 3 particiones** (la actual y dos con semillas nuevas, mismas reglas de estratificación) para la confirmación de
  los finalistas: la variabilidad entre particiones medida aquí antes fue ≈ 0,0004 en F1 (`10_optimismo.csv`).
- **Estratificar por Pobre × Dominio** hace que cada pliegue reproduzca la composición del test (que es proporcional a
  la del objetivo: razón test/entrenamiento entre 0,412 y 0,442 en los 24 dominios) y reduce la varianza entre
  pliegues. Los hogares de Bogotá se reparten en los pliegues (para cuando entran al entrenamiento con peso
  $\lambda>0$), pero **nunca** entran a la métrica.
- **Mismas particiones para todos los algoritmos** (comparaciones pareadas) y mismas semillas.
- **No hay agrupamiento a corregir**: la unidad es el hogar y las personas se agregan dentro del hogar (no hay fuga
  persona→hogar entre pliegues). El diseño muestral de la GEIH es por conglomerados (segmentos), pero el identificador
  de conglomerado no está disponible; los EE pueden estar levemente subestimados por esa razón (se declara).

### 2.3 Selección vs evaluación; CV anidada y sesgo de selección

*Selección* (elegir variables, hiperparámetros, $\lambda$ de Bogotá, umbral) necesita **ordenar** bien; *evaluación*
(reportar el F1 esperado del procedimiento ganador) necesita un **nivel** insesgado. Elegir el máximo de muchas
configuraciones evaluadas en los mismos pliegues sesga al alza el F1 del ganador (Varma y Simon 2006; Cawley y Talbot
2010, que muestran que el sobreajuste en la selección puede ser del mismo orden que las diferencias entre algoritmos).
Estrategia recomendada, de menor a mayor costo:

1. **Umbral siempre anidado** (ya implementado): el umbral del pliegue $k$ se elige con las predicciones OOF de los
   otros cuatro (sólo objetivo).
2. **Corrección de Tibshirani y Tibshirani (2009)** para el resto de la selección, gratis con los ensayos ya
   registrados: si $e_k(\theta)$ es el error del pliegue $k$ con la configuración $\theta$, $\hat\theta$ minimiza
   $\frac1K\sum_k e_k(\theta)$ y $\hat\theta_k$ minimiza $e_k(\theta)$, el sesgo estimado es
   $$\widehat{\mathrm{sesgo}}=\frac1K\sum_{k=1}^K\bigl[e_k(\hat\theta)-e_k(\hat\theta_k)\bigr]\;\ge0,$$
   y se reporta $\mathrm{CV}(\hat\theta)+\widehat{\mathrm{sesgo}}$ (para F1, con los signos invertidos).
3. **Re-evaluación en una partición nueva** de la configuración elegida (no vista durante la búsqueda): estima el
   optimismo directamente.
4. **CV anidada completa** (selección repetida dentro de cada pliegue externo; repetida, como recomiendan Krstajic
   et al. 2014) sólo para los 2–3 finalistas si el presupuesto lo permite.

### 2.4 Comparaciones pareadas con la corrección por dependencia

Las diferencias entre pliegues $d_j$ ($j=1,\dots,J$, $J=R\cdot K$) **no son independientes**: los conjuntos de
entrenamiento se solapan. Nadeau y Bengio (2003) modelan $\mathrm{corr}(d_j,d_{j'})=\rho\approx n_2/(n_1+n_2)$ y
obtienen $\mathrm{Var}(\bar d)=\sigma^2\bigl(\tfrac1J+\tfrac{\rho}{1-\rho}\bigr)$, de donde el **t de remuestreo
corregido**
$$t=\frac{\bar d}{\sqrt{\bigl(\tfrac1J+\tfrac{n_2}{n_1}\bigr)\hat\sigma_d^2}},\qquad gl=J-1,$$
con $n_2/n_1=1/(K-1)=0{,}25$ para K = 5. Con $J=15$ ($R=3$) el factor es $0{,}317$ frente a $0{,}067$ del t ingenuo: el EE
se multiplica por ≈ 2,2. Bouckaert y Frank (2004) encuentran que las versiones corregidas con repeticiones son las más replicables; Bengio y Grandvalet
(2004) prueban que no existe estimador insesgado de la varianza de la CV K-fold, así que la corrección es heurística
(conservadora en la práctica). **Prueba bayesiana correlacionada** (Corani y Benavoli 2015; Benavoli et al. 2017):
posterior $\mu\mid d\sim t_{J-1}\bigl(\bar d,\ (\tfrac1J+\tfrac{\rho}{1-\rho})\hat\sigma^2\bigr)$ con $\rho=1/K$, y una
**ROPE** (región de equivalencia práctica). Se reportan $P(\mu<-r)$, $P(|\mu|\le r)$ y $P(\mu>r)$.

**ROPE recomendada:** $r=0{,}002$ en F1 (≈ un EE de diferencia pareada en el tablero público y dos en el privado; nada
menor es observable en la competencia) y $r=0{,}0005$ en log-loss. **Bootstrap pareado** de las predicciones OOF (o
la fórmula $\psi$ de §2.1) mide la incertidumbre *de evaluación* condicional a los modelos ajustados: responde «¿le
ganará A a B en el tablero privado?»; el t corregido mide la incertidumbre del *procedimiento* (incluye
entrenamiento). Se reportan ambos. McNemar (Dietterich 1998) sobre los mismos hogares es una alternativa exacta para
dos envíos concretos. **Multiplicidad**: al comparar siete algoritmos contra el mejor, Holm sobre los p-valores
corregidos (Demšar 2006 para el razonamiento de comparaciones múltiples).

**Semillas.** Las semillas se promedian **dentro** de cada par (partición, pliegue) antes de diferenciar (no son
réplicas independientes de la muestra de evaluación). Bouthillier et al. (2021) muestran que ignorar la varianza por
semillas y particiones produce conclusiones no replicables; aquí, un LightGBM de una sola semilla tiene una desviación
entre semillas de 0,0007–0,0008 en F1 anidado y 0,0001–0,0002 en log-loss (§1.6): del mismo orden que muchas
diferencias que se querrían «detectar».

### 2.5 Regla de un error estándar, CV con confianza y parsimonia

La regla 1-EE (Breiman et al. 1984; Hastie et al. 2009, §7.10) elige el modelo más simple cuyo error CV está a menos de
un EE del mínimo. Con F1 y modelos anidados conviene la versión **pareada**: el EE relevante es el de la diferencia
$\mathrm{F1}(\text{mejor})-\mathrm{F1}(k)$, mucho menor que el EE marginal (0,0006 frente a 0,0018). **CV con confianza**
(Lei 2020) formaliza esto: prueba para cada candidato la hipótesis de que no es peor que el resto (con bootstrap
gaussiano de las diferencias de pérdida por observación) y devuelve un *conjunto de confianza* de modelos; dentro de él
se elige el más parsimonioso. Para este proyecto: **el modelo parsimonioso es el de menor $c(S)$ dentro de la ROPE del
mejor**, y se reporta además el conjunto de confianza CVC al 90 %.

### 2.6 ¿Hay algo mejor que la CV aquí?

- **CV ponderada por importancia (IWCV)**: es *exactamente* lo que se hace al medir sólo en los 24 dominios objetivo
  (§1.2); la post-estratificación por la composición del test (pesos entre 0,962 y 1,032) es un ajuste fino de
  robustez.
- **Error OOB** para Random Forest: estimador casi gratuito y válido con $n$ grande (§9); acelera la búsqueda de
  hiperparámetros, pero la comparación final entre algoritmos se hace con la CV común (pareada).
- **Bootstrap .632+** (Efron y Tibshirani 1997): diseñado para $n$ pequeño; aquí no mejora a la CV (§9).
- **Ensambles de la selección** (promediar las mejores configuraciones del *mismo* algoritmo; Caruana et al. 2004) y
  **promedio de semillas**: reducen la varianza de selección y de ajuste (§8).
- **Estabilidad** (Yu 2013; Yu y Kumbier 2020): reportar cuánto cambian el ranking de modelos, las variables y el umbral
  entre pliegues, semillas y particiones.

### 2.7 Recomendación concreta

Métrica principal: **F1 agregado anidado en el dominio objetivo** (24 dominios), sobre $R\times K=3\times5$ con 3
semillas promediadas dentro de cada (partición, pliegue). Métricas secundarias en el objetivo: log-loss, Brier,
PR-AUC, ROC-AUC, ECE (bins de igual masa), tasa de positivos predicha. Comparaciones: diferencias pareadas con t
corregido de Nadeau–Bengio y prueba bayesiana correlacionada con ROPE 0,002, más bootstrap pareado de las OOF
($B=2.000$). Sesgo de selección: corrección de Tibshirani–Tibshirani + re-evaluación en partición nueva; CV anidada
completa para los finalistas si hay tiempo. Elección del envío final: por CV.

### 2.8 Cómo verificarlo y riesgos

- *Verificación*: (i) repetir la curva de aprendizaje (60 % vs 80 %) con el modelo final de cada algoritmo; (ii) que el F1 de la partición 2 y 3
  caiga dentro del IC de la partición 1; (iii) que el EE analítico $\psi$ coincida con el bootstrap (aquí: 0,0062 vs
  0,0062 para $n=13.234$).
- *Riesgos*: reutilizar los mismos 5 pliegues para cientos de decisiones (sobreajuste de la partición): confirmar en
  particiones nuevas; interpretar diferencias de 0,001 como reales: usar ROPE; comparar algoritmos con esfuerzos de
  ajuste distintos (Sculley et al. 2018 piden ajustar los hiperparámetros clave de todos los modelos, incluidas las
  líneas base, y reportar el procedimiento): igualar presupuestos o al menos reportarlos.

---

## 3. Selección de variables por familia de modelo

### 3.1 Síntesis de la teoría

**La selección es parte del procedimiento de aprendizaje.** Si las variables se escogen mirando la etiqueta de toda la
muestra y después se valida por CV, el error estimado es optimista: Ambroise y McLachlan (2002) lo mostraron con
microarreglos, donde una regla con selección previa «perfecta» tenía error CV casi nulo con etiquetas aleatorias. En
términos formales, si $\mathcal A$ es el procedimiento completo (selección + ajuste + umbral), la CV estima
$\mathrm{Err}(\mathcal A)$ sólo si cada paso de $\mathcal A$ se re-ejecuta dentro de cada pliegue (Varma y Simon 2006;
Hastie, Tibshirani y Friedman 2009, §7.10.2). Corolario operativo: el ranking (SHAP, permutación, información mutua,
camino L1) se calcula con el entrenamiento del pliegue externo $k$ y se evalúa en el pliegue $k$; el conjunto final se
elige con la curva de parsimonia CV y luego se recalcula sobre toda la muestra objetivo.

**Tres familias de métodos (Guyon y Elisseeff 2003; Kohavi y John 1997).**

1. *Filtros* (independientes del modelo). Información mutua $I(X_j;Y)$ univariada: barata, capta no linealidad, ignora
   redundancia e interacciones. **mRMR** (Peng, Long y Ding 2005) añade variables de forma voraz maximizando
   $$J(X_j)=I(X_j;Y)-\frac{1}{|S|}\sum_{s\in S} I(X_j;X_s),$$
   que es la aproximación incremental de primer orden del criterio de máxima dependencia $I(X_S;Y)$. Es el filtro natural
   para Naive Bayes, cuyo supuesto de independencia condicional se rompe justamente con variables redundantes.
2. *Envoltorios* (wrappers). Búsqueda en el espacio de subconjuntos usando el propio modelo y la CV como criterio:
   selección hacia adelante, eliminación hacia atrás, RFE (Guyon et al. 2002; para RF, Díaz-Uriarte y Alvarez de Andrés
   2006). Son los que mejor optimizan el criterio, pero los más caros y los más expuestos a sobreajustar la partición
   (cada comparación reutiliza los mismos pliegues; Cawley y Talbot 2010). Se controlan con pasos por **bloques** (pocas
   decisiones, cada una con mucha señal) y con confirmación en una segunda partición.
3. *Integrados* (embedded). La penalización $\ell_1$ (lasso; Tibshirani 1996) produce soluciones dispersas; la red
   elástica (Zou y Hastie 2005) añade $\ell_2$ y conserva grupos de variables correlacionadas (efecto de agrupamiento),
   a costa de menos parsimonia; el *group lasso* (Yuan y Lin 2006; para logit, Meier, van de Geer y Bühlmann 2008)
   selecciona bloques completos (p. ej. todas las dummies de una categórica o todas las variables derivadas de una
   misma pregunta). El lasso encoge los coeficientes seleccionados; el *relaxed lasso* (Meinshausen 2007) o el
   «post-lasso» (Belloni y Chernozhukov 2013) re-estiman sin penalizar sobre el soporte. Hastie, Tibshirani y Tibshirani
   (2020) muestran que el relaxed lasso es el ganador más robusto frente a best subset y forward stepwise en relaciones
   señal/ruido bajas y altas. Los árboles, RF y boosting seleccionan implícitamente al elegir particiones.

**Estabilidad de la selección.** *Stability selection* (Meinshausen y Bühlmann 2010) ejecuta el selector base en
$B$ submuestras de tamaño $\lfloor n/2\rfloor$ y retiene las variables con frecuencia de selección $\hat\Pi_k\ge\pi_{thr}$.
Bajo intercambiabilidad y un selector «no peor que el azar», el número esperado de falsos seleccionados cumple
$E(V)\le q^2/\{(2\pi_{thr}-1)p\}$, con $q$ el tamaño medio del conjunto seleccionado en cada submuestra y $p$ el número
de candidatas (su Teorema 1). Shah y Samworth (2013) proponen *complementary pairs stability selection* (CPSS): con
$B$ pares de mitades disjuntas, **sin ningún supuesto**, para $\tau\in(1/2,1]$,
$$E\,|\hat S^{CPSS}_{n,\tau}\cap L_\theta|\;\le\;\frac{\theta}{2\tau-1}\,E\,|\hat S_{\lfloor n/2\rfloor}\cap L_\theta|,$$
donde $L_\theta$ es el conjunto de variables con probabilidad de selección $\le\theta$ (su Teorema 1(i)); con
$\theta=q/p$ se recupera $q^2/\{(2\tau-1)p\}$, y bajo unimodalidad la cota mejora por un factor $C(\tau,B)\theta$
(Teorema 2). En la práctica recomiendan $B=50$ pares. **Knockoffs** (Barber y Candès 2015; modelo-X: Candès, Fan, Janson
y Lv 2018) controlan la tasa de falsos descubrimientos para la hipótesis de independencia condicional
$X_j\perp Y\mid X_{-j}$; requieren conocer o modelar bien la distribución conjunta de $X$ (aquí mezcla de conteos,
proporciones y dummies, muy correlacionados), y responden una pregunta de *inferencia*, no de *pronóstico*: no se usan
como selector principal.

**Medidas de importancia y correlación.** (i) La importancia por impureza (Gini) de RF está sesgada hacia variables
continuas o de muchas categorías (Strobl et al. 2007). (ii) La importancia por permutación marginal rompe la dependencia
entre $X_j$ y las demás: obliga al modelo a extrapolar a regiones sin datos y sobrevalora variables correlacionadas
(Strobl et al. 2008; Hooker, Mentch y Zhou 2021). Las alternativas recomendadas por Hooker et al. requieren «un modelo
más»: permutación **condicional** (Strobl et al. 2008: permutar $X_j$ dentro de la rejilla formada por los cortes de las
covariables correlacionadas; Debeer y Strobl 2020) o **reajustar sin la variable** (LOCO: Lei et al. 2018; inferencia
general en Williamson et al. 2023). (iii) SHAP (Lundberg y Lee 2017) y TreeSHAP (Lundberg et al. 2020) descomponen cada
predicción de forma aditiva, $f(x)=\phi_0+\sum_j\phi_j(x)$; por esa aditividad, la suma de los $\phi_j$ de un **grupo**
es la contribución del grupo a cada predicción, lo que permite rankings por pregunta de encuesta (ojo: la media de
$|\phi_j|$ no es aditiva por grupos; hay que sumar primero $\phi$ dentro del grupo y luego promediar el valor absoluto).
La versión «interventional» de TreeSHAP también impone independencia entre $x_S$ y el resto y comparte, en menor grado,
el problema de extrapolación; con variables casi duplicadas el crédito se reparte entre ellas y ninguna parece
importante por sí sola. (iv) Fisher, Rudin y Dominici (2019) muestran que la importancia de una variable depende del
modelo elegido dentro de un «conjunto Rashomon» de modelos casi igual de buenos: la importancia de *un* ajuste no es una
propiedad de los datos. SAGE (Covert, Lundberg y Lee 2020) da importancias **globales** aditivas en términos de
reducción de la pérdida, que es lo que interesa para decidir qué preguntas conservar. (v) **Boruta** (Kursa y Rudnicki 2010) busca el conjunto *all-relevant* (incluye variables
débilmente relevantes y redundantes), lo contrario del objetivo de parsimonia.

### 3.2 La unidad de parsimonia es la pregunta de la encuesta

El enunciado premia el «mínimo número de variables» y pregunta si el modelo se puede aplicar con una **encuesta corta**.
Para la política pública el costo no es el número de columnas de la matriz, sino el número de **preguntas** del
cuestionario que hay que hacer (y a cuántas personas). Se recomienda que la fase B entregue un mapa
$Q:\text{variable}\to\{\text{preguntas fuente}\}$ (p. ej. `ocupados_pc` $\to$ {Oc, Nper}; `log_arriendo_sobre_linea`
$\to$ {P5130 o P5140, P5090, Lp, Npersug}) y que el costo de un conjunto sea
$$c(S)=\Bigl|\bigcup_{j\in S}Q(j)\Bigr|.$$
Agregar una variable derivada de preguntas ya incluidas tiene costo marginal cero; por eso la selección debe hacerse
**por bloques de pregunta**: el paso hacia adelante agrega la pregunta $q$ (con todas sus variables derivadas
compatibles) que más mejora la métrica CV por unidad de costo. Se reportan dos curvas de parsimonia: F1 frente a número
de variables y F1 frente a $c(S)$ (número de preguntas). Las líneas $L_p$ y $L_i$ y el dominio no son preguntas (los
conoce el Estado) y tienen costo cero.

**Contexto de focalización.** Es la lógica de los *proxy means tests* (Grosh y Baker 1995; Coady, Grosh y Hoddinott
2004): pocas preguntas observables y verificables que predicen el ingreso. La literatura reciente aporta tres lecciones
aplicables: (i) la validación **fuera de muestra** mejora la focalización frente a los PMT ajustados en muestra (McBride
y Nichols 2018, que además advierten que lo validado sirve «fuera de muestra pero no fuera de población», el mismo
punto de §1); (ii) restringir un logit con red elástica a **10 preguntas** no deteriora el desempeño frente al modelo
completo (Kshirsagar et al. 2017, con selección por bootstrap tipo stability selection y una tarjeta de puntaje
entera); (iii) los PMT tienden a errar justamente cerca de la línea y entre los más pobres (Brown, Ravallion y van de
Walle 2018), y la elección del instrumento tiene consecuencias de legitimidad y costo (Alatas et al. 2012; Hanna y
Olken 2018; con datos no tradicionales, Aiken et al. 2022).

### 3.3 Recomendación por algoritmo

| Algoritmo | Selección recomendada (todo dentro del pliegue) | Por qué |
|---|---|---|
| **LPM** | Camino lasso sobre MCO (mínimos cuadrados con $\ell_1$) para ordenar; soporte en $\lambda_{1se}$; reajuste MCO sin penalizar (post-lasso, Belloni–Chernozhukov 2013). Alternativa: hacia adelante por bloques de pregunta con CV. | Sin hiperparámetros propios: la única «complejidad» es el soporte. El post-lasso elimina el encogimiento que distorsiona el ranking cerca del umbral. |
| **Logit** | Camino L1-logit (glmnet; Friedman, Hastie y Tibshirani 2010) para ordenar; reajuste MV sin penalizar sobre el soporte (relaxed lasso con $\gamma=0$). Variante: group lasso por pregunta. BIC como referencia ($\log n\approx 11{,}9$ por parámetro). | El logit sin penalizar con ~200 columnas muy colineales es inestable; la selección L1 + reajuste da un modelo interpretable (EMP con EE sándwich) que sirve para la diapositiva de política. |
| **Elastic Net** | Integrada: $(\alpha,\lambda)$ por CV de log-loss; $\lambda_{1se}$ para la versión parsimoniosa; *stability selection* CPSS ($B=50$ pares, $\tau=0{,}75$) para reportar variables estables. Group lasso para dummies de una misma categórica. | La EN es el selector; reportar el soporte estable cuantifica la incertidumbre de la selección. |
| **CART** | Integrada por particiones + **poda de costo-complejidad** con CV y regla 1-EE (Breiman et al. 1984). Para la versión «política», profundidad $\le 4$–5. No hace falta preselección. | El árbol podado *es* el conjunto de variables usado; CART es inestable (Breiman 1996b): reportar la frecuencia de cada variable en los árboles de los 5 pliegues × semillas. |
| **Random Forest** | RFE por bloques (eliminar el 20 % menos importante y re-rankear) con importancia por **permutación agrupada sobre OOB** o, para la decisión final, **LOCO por bloques** (reajustar sin el bloque). Nunca Gini. Boruta sólo como cribado inicial de irrelevantes. | RF es robusto a variables irrelevantes; la selección busca parsimonia, no exactitud. La permutación marginal se sesga con correlación (Strobl 2008; Hooker 2021). |
| **Boosting** | Ranking por **SHAP agrupado** (suma de $\phi$ por pregunta) calculado en el entrenamiento del pliegue; RFE por bloques re-rankeando en cada paso; confirmación final con LOCO por bloques. Curva de parsimonia $k\in\{5,10,15,20,30,40,60,80,\text{todas}\}$. | TreeSHAP es exacto y rápido; agrupar resuelve el reparto de crédito entre variables casi duplicadas. |
| **Naive Bayes** | **Filtro mRMR** para ordenar + **envoltorio hacia adelante** (Bayes selectivo; Langley y Sage 1994) con CV de log-loss; discretización dentro del pliegue: igual frecuencia con el número de intervalos como hiperparámetro, en la línea de la discretización proporcional de Yang y Webb (2009) (intervalos ≈ tamaño por intervalo ≈ $\sqrt n$), que en NB supera a la MDL de Fayyad e Irani (1993); ver también Dougherty, Kohavi y Sahami (1995). | NB cuenta dos veces la evidencia de variables redundantes, lo que distorsiona las probabilidades aunque el orden pueda seguir siendo bueno (Domingos y Pazzani 1997; Hand y Yu 2001); la selección es la palanca principal de exactitud (no sólo de parsimonia). Como contraparte generativa del logit, converge más rápido pero a una asíntota peor (Ng y Jordan 2002): con 154 mil hogares se espera que quede por debajo del Logit. |

**Valores por defecto.** Curva de parsimonia con $k$ en una rejilla logarítmica; regla de elección: el menor $k$ (o
menor $c(S)$) cuyo F1 anidado no sea peor que el del mejor en más de una ROPE de 0,002 (o, en forma clásica, que esté a
menos de un error estándar **pareado** del mejor; §2.5). Stability selection: $B=50$ pares complementarios,
$\tau=0{,}75$, $q\approx 20$; reportar la cota CPSS de $E(V)$.

### 3.4 Cómo verificarlo aquí

1. **Fuga por selección:** comparar el F1 de la versión parsimoniosa con selección *dentro* del pliegue contra la misma
   con selección en toda la muestra; la diferencia estima el sesgo de Ambroise–McLachlan (debe ser pequeña con $n$
   grande, pero se documenta).
2. **Estabilidad:** índice de Jaccard medio entre los conjuntos elegidos en los 5 pliegues × 3 semillas
   ($\bar J=\frac{2}{m(m-1)}\sum_{a<b}|S_a\cap S_b|/|S_a\cup S_b|$) con su error estándar bootstrap; frecuencia de
   selección por variable. Un conjunto con $\bar J<0{,}5$ no debe presentarse como «las variables que importan».
3. **LOCO agrupado** sobre el modelo final: caída de log-loss y de F1 al reajustar sin cada bloque teórico (A–F e IPM),
   con IC pareado.
4. **Encuesta corta:** el mejor modelo restringido a $c(S)\le 15$ preguntas frente al mejor sin restricción; la
   diferencia es el «precio de la encuesta corta» que se lleva a la diapositiva de política.

### 3.5 Riesgos

- Sobreajustar la partición con envoltorios de muchos pasos: usar pasos por bloque y confirmar el conjunto final en una
  segunda partición (semilla distinta).
- Presentar importancias de un solo ajuste como hechos: la importancia depende del modelo (Fisher et al. 2019) y de la
  semilla (§7, infraespecificación); reportar promedios sobre pliegues × semillas con su dispersión.
- Variables casi duplicadas (p. ej. `arriendo`, `log_arriendo`, `arriendo_x_cabecera`): sin agrupar, la importancia se
  diluye y la selección por ranking individual puede descartar el bloque entero.

---

## 4. Ajuste de hiperparámetros

### 4.1 Síntesis de la teoría

**Búsqueda aleatoria frente a rejilla.** La respuesta $\Psi(\lambda)$ suele tener **baja dimensión efectiva**: sólo unos
pocos hiperparámetros importan y cuáles importan cambia entre conjuntos de datos. Una rejilla de 9 puntos en 2
dimensiones sólo prueba 3 valores distintos del hiperparámetro relevante; 9 puntos aleatorios prueban 9 (Bergstra y
Bengio 2012, §1 y Fig. 1; con procesos gaussianos con ARD estiman una dimensión efectiva de 1 a 4 sobre 7). La búsqueda
**cuasi-aleatoria** (secuencias de baja discrepancia, p. ej. Sobol) es una «rejilla jitterizada y barajada»:
no adaptativa, reproducible y, según el *Deep Learning Tuning Playbook* de Google (Godbole et al. 2023), «es muy difícil
que un algoritmo adaptativo le gane a una búsqueda cuasi-aleatoria con el doble de presupuesto»; por eso la recomiendan
en la fase de **exploración** y la optimización bayesiana en la de **explotación**.

**Optimización bayesiana.** Modela $\Psi$ con un sustituto y elige el siguiente punto maximizando una función de
adquisición: proceso gaussiano con mejora esperada (Snoek, Larochelle y Adams 2012; es el algoritmo por defecto de Google
Vizier para estudios de menos de mil ensayos: *batched GP bandits* con kernel Matérn con ARD; Golovin et al. 2017) o
estimadores de Parzen en árbol (TPE; Bergstra et al. 2011; Watanabe 2023), que es el muestreador por defecto de Optuna
(Akiba et al. 2019) y maneja espacios condicionales y mixtos.

**Multi-fidelidad.** *Successive halving* (Jamieson y Talwalkar 2016): con $n$ configuraciones y recurso mínimo $r$,
entrenar todas con $r$, quedarse con la mejor fracción $1/\eta$ y multiplicar el recurso por $\eta$ ($\eta=3$ por
defecto). *Hyperband* (Li et al. 2018) recorre varios compromisos $(n,r)$; ASHA (Li et al. 2020) lo hace asíncrono;
BOHB (Falkner, Klein y Hutter 2018) combina Hyperband con un sustituto bayesiano. La **regla de la mediana** de Vizier
(§3.2.2) detiene el ensayo $x_t$ en el paso $s$ si su mejor valor hasta $s$ es estrictamente peor que la mediana de los
promedios acumulados $\bar o^{\tau}_{1:s}=\frac1s\sum_{i\le s}o^\tau_i$ de los ensayos completados; es libre de modelo y
da aceleraciones de 2–3× frente a búsqueda aleatoria sin perder el mejor ensayo. Aquí el «recurso» natural es el
**número de pliegues evaluados** (evaluar el pliegue 1, podar si va peor que la mediana; Optuna `MedianPruner`) o el
tamaño de submuestra.

**Qué hiperparámetros importan (tunability).** Probst, Boulesteix y Bischl (2019) definen la *tunability* como la
ganancia esperada de ajustar frente a usar valores por defecto y la estiman en 38 conjuntos binarios de OpenML (AUC):

| Algoritmo | Tunability (AUC, sobre los defaults del paquete → sobre defaults óptimos) | Más ajustables | Defaults óptimos (su Tabla 3) |
|---|---|---|---|
| glmnet (EN) | 0,069 → 0,024 | $\lambda$ (0,021) ≫ $\alpha$ (0,006) | $\alpha=0{,}403$, $\lambda=0{,}004$ |
| rpart (CART) | 0,038 → 0,012 | `minbucket`, `minsplit` | cp = 0, maxdepth = 21, minbucket = 12, minsplit = 24 |
| ranger (RF) | 0,010 → 0,006 | `mtry` (0,003) > `sample.fraction` > `replace`, `min.node.size` | 983 árboles, sin reemplazo, sample.fraction = 0,703, mtry = 0,257·p, min.node.size = 1 |
| xgboost | 0,043 → 0,014 | `eta` (con `nrounds`), luego el resto (0,001–0,002) | nrounds = 4.168, eta = 0,018, subsample = 0,839, max_depth = 13, min_child_weight = 2,06, colsample_bytree = 0,752, colsample_bylevel = 0,585, λ = 0,982, α = 1,113 |

Lecturas: (i) RF es el algoritmo **menos** ajustable (sus defaults son casi óptimos); (ii) EN y boosting ganan más con
el ajuste; (iii) el número de árboles de RF **no se ajusta**: para Brier y log-loss el error esperado es monótono
decreciente en el número de árboles (Probst y Boulesteix 2018; la mayor ganancia ocurre en los primeros 100–250); (iv)
para RF, Probst, Wright y Boulesteix (2019) recomiendan ajustar `mtry`, tamaño de muestra y tamaño de nodo
**con predicciones OOB** (no CV) mediante optimización secuencial (30 puntos iniciales + 70 iteraciones), con Brier como
medida y el promedio del 5 % mejor como valor final.

**Sobreajuste del criterio de validación.** Cada ensayo reutiliza los mismos pliegues; elegir el máximo de $m$
estimaciones ruidosas con EE $\sigma$ sesga al alza el ganador en $\approx\sigma\sqrt{2\log m}$ cuando las diferencias
reales son menores que $\sigma$ (Cawley y Talbot 2010). Bischl et al. (2023) recomiendan remuestreo anidado para evaluar
el modelo ajustado y presupuestos explícitos.

### 4.2 Objetivo de la búsqueda: log-loss, no F1

La log-loss $\ell(q,y)=-y\log q-(1-y)\log(1-q)$ es **estrictamente propia**: $E[\ell(q,Y)\mid X=x]$ se minimiza sólo en
$q=\eta(x)$ (Gneiting y Raftery 2007). Por §5.2, el F1 del plug-in converge a $F^*$ cuando $E|\hat\eta-\eta|\to0$, y la
log-loss controla esa distancia: el exceso de log-loss es $E_X[\mathrm{KL}(\eta(X)\,\|\,\hat\eta(X))]$ y, por la
desigualdad de Pinsker para Bernoulli, $\mathrm{KL}(\eta\|\hat\eta)\ge2(\eta-\hat\eta)^2$, de modo que
$E|\hat\eta-\eta|\le\sqrt{\Delta_{\text{log-loss}}/2}$.
El F1 en el umbral óptimo, en cambio, depende sólo del orden cerca del umbral, es escalonado en los hiperparámetros y
tiene más ruido relativo. **Evidencia propia**: en las búsquedas anteriores (cuadernos 10–12, 9–19 ensayos completos por
algoritmo) el rango de F1 anidado entre *todas* las configuraciones fue de sólo 0,0012–0,0015, comparable con la
desviación entre semillas de un mismo modelo (≈ 0,0007), y la correlación de Spearman entre log-loss y F1 a través de los
ensayos fue −0,68 (LightGBM), −0,22 (XGBoost) y −0,07 (RF): cerca del óptimo, **las diferencias de F1 entre
configuraciones son mayoritariamente ruido**. Elegir por F1 persigue ese ruido; elegir por log-loss usa una señal más
estable y teóricamente alineada.

**Regla**: la búsqueda minimiza la **log-loss OOF en el dominio objetivo**; las 5 mejores configuraciones se re-evalúan
con $R\times K\times S=3\times5\times3$ y se elige por log-loss media, desempatando por F1 anidado (dentro de la ROPE,
se prefiere la más simple o la más regularizada). Brier es una alternativa igualmente válida (también propia y más
robusta a probabilidades extremas).

### 4.3 Semillas, parada temprana y elección robusta

- **Semillas**: con submuestreo de filas y columnas, el LightGBM de una sola semilla varía 0,0007–0,0008 en F1 anidado
  entre semillas (§1.6). Toda configuración finalista se evalúa con ≥ 3 semillas; el modelo final promedia 5–10 (§8).
- **Parada temprana**: con un 10 % interno (≈ 12 mil hogares) el número óptimo de árboles varió entre 261 y 534 entre
  semillas para la misma configuración: la parada temprana es ruidosa. Recomendaciones: reservar 15–20 % interno
  **sólo de hogares objetivo**, paciencia ≥ 200 iteraciones con tasa de aprendizaje 0,02–0,05. Para el reajuste
  final con el 100 % de los datos (§8.3) no hay conjunto de parada: usar la mediana de las iteraciones óptimas de los
  pliegues (a la tasa final) multiplicada por ≈ 1,2, entre el escalamiento lineal con el tamaño muestral
  ($1/(1-1/K)=1{,}25$) y el de raíz (1,12); la log-loss es plana cerca del óptimo, así que la elección no es crítica.
- **Elección robusta**: preferir configuraciones en una **meseta** (el vecindario de la mejor también es bueno) a un
  pico aislado; operativamente, promediar los hiperparámetros numéricos del 5 % mejor (Probst, Wright y Boulesteix
  2019) o elegir la configuración con mejor log-loss media sobre semillas.
- **Límites del espacio**: si el mejor punto está cerca de un borde, ampliar el espacio y repetir (Playbook, «Identifying
  bad search space boundaries»).
- **Hiperparámetros científicos, de molestia y fijos** (Playbook): cuando se compara una decisión «científica» (peso de
  Bogotá, conjunto de variables, restricciones monótonas, pesos de clase), los hiperparámetros de molestia (tasa de
  aprendizaje, hojas, regularización) deben estar re-optimizados para cada valor, o al menos verificarse con una
  *isolation plot* (mejor ensayo por valor del científico) que la conclusión no cambia.

### 4.4 Restricciones monótonas por teoría económica (boosting y RF)

**Fundamento.** La pobreza se define como $Y=\mathbb 1\{I<R\}$, con $I$ el ingreso del hogar y $R$ el ingreso requerido
(observable: $R_h=N_{ug}\,L_p-A_h$). Si $F(\cdot\mid x)$ es la distribución de $I$ dado las demás covariables,
$\eta(x,R)=F(R\mid x)$, **creciente en $R$** a $x$ fijo. Análogamente, si una variable $z$ desplaza la distribución de
$I$ en el sentido de dominancia estocástica de primer orden (más ocupados por miembro, más cotizantes, más educación a
igual tamaño del hogar), entonces $\eta$ es decreciente en $z$ ceteris paribus; si $z$ es un marcador de focalización
(régimen subsidiado del SISBÉN) o de privación (hacinamiento, dependencia), $\eta$ es creciente. Estos son supuestos
*ceteris paribus* plausibles, no identidades; por eso se **contrastan**.

**Lista candidata** (≈ 6–10 variables, sólo con signo inequívoco): `log_ingreso_requerido` (+), `ocupados_pc` (−),
`formales_pc` o `n_cotiza_pension` (−), `educ_max_adultos` y `jefe_anios_educ` (−), `personas_por_cuarto` (+),
`dependencia` (+), `share_subsidiado` (+). No restringir variables cuyo efecto parcial cambia de signo con la
composición del hogar (p. ej. `Nper` sin controlar ocupados) ni las interacciones.

**Implementación** (documentación oficial de LightGBM y XGBoost): LightGBM `monotone_constraints` con
`monotone_constraints_method="advanced"` (la opción `basic` sobre-restringe); XGBoost `monotone_constraints`; CatBoost `monotone_constraints`; scikit-learn ≥ 1.4
`RandomForestClassifier(monotonic_cst=...)` (la restricción opera sobre la probabilidad de la clase positiva).

**Regla de adopción** (no inferioridad): adoptar si la diferencia pareada de F1 anidado es $\ge-0{,}002$ (ROPE) y la de
log-loss $\le+0{,}0005$, ambas con el t corregido. **Por qué vale la pena aunque no mejore**: (i) reduce la varianza
(es información a priori correcta); (ii) robustez fuera de muestra; (iii) **legitimidad de política**: una regla de
focalización no debe quitarle elegibilidad a un hogar porque tenga un hijo más o un ocupado menos (Gupta et al. 2016;
Wang y Gupta 2020 lo formulan como restricción ética de forma; en ensambles bayesianos de árboles, Chipman et al.
2022 muestran cómo imponer monotonía multidimensional). Para LPM/Logit/EN los signos se *verifican* (no se
imponen) y se reportan las violaciones.

### 4.5 Espacios recomendados y presupuestos

Los espacios concretos por algoritmo están en la tabla final (§11). Principios: escalas logarítmicas para tasas,
regularizaciones y tamaños mínimos de hoja; tasa de aprendizaje fija en la búsqueda (0,05) y más baja en el modelo final
(0,02 con más árboles; Friedman 2001 muestra que una contracción menor con más iteraciones mejora el error); número de
árboles por parada temprana en boosting y fijo y grande en RF. Presupuestos orientativos: LPM/Logit, sin búsqueda
(sólo la especificación); EN, rejilla completa $\alpha\times$ camino de 50–100 $\lambda$ (barata con *warm starts*);
CART, rejilla de ≈ 100 configuraciones + camino de poda; RF, 60 ensayos TPE con OOB + confirmación CV de los 5 mejores;
Boosting, 64 cuasi-aleatorios + 100–150 TPE con poda por mediana + confirmación de los 5 mejores; NB, envoltorio de
selección + rejilla pequeña de suavizado y discretización.

### 4.6 Cómo verificarlo y riesgos

- Gráficos de eje por hiperparámetro y de importancia (fANOVA; Hutter, Hoos y Leyton-Brown 2014, la que reporta
  Optuna) para mostrar qué importó y que el óptimo no está en un borde.
- Optimismo de la búsqueda: re-evaluar el ganador en una partición nueva y reportar la corrección de
  Tibshirani–Tibshirani (§2.3).
- Riesgo de comparar algoritmos con presupuestos distintos: Sculley et al. (2018) piden ajustar todos los
  hiperparámetros clave de todos los modelos, incluidas las líneas base, y reportar el procedimiento; aquí se reporta el
  presupuesto (ensayos, tiempo) de cada algoritmo en la comparación.

---

## 5. F1 y el umbral: teoría de decisión, calibración y prevalencia

### 5.1 El clasificador F1-óptimo es un umbral sobre la probabilidad (derivación)

Sea $\eta(x)=P(Y=1\mid X=x)$, $\pi=P(Y=1)$ y un clasificador (posiblemente aleatorizado) $\theta:\mathcal X\to[0,1]$.
En la población, $TP(\theta)=E[\eta(X)\theta(X)]$, $FP(\theta)=E[(1-\eta(X))\theta(X)]$ y $FN(\theta)=\pi-TP(\theta)$, de modo
que
$$\mathrm{F1}(\theta)=\frac{2TP}{2TP+FP+FN}=\frac{2\,E[\eta\theta]}{E[\theta]+\pi}.$$
Es un cociente de funcionales lineales en $\theta$. Sea $F^*=\sup_\theta \mathrm{F1}(\theta)$. Por definición, para todo
$\theta$, $2E[\eta\theta]-F^*(E[\theta]+\pi)\le0$, con igualdad en el óptimo (argumento de Dinkelbach para programas
lineal-fraccionales). Luego $\theta^*$ maximiza el funcional **lineal**
$$L(\theta)=E\bigl[(2\eta(X)-F^*)\,\theta(X)\bigr]-F^*\pi,$$
que se maximiza punto a punto con
$$\boxed{\theta^*(x)=\mathbb 1\{\eta(x)>F^*/2\}}$$
(cualquier valor si $\eta(x)=F^*/2$). Es el corolario 1 de Lipton, Elkan y Naryanaswamy (2014) (su teorema 1 lo
enuncia en términos de razón de verosimilitudes: positivo si $\frac{b\,p(s\mid t=1)}{(1-b)\,p(s\mid t=0)}\ge J$, con $J$ el
índice de Jaccard del clasificador óptimo) y el caso $F_1$ del teorema 2 de Koyejo et al. (2014), que para cualquier
métrica lineal-fraccional $L=\frac{c_0+c_1TP+c_2\gamma}{d_0+d_1TP+d_2\gamma}$ ($\gamma=P(\hat Y=1)$) da
$\theta^*=\mathrm{sign}(\eta-\delta^*)$ con $\delta^*=\frac{d_2L^*-c_2}{c_1-d_1L^*}$ ($F_\beta$: $\delta^*=L^*/(1+\beta^2)$; Jaccard:
$L^*/(1+L^*)$); ver también Zhao et al. (2013) y Ye et al. (2012, teorema 4). Consecuencias inmediatas:

1. **Sólo importan el orden de $\eta$ y un umbral.** Ninguna otra regla (por dominio, por subgrupo) mejora el F1 global
   si $\eta$ se conoce: el umbral óptimo es **el mismo** para todos los hogares y todos los dominios.
2. **El umbral óptimo es $F^*/2\le1/2$**: con $F^*\approx0{,}77$, $\delta^*\approx0{,}38$–$0{,}39$, y el clasificador
   predice «pobre» a hogares con probabilidad menor que 1/2; la tasa de positivos predicha supera a la prevalencia (aquí
   0,223 frente a 0,207). El umbral 0,5 es subóptimo (los primeros envíos con EN sacaron 0,44 por esto).
3. **Diagnóstico de calibración:** si las probabilidades están calibradas cerca del umbral, el umbral empírico debe
   cumplir $\hat t\approx\hat F/2$. Con el LightGBM de referencia: $2\hat t=0{,}783$ frente a $\hat F=0{,}768$; la
   pequeña brecha es compatible con la meseta de la curva F1 (siguiente punto).

### 5.2 Consistencia del umbral «plug-in»

El procedimiento en dos pasos —partir la muestra en $S_1$ y $S_2$, estimar $\hat\eta$ en $S_1$ y elegir
$\hat\delta=\arg\max_\delta\widehat{\mathrm{F1}}_{S_2}(\mathrm{sign}(\hat\eta-\delta))$ en $S_2$— es **consistente**:
si $\hat\eta\to\eta$ en probabilidad, el F1 del clasificador resultante converge a $F^*$ (Koyejo et al. 2014, algoritmo 1
«Two-Step EUM» y teorema 10). Narasimhan, Vaish y Agarwal (2014, teorema 1 y teorema 8 para $F_\beta$) dan las
condiciones precisas: el estimador de probabilidad es consistente en $L_1$, $E_X|\hat\eta(X)-\eta(X)|\to0$; el umbral
óptimo $t^*$ existe en $(0,1)$; y las distribuciones de $\eta(X)$ condicionales a $Y=1$ y a $Y=0$ son **continuas en
$t^*$** (sin átomos en el umbral). Las OOF de la CV son exactamente un esquema de dos pasos *cross-fitted*: cada
$\hat\eta$ se evalúa en hogares que no usó, y el umbral anidado se elige en pliegues distintos del evaluado. Dos
lecturas aplicables:

- **El umbral se estima con predicciones honestas** (OOF o OOB), nunca con las predicciones de entrenamiento, que son
  demasiado extremas; y la muestra que estima el umbral debe tener la misma distribución que el test: **sólo los 24
  dominios objetivo**.
- **La calidad del F1 depende de $E|\hat\eta-\eta|$**: por eso el ajuste de hiperparámetros optimiza una regla de
  puntuación estrictamente propia (log-loss o Brier; Gneiting y Raftery 2007), cuyo minimizador poblacional es
  justamente $\eta$ (§4.2).
- **La condición de continuidad falla en modelos con pocas puntuaciones distintas** (CART podado, NB muy discretizado):
  hay átomos en el umbral, la curva F1 es escalonada y el umbral óptimo puede requerir aleatorizar dentro de una hoja.
  En la práctica se elige entre los valores observados y se acepta la pérdida.

**El umbral está en una meseta.** Si $\mathrm{F1}(t)$ es dos veces diferenciable en $t^*$,
$\mathrm{F1}(t)\approx F^*-\tfrac12|\mathrm{F1}''(t^*)|(t-t^*)^2$: un error en el umbral cuesta F1 de **segundo orden**.
Aquí el F1 OOF está a menos de 0,001 de su máximo para todo $t\in[0{,}347;\,0{,}410]$ (tasas de positivos predichas
entre 0,217 y 0,238). Por eso el umbral anidado pierde muy poco frente al «oráculo» (≈ 0,0005–0,001), y por eso
ajustar el umbral con el tablero público (EE 0,006) es inútil y peligroso.

**Expected utility vs decision-theoretic.** Ye, Chai, Lee y Chieu (2012) distinguen maximizar el F1 *poblacional* (EUM,
un umbral) de maximizar el F1 *esperado del conjunto de test* (DTA; algoritmo exacto de Dembczyński et al. 2011) y
prueban (teorema 8) que, si $P(\eta(X)=\delta)=0$ para todo $\delta$, ambos son asintóticamente equivalentes cuando el
conjunto de test crece. Con 66 mil hogares (o 53 mil en el privado) la regla de umbral es la correcta. Puthiya
Parambath, Usunier y Grandvalet (2014) llegan a la misma forma reduciendo la maximización del F a una sucesión de
problemas sensibles al costo, y Dembczyński et al. (2017) revisan la consistencia de ambos enfoques.

### 5.3 El papel de la calibración

Si $s(x)=g(\eta(x))$ con $g$ estrictamente creciente, la regla $\mathbb 1\{s>g(F^*/2)\}$ es idéntica a la óptima: **para F1
basta el orden** si el umbral se estima en la misma escala de puntuación. La calibración importa cuando:

1. **El umbral se traslada entre escalas**: de los modelos de los pliegues (OOF) al modelo final (reajustado con más
   datos, más árboles o promediado), o entre semillas. Si la escala cambia, un umbral fijo deja de ser óptimo; la
   solución es calibrar o igualar la tasa de positivos (§5.5).
2. **Hay descalibración distinta por subgrupo** ($s=g_d(\eta)$ con $g_d$ dependiente del dominio $d$): ninguna
   transformación monótona global la corrige; un umbral por dominio $t_d=g_d(F^*/2)$ sí, pero añade 24 parámetros con
   ≈ 6.400 hogares cada uno. Sólo se adopta si gana en CV por encima de la ROPE (aquí ya se probó y no se adoptó:
   −0,0002, `10_umbral_dominio.csv`). Incluir el dominio como variable del modelo es la forma estadísticamente más
   eficiente de lograr calibración por dominio.
3. **Se promedian modelos** (semillas, pliegues): promediar probabilidades descalibradas mezcla escalas.

**Métodos.** Platt (2000; manuscrito de 1999): ajuste logístico $\sigma(a\,s+b)$, adecuado para distorsiones sigmoides (boosting con
pocas iteraciones, SVM); isotónica (Zadrozny y Elkan 2002): no paramétrica y monótona, necesita miles de casos (aquí
sobran). Niculescu-Mizil y Caruana (2005): los árboles potenciados y los bosques comprimen las probabilidades hacia el
centro, Naive Bayes las empuja a los extremos, y la regresión logística sale bien calibrada; el boosting moderno con
pérdida logarítmica y parada temprana suele salir calibrado (aquí ECE ≈ 0,001 en el LightGBM previo). **Regla**: la
calibración se ajusta *dentro* del pliegue (con un subconjunto interno o con predicciones OOF internas), se evalúa con
Brier (1950), log-loss, ECE con bins de **igual masa** (Nixon et al. 2019) y curvas de confiabilidad, y se **adopta sólo si
mejora el F1 anidado o es necesaria para trasladar el umbral**. Para NB y RF se espera que la isotónica ayude a la
log-loss pero no al F1 (que sólo depende del orden). El objetivo de un pronóstico probabilístico es «maximizar la
nitidez sujeta a la calibración» (Gneiting, Balabdaoui y Raftery 2007): la calibración se exige; la discriminación
(nitidez) es lo que el F1 premia.

### 5.4 Prevalencia y desplazamiento de etiquetas

Si la prevalencia del test fuera $\pi'\neq\pi$ con $P(X\mid Y)$ fija (prior shift), la posterior correcta es (Saerens,
Latinne y Decaestecker 2002)
$$\eta'(x)=\frac{\frac{\pi'}{\pi}\eta(x)}{\frac{\pi'}{\pi}\eta(x)+\frac{1-\pi'}{1-\pi}\bigl(1-\eta(x)\bigr)},$$
el F1 alcanzable cambia a $F'^*$ y el umbral óptimo pasa a $\eta'>F'^*/2$. Aquí **no hay desplazamiento de etiquetas
entre el objetivo y el test**: la prevalencia del objetivo es 20,72 % y, post-estratificada por la composición del
test, 20,75 % (EE muestral 0,16 pp; se esperan ≈ 13.730 pobres en el test). Sí lo habría entre *toda* la muestra de
entrenamiento (20,02 %) y el test si el modelo no supiera distinguir Bogotá: por eso el umbral se elige con OOF del
objetivo. Elkan (2001, teorema 2) da la misma corrección en forma cerrada para un cambio de tasa base $b\to b'$:
$p'=b'(p-pb)/(b-pb+b'p-bb')$. La estimación de $\pi_{test}$ por EM (Saerens), BBSE (Lipton, Wang y Smola 2018) o
máxima verosimilitud con recalibración corregida por sesgo (Alexandari, Kundaje y Shrikumar 2020) sirve como
**verificación**:
la media de las probabilidades calibradas en el test debe caer en $0{,}2075\pm0{,}003$.

### 5.5 Umbral fijo frente a tasa predicha (cuantil)

Dos formas de llevar el umbral al test:

- **Umbral fijo** $\hat t$ sobre la escala de probabilidad (válido si el predictor del test tiene la misma escala que el
  OOF);
- **Igualar la tasa de positivos**: $\hat q=\frac1{n_{\mathcal T}}\sum_{i\in\mathcal T}\mathbb 1\{s_i^{OOF}\ge\hat t\}$ y
  $t_{test}=$ cuantil $1-\hat q$ de las puntuaciones del test.

Si el test y el objetivo son intercambiables (aquí: AUC adversarial 0,499 sin Bogotá) y el predictor del test es una
transformación monótona del predictor OOF, igualar la tasa reproduce exactamente la misma regla de decisión y es
**invariante a cambios de escala**; si hubiera desplazamiento de etiquetas, igualar la tasa sería sesgado y el umbral
fijo sobre probabilidades recalibradas sería lo correcto. **Recomendación** (coherente con §8.3, donde el predictor
final se reajusta con el 100 % de los datos): (i) boosting, reajustado con más datos y otro número de árboles, cambia
algo de escala → **tasa igualada** $\hat q$, con el umbral fijo $\hat t$ como contraste; (ii) RF → umbral elegido en las
probabilidades **OOB del propio bosque final** (misma escala; §9.3); (iii) modelos deterministas (Logit, LPM, EN, NB,
CART) → umbral fijo $\hat t$, verificando que la tasa predicha en test caiga en
$\hat q\pm2\sqrt{\hat q(1-\hat q)/66.168}=\hat q\pm0{,}003$ y pasando a la tasa igualada si no cae. En todos los casos se
reportan ambas reglas y la proporción de hogares de test en que difieren.

**Umbral suavizado.** El argmax de la curva empírica es ruidoso; se recomienda la **mediana bootstrap** del argmax
($B=500$ remuestreos de los hogares objetivo OOF) o el punto medio del intervalo donde $\mathrm{F1}\ge\max-0{,}0005$.

### 5.6 F1 frente a los costos de la focalización (para la etapa 4 del enunciado)

El F1 es la media armónica de precisión y *recall*: pondera igual la filtración (falsos positivos) y la subcobertura
(falsos negativos) en el sentido de $2TP/(2TP+FP+FN)$. Un gobierno que valore más evitar la subcobertura usaría $F_\beta$
con $\beta>1$, cuyo umbral óptimo es $\delta^*=F_\beta^*/(1+\beta^2)$ (Koyejo et al. 2014): **más bajo** que $F_1^*/2$, con
más hogares cubiertos y más filtración; con costos explícitos $c_{FP},c_{FN}$ el umbral es $c_{FP}/(c_{FP}+c_{FN})$
(Elkan 2001). Para la presentación: graficar la frontera precisión-*recall* del mejor modelo marcando los umbrales de
$F_1$, $F_2$ y de un costo explícito, y reportar en cada uno las tasas de subcobertura y de filtración, todo con las OOF
del objetivo.

### 5.7 Riesgos del tablero público

El público (≈ 13.234 hogares) tiene EE de 0,006 en nivel y ≈ 0,002 en diferencias pareadas (§2.1). Probar umbrales en
el público es sobreajuste adaptativo puro (Blum y Hardt 2015; Dwork et al. 2015): la meseta de la curva F1 hace que
cualquier «mejora» observada sea ruido. **Regla**: ningún umbral, variable o hiperparámetro se elige por el público;
el público sólo se usa para verificar que el F1 observado caiga en el IC de la CV
($\hat F_{CV}\pm2\times0{,}006$).

### 5.8 Recomendación concreta y valores por defecto

1. Umbral = argmax de F1 sobre las OOF **de los 24 dominios objetivo**, suavizado por mediana bootstrap; F1 reportado =
   anidado (umbral de cada pliegue elegido en los otros 4).
2. Un solo umbral global (no por dominio) salvo ganancia CV > ROPE.
3. Diagnósticos obligatorios: $\hat t$ vs $\hat F/2$; tasa predicha OOF vs test; media de probabilidades en test vs
   0,2075; curva F1(t) con su meseta.
4. Calibración (Platt o isotónica, dentro del pliegue) sólo si mejora el F1 anidado o si el umbral debe trasladarse.

### 5.9 Cómo verificarlo y riesgos

- Verificar con las OOF que el F1 anidado y el F1 con umbral «oráculo» difieran en < 0,001 (meseta).
- Verificar que la tasa predicha en test con el umbral fijo coincida con la OOF (±0,003); si no, hay un problema de
  escala o un error de código (Zinkevich, regla #37: medir la brecha entre entrenamiento y servicio).
- Riesgo: modelos con pocas puntuaciones distintas (CART podado, NB discretizado) tienen curvas F1 escalonadas; el
  umbral debe elegirse entre los valores observados y la tasa predicha puede no poder igualarse exactamente (empates).

---

## 6. Desbalance de clases (20 % de pobres)

### 6.1 Síntesis de la teoría

**Pesos de clase = desplazamiento del umbral (Elkan 2001, teorema 1).** Si se multiplica el peso de los negativos por $k$ (o el de
los positivos por $1/k$), el minimizador poblacional de una pérdida propia ponderada es
$$\eta_k(x)=\frac{\eta(x)}{\eta(x)+k\,\bigl(1-\eta(x)\bigr)},$$
función **estrictamente creciente** de $\eta(x)$. Por tanto, umbralizar $\eta_k$ en $\delta$ equivale a umbralizar $\eta$ en
$\delta'=k\delta/(1-\delta+k\delta)$: en la población, ponderar las clases sólo cambia el umbral. Con costos
$c_{FP},c_{FN}$, la decisión óptima es $\eta(x)\ge c_{FP}/(c_{FP}+c_{FN})$ y la reponderación por $k$ es la forma de
lograr ese umbral con un clasificador que umbraliza en 1/2. Como el F1 se maximiza con un umbral elegido
explícitamente sobre las OOF (§5), **los pesos de clase no pueden mejorar el F1 poblacional**; en muestra finita sólo
cambian la regularización efectiva, las particiones de los árboles y la parada temprana, y empeoran la calibración
(la documentación de LightGBM lo advierte para `is_unbalance` y `scale_pos_weight`).

**Remuestreo.** Submuestrear los negativos con tasa $r$ produce $\eta_r=\eta/(\eta+r(1-\eta))$: descalibra y bota
información (varianza mayor). Si se submuestrea por costo computacional, hay que **reponderar** por $1/r$ (Google MLCC,
*downsample and upweight*; Zinkevich, regla #30: «Importance-weight sampled data, don't arbitrarily drop it!») o
corregir el intercepto (King y Zeng 2001: $\tilde\beta_0=\hat\beta_0-\log\bigl[\frac{1-\tau}{\tau}\frac{\bar y}{1-\bar y}\bigr]$,
con $\tau$ la prevalencia poblacional y $\bar y$ la de la muestra). SMOTE (Chawla et al. 2002) interpola entre hogares
pobres y crea hogares imposibles (conteos no enteros de personas, combinaciones de tenencia y arriendo inexistentes),
y distorsiona $P(X\mid Y)$. Van den Goorbergh et al. (2022) muestran, en regresión logística, que ROS, RUS y SMOTE no
mejoran el AUROC y producen una sobreestimación fuerte y sistemática del riesgo de la clase minoritaria (intercepto de
calibración ≤ −4,5 con 1 % de eventos en su simulación), y que la misma ganancia de sensibilidad se obtiene moviendo el
umbral; Carriero et al. (2025) extienden el resultado a modelos de aprendizaje automático, y Elor y Averbuch-Elor (2022) encuentran que el balanceo no ayuda a clasificadores fuertes (boosting) cuando
el umbral se optimiza. **Focal loss** (Lin et al. 2017), $-(1-p)^\gamma\log p$, se diseñó para desbalances de 1:1000 en
detección de objetos; es calibrada para la clasificación pero **no** es estrictamente propia para ningún $\gamma>0$, y en el caso
binario su minimizador es sistemáticamente subconfiado (Charoenphakdee et al. 2021, teorema 5 y corolario 10); sus
salidas sólo se vuelven probabilidades con la transformación cerrada de su teorema 11.

**Magnitud aquí.** 20 % de positivos es un desbalance leve: hay ≈ 32 mil pobres en el objetivo, de sobra para estimar
$\eta$. **Evidencia propia** (`10_desbalance.csv`, LightGBM con umbral optimizado): `is_unbalance` cambió el F1 anidado en
+0,00002 y $\sqrt{n_0/n_1}$ en −0,0004 (EE pareado ≈ 0,0004), con log-loss y Brier peores.

### 6.2 Recomendación concreta

- **Especificación principal de todos los algoritmos: sin pesos de clase y sin remuestreo**; el desbalance se maneja con
  el umbral F1-óptimo (§5) y con métricas adecuadas (F1, PR-AUC, log-loss; nunca exactitud).
- **Un experimento de sensibilidad por algoritmo** (lo pide el enunciado en la etapa 3: «Consider whether a class
  imbalance strategy made a difference»): pesos balanceados ($w_1/w_0=n_0/n_1\approx3{,}8$) frente a ninguno, con el
  umbral re-optimizado en ambos; reportar ΔF1 anidado (pareado, con ROPE), ΔBrier, ΔECE. Predicción teórica: ΔF1 dentro
  de la ROPE y calibración peor. En CART el efecto puede ser algo mayor (los pesos cambian el criterio de Gini y la poda),
  y en NB los pesos equivalen a cambiar la probabilidad a priori, es decir, otra vez el umbral.
- Prohibido: SMOTE y cualquier remuestreo sin reponderación; ajustar el umbral a 0,5 tras balancear (es una forma
  indirecta y ruidosa de elegir el umbral).

### 6.3 Cómo verificarlo y riesgos

- Verificación: la tabla de sensibilidad anterior con IC pareados; las curvas de confiabilidad con y sin pesos (la
  versión ponderada debe mostrar sobreestimación sistemática de $\eta$).
- Riesgo: confundir una ganancia de F1 con umbral 0,5 (que sí mejora al balancear) con una ganancia real; la comparación
  válida es siempre con el umbral óptimo en ambas versiones.

---

## 7. Investigación de Google aplicable (sólo lo que sirve con los algoritmos permitidos)

| Trabajo | Qué dice | Qué se hace aquí |
|---|---|---|
| **TabNet** (Arik y Pfister 2021, AAAI) | Red neuronal con atención secuencial y máscaras dispersas (sparsemax) que selecciona variables por observación; interpretable. | No es un algoritmo permitido. Lección: la selección de variables por observación y las máscaras dispersas son la versión neuronal de lo que el boosting hace con las particiones; las comparaciones independientes posteriores (Shwartz-Ziv y Armon 2022; Grinsztajn, Oyallon y Varoquaux 2022) muestran que los árboles potenciados siguen siendo superiores o equivalentes en datos tabulares de tamaño medio. Justifica concentrar el esfuerzo en boosting. |
| **Infraespecificación** (D'Amour et al. 2022, JMLR) | Un *pipeline* está infraespecificado si muchos predictores satisfacen igual de bien su criterio de validación (§2.2); difieren en pruebas de estrés: por subgrupos, bajo desplazamiento y contrastivas (§2.4). Protocolo: reentrenar cambiando sólo la semilla, verificar que el desempeño iid es igual y medir la variabilidad en las pruebas de estrés (§4). | (i) Medir la dispersión entre semillas del F1 por dominio, por zona (cabecera/rural) y por tamaño del hogar; (ii) el «deja-un-dominio-fuera» de §1.7 como prueba de estrés de desplazamiento; (iii) promediar semillas (§8); (iv) no interpretar importancias de un solo ajuste. |
| **Incertidumbre bajo desplazamiento** (Ovadia et al. 2019, NeurIPS) | La calibración se degrada con el desplazamiento; los ensambles son los más robustos. | El test no está desplazado respecto al objetivo, pero refuerza el promedio de semillas y la evaluación de calibración. |
| **Medir la calibración** (Nixon et al. 2019, CVPRW) | El ECE con bins fijos deja que pocos bins dominen, confunde calibración con nitidez y depende del número de bins; proponen ACE (bins de igual masa) y medidas por clase. | Reportar ECE con bins de igual masa (ACE) y curvas de confiabilidad; Brier y log-loss como medidas principales de calibración. |
| **Google Vizier** (Golovin et al. 2017, KDD; Song et al. 2022) | Por defecto, *batched GP bandits* (Matérn-ARD, mejora esperada) para < 1.000 ensayos; parada automática por la regla de la mediana; aceleración de 2–3× frente a búsqueda aleatoria. | TPE de Optuna (equivalente práctico) con `MedianPruner` por pliegues; búsqueda cuasi-aleatoria inicial. |
| **Deep Learning Tuning Playbook** (Godbole et al. 2023) | Hiperparámetros científicos, de molestia y fijos; cuasi-aleatoria para explorar y bayesiana para explotar; revisar los bordes del espacio; *isolation plots*. | §4.3: el peso de Bogotá, el conjunto de variables, las restricciones monótonas y los pesos de clase se tratan como científicos y se comparan con los de molestia re-optimizados. |
| **Rules of Machine Learning** (Zinkevich) | Regla #4: «Keep the first model simple and get the infrastructure right»; #12: «Don't overthink which objective you choose to directly optimize»; #14: los modelos lineales/logísticos convergidos sin regularizar están calibrados; #24: «Measure the delta between models»; #29 y #32: entrenar como se sirve, reutilizar el código entre entrenamiento y servicio; #30: ponderar, no descartar, los datos submuestreados; #37: medir la brecha entrenamiento/servicio; #40: «Keep ensembles simple». | El mismo código construye las variables de train y test (fase B ya lo hace); medir la diferencia de predicciones entre versiones de un modelo (#24) y la tasa predicha OOF frente a test (#37, §5.8); ensamble sólo dentro de un algoritmo (#40, §8); Logit como línea base calibrada (#14). |
| **Deuda técnica oculta** (Sculley et al. 2015, NeurIPS) | En sistemas de ML «cambiar cualquier cosa cambia todo» (entrelazamiento), y las tuberías de datos y las configuraciones acumulan deuda. | Un solo módulo de variables para train y test, configuración versionada en JSON y comparaciones que cambian **una** cosa a la vez (§4.3). |
| **Calibración de modelos modernos** (Minderer et al. 2021, NeurIPS) | Revisitan la calibración de redes modernas, también bajo desplazamiento de distribución, y la reportan junto con la exactitud. | Se reporta la calibración de todos los algoritmos (Brier, ACE, curvas) aunque el F1 sólo dependa del orden. |
| **Monotonía como restricción de forma** (Gupta et al. 2016, JMLR; You et al. 2017; Wang y Gupta 2020, AISTATS) | Las restricciones monótonas mejoran la robustez y la interpretabilidad, y pueden encarnar principios normativos («a igualdad de lo demás, más X nunca debe reducir el puntaje»). | §4.4: restricciones monótonas por teoría económica en boosting y RF, adoptadas si no empeoran la CV; argumento de legitimidad para la diapositiva de política. |
| **AutoML Tables en KaggleDays** (Lu 2019, Google AI Blog) | Una solución automática de extremo a extremo para datos tabulares (búsqueda de arquitecturas y ensambles) compitió con equipos humanos. | Lección limitada: el ensamble y la búsqueda automática ayudan, pero aquí no se pueden mezclar algoritmos (§8); se usa sólo la idea de presupuestos de búsqueda sistemáticos. |
| **Data valuation (DVRL)** (Yoon, Arik y Pfister 2020, ICML) | Aprende un valor por observación de entrenamiento con un conjunto de validación del dominio objetivo; sirve para adaptación de dominio. | Es la generalización por observación del peso de Bogotá $\lambda_B$; aquí basta la versión de un parámetro elegida por CV en el objetivo (§1). |
| **Machine Learning Crash Course** (Google for Developers) | Umbrales: 0,5 no es buena idea con clases desbalanceadas o costos asimétricos; desbalance: submuestrear y ponderar al alza por el mismo factor. | §5 y §6. |

**Lo que no se aplica**: redes neuronales tabulares, AutoML con mezcla de algoritmos y cualquier dato externo.

---

## 8. Ensamble y estabilidad dentro de un mismo algoritmo

### 8.1 Síntesis de la teoría

**Promedio de semillas y bagging.** Sea $\hat p^{(s)}(x)$ la predicción con semilla $s$ (submuestreo de filas y columnas,
partición interna de parada temprana). Con $m(x)=E_s[\hat p^{(s)}(x)\mid\mathcal D]$ y $v(x)=\mathrm{Var}_s[\hat p^{(s)}(x)\mid\mathcal D]$,
el promedio de $S$ semillas cumple
$$E\bigl[(\bar p_S(x)-\eta(x))^2\bigr]=E\bigl[(m(x)-\eta(x))^2\bigr]+\frac{E[v(x)]}{S},$$
y, por la desigualdad de Jensen, para cualquier pérdida convexa en la probabilidad (log-loss, Brier)
$\ell(\bar p_S,y)\le\frac1S\sum_s\ell(\hat p^{(s)},y)$: **promediar nunca empeora la log-loss media de los miembros**.
Breiman (1996a) mostró que el bagging reduce la varianza de procedimientos inestables; Bühlmann y Yu (2002) lo
formalizan (suaviza decisiones duras como las particiones) y muestran que el *subagging* (submuestras sin reemplazo,
p. ej. del 50 %) es casi equivalente y más barato. Los ensambles de semillas son también la técnica más robusta de
cuantificación de incertidumbre bajo desplazamiento (Lakshminarayanan, Pritzel y Blundell 2017; Ovadia et al. 2019).

**Evidencia propia (§1.6).** LightGBM con las variables permitidas, 5 pliegues, sin Bogotá: una semilla da log-loss
objetivo 0,2174 y F1 OOF óptimo 0,7660 (promedio de 3 semillas individuales); el **promedio de 3 semillas** da
0,2158 y 0,7680. Es la mejora más grande y más barata de todas las comprobaciones de este memo (+0,002 de F1), y se
explica por la varianza de la parada temprana y del submuestreo (§4.3).

**Promedio de los modelos de pliegue** («CV-bagging»): la predicción de test es el promedio de los $K$ modelos
entrenados en los pliegues (cada uno con 80 % de los datos) y de sus $S$ semillas. Es subagging con fracción 0,8, usa las
mismas funciones que produjeron las OOF (el umbral se traslada sin cambio de escala, §5.5) y no requiere decidir el
número de árboles del reajuste. **Pero desperdicia el 20 % de los datos en cada modelo**: la curva de aprendizaje
medida (§2.2) indica que pasar de 80 % a 100 % ganaría ≈ 0,001 de F1 y de log-loss. El **reajuste con el 100 % y $S$
semillas** obtiene a la vez la reducción de varianza (semillas) y la ganancia de datos; su único costo es fijar el
número de árboles sin conjunto de parada (§4.3) y trasladar el umbral por tasa igualada (§5.5). Por eso es el
predictor principal, y el promedio de pliegues queda como contraste.

**Ensamble de la selección** (Caruana et al. 2004): promediar las $k$ mejores configuraciones de hiperparámetros del
**mismo** algoritmo reduce la varianza de la selección (§4.1). Útil cuando las mejores configuraciones están en la
ROPE entre sí.

**Estabilidad** (Yu 2013; Yu y Kumbier 2020, marco PCS): un resultado es creíble si es estable frente a perturbaciones
razonables de los datos y del algoritmo. Medidas: (i) proporción de hogares de test cuya clasificación cambia entre
semillas (inestabilidad de decisión); (ii) Jaccard de las variables seleccionadas entre pliegues y semillas (§3.4);
(iii) dispersión del umbral elegido entre pliegues; (iv) ranking de algoritmos entre particiones.

### 8.2 Qué está permitido y qué no

- **Permitido (sigue siendo el mismo algoritmo)**: promedio de semillas; promedio de los modelos de pliegue; promedio de
  las mejores configuraciones del mismo algoritmo; más árboles en RF (en RF, «más semillas» es lo mismo que más
  árboles). El nombre del archivo de envío debe declararlo (p. ej. `LGBM_leaves_31_lr_0_02_..._s10.csv` para 10 semillas).
- **CART: no promediar.** Un promedio de árboles CART con bootstrap es *bagged trees* (un RF con `mtry = p`), es decir,
  otro algoritmo; además destruye la interpretabilidad, que es la razón de ser de CART. CART se estabiliza con la poda
  1-EE y se reporta su inestabilidad (§9.4).
- **Logit, LPM, EN, NB** son deterministas: no hay semillas que promediar (el promedio de los modelos de pliegue es
  admisible pero casi idéntico al reajuste con todo).
- **Boosting**: LightGBM, XGBoost y CatBoost son implementaciones del mismo algoritmo («Boosting», regla 4 del plan);
  un promedio de las tres es defendible como «Boosting (ensamble de implementaciones)», pero debe declararse así y cuenta
  como **un** algoritmo.
- **No permitido como algoritmo de la lista**: mezclar familias (p. ej. promedio de Logit y LightGBM, o *stacking*,
  Wolpert 1992). No es ninguno de los siete algoritmos del enunciado, no cuenta para el mínimo de cinco algoritmos y
  rompe la comparación «en igualdad de condiciones» que pide la presentación. Si se quisiera reportar, sólo como
  análisis complementario fuera de los envíos (Zinkevich, regla #40: «Keep ensembles simple»).

### 8.3 Recomendación concreta y valores por defecto

- Boosting: $S=3$ semillas en la confirmación CV y $S=10$ en el modelo final, **reajustado con el 100 %** de los datos
  (tasa 0,02, árboles = mediana CV × 1,2); predicción de test = promedio de las 10 semillas; umbral por tasa igualada a
  la de las OOF promediadas por semilla. Contraste: promedio de los $K\times S$ modelos de pliegue con umbral fijo;
  reportar la proporción de hogares de test clasificados distinto por ambos (se espera < 2 %).
- RF: reajuste con el 100 % y 1.000 árboles (verificar con la curva OOB que la log-loss se estabilizó); umbral en las
  OOB de ese mismo bosque.
- Modelos deterministas: reajuste con el 100 %; umbral fijo de las OOF con verificación de tasa (§5.5).
- Reportar, para el modelo final, la inestabilidad de decisión entre semillas y la ganancia del promedio
  ($S=1$ frente a $S=10$) con IC pareado.

### 8.4 Cómo verificarlo y riesgos

- Curva de log-loss y F1 OOF frente a $S\in\{1,2,3,5,10\}$: debe ser decreciente y aplanarse.
- Riesgo: promediar probabilidades de modelos con calibraciones distintas (p. ej. semillas con número de árboles muy
  distinto) puede mover el umbral óptimo; por eso el umbral se elige **después** de promediar.

---

## 9. Error fuera de bolsa (OOB) y bootstrap .632/.632+ frente a la validación cruzada

### 9.1 Síntesis de la teoría

**OOB en bagging y Random Forest.** Cada árbol se entrena con una muestra bootstrap; la probabilidad de que el hogar $i$
quede fuera de la muestra de un árbol es $(1-1/n)^n\to e^{-1}\approx0{,}368$ (con `max_samples` $=s\cdot n$, ≈ $e^{-s}$).
La predicción OOB de $i$ promedia sólo los árboles en los que $i$ no participó; como en RF los árboles son
condicionalmente independientes dados los datos, esa predicción **no usa $y_i$** y es tan honesta como una predicción
fuera de pliegue (Breiman 1996c, 2001; Wolpert y Macready 1999 la proponen como estimador eficiente del error de
bagging). Ventajas: un solo ajuste en lugar de $K$, y el ensamble evaluado se entrena con muestras del 100 % de los
datos (no del 80 %), así que es menos pesimista que la CV de 5 pliegues.

**Sesgos conocidos.** (i) *Menos árboles*: cada predicción OOB promedia ≈ 37 % de los árboles; como la log-loss y el
Brier esperados decrecen monótonamente con el número de árboles (Probst y Boulesteix 2018), el OOB es levemente
pesimista para esas medidas; con 500 árboles (≈ 185 por predicción OOB) la diferencia es despreciable, y con pocos
árboles puede haber hogares sin predicción OOB (`NaN` en `oob_decision_function_`). (ii) *Sesgo en clasificación con
muestras pequeñas*: Mitchell (2011) y Janitza y Hornung (2018) muestran que el OOB **sobrestima** el error con $n$
pequeño, clases balanceadas, muchos predictores y efectos débiles (el submuestreo desbalancea por azar las clases y los
árboles votan por la clase contraria); el sesgo es «casi despreciable para $n=1000$» y desaparece con $n$ grande (la CV
no estratificada sufre el mismo problema). Bylander (2002) encontró sesgos similares en OOB y CV-10 en conjuntos
pequeños de dos clases y propuso una corrección basada en la distribución de votos OOB. **Con 154 mil hogares y 20 % de
positivos ninguno de estos sesgos es relevante**: el OOB es aquí un buen sustituto de la CV para Random Forest.

**Boosting con submuestreo.** El *stochastic gradient boosting* (Friedman 2002) ajusta cada árbol con una submuestra
sin reemplazo, lo que reduce la varianza; pero el boosting es **secuencial**: el árbol $m$ se ajusta a los residuos de un ensamble que
ya usó al hogar $i$ en iteraciones anteriores, así que no existe un sub-ensamble independiente de $y_i$ y **no hay
predicción OOB honesta**. Lo que ofrece scikit-learn (`GradientBoostingClassifier(subsample<1)`) es `oob_improvement_`,
la mejora de la pérdida en los hogares fuera de la submuestra de la iteración $m$ respecto a la iteración anterior (y,
desde la versión 1.3, `oob_scores_` y `oob_score_`). Su documentación advierte que es «un estimador pesimista de la
pérdida de test, pero una aproximación razonable para pocos árboles», que sigue a la pérdida de test en las primeras
≈ 100 iteraciones y luego diverge, y que la CV estima mejor. Sirve como heurística para elegir el número de
iteraciones, no para evaluar. LightGBM (`bagging_fraction`, muestreo sin reemplazo), XGBoost (`subsample`) y CatBoost
no producen estimaciones OOB; su análogo es la **parada temprana con un subconjunto interno** (ya implementada) y la
CV.

**Bootstrap .632 y .632+.** Con $B$ remuestreos bootstrap y $C^{-i}$ el conjunto de remuestreos que no contienen a $i$
(Efron 1983; Efron y Tibshirani 1997; Hastie et al. 2009, §7.11, ecs. 7.56–7.61):
$$\widehat{\mathrm{Err}}^{(1)}=\frac1n\sum_i\frac1{|C^{-i}|}\sum_{b\in C^{-i}}L\bigl(y_i,\hat f^{*b}(x_i)\bigr),\qquad
\widehat{\mathrm{Err}}^{(.632)}=0{,}368\,\overline{\mathrm{err}}+0{,}632\,\widehat{\mathrm{Err}}^{(1)},$$
donde $\overline{\mathrm{err}}$ es el error aparente (en entrenamiento). El .632+ corrige el caso de sobreajuste fuerte:
con la tasa de no información $\hat\gamma$ (en binaria, $\hat\gamma=\hat p_1(1-\hat q_1)+(1-\hat p_1)\hat q_1$), la tasa
relativa de sobreajuste $\hat R=(\widehat{\mathrm{Err}}^{(1)}-\overline{\mathrm{err}})/(\hat\gamma-\overline{\mathrm{err}})$
y $\hat w=0{,}632/(1-0{,}368\hat R)$,
$\widehat{\mathrm{Err}}^{(.632+)}=(1-\hat w)\overline{\mathrm{err}}+\hat w\,\widehat{\mathrm{Err}}^{(1)}$. Se diseñó para $n$
pequeño, donde la CV tiene mucha varianza y $\widehat{\mathrm{Err}}^{(1)}$ (que entrena con ≈ 63,2 % de observaciones
distintas) es muy pesimista.

### 9.2 ¿Cuándo es el OOB un buen sustituto de la CV aquí?

| Algoritmo | OOB disponible | Uso recomendado |
|---|---|---|
| Random Forest | Sí, honesto (`oob_score=True`, `oob_decision_function_`) | **Búsqueda de hiperparámetros con OOB** (1 ajuste por configuración); umbral y F1 con OOB del objetivo; confirmación de los 3–5 mejores con la CV común |
| Boosting sklearn (`GradientBoostingClassifier`, `subsample<1`) | Sólo `oob_improvement_` / `oob_scores_` (sesgado, pesimista) | Heurística para el número de iteraciones; no para evaluar ni comparar |
| LightGBM / XGBoost / CatBoost | No | Parada temprana interna (sólo objetivo) + CV |
| CART, Logit, LPM, EN, NB | No (no remuestrean) | CV; .632+ no recomendado (§9.4) |

### 9.3 Procedimiento OOB para Random Forest (con Bogotá y F1)

1. **Datos de ajuste**: con $\lambda_B=0$, **eliminar físicamente** a Bogotá del conjunto de ajuste (en scikit-learn
   1.7, `max_samples` flotante se mide sobre `X.shape[0]` y las filas con peso cero igual ocupan lugares del bootstrap).
   Con $\lambda_B\in(0,1]$, pasar `sample_weight` ($\lambda_B$ para Bogotá, 1 para el resto); las predicciones OOB de
   Bogotá existen pero **no entran a ninguna métrica**.
2. **Ajuste**: `RandomForestClassifier(oob_score=True, bootstrap=True, n_estimators=500, ...)`; verificar que no haya
   `NaN` en `oob_decision_function_`.
3. **Métricas OOB en el objetivo** (24 dominios): log-loss y Brier (criterio de búsqueda), AUC, PR-AUC; **F1 con umbral
   anidado usando los pliegues comunes**: para cada pliegue $k$, umbral elegido con las probabilidades OOB de los
   hogares objetivo fuera de $k$ y aplicado a los de $k$ (las OOB ya son honestas; anidar sólo elimina el optimismo de
   elegir el umbral en la misma muestra, ≈ 0,0005).
4. **Búsqueda**: TPE (60 ensayos) sobre `max_features`, `min_samples_leaf`, `max_samples`, `criterion`, minimizando la
   log-loss OOB del objetivo, con 300 árboles; luego 1.000 árboles para los finalistas (Probst, Wright y Boulesteix 2019
   recomiendan exactamente ajustar con OOB y Brier).
5. **Confirmación con CV**: los 3–5 mejores se re-evalúan con la CV común (mismos pliegues que los demás algoritmos),
   que es la que entra a la comparación pareada entre algoritmos. Reportar la diferencia OOB − CV como verificación
   (se espera que el OOB sea igual o algo mejor que la CV de 5 pliegues, porque entrena con el 100 %).
6. **Umbral de envío**: puede elegirse con las OOB del modelo final ajustado con todo el objetivo (misma escala que las
   predicciones de test del mismo bosque), lo que evita el traslado de escala de §5.5.

### 9.4 ¿Conviene el .632+ para CART, Logit, LPM, EN y NB?

**No como criterio principal.** Razones: (i) con $n=154.393$ la pendiente de la curva de aprendizaje es pequeña
(§2.2: ≈ 0,002 de log-loss entre 60 % y 80 % en el modelo flexible, menos en los paramétricos), de modo que el sesgo
pesimista de la CV-5 es pequeño y común a todos los algoritmos; el .632+ corrige un sesgo que aquí no decide nada;
(ii) para LPM, Logit y NB, con $n\gg p$, el sobreajuste es mínimo ($\overline{\mathrm{err}}\approx$ CV, $\hat R\approx0$),
así que el .632+ se reduce a un promedio de dos números casi iguales; (iii) para EN, cada remuestreo exigiría repetir la
elección de $\lambda$ (si no, es optimista), con $B\ge50$ ajustes; (iv) el F1 no es una pérdida aditiva y depende del
umbral, y las fórmulas del .632+ no están validadas para F1 (habría que definir la tasa de no información del F1:
para un clasificador aleatorio con tasa predicha $q$ y prevalencia $\pi$, $TP=\pi q$ y
$\mathrm{F1}_0=2\pi q/(\pi+q)$; con $\pi=q=0{,}207$, $\mathrm{F1}_0=0{,}207$); (v) las comparaciones pareadas exigen los
mismos pliegues para todos los algoritmos. **Uso útil del bootstrap**: como diagnóstico de **inestabilidad** de CART
(frecuencia con la que cada variable aparece en las primeras particiones en 200 remuestreos; distribución del número
de hojas tras la poda) y, para Logit/LPM, como verificación de que el optimismo aparente
$\mathrm{CV}-\overline{\mathrm{err}}$ es despreciable (si es < 0,001 en log-loss, el .632+ no aporta nada).

### 9.5 Riesgos

- Comparar el OOB de RF con la CV de otros algoritmos: no es pareado ni el mismo estimando (100 % vs 80 % de los
  datos). La comparación entre algoritmos usa siempre la CV común.
- Usar `oob_improvement_` de boosting como si fuera un error de test: es pesimista y está sesgado por construcción.
- Olvidar excluir a Bogotá de las métricas OOB (las OOB se calculan para todas las filas del ajuste).

---

## 10. Protocolo recomendado (para programar en la fase C)

Notación: $\mathcal T$ = hogares de entrenamiento fuera de Bogotá (24 dominios, 154.393); $\mathcal B$ = Bogotá (10.567);
ROPE: 0,002 en F1 y 0,0005 en log-loss. Todo número que se reporte lleva su EE o IC.

**Paso 0. Entradas y verificaciones de integridad.**
`modelo_{train,test}_v7.parquet`, `conjuntos_variables_v7.json` y, si la fase B lo entrega, el mapa
variable → preguntas fuente. Afirmar con `assert`: (a) dominios del test ⊂ dominios de $\mathcal T$; (b) ningún valor
de $L_p$ de Bogotá aparece en el test; (c) ninguna variable prohibida (las de sólo-train y las derivadas de modelos de
ingreso); (d) la construcción de variables de train y test usa el mismo código (Zinkevich #29/#32).

**Paso 1. Máscaras y particiones.** `mask_T = Dominio != "BOGOTA"`. Partición $r=1$: la columna `fold` actual; $r=2,3$:
`StratifiedKFold(5, shuffle=True)` con semillas 20263 y 20264 sobre el estrato Pobre × Dominio de los 164.960 hogares
(Bogotá incluida, para cuando entre al entrenamiento). Guardar `pliegues_v7.csv` con `fold_r1`, `fold_r2`, `fold_r3`.
Las mismas particiones y semillas para todos los algoritmos.

**Paso 2. Motor de CV común** `cv(algoritmo, params, variables, lambda_B, r, semillas)`:
- para cada pliegue $k$: entrenamiento = (pliegues $\ne k$) ∩ ($\mathcal T$ ∪ $\mathcal B$ si $\lambda_B>0$), con peso
  $\lambda_B$ para $\mathcal B$ y 1 para $\mathcal T$ (si $\lambda_B=0$, eliminar $\mathcal B$ físicamente);
- preprocesamiento ajustado en el entrenamiento del pliegue (tabla §11.1); codificaciones con la etiqueta *cross-fitted*
  con un K interno (§1.8);
- parada temprana (boosting) con un 15–20 % interno **sólo de $\mathcal T$**;
- devuelve: puntajes OOF de todas las filas, puntajes de test promediados sobre los $K\times S$ modelos, número de
  iteraciones, tiempos.

**Paso 3. Métricas (sólo $\mathcal T$).** F1 **agregado anidado** (umbral del pliegue $k$ = argmax de F1 en las OOF de
$\mathcal T$ fuera de $k$); F1 por pliegue; log-loss; Brier; ROC-AUC; PR-AUC; ACE con 20 bins de igual masa; tasa
predicha; umbral. Incertidumbre: EE de F1 por función de influencia $\psi$ y IC bootstrap ($B=2.000$); F1
post-estratificado por la composición del test (robustez). Promediar semillas **antes** de calcular métricas por
(partición, pliegue).

**Paso 4. Comparaciones.** Para dos procedimientos A y B: diferencias por (partición, pliegue) $d_j$, $J=R\cdot K$; t
corregido de Nadeau–Bengio (factor $1/J+1/4$, $J-1$ g.l.); prueba bayesiana correlacionada con ROPE ($P$(izq),
$P$(ROPE), $P$(der)); bootstrap pareado de las OOF (incertidumbre de evaluación; «¿ganaría en el privado?»). Para
siete algoritmos contra el mejor: Holm. Sesgo de selección de cada búsqueda: corrección de Tibshirani–Tibshirani con
los registros de ensayos.

**Paso 5. Bogotá** (detalle y justificación en §1):
1. Diagnósticos (una vez): clasificador de dominio Bogotá vs otras cabeceras (AUC y tamaño efectivo de muestra de las
   razones de densidad); prueba de transferencia (entrenar en $\mathcal T$, predecir $\mathcal B$: AUC y calibración en
   el agregado frente a la CV dentro de Bogotá).
2. Para **cada algoritmo**, con su configuración ajustada: $\lambda_B\in\{0,1\}$ (y 0,5 en RF y boosting) evaluado con
   $R\times K\times S=3\times5\times3$; criterio = log-loss media en $\mathcal T$; regla (§1.9): para «el mejor de cada
   algoritmo», $\hat\lambda_B=\arg\min$ (regla de Bayes); reportar la diferencia con IC corregido y $P$(ROPE) de la
   prueba bayesiana (si $P$(ROPE) > 0,95, declarar «equivalentes en la práctica»); para la comparación común entre
   algoritmos, **$\lambda_B=1$** para todos; valor por defecto si no se corre el contraste: $\lambda_B=1$.
3. Prueba de estrés (sólo para los 2 finalistas): placebo «dominio extra» y deja-un-dominio-fuera (§1.7).
4. Umbral, prevalencia y todas las métricas: siempre sólo $\mathcal T$.

**Paso 6. Selección de variables por familia** (tabla §11.1), siempre dentro del pliegue externo: ranking → RFE o hacia
adelante por bloques de pregunta → curva de parsimonia (F1 anidado y log-loss frente a número de variables y de
preguntas $c(S)$, con IC pareados respecto al mejor) → conjunto parsimonioso = menor $c(S)$ dentro de la ROPE del mejor
(y conjunto de confianza CVC al 90 % como reporte) → estabilidad (Jaccard entre pliegues × semillas, frecuencia por
variable) → LOCO por bloque teórico sobre el modelo final. Dos conjuntos comunes para la comparación «en igualdad de
condiciones»: (a) todas las variables v7; (b) la encuesta corta. Más el «mejor de cada algoritmo» con su propia
selección.

**Paso 7. Ajuste de hiperparámetros** (§4 y tabla §11.2): Optuna con muestreador TPE, 10–64 puntos iniciales
cuasi-aleatorios (QMC/Sobol), `MedianPruner` por pliegues; objetivo = log-loss OOF de $\mathcal T$ en la partición 1
con 1 semilla; registrar todos los ensayos (parámetros, log-loss y F1 por pliegue); confirmar los 5 mejores con
$3\times5\times3$; elegir por log-loss media, desempate por F1 anidado; gráficos de eje e importancia fANOVA; ampliar el
espacio si el óptimo toca un borde. Hiperparámetros científicos ($\lambda_B$, monotonía, pesos de clase, conjunto de
variables) comparados con los de molestia re-optimizados (o verificados con *isolation plot*).

**Paso 8. Random Forest con OOB** (§9.3): búsqueda con log-loss OOB de $\mathcal T$; F1 OOB con umbral anidado por los
pliegues comunes; confirmación con la CV común para la comparación.

**Paso 9. Restricciones monótonas** (boosting y RF; §4.4): lista teórica de 6–10 variables; adoptar por no inferioridad
(ΔF1 ≥ −0,002 y Δlog-loss ≤ +0,0005, t corregido).

**Paso 10. Desbalance** (§6): especificación principal sin pesos; una corrida de sensibilidad con pesos balanceados por
algoritmo; reportar ΔF1, ΔBrier y ΔACE pareados.

**Paso 11. Umbral y predictor final** (§5.5, §8.3): OOF promediadas por semilla → umbral $\hat t$ = mediana bootstrap
($B=500$) del argmax de F1 en $\mathcal T$, y tasa $\hat q$ de positivos que implica. Predictor final = **reajuste con el
100 %** de los datos de entrenamiento (con $\lambda_B$) y, en boosting, 10 semillas con árboles = mediana CV × 1,2 a
tasa 0,02. Regla de umbral en test: boosting → tasa igualada $\hat q$; RF → umbral en las OOB del bosque final;
deterministas → $\hat t$ fijo con verificación de tasa. Contraste obligatorio: promedio de los $K\times S$ modelos de
pliegue con $\hat t$ fijo, y proporción de hogares de test clasificados distinto. Diagnósticos: $\hat t$ vs $\hat F/2$;
tasa predicha en test vs $\hat q$ ($\pm0{,}003$); media de probabilidades en test vs 0,2075 ($\pm0{,}003$).

**Paso 12. Registro y salidas.** Usar `registro.py` (mismo protocolo para todos): OOF, test, metadatos con
hiperparámetros, $\lambda_B$, conjunto de variables, $c(S)$, semillas, tiempos, EE e IC; `protocolo_v7.json` con todas
las decisiones y sus pruebas; CSV con el nombre `ALG_hiperparámetros_n<vars>_lb<λ>_s<S>.csv` (sin subir).
Reproducibilidad: semillas explícitas en cada ajuste; versiones de Python y paquetes registradas en el JSON (en este
equipo: Python 3.13.5, scikit-learn 1.7.2, LightGBM 4.7.0); cachés reanudables por clave (algoritmo, configuración,
conjunto de variables, $\lambda_B$, partición, semilla), de modo que una corrida interrumpida continúe sin repetir
ajustes.

**Paso 13. Política de envíos (para el coordinador).** Elegir por CV, nunca por el público; usar el público sólo para
verificar que el F1 cae en $\hat F_{CV}\pm0{,}012$ (2 EE). Sugerencia de los ≥ 15 envíos: el mejor de cada uno de los 7
algoritmos (7); la versión parsimoniosa/encuesta corta de los 3 mejores (3); sensibilidades informativas del mejor
algoritmo: $\lambda_B=0$ vs 1, monótono vs libre, pesos de clase (3); y 2 de reserva. Envío final marcado: el de mayor F1
anidado CV (o, dentro de la ROPE, el más parsimonioso o monótono).

**Paso 14. Pruebas del código** (antes de la fase D): datos sintéticos (`tools/sintetico.py`) con una señal conocida y
un «dominio extra» con distinta prevalencia: verificar que (a) ninguna métrica usa filas de $\mathcal B$; (b) el umbral
anidado se calcula sin el pliegue evaluado; (c) con etiquetas permutadas el F1 anidado cae al F1 del mejor clasificador no
informativo: como $2\pi q/(\pi+q)$ crece con la tasa predicha $q$, el óptimo es predecir todo positivo, con
$\mathrm{F1}=2\pi/(1+\pi)\approx0{,}34$ (Lipton et al. 2014); (d) los resultados son reproducibles con las mismas semillas.

---

## 11. Tabla por algoritmo

### 11.1 Selección de variables, preprocesamiento y evaluación

Todo lo que aprende de los datos (imputación, escalas, discretización, codificaciones con la etiqueta, ranking de
variables, calibración, umbral) se ajusta **dentro** del entrenamiento de cada pliegue. «Objetivo» = 24 dominios sin
Bogotá.

| Algoritmo | Selección de variables | Preprocesamiento | Evaluación | Peso de Bogotá $\lambda_B$ | Calibración / umbral |
|---|---|---|---|---|---|
| **LPM** (MCO sobre 0/1) | Camino lasso-MCO para ordenar → soporte en $\lambda_{1se}$ → MCO sin penalizar (post-lasso); curva de parsimonia por preguntas; stability selection CPSS para reportar | Mediana + indicador de faltante; winsorizar 0,5–99,5 %; logs ya construidos; dummies (Dominio y categóricas, una base); estandarizar; *splines* naturales (4 g.l.) opcionales en 3–5 continuas clave | CV común $R\times K$; sin semillas; .632+ no | $\{0,1\}$ por CV; empate → 1 | Sin calibrar (puntaje lineal); umbral sobre el puntaje en las OOF del objetivo |
| **Logit** | Camino L1-logit → soporte $\lambda_{1se}$ → logit MV sin penalizar (relaxed lasso $\gamma=0$); alternativa group lasso por pregunta; BIC como referencia | Igual que LPM | CV común; .632+ no (optimismo aparente ≈ 0: verificar que log-loss en entrenamiento − CV < 0,001) | $\{0,1\}$; empate → 1 | Calibrado por construcción (Zinkevich #14); umbral F1 de OOF ≈ $\hat F/2$ |
| **Elastic Net** (logística) | Integrada; $\lambda_{1se}$ para la versión parsimoniosa; CPSS ($B=50$ pares, $\tau=0{,}75$); group lasso para dummies | Igual que LPM (estandarizar es obligatorio) | CV común (la elección de $\lambda$ anidada dentro del pliegue); .632+ no (exigiría re-elegir $\lambda$ en cada remuestreo) | $\{0,1\}$; empate → 1 | Encogimiento sesga las probabilidades hacia la media: umbral empírico (no $F/2$) |
| **CART** | Integrada: particiones + poda costo-complejidad con CV y regla 1-EE; versión «política» con profundidad ≤ 4–5 | Sin escalar; faltantes con indicador o valor centinela; categóricas como códigos ordenados por tasa de pobreza (dentro del pliegue) | CV común; bootstrap ($B=200$) sólo como diagnóstico de inestabilidad | $\{0,1\}$; empate → 1 | Puntajes = proporciones de hoja (pocos valores; curva F1 escalonada); umbral entre valores observados |
| **Random Forest** | RFE por bloques (20 % por paso) con permutación **agrupada** sobre OOB; confirmación con LOCO por bloques; nunca Gini | Sin escalar; faltantes con centinela + indicador; categóricas de alta cardinalidad con codificación con la etiqueta encogida y *cross-fitted* | **OOB para la búsqueda**; CV común para confirmar y comparar | $\{0,0{,}5,1\}$; empate → 1 | Isotónica dentro del pliegue si mejora la log-loss; umbral en OOB del objetivo |
| **Boosting** (LightGBM principal; XGBoost y CatBoost como variantes) | SHAP **agrupado** por pregunta dentro del pliegue → RFE por bloques re-rankeando → LOCO final; curva de parsimonia $k\in\{5,10,15,20,30,40,60,80,\text{todas}\}$ | Sin escalar; faltantes nativos; categóricas nativas (LightGBM `cat_smooth`, CatBoost estadísticas ordenadas); restricciones monótonas candidatas (§4.4) | CV común con parada temprana interna (15–20 %, sólo objetivo). **Sin OOB**: LightGBM, XGBoost y CatBoost no lo producen; `oob_improvement_` de `GradientBoostingClassifier` (sklearn, `subsample<1`) es pesimista y sólo orienta el número de iteraciones | $\{0,0{,}5,1\}$; empate → 1 | Calibrado en general (log-loss); umbral en OOF promediadas por semilla, trasladado por tasa igualada |
| **Naive Bayes** | Filtro mRMR para ordenar + envoltorio hacia adelante (Bayes selectivo) con CV de log-loss; $k\le 15$–25 | **Discretización** dentro del pliegue (igual frecuencia; número de intervalos como hiperparámetro, hasta ≈ $\sqrt n$ según Yang y Webb 2009; MDL como alternativa) + `CategoricalNB`; alternativa `GaussianNB` sobre variables transformadas a normal por cuantiles | CV común; .632+ no | $\{0,1\}$; empate → 1 | Probabilidades extremas: isotónica dentro del pliegue para la log-loss; el F1 sólo depende del orden |

### 11.2 Hiperparámetros, búsqueda y presupuesto

| Algoritmo | Espacio de búsqueda (escala) | Estrategia | Presupuesto | Modelo final |
|---|---|---|---|---|
| **LPM** | Ninguno; opciones de especificación: tamaño del soporte (camino $\lambda$), *splines* sí/no, interacciones teóricas sí/no | Camino completo de lasso + comparación pareada de 2–4 especificaciones | Minutos | Reajuste con todo el objetivo (+ Bogotá si $\lambda_B=1$) |
| **Logit** | Ninguno (o L2 muy débil, $C=10^4$, por estabilidad numérica); mismas opciones de especificación | Igual | Minutos | Igual |
| **Elastic Net** | $\alpha$ (`l1_ratio`) $\in\{0;0{,}1;0{,}25;0{,}5;0{,}75;0{,}9;1\}$; $\lambda$: 50–100 valores log desde $\lambda_{max}$ hasta $10^{-4}\lambda_{max}$ (defaults óptimos de Probst et al.: $\alpha=0{,}4$; el más ajustable es $\lambda$) | Rejilla completa con *warm starts* (glmnet/saga) | 7 caminos × 5 pliegues | $\lambda_{min}$ (versión de pronóstico) y $\lambda_{1se}$ (parsimoniosa) |
| **CART** | `max_depth` 2–30; `min_samples_leaf` 5–2.000 (log); `min_samples_split` 10–4.000 (log); `criterion` {gini, entropy}; `ccp_alpha` del camino de poda del pliegue (defaults óptimos rpart: cp = 0, maxdepth = 21, minbucket = 12; aquí, con 120 mil hogares por pliegue, hojas mayores) | Rejilla (≈ 100 combinaciones) + camino de poda; regla 1-EE | ≈ 500 ajustes baratos | Árbol podado reajustado con el 100 % (y versión de profundidad ≤ 4 para política); umbral fijo con verificación de tasa |
| **Random Forest** | `max_features` fracción 0,03–0,7 (log; default óptimo ≈ 0,26·p; el más ajustable); `min_samples_leaf` 1–200 (log); `max_samples` 0,2–1,0 (default óptimo 0,7); `criterion` {gini, log_loss}; `n_estimators` **fijo** (300 en búsqueda, 1.000 final); `class_weight` None | TPE (Optuna) sobre log-loss **OOB** del objetivo, 10 iniciales cuasi-aleatorios | 60 ensayos + CV de los 5 mejores ($3\times5\times1$) | Reajuste con el 100 %, 1.000 árboles; umbral en las OOB del bosque final |
| **LightGBM** (Ke et al. 2017) | `learning_rate` 0,05 fija en búsqueda; `num_leaves` 8–256 (log); `min_child_samples` 10–1.000 (log); `feature_fraction` 0,3–1; `bagging_fraction` 0,5–1 con `bagging_freq` = 1; `lambda_l2` $10^{-3}$–100 (log); `lambda_l1` $10^{-3}$–10 (log); `min_gain_to_split` 0–2; `max_bin` {63, 127, 255}; `cat_smooth` 1–100 (log); `max_cat_to_onehot` {4, 8, 32}; árboles por parada temprana (paciencia 200, máx. 10.000). Científicos: $\lambda_B$, monotonía, conjunto de variables | 64 cuasi-aleatorios (Sobol) + 100–150 TPE con `MedianPruner` por pliegue | ≈ 200 ensayos × 5 pliegues + confirmación $3\times5\times3$ de los 5 mejores | Reajuste con el 100 %: `learning_rate` 0,02, árboles = mediana CV × 1,2, promedio de 10 semillas; umbral por tasa igualada (contraste: promedio de 5 pliegues × 10 semillas) |
| **XGBoost** (Chen y Guestrin 2016) | `eta` 0,05 en búsqueda; `max_depth` 3–12; `min_child_weight` 1–128 (log); `subsample` 0,5–1; `colsample_bytree` 0,3–1; `colsample_bylevel` 0,3–1; `lambda` $10^{-3}$–100 (log); `alpha` $10^{-3}$–10 (log); `gamma` 0–5; `tree_method` hist; `max_bin` 256 (defaults óptimos de Probst et al.: eta 0,018 con 4.168 rondas, max_depth 13, subsample 0,84, colsample_bytree 0,75) | TPE con poda | 60–80 ensayos | Igual que LightGBM |
| **CatBoost** (Prokhorenkova et al. 2018) | `depth` 4–10; `l2_leaf_reg` 1–30 (log); `random_strength` $10^{-2}$–10 (log); `bagging_temperature` 0–1; `border_count` {128, 254}; `one_hot_max_size` {2, 10}; `learning_rate` 0,05; parada temprana (`od_wait` 200) | TPE con poda | 40–60 ensayos | Igual |
| **Naive Bayes** | Número de variables $k$ (principal, por el envoltorio); intervalos {5, 10, 20, 50, 100, $\sqrt n$} y estrategia {igual frecuencia, MDL}; `alpha` (Laplace) 0,01–10 (log); `GaussianNB.var_smoothing` $10^{-12}$–$10^{-2}$ (log) | Envoltorio hacia adelante + rejilla pequeña | ≈ 100 ajustes baratos | Reajuste con todo |

**Reglas comunes**: objetivo de la búsqueda = log-loss OOF del objetivo; los 5 mejores se confirman con $R\times K\times S$;
elección final por log-loss media con desempate por F1 anidado; reportar presupuesto (ensayos y tiempo) de cada
algoritmo; verificar que el óptimo no esté en un borde del espacio.

---

## 12. Referencias

Todas las referencias están en `docs/metodologia/referencias_metodologia.bib` con su clave (entre corchetes abajo).
Cada entrada se verificó el 19 de octubre de 2026 contra una fuente primaria legítima (Crossref, página del editor,
JMLR/PMLR/NeurIPS/ACL/AAAI/IJCAI, arXiv, documentación oficial o sitios de Google Research); la URL usada está en el
campo `note` de la entrada. Ninguna quedó marcada «NO VERIFICADO». Notas sobre la verificación:

- «Nan et al. (2012)», como suele citarse, es **Ye, Chai, Lee y Chieu (2012)**: el primer autor se llama Nan Ye.
- Bates, Hastie y Tibshirani salió en línea en 2023 y en el volumen 119 de JASA en 2024; Lei (2020) salió en línea en
  2019.
- Hanna y Olken (2018) ocupa las páginas 201–226 del JEP 32(4) (no 157–180).
- Kohavi (1995): páginas 1137–1143 del vol. 2 de IJCAI-95.
- Del artículo de Shimodaira (2000) se verificaron los metadatos; su contenido se resume a partir de Sugiyama,
  Krauledat y Müller (2007, §2.2), que lo citan. De Nadeau y Bengio (2003), Bylander (2002) y Efron y Tibshirani (1997)
  las fórmulas se cotejaron en la versión NIPS 1999 de los mismos autores, en el resumen del editor y en Hastie et al.
  (2009, §7.11), respectivamente.
- Las páginas web (Rules of ML, Machine Learning Crash Course, documentación de LightGBM, XGBoost y scikit-learn) se
  citan con la fecha de su última actualización.

- Aiken, E., Bellue, S., Karlan, D., Udry, C. y Blumenstock, J. E. (2022). Machine Learning and Phone Data Can Improve Targeting of Humanitarian Aid. *Nature 603(7903): 864–870*. doi:10.1038/s41586-022-04484-9. `[aiken2022]`
- Akiba, T., Sano, S., Yanase, T., Ohta, T. y Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '19): 2623–2631*. doi:10.1145/3292500.3330701. arXiv:1907.10902. `[akiba2019]`
- Alatas, V., Banerjee, A., Hanna, R., Olken, B. A. y Tobias, J. (2012). Targeting the Poor: Evidence from a Field Experiment in Indonesia. *American Economic Review 102(4): 1206–1240*. doi:10.1257/aer.102.4.1206. `[alatas2012]`
- Alexandari, A., Kundaje, A. y Shrikumar, A. (2020). Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation. *Proceedings of the 37th International Conference on Machine Learning (ICML 2020) 119: 222–232*. arXiv:1901.06852. `[alexandari2020]`
- Ambroise, C. y McLachlan, G. J. (2002). Selection Bias in Gene Extraction on the Basis of Microarray Gene-Expression Data. *Proceedings of the National Academy of Sciences of the United States of America 99(10): 6562–6566*. doi:10.1073/pnas.102102699. `[ambroise2002]`
- Arik, S. Ö. y Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. *Proceedings of the AAAI Conference on Artificial Intelligence 35(8): 6679–6687*. doi:10.1609/aaai.v35i8.16826. arXiv:1908.07442. `[arik2021tabnet]`
- Arlot, S. y Celisse, A. (2010). A Survey of Cross-Validation Procedures for Model Selection. *Statistics Surveys 4: 40–79*. doi:10.1214/09-SS054. `[arlot2010]`
- Barber, R. F. y Candès, E. J. (2015). Controlling the False Discovery Rate via Knockoffs. *The Annals of Statistics 43(5): 2055–2085*. doi:10.1214/15-AOS1337. arXiv:1404.5609. `[barber2015]`
- Bates, S., Hastie, T. y Tibshirani, R. (2024). Cross-Validation: What Does It Estimate and How Well Does It Do It? *Journal of the American Statistical Association 119(546): 1434–1445*. doi:10.1080/01621459.2023.2197686. arXiv:2104.00673. `[bates2024cv]`
- Belloni, A. y Chernozhukov, V. (2013). Least Squares after Model Selection in High-Dimensional Sparse Models. *Bernoulli 19(2): 521–547*. doi:10.3150/11-BEJ410. arXiv:1001.0188. `[belloni2013]`
- Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F. y Vaughan, J. W. (2010). A theory of learning from different domains. *Machine Learning 79(1–2): 151–175*. doi:10.1007/s10994-009-5152-4. `[bendavid2010theory]`
- Ben-David, S., Lu, T., Luu, T. y Pál, D. (2010). Impossibility Theorems for Domain Adaptation. *Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS 2010) 9: 129–136*. <https://proceedings.mlr.press/v9/david10a.html> `[bendavid2010impossibility]`
- Benavoli, A., Corani, G., Demšar, J. y Zaffalon, M. (2017). Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis. *Journal of Machine Learning Research 18(77): 1–36*. <https://jmlr.org/papers/v18/16-305.html> `[benavoli2017]`
- Bengio, Y. y Grandvalet, Y. (2004). No Unbiased Estimator of the Variance of K-Fold Cross-Validation. *Journal of Machine Learning Research 5: 1089–1105*. <https://jmlr.org/papers/v5/grandvalet04a.html> `[bengio2004novariance]`
- Bergstra, J., Bardenet, R., Bengio, Y. y Kégl, B. (2011). Algorithms for Hyper-Parameter Optimization. *Advances in Neural Information Processing Systems 24 (NIPS 2011) 24*. <https://proceedings.neurips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html> `[bergstra2011]`
- Bergstra, J. y Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. *Journal of Machine Learning Research 13(10): 281–305*. <https://www.jmlr.org/papers/v13/bergstra12a.html> `[bergstra2012]`
- Bickel, S., Brückner, M. y Scheffer, T. (2009). Discriminative Learning Under Covariate Shift. *Journal of Machine Learning Research 10(75): 2137–2155*. <https://jmlr.org/papers/v10/bickel09a.html> `[bickel2009]`
- Bischl, B., Binder, M., Lang, M., Pielok, T., Richter, J., Coors, S. et al. (2023). Hyperparameter Optimization: Foundations, Algorithms, Best Practices, and Open Challenges. *WIREs Data Mining and Knowledge Discovery 13(2): e1484*. doi:10.1002/widm.1484. arXiv:2107.05847. `[bischl2023hpo]`
- Blitzer, J., Crammer, K., Kulesza, A., Pereira, F. y Wortman, J. (2008). Learning Bounds for Domain Adaptation. *Advances in Neural Information Processing Systems 20 (NIPS 2007) 20*. <https://proceedings.neurips.cc/paper_files/paper/2007/hash/42e77b63637ab381e8be5f8318cc28a2-Abstract.html> `[blitzer2008bounds]`
- Blum, A. y Hardt, M. (2015). The Ladder: A Reliable Leaderboard for Machine Learning Competitions. *Proceedings of the 32nd International Conference on Machine Learning 37: 1006–1014*. arXiv:1502.04585. `[blum2015ladder]`
- Bouckaert, R. R. y Frank, E. (2004). Evaluating the Replicability of Significance Tests for Comparing Learning Algorithms. *Advances in Knowledge Discovery and Data Mining: 8th Pacific-Asia Conference, PAKDD 2004, Sydney, Australia, May 26–28, 2004, Proceedings 3056: 3–12*. doi:10.1007/978-3-540-24775-3_3. `[bouckaert2004]`
- Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J. et al. (2021). Accounting for Variance in Machine Learning Benchmarks. *Proceedings of Machine Learning and Systems (MLSys 2021) 3: 747–769*. arXiv:2103.03098. `[bouthillier2021]`
- Breiman, L., Friedman, J. H., Olshen, R. A. y Stone, C. J. (1984). Classification and Regression Trees. *Wadsworth International Group*. `[breiman1984cart]`
- Breiman, L. (1996c). Out-of-Bag Estimation. *Statistics Department, University of California, Berkeley*. <https://www.stat.berkeley.edu/~breiman/OOBestimation.pdf> `[breiman1996oob]`
- Breiman, L. (1996a). Bagging Predictors. *Machine Learning 24(2): 123–140*. doi:10.1007/BF00058655. `[breiman1996bagging]`
- Breiman, L. (1996b). Heuristics of Instability and Stabilization in Model Selection. *The Annals of Statistics 24(6): 2350–2383*. doi:10.1214/aos/1032181158. `[breiman1996instability]`
- Breiman, L. (2001). Random Forests. *Machine Learning 45(1): 5–32*. doi:10.1023/A:1010933404324. `[breiman2001rf]`
- Brier, G. W. (1950). Verification of Forecasts Expressed in Terms of Probability. *Monthly Weather Review 78(1): 1–3*. doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2. `[brier1950]`
- Brown, C., Ravallion, M. y van de Walle, D. (2018). A Poor Means Test? Econometric Targeting in Africa. *Journal of Development Economics 134: 109–124*. doi:10.1016/j.jdeveco.2018.05.004. `[brown2018]`
- Bühlmann, P. y Yu, B. (2002). Analyzing Bagging. *The Annals of Statistics 30(4): 927–961*. doi:10.1214/aos/1031689014. `[buhlmann2002]`
- Bylander, T. (2002). Estimating Generalization Error on Two-Class Datasets Using Out-of-Bag Estimates. *Machine Learning 48(1-3): 287–297*. doi:10.1023/A:1013964023376. `[bylander2002]`
- Candès, E., Fan, Y., Janson, L. y Lv, J. (2018). Panning for Gold: ‘Model-X' Knockoffs for High Dimensional Controlled Variable Selection. *Journal of the Royal Statistical Society: Series B (Statistical Methodology) 80(3): 551–577*. doi:10.1111/rssb.12265. arXiv:1610.02351. `[candes2018]`
- Carriero, A., Luijken, K., de Hond, A., Moons, K. G. M., van Calster, B. y van Smeden, M. (2025). The Harms of Class Imbalance Corrections for Machine Learning Based Prediction Models: A Simulation Study. *Statistics in Medicine 44(3–4): e10320*. doi:10.1002/sim.10320. arXiv:2404.19494. `[carriero2025]`
- Caruana, R., Niculescu-Mizil, A., Crew, G. y Ksikes, A. (2004). Ensemble Selection from Libraries of Models. *Proceedings of the Twenty-First International Conference on Machine Learning (ICML '04): 18*. doi:10.1145/1015330.1015432. `[caruana2004]`
- Cawley, G. C. y Talbot, N. L. C. (2010). On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. *Journal of Machine Learning Research 11(70): 2079–2107*. <https://jmlr.org/papers/v11/cawley10a.html> `[cawley2010]`
- Charoenphakdee, N., Vongkulbhisal, J., Chairatanakul, N. y Sugiyama, M. (2021). On Focal Loss for Class-Posterior Probability Estimation: A Theoretical Perspective. *2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR): 5198–5207*. doi:10.1109/CVPR46437.2021.00516. arXiv:2011.09172. `[charoenphakdee2021]`
- Chawla, N. V., Bowyer, K. W., Hall, L. O. y Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research 16: 321–357*. doi:10.1613/jair.953. `[chawla2002]`
- Chen, T. y Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16): 785–794*. doi:10.1145/2939672.2939785. arXiv:1603.02754. `[chen2016]`
- Chipman, H. A., George, E. I., McCulloch, R. E. y Shively, T. S. (2022). mBART: Multidimensional Monotone BART. *Bayesian Analysis 17(2): 515–544*. doi:10.1214/21-BA1259. arXiv:1612.01619. `[chipman2022monotone]`
- Coady, D., Grosh, M. y Hoddinott, J. (2004). Targeting of Transfers in Developing Countries: Review of Lessons and Experience. *The World Bank*. doi:10.1596/0-8213-5769-7. `[coady2004]`
- Corani, G. y Benavoli, A. (2015). A Bayesian Approach for Comparing Cross-Validated Algorithms on Multiple Data Sets. *Machine Learning 100(2-3): 285–304*. doi:10.1007/s10994-015-5486-z. `[corani2015correlated]`
- Cortes, C., Mohri, M., Riley, M. y Rostamizadeh, A. (2008). Sample Selection Bias Correction Theory. *Algorithmic Learning Theory: 19th International Conference, ALT 2008 5254: 38–53*. doi:10.1007/978-3-540-87987-9_8. arXiv:0805.2775. `[cortes2008ssb]`
- Cortes, C., Mansour, Y. y Mohri, M. (2010). Learning Bounds for Importance Weighting. *Advances in Neural Information Processing Systems 23*. <https://proceedings.neurips.cc/paper_files/paper/2010/hash/59c33016884a62116be975a9bb8257e3-Abstract.html> `[cortes2010iw]`
- Covert, I., Lundberg, S. M. y Lee, S. I. (2020). Understanding Global Feature Contributions With Additive Importance Measures. *Advances in Neural Information Processing Systems 33 (NeurIPS 2020) 33: 17212–17223*. arXiv:2004.00668. `[covert2020sage]`
- D'Amour, A., Heller, K., Moldovan, D., Adlam, B., Alipanahi, B., Beutel, A. et al. (2022). Underspecification Presents Challenges for Credibility in Modern Machine Learning. *Journal of Machine Learning Research 23(226): 1–61*. arXiv:2011.03395. `[damour2022]`
- Daumé III, H. (2007). Frustratingly Easy Domain Adaptation. *Proceedings of the 45th Annual Meeting of the Association of Computational Linguistics: 256–263*. arXiv:0907.1815. `[daume2007]`
- Debeer, D. y Strobl, C. (2020). Conditional Permutation Importance Revisited. *BMC Bioinformatics 21: 307*. doi:10.1186/s12859-020-03622-2. `[debeer2020]`
- Dembczyński, K., Waegeman, W., Cheng, W. y Hüllermeier, E. (2011). An Exact Algorithm for F-Measure Maximization. *Advances in Neural Information Processing Systems 24: 1404–1412*. <https://proceedings.neurips.cc/paper_files/paper/2011/hash/71ad16ad2c4d81f348082ff6c4b20768-Abstract.html> `[dembczynski2011]`
- Dembczyński, K., Kotłowski, W., Koyejo, O. y Natarajan, N. (2017). Consistency Analysis for Binary Classification Revisited. *Proceedings of the 34th International Conference on Machine Learning 70: 961–969*. <https://proceedings.mlr.press/v70/dembczynski17a.html> `[dembczynski2017]`
- Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. *Journal of Machine Learning Research 7(1): 1–30*. <https://jmlr.org/papers/v7/demsar06a.html> `[demsar2006]`
- Díaz-Uriarte, R. y Alvarez de Andrés, S. (2006). Gene Selection and Classification of Microarray Data Using Random Forest. *BMC Bioinformatics 7: 3*. doi:10.1186/1471-2105-7-3. `[diaz2006]`
- Dietterich, T. G. (1998). Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Computation 10(7): 1895–1923*. doi:10.1162/089976698300017197. `[dietterich1998]`
- Domingos, P. y Pazzani, M. (1997). On the Optimality of the Simple Bayesian Classifier under Zero-One Loss. *Machine Learning 29(2–3): 103–130*. doi:10.1023/A:1007413511361. `[domingos1997]`
- Dougherty, J., Kohavi, R. y Sahami, M. (1995). Supervised and Unsupervised Discretization of Continuous Features. *Machine Learning: Proceedings of the Twelfth International Conference (ICML 1995): 194–202*. doi:10.1016/B978-1-55860-377-6.50032-3. `[dougherty1995]`
- Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, O. y Roth, A. (2015). The Reusable Holdout: Preserving Validity in Adaptive Data Analysis. *Science 349(6248): 636–638*. doi:10.1126/science.aaa9375. `[dwork2015]`
- Efron, B. y Morris, C. (1975). Data Analysis Using Stein's Estimator and its Generalizations. *Journal of the American Statistical Association 70(350): 311–319*. doi:10.1080/01621459.1975.10479864. `[efron1975]`
- Efron, B. (1983). Estimating the Error Rate of a Prediction Rule: Improvement on Cross-Validation. *Journal of the American Statistical Association 78(382): 316–331*. doi:10.1080/01621459.1983.10477973. `[efron1983]`
- Efron, B. y Tibshirani, R. (1997). Improvements on Cross-Validation: The .632+ Bootstrap Method. *Journal of the American Statistical Association 92(438): 548–560*. doi:10.1080/01621459.1997.10474007. `[efron1997]`
- Elbers, C., Lanjouw, J. O. y Lanjouw, P. (2003). Micro-Level Estimation of Poverty and Inequality. *Econometrica 71(1): 355–364*. doi:10.1111/1468-0262.00399. `[elbers2003]`
- Elkan, C. (2001). The Foundations of Cost-Sensitive Learning. *Proceedings of the Seventeenth International Joint Conference on Artificial Intelligence (IJCAI-01) 2: 973–978*. <https://www.ijcai.org/Proceedings/01/IJCAI-2001-k.pdf> `[elkan2001]`
- Elor, Y. y Averbuch-Elor, H. (2022). To SMOTE, or not to SMOTE? arXiv:2201.08528. `[elor2022]`
- Falkner, S., Klein, A. y Hutter, F. (2018). BOHB: Robust and Efficient Hyperparameter Optimization at Scale. *Proceedings of the 35th International Conference on Machine Learning (ICML) 80: 1437–1446*. arXiv:1807.01774. `[falkner2018bohb]`
- Fay, R. E. y Herriot, R. A. (1979). Estimates of Income for Small Places: An Application of James–Stein Procedures to Census Data. *Journal of the American Statistical Association 74(366): 269–277*. doi:10.1080/01621459.1979.10482505. `[fay1979]`
- Fayyad, U. M. y Irani, K. B. (1993). Multi-Interval Discretization of Continuous-Valued Attributes for Classification Learning. *Proceedings of the Thirteenth International Joint Conference on Artificial Intelligence (IJCAI-93) 2: 1022–1027*. <https://www.ijcai.org/Proceedings/93-2/Papers/022.pdf> `[fayyad1993]`
- Fisher, A., Rudin, C. y Dominici, F. (2019). All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously. *Journal of Machine Learning Research 20(177): 1–81*. arXiv:1801.01489. `[fisher2019]`
- Forman, G. y Scholz, M. (2010). Apples-to-Apples in Cross-Validation Studies: Pitfalls in Classifier Performance Measurement. *ACM SIGKDD Explorations Newsletter 12(1): 49–57*. doi:10.1145/1882471.1882479. `[forman2010]`
- Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *The Annals of Statistics 29(5): 1189–1232*. doi:10.1214/aos/1013203451. `[friedman2001]`
- Friedman, J. H. (2002). Stochastic Gradient Boosting. *Computational Statistics & Data Analysis 38(4): 367–378*. doi:10.1016/S0167-9473(01)00065-2. `[friedman2002]`
- Friedman, J. H., Hastie, T. y Tibshirani, R. (2010). Regularization Paths for Generalized Linear Models via Coordinate Descent. *Journal of Statistical Software 33(1): 1–22*. doi:10.18637/jss.v033.i01. `[friedman2010glmnet]`
- Gneiting, T. y Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *Journal of the American Statistical Association 102(477): 359–378*. doi:10.1198/016214506000001437. `[gneiting2007proper]`
- Gneiting, T., Balabdaoui, F. y Raftery, A. E. (2007). Probabilistic Forecasts, Calibration and Sharpness. *Journal of the Royal Statistical Society: Series B (Statistical Methodology) 69(2): 243–268*. doi:10.1111/j.1467-9868.2007.00587.x. `[gneiting2007calibration]`
- Godbole, V., Dahl, G. E., Gilmer, J., Shallue, C. J. y Nado, Z. (2023). Deep Learning Tuning Playbook. *Repositorio de GitHub google-research/tuning_playbook*. <https://github.com/google-research/tuning_playbook> `[godbole2023]`
- Golovin, D., Solnik, B., Moitra, S., Kochanski, G., Karro, J. y Sculley, D. (2017). Google Vizier: A Service for Black-Box Optimization. *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '17): 1487–1495*. doi:10.1145/3097983.3098043. `[golovin2017]`
- Google for Developers (2025). Datasets: Class-imbalanced datasets. *Machine Learning Crash Course (módulo Datasets, generalization, and overfitting)*. <https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets> `[google_mlcc_imbalanced]`
- Google for Developers (2026). Thresholds and the confusion matrix. *Machine Learning Crash Course (módulo Classification)*. <https://developers.google.com/machine-learning/crash-course/classification/thresholding> `[google_mlcc_thresholds]`
- Grinsztajn, L., Oyallon, E. y Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *Advances in Neural Information Processing Systems 35 (NeurIPS 2022), Datasets and Benchmarks Track 35: 507–520*. doi:10.52202/068431-0037. arXiv:2207.08815. `[grinsztajn2022]`
- Grosh, M. E. y Baker, J. L. (1995). Proxy Means Tests for Targeting Social Programs: Simulations and Speculation. *The World Bank*. <https://documents.worldbank.org/en/publication/documents-reports/documentdetail/750401468776352539> `[grosh1995]`
- Gulrajani, I. y Lopez-Paz, D. (2021). In Search of Lost Domain Generalization. *International Conference on Learning Representations (ICLR 2021)*. arXiv:2007.01434. `[gulrajani2021]`
- Gupta, M., Cotter, A., Pfeifer, J., Voevodski, K., Canini, K., Mangylov, A. et al. (2016). Monotonic Calibrated Interpolated Look-Up Tables. *Journal of Machine Learning Research 17(109): 1–47*. arXiv:1505.06378. `[gupta2016lattice]`
- Guyon, I., Weston, J., Barnhill, S. y Vapnik, V. (2002). Gene Selection for Cancer Classification using Support Vector Machines. *Machine Learning 46(1–3): 389–422*. doi:10.1023/A:1012487302797. `[guyon2002rfe]`
- Guyon, I. y Elisseeff, A. (2003). An Introduction to Variable and Feature Selection. *Journal of Machine Learning Research 3: 1157–1182*. <https://www.jmlr.org/papers/v3/guyon03a.html> `[guyon2003]`
- Hand, D. J. y Yu, K. (2001). Idiot's Bayes—Not So Stupid After All? *International Statistical Review 69(3): 385–398*. doi:10.1111/j.1751-5823.2001.tb00465.x. `[hand2001]`
- Hanna, R. y Olken, B. A. (2018). Universal Basic Incomes versus Targeted Transfers: Anti-Poverty Programs in Developing Countries. *Journal of Economic Perspectives 32(4): 201–226*. doi:10.1257/jep.32.4.201. `[hanna2018]`
- Hastie, T., Tibshirani, R. y Friedman, J. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction. *Springer*. doi:10.1007/978-0-387-84858-7. `[hastie2009esl]`
- Hastie, T., Tibshirani, R. y Tibshirani, R. J. (2020). Best Subset, Forward Stepwise or Lasso? Analysis and Recommendations Based on Extensive Comparisons. *Statistical Science 35(4): 579–592*. doi:10.1214/19-STS733. arXiv:1707.08692. `[hastie2020bestsubset]`
- Hooker, G., Mentch, L. y Zhou, S. (2021). Unrestricted Permutation Forces Extrapolation: Variable Importance Requires at Least One More Model, or There Is No Free Variable Importance. *Statistics and Computing 31(6): 82*. doi:10.1007/s11222-021-10057-z. arXiv:1905.03151. `[hooker2021]`
- Huang, J., Smola, A. J., Gretton, A., Borgwardt, K. M. y Schölkopf, B. (2007). Correcting Sample Selection Bias by Unlabeled Data. *Advances in Neural Information Processing Systems 19 (NIPS 2006) 19: 601–608*. doi:10.7551/mitpress/7503.003.0080. `[huang2007kmm]`
- Hutter, F., Hoos, H. y Leyton-Brown, K. (2014). An Efficient Approach for Assessing Hyperparameter Importance. *Proceedings of the 31st International Conference on Machine Learning (ICML) 32(1): 754–762*. <https://proceedings.mlr.press/v32/hutter14.html> `[hutter2014fanova]`
- Jamieson, K. y Talwalkar, A. (2016). Non-stochastic Best Arm Identification and Hyperparameter Optimization. *Proceedings of the 19th International Conference on Artificial Intelligence and Statistics (AISTATS) 51: 240–248*. arXiv:1502.07943. `[jamieson2016]`
- Janitza, S. y Hornung, R. (2018). On the Overestimation of Random Forest's Out-of-Bag Error. *PLoS ONE 13(8): e0201904*. doi:10.1371/journal.pone.0201904. `[janitza2018]`
- Kanamori, T., Hido, S. y Sugiyama, M. (2009). A Least-squares Approach to Direct Importance Estimation. *Journal of Machine Learning Research 10(48): 1391–1445*. <https://jmlr.org/papers/v10/kanamori09a.html> `[kanamori2009ulsif]`
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems 30 (NIPS 2017) 30*. <https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html> `[ke2017]`
- King, G. y Zeng, L. (2001). Logistic Regression in Rare Events Data. *Political Analysis 9(2): 137–163*. doi:10.1093/oxfordjournals.pan.a004868. `[king2001]`
- Kohavi, R. (1995). A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection. *Proceedings of the Fourteenth International Joint Conference on Artificial Intelligence (IJCAI-95) 2: 1137–1143*. <https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf> `[kohavi1995]`
- Kohavi, R. y John, G. H. (1997). Wrappers for Feature Subset Selection. *Artificial Intelligence 97(1–2): 273–324*. doi:10.1016/S0004-3702(97)00043-X. `[kohavi1997wrappers]`
- Koyejo, O., Natarajan, N., Ravikumar, P. y Dhillon, I. S. (2014). Consistent Binary Classification with Generalized Performance Metrics. *Advances in Neural Information Processing Systems 27*. <https://proceedings.neurips.cc/paper_files/paper/2014/hash/98053046e0dce5c7d946c67b96a85e18-Abstract.html> `[koyejo2014]`
- Krstajic, D., Buturovic, L. J., Leahy, D. E. y Thomas, S. (2014). Cross-Validation Pitfalls When Selecting and Assessing Regression and Classification Models. *Journal of Cheminformatics 6: 10*. doi:10.1186/1758-2946-6-10. `[krstajic2014]`
- Kshirsagar, V., Wieczorek, J., Ramanathan, S. y Wells, R. (2017). Household Poverty Classification in Data-Scarce Environments: A Machine Learning Approach. *NIPS 2017 Workshop on Machine Learning for the Developing World*. arXiv:1711.06813. `[kshirsagar2017]`
- Kursa, M. B. y Rudnicki, W. R. (2010). Feature Selection with the Boruta Package. *Journal of Statistical Software 36(11): 1–13*. doi:10.18637/jss.v036.i11. `[kursa2010]`
- Lakshminarayanan, B., Pritzel, A. y Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *Advances in Neural Information Processing Systems 30 (NIPS 2017) 30*. arXiv:1612.01474. `[lakshminarayanan2017]`
- Langley, P. y Sage, S. (1994). Induction of Selective Bayesian Classifiers. *Uncertainty in Artificial Intelligence: Proceedings of the Tenth Conference (UAI 1994): 399–406*. doi:10.1016/B978-1-55860-332-5.50055-9. `[langley1994]`
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. y Wasserman, L. (2018). Distribution-Free Predictive Inference for Regression. *Journal of the American Statistical Association 113(523): 1094–1111*. doi:10.1080/01621459.2017.1307116. arXiv:1604.04173. `[lei2018loco]`
- Lei, J. (2020). Cross-Validation With Confidence. *Journal of the American Statistical Association 115(532): 1978–1997*. doi:10.1080/01621459.2019.1672556. arXiv:1703.07904. `[lei2020cvc]`
- Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A. y Talwalkar, A. (2018). Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization. *Journal of Machine Learning Research 18(185): 1–52*. arXiv:1603.06560. `[li2018hyperband]`
- Li, L., Jamieson, K., Rostamizadeh, A., Gonina, E., Ben-tzur, J., Hardt, M. et al. (2020). A System for Massively Parallel Hyperparameter Tuning. *Proceedings of Machine Learning and Systems 2 (MLSys 2020) 2: 230–246*. arXiv:1810.05934. `[li2020asha]`
- LightGBM developers (2026). Parameters: monotone_constraints, monotone_constraints_method. <https://lightgbm.readthedocs.io/en/latest/Parameters.html> `[lightgbm_monotone_doc]`
- Lin, T. Y., Goyal, P., Girshick, R., He, K. y Dollár, P. (2017). Focal Loss for Dense Object Detection. *2017 IEEE International Conference on Computer Vision (ICCV): 2999–3007*. doi:10.1109/ICCV.2017.324. arXiv:1708.02002. `[lin2017focal]`
- Lipton, Z. C., Elkan, C. y Naryanaswamy, B. (2014). Optimal Thresholding of Classifiers to Maximize F1 Measure. *Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2014 8725: 225–239*. doi:10.1007/978-3-662-44851-9_15. arXiv:1402.1892. `[lipton2014]`
- Lipton, Z., Wang, Y. X. y Smola, A. (2018). Detecting and Correcting for Label Shift with Black Box Predictors. *Proceedings of the 35th International Conference on Machine Learning (ICML 2018) 80: 3122–3130*. arXiv:1802.03916. `[lipton2018bbse]`
- Lopez-Paz, D. y Oquab, M. (2017). Revisiting Classifier Two-Sample Tests. *International Conference on Learning Representations (ICLR 2017)*. arXiv:1610.06545. `[lopezpaz2017c2st]`
- Lu, Y. (2019). An End-to-End AutoML Solution for Tabular Data at KaggleDays. *Google AI Blog (hoy Google Research Blog)*. <https://research.google/blog/an-end-to-end-automl-solution-for-tabular-data-at-kaggledays/> `[lu2019automl]`
- Lundberg, S. M. y Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems 30 (NIPS 2017) 30: 4765–4774*. arXiv:1705.07874. `[lundberg2017]`
- Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B. et al. (2020). From Local Explanations to Global Understanding with Explainable AI for Trees. *Nature Machine Intelligence 2(1): 56–67*. doi:10.1038/s42256-019-0138-9. arXiv:1905.04610. `[lundberg2020]`
- Ma, C., Pathak, R. y Wainwright, M. J. (2023). Optimally tackling covariate shift in RKHS-based nonparametric regression. *The Annals of Statistics 51(2): 738–761*. doi:10.1214/23-AOS2268. arXiv:2205.02986. `[ma2023covshift]`
- Mansour, Y., Mohri, M. y Rostamizadeh, A. (2009). Domain Adaptation: Learning Bounds and Algorithms. *Proceedings of the Conference on Learning Theory (COLT 2009)*. arXiv:0902.3430. `[mansour2009discrepancy]`
- McBride, L. y Nichols, A. (2018). Retooling Poverty Targeting Using Out-of-Sample Validation and Machine Learning. *The World Bank Economic Review 32(3): 531–550*. doi:10.1093/wber/lhw056. `[mcbride2018]`
- Meier, L., van de Geer, S. y Bühlmann, P. (2008). The Group Lasso for Logistic Regression. *Journal of the Royal Statistical Society: Series B (Statistical Methodology) 70(1): 53–71*. doi:10.1111/j.1467-9868.2007.00627.x. `[meier2008]`
- Meinshausen, N. (2007). Relaxed Lasso. *Computational Statistics & Data Analysis 52(1): 374–393*. doi:10.1016/j.csda.2006.12.019. `[meinshausen2007relaxed]`
- Meinshausen, N. y Bühlmann, P. (2010). Stability Selection. *Journal of the Royal Statistical Society: Series B (Statistical Methodology) 72(4): 417–473*. doi:10.1111/j.1467-9868.2010.00740.x. arXiv:0809.2932. `[meinshausen2010]`
- Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. *ACM SIGKDD Explorations Newsletter 3(1): 27–32*. doi:10.1145/507533.507538. `[micci2001]`
- Minderer, M., Djolonga, J., Romijnders, R., Hubis, F., Zhai, X., Houlsby, N. et al. (2021). Revisiting the Calibration of Modern Neural Networks. *Advances in Neural Information Processing Systems 34: 15682–15694*. arXiv:2106.07998. `[minderer2021]`
- Mitchell, M. W. (2011). Bias of the Random Forest Out-of-Bag (OOB) Error for Certain Input Parameters. *Open Journal of Statistics 1(3): 205–211*. doi:10.4236/ojs.2011.13024. `[mitchell2011]`
- Molina, I. y Rao, J. N. K. (2010). Small area estimation of poverty indicators. *Canadian Journal of Statistics 38(3): 369–385*. doi:10.1002/cjs.10051. `[molina2010]`
- Moreno-Torres, J. G., Raeder, T., Alaiz-Rodríguez, R., Chawla, N. V. y Herrera, F. (2012). A unifying view on dataset shift in classification. *Pattern Recognition 45(1): 521–530*. doi:10.1016/j.patcog.2011.06.019. `[morenotorres2012]`
- Nadeau, C. y Bengio, Y. (2003). Inference for the Generalization Error. *Machine Learning 52(3): 239–281*. doi:10.1023/A:1024068626366. `[nadeau2003]`
- Narasimhan, H., Vaish, R. y Agarwal, S. (2014). On the Statistical Consistency of Plug-in Classifiers for Non-decomposable Performance Measures. *Advances in Neural Information Processing Systems 27*. <https://proceedings.neurips.cc/paper_files/paper/2014/hash/3644e33a5161ec5f3997a6acb98d4447-Abstract.html> `[narasimhan2014]`
- Ng, A. Y. y Jordan, M. I. (2002). On Discriminative vs. Generative Classifiers: A Comparison of Logistic Regression and Naive Bayes. *Advances in Neural Information Processing Systems 14 14*. <https://proceedings.neurips.cc/paper_files/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html> `[ng2002]`
- Niculescu-Mizil, A. y Caruana, R. (2005). Predicting Good Probabilities with Supervised Learning. *Proceedings of the 22nd International Conference on Machine Learning (ICML '05): 625–632*. doi:10.1145/1102351.1102430. `[niculescu2005]`
- Nixon, J., Dusenberry, M. W., Zhang, L., Jerfel, G. y Tran, D. (2019). Measuring Calibration in Deep Learning. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops: 38–41*. arXiv:1904.01685. `[nixon2019]`
- Ovadia, Y., Fertig, E., Ren, J., Nado, Z., Sculley, D., Nowozin, S. et al. (2019). Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift. *Advances in Neural Information Processing Systems 32*. arXiv:1906.02530. `[ovadia2019]`
- Pan, J., Pham, V., Dorairaj, M., Chen, H. y Lee, J. Y. (2020). Adversarial Validation Approach to Concept Drift Problem in User Targeting Automation Systems at Uber. arXiv:2004.03045. `[pan2020adversarial]`
- Pargent, F., Pfisterer, F., Thomas, J. y Bischl, B. (2022). Regularized target encoding outperforms traditional methods in supervised machine learning with high cardinality features. *Computational Statistics 37(5): 2671–2692*. doi:10.1007/s00180-022-01207-6. arXiv:2104.00629. `[pargent2022]`
- Peng, H., Long, F. y Ding, C. (2005). Feature Selection Based on Mutual Information: Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy. *IEEE Transactions on Pattern Analysis and Machine Intelligence 27(8): 1226–1238*. doi:10.1109/TPAMI.2005.159. `[peng2005mrmr]`
- Platt, J. C. (2000). Probabilities for SV Machines. *Advances in Large-Margin Classifiers: 61–74*. doi:10.7551/mitpress/1113.003.0008. `[platt1999]`
- Probst, P. y Boulesteix, A. L. (2018). To Tune or Not to Tune the Number of Trees in Random Forest. *Journal of Machine Learning Research 18(181): 1–18*. arXiv:1705.05654. `[probst2018numtrees]`
- Probst, P., Boulesteix, A. L. y Bischl, B. (2019). Tunability: Importance of Hyperparameters of Machine Learning Algorithms. *Journal of Machine Learning Research 20(53): 1–32*. arXiv:1802.09596. `[probst2019tunability]`
- Probst, P., Wright, M. N. y Boulesteix, A. L. (2019). Hyperparameters and Tuning Strategies for Random Forest. *WIREs Data Mining and Knowledge Discovery 9(3): e1301*. doi:10.1002/widm.1301. arXiv:1804.03515. `[probst2019rf]`
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V. y Gulin, A. (2018). CatBoost: Unbiased Boosting with Categorical Features. *Advances in Neural Information Processing Systems 31 (NeurIPS 2018) 31*. arXiv:1706.09516. `[prokhorenkova2018]`
- Puthiya Parambath, S. A., Usunier, N. y Grandvalet, Y. (2014). Optimizing F-Measures by Cost-Sensitive Classification. *Advances in Neural Information Processing Systems 27*. <https://proceedings.neurips.cc/paper_files/paper/2014/hash/5c0314ec1b57fcd36bbb013f3f025868-Abstract.html> `[puthiya2014]`
- Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A. y Lawrence, N. D. (eds.) (2009). Dataset Shift in Machine Learning. *MIT Press*. doi:10.7551/mitpress/9780262170055.001.0001. `[quinonero2009]`
- Rabanser, S., Günnemann, S. y Lipton, Z. (2019). Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift. *Advances in Neural Information Processing Systems 32*. arXiv:1810.11953. `[rabanser2019]`
- Roelofs, R., Shankar, V., Recht, B., Fridovich-Keil, S., Hardt, M., Miller, J. et al. (2019). A Meta-Analysis of Overfitting in Machine Learning. *Advances in Neural Information Processing Systems 32 (NeurIPS 2019) 32: 9179–9189*. <https://papers.nips.cc/paper_files/paper/2019/hash/ee39e503b6bedf0c98c388b7e8589aca-Abstract.html> `[roelofs2019]`
- Saerens, M., Latinne, P. y Decaestecker, C. (2002). Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure. *Neural Computation 14(1): 21–41*. doi:10.1162/089976602753284446. `[saerens2002]`
- scikit-learn developers (2026). Gradient Boosting Out-of-Bag estimates (ejemplo) y documentación de GradientBoostingClassifier (oob_improvement_, oob_scores_, oob_score_). *Documentación oficial de scikit-learn*. <https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_oob.html> `[sklearn_gb_oob]`
- scikit-learn developers (2026). RandomForestClassifier: oob_score, oob_decision_function_, max_samples, monotonic_cst. *Documentación oficial de scikit-learn*. <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html> `[sklearn_rf_doc]`
- Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D. et al. (2015). Hidden Technical Debt in Machine Learning Systems. *Advances in Neural Information Processing Systems 28*. <https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html> `[sculley2015]`
- Sculley, D., Snoek, J., Wiltschko, A. y Rahimi, A. (2018). Winner's Curse? On Pace, Progress, and Empirical Rigor. *International Conference on Learning Representations (ICLR 2018), Workshop Track*. <https://openreview.net/forum?id=rJWF0Fywf> `[sculley2018winners]`
- Shah, R. D. y Samworth, R. J. (2013). Variable Selection with Error Control: Another Look at Stability Selection. *Journal of the Royal Statistical Society: Series B (Statistical Methodology) 75(1): 55–80*. doi:10.1111/j.1467-9868.2011.01034.x. arXiv:1105.5578. `[shah2013]`
- Shimodaira, H. (2000). Improving predictive inference under covariate shift by weighting the log-likelihood function. *Journal of Statistical Planning and Inference 90(2): 227–244*. doi:10.1016/S0378-3758(00)00115-4. `[shimodaira2000]`
- Shwartz-Ziv, R. y Armon, A. (2022). Tabular Data: Deep Learning is Not All You Need. *Information Fusion 81: 84–90*. doi:10.1016/j.inffus.2021.11.011. arXiv:2106.03253. `[shwartzziv2022]`
- Snoek, J., Larochelle, H. y Adams, R. P. (2012). Practical Bayesian Optimization of Machine Learning Algorithms. *Advances in Neural Information Processing Systems 25 (NIPS 2012) 25*. arXiv:1206.2944. `[snoek2012]`
- Song, X., Perel, S., Lee, C., Kochanski, G. y Golovin, D. (2022). Open Source Vizier: Distributed Infrastructure and API for Reliable and Flexible Blackbox Optimization. *Proceedings of the First International Conference on Automated Machine Learning 188: 8/1–17*. arXiv:2207.13676. `[song2022ossvizier]`
- Strobl, C., Boulesteix, A. L., Zeileis, A. y Hothorn, T. (2007). Bias in Random Forest Variable Importance Measures: Illustrations, Sources and a Solution. *BMC Bioinformatics 8: 25*. doi:10.1186/1471-2105-8-25. `[strobl2007]`
- Strobl, C., Boulesteix, A. L., Kneib, T., Augustin, T. y Zeileis, A. (2008). Conditional Variable Importance for Random Forests. *BMC Bioinformatics 9: 307*. doi:10.1186/1471-2105-9-307. `[strobl2008]`
- Sugiyama, M., Krauledat, M. y Müller, K. R. (2007). Covariate Shift Adaptation by Importance Weighted Cross Validation. *Journal of Machine Learning Research 8(35): 985–1005*. <https://jmlr.org/papers/v8/sugiyama07a.html> `[sugiyama2007iwcv]`
- Sugiyama, M. y Kawanabe, M. (2012). Machine Learning in Non-Stationary Environments: Introduction to Covariate Shift Adaptation. *MIT Press*. doi:10.7551/mitpress/9780262017091.001.0001. `[sugiyama2012book]`
- Sugiyama, M., Suzuki, T. y Kanamori, T. (2012). Density Ratio Estimation in Machine Learning. *Cambridge University Press*. doi:10.1017/CBO9781139035613. `[sugiyama2012dre]`
- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the Royal Statistical Society: Series B (Methodological) 58(1): 267–288*. doi:10.1111/j.2517-6161.1996.tb02080.x. `[tibshirani1996]`
- Tibshirani, R. J. y Tibshirani, R. (2009). A Bias Correction for the Minimum Error Rate in Cross-Validation. *The Annals of Applied Statistics 3(2): 822–829*. doi:10.1214/08-AOAS224. arXiv:0908.2904. `[tibshirani2009bias]`
- van den Goorbergh, R., van Smeden, M., Timmerman, D. y Van Calster, B. (2022). The Harm of Class Imbalance Corrections for Risk Prediction Models: Illustration and Simulation Using Logistic Regression. *Journal of the American Medical Informatics Association 29(9): 1525–1534*. doi:10.1093/jamia/ocac093. arXiv:2202.09101. `[goorbergh2022]`
- Varma, S. y Simon, R. (2006). Bias in Error Estimation When Using Cross-Validation for Model Selection. *BMC Bioinformatics 7: 91*. doi:10.1186/1471-2105-7-91. `[varma2006]`
- Wager, S. (2020). Cross-Validation, Risk Estimation, and Model Selection: Comment on a Paper by Rosset and Tibshirani. *Journal of the American Statistical Association 115(529): 157–160*. doi:10.1080/01621459.2020.1727235. `[wager2020]`
- Wang, S. y Gupta, M. (2020). Deontological Ethics By Monotonicity Shape Constraints. *Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics (AISTATS) 108: 2043–2054*. arXiv:2001.11990. `[wang2020monotone]`
- Watanabe, S. (2023). Tree-Structured Parzen Estimator: Understanding Its Algorithm Components and Their Roles for Better Empirical Performance. *arXiv preprint arXiv:2304.11127*. arXiv:2304.11127. `[watanabe2023tpe]`
- Williamson, B. D., Gilbert, P. B., Simon, N. R. y Carone, M. (2023). A General Framework for Inference on Algorithm-Agnostic Variable Importance. *Journal of the American Statistical Association 118(543): 1645–1658*. doi:10.1080/01621459.2021.2003200. arXiv:2004.03683. `[williamson2023]`
- Wolpert, D. H. (1992). Stacked Generalization. *Neural Networks 5(2): 241–259*. doi:10.1016/S0893-6080(05)80023-1. `[wolpert1992]`
- Wolpert, D. H. y Macready, W. G. (1999). An Efficient Method To Estimate Bagging's Generalization Error. *Machine Learning 35(1): 41–55*. doi:10.1023/A:1007519102914. `[wolpert1999]`
- XGBoost developers (2026). Monotonic Constraints. <https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html> `[xgboost_monotone_doc]`
- Yang, Y. y Webb, G. I. (2009). Discretization for Naive-Bayes Learning: Managing Discretization Bias and Variance. *Machine Learning 74(1): 39–74*. doi:10.1007/s10994-008-5083-5. `[yang2009discretization]`
- Ye, N., Chai, K. M. A., Lee, W. S. y Chieu, H. L. (2012). Optimizing F-Measures: A Tale of Two Approaches. *Proceedings of the 29th International Conference on Machine Learning (ICML 2012)*. arXiv:1206.4625. `[ye2012]`
- Yoon, J., Arik, S. O. y Pfister, T. (2020). Data Valuation using Reinforcement Learning. *Proceedings of the 37th International Conference on Machine Learning (ICML 2020) 119: 10842–10851*. arXiv:1909.11671. `[yoon2020dvrl]`
- You, S., Ding, D., Canini, K., Pfeifer, J. y Gupta, M. (2017). Deep Lattice Networks and Partial Monotonic Functions. *Advances in Neural Information Processing Systems 30 (NIPS 2017) 30*. arXiv:1709.06680. `[you2017dln]`
- Yu, B. (2013). Stability. *Bernoulli 19(4): 1484–1500*. doi:10.3150/13-BEJSP14. arXiv:1310.0150. `[yu2013stability]`
- Yu, B. y Kumbier, K. (2020). Veridical Data Science. *Proceedings of the National Academy of Sciences 117(8): 3920–3929*. doi:10.1073/pnas.1901326117. arXiv:1901.08152. `[yu2020pcs]`
- Yuan, M. y Lin, Y. (2006). Model Selection and Estimation in Regression with Grouped Variables. *Journal of the Royal Statistical Society: Series B (Statistical Methodology) 68(1): 49–67*. doi:10.1111/j.1467-9868.2005.00532.x. `[yuan2006]`
- Zadrozny, B. y Elkan, C. (2002). Transforming Classifier Scores into Accurate Multiclass Probability Estimates. *Proceedings of the Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '02): 694–699*. doi:10.1145/775047.775151. `[zadrozny2002]`
- Zadrozny, B. (2004). Learning and Evaluating Classifiers under Sample Selection Bias. *Proceedings of the Twenty-First International Conference on Machine Learning (ICML 2004): 114*. doi:10.1145/1015330.1015425. `[zadrozny2004]`
- Zhao, M. J., Edakunni, N., Pocock, A. y Brown, G. (2013). Beyond Fano's Inequality: Bounds on the Optimal F-Score, BER, and Cost-Sensitive Risk and Their Implications. *Journal of Machine Learning Research 14(32): 1033–1090*. <https://jmlr.org/papers/v14/zhao13a.html> `[zhao2013fano]`
- Zinkevich, M. (2025). Rules of Machine Learning: Best Practices for ML Engineering. *Google for Developers, Machine Learning Guides*. <https://developers.google.com/machine-learning/guides/rules-of-ml> `[zinkevich_rules]`
- Zou, H. y Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. *Journal of the Royal Statistical Society: Series B (Statistical Methodology) 67(2): 301–320*. doi:10.1111/j.1467-9868.2005.00503.x. `[zou2005]`
