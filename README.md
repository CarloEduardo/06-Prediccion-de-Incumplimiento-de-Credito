# Predicción de Incumplimiento de Credito 💳

Clasificación del incumplimiento de pago de clientes de tarjetas de crédito utilizando el [conjunto de datos de la UCI](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) y técnicas de aprendizaje automático.

Para conocer en detalle las técnicas y metodologías empleadas, consulta el [**informe técnico**](https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/blob/main/Default_of_Credit_Card_Clients.ipynb) o la [**versión web del informe**](https://htmlpreview.github.io/?https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/blob/main/Default_of_Credit_Card_Clients.html).

# Nombre del Proyecto

Breve descripción del problema y del objetivo del proyecto.

## Dataset
- Fuente de los datos
- Número de observaciones y variables
- Variable objetivo

## Objetivo
Descripción del problema de negocio y del modelo que se busca desarrollar.

## Tecnologías
- Python
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

## Metodología
1. Carga de datos
2. Análisis exploratorio (EDA)
3. Preprocesamiento
4. Ingeniería de características
5. Entrenamiento de modelos
6. Optimización de hiperparámetros
7. Evaluación del modelo

## Resultados
- Modelo ganador
- Principales métricas (Accuracy, Precision, Recall, F1, ROC-AUC)
- Matriz de confusión (opcional)

## Estructura del proyecto
```text
├── data/
├── notebooks/
├── images/
├── models/
├── src/
├── requirements.txt
└── README.md
```

## Contents:

1.  [Introduction](#introduction)
2.  [Exploratory Data Analysis](#exploratory-data-analysis)
    * [Feature Analysis](#feature-analysis)
    * [Data cleaning](#data-cleaning)
    * [Correlation](#correlation)
    * [Box plot](#box-plot)
3.  [Data Preprocessing](#data-preprocessing)
    * [One-hot encoding for categorical variables](#one-hot-encoding-for-categorical-variables)
    * [Feature Scaling](#feature-scaling)
    * [Train-test split](#train-test-split)
    * [Dimensionality Reduction](#dimensionality-reduction)
        * [Feature selection](#feature-selection)
        * [Principal Component Analysis](#principal-component-analysis)
    * [Oversampling and Undersampling](#oversampling-and-undersampling)
4.  [Classification algorithms](#classification-algorithms)
    * [Evaluation Metrics](#evaluation-methods)
    * [Models](#models)
        * [Logistic Regression](#logistic-regression)
        * [Random Forests](#random-forest)
        * [Support Vector Machines](#svm)
5.  [Results](#results)
6.  [Conclusions](#conclusions)

## Introducción

El objetivo de este estudio es aplicar algoritmos de aprendizaje automático supervisado para identificar los principales factores que determinan la probabilidad de incumplimiento en el pago de tarjetas de crédito, destacando los fundamentos matemáticos de los métodos utilizados.

El incumplimiento de una tarjeta de crédito ocurre cuando un cliente presenta un atraso significativo en el pago de sus obligaciones. En Taiwán, con el propósito de aumentar su participación en el mercado, las entidades emisoras de tarjetas de crédito otorgaron un gran número de tarjetas y líneas de crédito a solicitantes que no cumplían con los criterios adecuados de elegibilidad. Al mismo tiempo, muchos titulares, independientemente de su capacidad de pago, hicieron un uso excesivo del crédito para financiar su consumo, acumulando elevados niveles de endeudamiento.

El propósito de este proyecto es desarrollar un modelo automatizado capaz de identificar los factores más influyentes y predecir el incumplimiento en el pago de tarjetas de crédito utilizando información demográfica de los clientes y su historial de transacciones. Asimismo, se presentan los conceptos fundamentales del aprendizaje automático supervisado, junto con una explicación detallada de las técnicas y algoritmos empleados para construir los modelos predictivos. En particular, se implementan y comparan los algoritmos de Regresión Logística (Logistic Regression), Bosques Aleatorios (Random Forest) y Máquinas de Vectores de Soporte (Support Vector Machines, SVM).

## Entorno de desarrollo

El análisis se ha realizado íntegramente utilizando el lenguaje de programación **Python**, aprovechando diversos paquetes de aprendizaje automático y análisis estadístico, como `scikit-learn`, `numpy`, `pandas` e `imblearn`, junto con otras bibliotecas de visualización de datos (`matplotlib` y `seaborn`).

Se han incluido algunos fragmentos de código para facilitar la comprensión de cada etapa del análisis. Además, la implementación completa del código de este análisis está disponible públicamente en [github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys](https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys).

## Análisis Exploratorio de Datos

### Análisis de las Variables

El conjunto de datos utilizado en este estudio corresponde al conjunto **_Default of Credit Card Clients_** del repositorio de aprendizaje automático de la **UCI (University of California, Irvine)**, disponible en el siguiente [enlace](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).

Este conjunto de datos está compuesto por **30 000 observaciones**, cada una de las cuales representa a un cliente distinto de tarjeta de crédito. Cada observación contiene **24 variables** que incluyen información sobre incumplimientos de pago, características demográficas, datos crediticios, historial de pagos y estados de cuenta de clientes de tarjetas de crédito en Taiwán, correspondientes al período comprendido entre **abril y septiembre de 2005**.

El primer grupo de variables contiene información relacionada con las **características personales del cliente**:

1. `ID`: Identificador único de cada cliente, variable categórica.
2. `LIMIT_BAL`: Monto del límite de crédito otorgado en dólares taiwaneses (NT$), incluyendo el crédito individual y el crédito complementario o familiar.
3. `SEX`: Sexo, variable categórica (1 = hombre, 2 = mujer).
4. `EDUCATION`: Nivel educativo, variable categórica (1 = posgrado, 2 = universidad, 3 = educación secundaria, 4 = otros, 5 = desconocido, 6 = desconocido).
5. `MARRIAGE`: Estado civil, variable categórica (1 = casado, 2 = soltero, 3 = otros).
6. `AGE`: Edad en años, variable numérica.

Las siguientes variables contienen información sobre el **estado de retraso en los pagos** correspondiente a un mes específico:

1. `PAY_0`: Estado de pago en septiembre de 2005 (-1 = pago puntual, 1 = retraso de un mes, 2 = retraso de dos meses, …, 8 = retraso de ocho meses, 9 = retraso de nueve meses o más).
2. `PAY_2`: Estado de pago en agosto de 2005 (misma escala que la anterior).
3. `PAY_3`: Estado de pago en julio de 2005 (misma escala que la anterior).
4. `PAY_4`: Estado de pago en junio de 2005 (misma escala que la anterior).
5. `PAY_5`: Estado de pago en mayo de 2005 (misma escala que la anterior).
6. `PAY_6`: Estado de pago en abril de 2005 (misma escala que la anterior).

Las siguientes variables contienen información relacionada con el **monto del estado de cuenta** (es decir, el reporte mensual que las entidades emisoras de tarjetas de crédito entregan a sus clientes):

1. `BILL_AMT1`: Monto del estado de cuenta de septiembre de 2005 (NT$).
2. `BILL_AMT2`: Monto del estado de cuenta de agosto de 2005 (NT$).
3. `BILL_AMT3`: Monto del estado de cuenta de julio de 2005 (NT$).
4. `BILL_AMT4`: Monto del estado de cuenta de junio de 2005 (NT$).
5. `BILL_AMT5`: Monto del estado de cuenta de mayo de 2005 (NT$).
6. `BILL_AMT6`: Monto del estado de cuenta de abril de 2005 (NT$).

Las siguientes variables corresponden al **monto del pago realizado** en un mes específico:

1. `PAY_AMT1`: Monto del pago realizado en septiembre de 2005 (NT$).
2. `PAY_AMT2`: Monto del pago realizado en agosto de 2005 (NT$).
3. `PAY_AMT3`: Monto del pago realizado en julio de 2005 (NT$).
4. `PAY_AMT4`: Monto del pago realizado en junio de 2005 (NT$).
5. `PAY_AMT5`: Monto del pago realizado en mayo de 2005 (NT$).
6. `PAY_AMT6`: Monto del pago realizado en abril de 2005 (NT$).

La última variable corresponde a la **variable objetivo** que se desea predecir:

1. `default.payment.next.month`: Indica si el titular de la tarjeta de crédito incumplirá con el pago el mes siguiente (1 = sí, 0 = no).

El objetivo principal del conjunto de datos es identificar a los clientes que se espera que incumplan con el pago de su tarjeta de crédito el mes siguiente, utilizando como variable objetivo la columna `default.payment.next.month`, donde **"0"** representa a los clientes que **no incumplen** y **"1"** a los clientes que **sí incumplen**. Por lo tanto, se trata de un **_problema de clasificación binaria_** sobre un conjunto de datos **relativamente desbalanceado**, como se muestra en la siguiente figura.

<p align = "center">
<img height="300" src="https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/blob/main/images/imbalanced_plot.svg">
</p>
<p align = "center">
Class distribution
</p>
 
Como se puede observar, **6,636 de los 30,000 clientes (22.1%)** incumplirán con el pago de su tarjeta de crédito el mes siguiente (es decir, pertenecen a la categoría **1**).

Para obtener una primera visión de cómo se presentan los datos, en la siguiente tabla se muestran algunas observaciones del conjunto de datos. No se registran valores faltantes en ninguna de las variables para las **30,000** observaciones.

A primera vista, los datos parecen estar ya codificados y depurados. Sin embargo, al examinar con mayor detalle algunas muestras y las estadísticas descriptivas, se identifican varias observaciones que vale la pena destacar.

```python
# read input
data = pd.read_csv(INPUT_PATH)
data.head()
``` 

|     | ID  | LIMIT_BAL | SEX | EDUCATION | MARRIAGE | AGE | PAY_0 | PAY_2 | PAY_3 | PAY_4 | ... | BILL_AMT4 | BILL_AMT5 | BILL_AMT6 | PAY_AMT1 | PAY_AMT2 | PAY_AMT3 | PAY_AMT4 | PAY_AMT5 | PAY_AMT6 | default.payment.next.month |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0   | 1   | 20000.0 | 2   | 2   | 1   | 24  | 2   | 2   | -1  | -1  | ... | 0.0 | 0.0 | 0.0 | 0.0 | 689.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1   |
| 1   | 2   | 120000.0 | 2   | 2   | 2   | 26  | -1  | 2   | 0   | 0   | ... | 3272.0 | 3455.0 | 3261.0 | 0.0 | 1000.0 | 1000.0 | 1000.0 | 0.0 | 2000.0 | 1   |
| 2   | 3   | 90000.0 | 2   | 2   | 2   | 34  | 0   | 0   | 0   | 0   | ... | 14331.0 | 14948.0 | 15549.0 | 1518.0 | 1500.0 | 1000.0 | 1000.0 | 1000.0 | 5000.0 | 0   |
| 3   | 4   | 50000.0 | 2   | 2   | 1   | 37  | 0   | 0   | 0   | 0   | ... | 28314.0 | 28959.0 | 29547.0 | 2000.0 | 2019.0 | 1200.0 | 1100.0 | 1069.0 | 1000.0 | 0   |
| 4   | 5   | 50000.0 | 1   | 2   | 1   | 57  | -1  | 0   | -1  | 0   | ... | 20940.0 | 19146.0 | 19131.0 | 2000.0 | 36681.0 | 10000.0 | 9000.0 | 689.0 | 679.0 | 0   |

5 rows × 25 columns
```python
# Summary Statistics
data.describe()
```

|     | ID  | LIMIT_BAL | SEX | EDUCATION | MARRIAGE | AGE | PAY_0 | PAY_2 | PAY_3 | PAY_4 | ... | BILL_AMT4 | BILL_AMT5 | BILL_AMT6 | PAY_AMT1 | PAY_AMT2 | PAY_AMT3 | PAY_AMT4 | PAY_AMT5 | PAY_AMT6 | default.payment.next.month |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| count | 30000.00 | 30000.00 | 30000.00 | 30000.00 | 30000.00 | 30000.00 | 30000.00 | 30000.00 | 30000.00 | 30000.00 | ... | 30000.00 | 30000.00 | 30000.00 | 30000.00 | 3.00e+04 | 30000.00 | 30000.00 | 30000.00 | 30000.00 | 30000.00 |
| mean | 15000.50 | 167484.32 | 1.60 | 1.85 | 1.55 | 35.49 | -0.02 | -0.13 | -0.17 | -0.22 | ... | 43262.95 | 40311.40 | 38871.76 | 5663.58 | 5.92e+03 | 5225.68 | 4826.08 | 4799.39 | 5215.50 | 0.22 |
| std | 8660.40 | 129747.66 | 0.49 | 0.79 | 0.52 | 9.22 | 1.12 | 1.20 | 1.20 | 1.17 | ... | 64332.86 | 60797.16 | 59554.11 | 16563.28 | 2.30e+04 | 17606.96 | 15666.16 | 15278.31 | 17777.47 | 0.42 |
| min | 1.00 | 10000.00 | 1.00 | 0.00 | 0.00 | 21.00 | -2.00 | -2.00 | -2.00 | -2.00 | ... | -170000.00 | -81334.00 | -339603.00 | 0.00 | 0.00e+00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 25% | 7500.75 | 50000.00 | 1.00 | 1.00 | 1.00 | 28.00 | -1.00 | -1.00 | -1.00 | -1.00 | ... | 2326.75 | 1763.00 | 1256.00 | 1000.00 | 8.33e+02 | 390.00 | 296.00 | 252.50 | 117.75 | 0.00 |
| 50% | 15000.50 | 140000.00 | 2.00 | 2.00 | 2.00 | 34.00 | 0.00 | 0.00 | 0.00 | 0.00 | ... | 19052.00 | 18104.50 | 17071.00 | 2100.00 | 2.01e+03 | 1800.00 | 1500.00 | 1500.00 | 1500.00 | 0.00 |
| 75% | 22500.25 | 240000.00 | 2.00 | 2.00 | 2.00 | 41.00 | 0.00 | 0.00 | 0.00 | 0.00 | ... | 54506.00 | 50190.50 | 49198.25 | 5006.00 | 5.00e+03 | 4505.00 | 4013.25 | 4031.50 | 4000.00 | 0.00 |
| max | 30000.00 | 1000000.00 | 2.00 | 6.00 | 3.00 | 79.00 | 8.00 | 8.00 | 8.00 | 8.00 | ... | 891586.00 | 927171.00 | 961664.00 | 873552.00 | 1.68e+06 | 896040.00 | 621000.00 | 426529.00 | 528666.00 | 1.00 |

8 rows × 25 columns

Por ejemplo, pueden observarse las siguientes inconsistencias:

* El nombre de la columna `PAY_0` debería cambiarse por `PAY_1`.
* Las variables `EDUCATION` y `MARRIAGE` contienen algunas categorías no documentadas.
* Las columnas `PAY_n` presentan un valor mínimo de **-2** (no documentado) y un valor máximo de **8**. Por ello, es probable que estas variables requieran un reescalamiento.

### 2.2 Limpieza de los datos

La presencia de errores en el conjunto de datos puede abordarse de dos maneras:

1. Eliminar las filas asociadas a un error, considerando el número de registros afectados con respecto al total de observaciones.
2. Corregir el valor erróneo de la variable.

En este caso, el primer método se aplica a las variables categóricas, eliminando de forma segura las categorías desconocidas, ya que el número total de valores anómalos es de **399**:

* En la variable `EDUCATION` existen tres categorías (**0**, **5** y **6**) que no aparecen en la descripción oficial del conjunto de datos proporcionada por el repositorio UCI.
* En la variable `MARRIAGE` se observa la presencia de la categoría **0**, la cual no corresponde a ninguna de las categorías previamente descritas.

Por otro lado, el segundo enfoque se aplica a las variables `PAY_n`, corrigiéndolas mediante la suma de **1** a cada valor y agrupando posteriormente las categorías resultantes **0** y **-1**, ya que el valor **0** no corresponde a ninguna de las categorías descritas originalmente.

```python
data['MARRIAGE'].value_counts()
```
    Out [6]:

    2    15964
    1    13659
    3      323
    0       54
    Name: MARRIAGE, dtype: int64

```python
data['EDUCATION'].value_counts()
```
    Out [7]:

    2    14030
    1    10585
    3     4917
    5      280
    4      123
    6       51
    0       14
    Name: EDUCATION, dtype: int64

```python
# Payment delay description
data[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']].describe()
```

|     | PAY_1 | PAY_2 | PAY_3 | PAY_4 | PAY_5 | PAY_6 |
| --- | --- | --- | --- | --- | --- | --- |
| count | 29601.00 | 29601.00 | 29601.00 | 29601.00 | 29601.00 | 29601.00 |
| mean | -0.01 | -0.13 | -0.16 | -0.22 | -0.26 | -0.29 |
| std | 1.12 | 1.20 | 1.20 | 1.17 | 1.14 | 1.15 |
| min | -2.00 | -2.00 | -2.00 | -2.00 | -2.00 | -2.00 |
| 25% | -1.00 | -1.00 | -1.00 | -1.00 | -1.00 | -1.00 |
| 50% | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 75% | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| max | 8.00 | 8.00 | 8.00 | 8.00 | 8.00 | 8.00 |

``` python
# Amount of given credit in NT dollars
data.LIMIT_BAL.describe()
```

    Out[20]:

    count      29601.00
    mean      167550.54
    std       129944.02
    min        10000.00
    25%        50000.00
    50%       140000.00
    75%       240000.00
    max      1000000.00
    Name: LIMIT_BAL, dtype: float64

El valor muy elevado de la desviación estándar fue analizado con mayor detalle. Como puede observarse, la mayoría de los incumplimientos se concentra en clientes con límites de crédito entre **0 y 100,000**, y la densidad de clientes en este intervalo es mayor para quienes incumplen que para quienes no lo hacen.

<p align = "center">
<img height="300" src="https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/blob/main/images/continuosDistribution.svg">
</p>
<p align = "center">
Density plot of amount of given credit (LIMIT_BAL)
</p>

### Correlación entre variables

Otro aspecto relevante que puede afectar el rendimiento de los modelos de clasificación es la correlación entre las variables. La presencia de variables altamente correlacionadas puede reducir el desempeño de algunos algoritmos de clasificación que asumen que los predictores son independientes entre sí.

Además, identificar variables correlacionadas permite detectar información redundante que podría representarse con un menor número de atributos, dando lugar a modelos más simples. De hecho, algunos métodos son sensibles a conjuntos de datos con alta dimensionalidad (especialmente aquellos basados en distancias y con un número reducido de observaciones). Por ello, reducir la dimensionalidad de los datos puede hacer que los modelos sean más robustos y estables, siempre que las variables eliminadas no aporten información significativa.

El _coeficiente de correlación de Pearson_ es una medida estadística que cuantifica la relación lineal entre un par de variables aleatorias. En particular, dado el vector aleatorio $(X,Y)$:

$$\rho_{X,Y} := \frac{Cov(X,Y)}{\sigma_X \sigma_Y} \in [-1,1] \subset \mathbb{R}$$

donde $Cov(X,Y)$ representa la covarianza entre las variables y $\sigma_X$ y $\sigma_Y$ corresponden a las desviaciones estándar de $X$ y $Y$, respectivamente.

El coeficiente $\rho_{X,Y}$ toma valores entre **-1** y **1**. Cuanto mayor sea su valor absoluto, más fuerte será la relación lineal entre ambas variables. Esto significa que, en promedio, cuando una variable toma valores elevados respecto a su media, la otra también tiende a tomar valores elevados (o bajos, si la correlación es negativa).

Es importante destacar que el coeficiente de Pearson únicamente mide relaciones lineales. Por ello, dos variables pueden presentar una fuerte relación no lineal (por ejemplo, una relación cuadrática) y, aun así, obtener un coeficiente cercano a **0**.

En la práctica, la distribución de probabilidad subyacente es desconocida y solo se dispone de una muestra de observaciones del vector aleatorio $(X,Y)$. Por ello, el coeficiente se estima utilizando las covarianzas y desviaciones estándar muestrales para cada par de variables.

La siguiente figura muestra la **matriz de correlación** entre las variables. Dado que esta matriz es simétrica y el conjunto de datos contiene un número considerable de atributos, se presenta únicamente el **mapa de calor de la mitad inferior de la matriz** (excluyendo la diagonal principal) para facilitar su interpretación.

<p align = "center">
<img height="750" src="https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/blob/main/images/correlation.png">
</p>
<p align = "center">
Correlation matrix by means of the Pearson’s coefficient for all feature pairs.
</p>

Como se muestra en la matriz de correlación anterior, algunas variables presentan altas correlaciones entre sí. En particular, existe una fuerte correlación positiva entre las variables `BILL_AMTn`, por ejemplo:

* BILL\_AMT1 and BILL\_AMT2 have $\rho = 0.95$
* BILL\_AMT2 and BILL\_AMT3 have $\rho = 0.93$
* BILL\_AMT4 and BILL\_AMT5 have $\rho = 0.94$

Con el fin de eliminar variables correlacionadas, durante la fase de preprocesamiento de los datos se aplicarán algunas técnicas de reducción de dimensionalidad.

### Diagrama de caja (Boxplot)

Finalmente, se generan diagramas de caja para cada una de las variables numéricas con el objetivo de examinar con mayor detalle su distribución e identificar posibles valores atípicos (es decir, datos "anómalos" que podrían haberse calculado incorrectamente o que son excepcionalmente diferentes del rango esperado).

Un diagrama de caja representa un resumen de cinco valores de un conjunto de datos:

* Primer cuartil: el percentil 25.
* Mediana (o segundo cuartil): el valor central del conjunto de datos.
* Tercer cuartil: el percentil 75.
* Mínimo: el valor más pequeño, excluyendo los valores atípicos.
* Máximo: el valor más grande, excluyendo los valores atípicos.

En el diagrama de caja, el **rango intercuartílico (IQR)** se define como la distancia entre el tercer y el primer cuartil:
$$
IQR = Q_3 - Q_1 = q(0.75) - q(0.25)
$$

<p align = "center">
<img height="300" src="https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/blob/main/images/boxplot_theory.png?raw=true">
</p>
<p align = "center">
Boxplot description
</p>

Además, mediante estos gráficos es posible detectar la presencia de valores atípicos (*outliers*), es decir, observaciones individuales, utilizando la siguiente regla: toda muestra ubicada por debajo de $Q_1 - 1.5 \cdot IQR$ o por encima de $Q_3 + 1.5 \cdot IQR$ se considera un valor atípico.

_**Nota:** Para este análisis se aplicó una normalización a todos los atributos con el fin de llevarlos a la misma escala; de lo contrario, no habría sido posible compararlos entre sí._

### Escalamiento Min-Max para variables numéricas

Como se mencionó anteriormente, las variables de entrada pueden estar expresadas en diferentes unidades y, por lo tanto, presentar distintas escalas y magnitudes. Por esta razón, antes de generar un diagrama de cajas (*boxplot*), se aplica `MinMaxScaler()` para escalar las características al intervalo $(0, 1)$.

La idea fundamental de esta técnica de reescalamiento es que, para cada característica, el valor mínimo se transforma en 0, el valor máximo en 1 y todos los demás valores se convierten en números decimales comprendidos entre 0 y 1.

La transformación está dada por la siguiente fórmula:

$$ X_{scaled} = \frac{(X - X_{min})}{(X_{max} - X_{min})} $$

donde $X_{min}$ representa el valor mínimo de la columna y $X_{max}$ el valor máximo de la columna.

Esta transformación se aplica **únicamente a las variables numéricas**, ya que las variables categóricas ya fueron transformadas mediante codificación *one-hot* (*one-hot encoding*), lo que las reescala automáticamente al intervalo $(0,1)$.

### Estandarización

Otra técnica de escalamiento ampliamente utilizada es la **estandarización**, la cual transforma cada valor $x$ de la característica $X$ de la siguiente manera, de forma independiente para cada columna:

$$ x^{\prime} =\frac{x-\mu_{X}}{\sigma_{X}} $$

donde $\mu_{X}$ representa la media muestral y $\sigma_{X}$ la desviación estándar muestral de la característica $X$. De esta manera, se consigue que cada atributo tenga una nueva media empírica igual a 0 y una varianza igual a 1.

_**Nota:** Las estadísticas utilizadas para el escalamiento (por ejemplo, la media y la desviación estándar en el caso de `StandardScaler`) se calculan **únicamente sobre el conjunto de entrenamiento**. Posteriormente, tanto el conjunto de entrenamiento como el de prueba se transforman utilizando dichas estadísticas. Este procedimiento es fundamental, ya que evita la **fuga de información (*data leakage*)** desde el conjunto de prueba hacia el proceso de entrenamiento, lo cual sería metodológicamente incorrecto._

```python 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() \# or MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```
<p align = "center">
<img height="300" src="https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/blob/main/images/normalized.svg">
</p>
<p align = "center">
Diagramas de caja de las variables numéricas escaladas mediante normalización Min-Max (izquierda) y estandarización (derecha).
</p>

De acuerdo con el método descrito anteriormente, los diagramas de caja permiten identificar una gran cantidad de valores atípicos (outliers). Es posible modificar el coeficiente multiplicativo que determina el umbral a partir del cual una observación se considera un valor atípico; sin embargo, incluso al duplicar dicho coeficiente, se siguen identificando más de 4 000 posibles valores atípicos.

Debido a la falta de conocimiento específico del dominio y al elevado número de valores atípicos detectados, se decidió no eliminar ninguna observación del conjunto de datos.

## Preprocesamiento de los datos

### Codificación One-Hot para variables categóricas

La codificación mediante valores enteros puede imponer una relación ordinal entre variables categóricas que en realidad no existe. Por esta razón, se aplica una **codificación One-Hot**. Las variables categóricas, como `SEX`, `MARRIAGE` y `EDUCATION`, se transforman en variables binarias para eliminar cualquier orden implícito que, en este caso, carece de significado.

La codificación One-Hot representa las variables categóricas mediante vectores binarios. Para ello, primero cada categoría se asigna a un valor entero. Posteriormente, cada uno de estos valores enteros se convierte en un vector binario compuesto únicamente por ceros, excepto en la posición correspondiente a la categoría, donde se asigna un valor de 1.

|     | SEX | EDUCATION_1 | EDUCATION_2 | EDUCATION_3 | EDUCATION_4 | MARRIAGE_1 | MARRIAGE_2 | MARRIAGE_3 | LIMIT_BAL | ... | BILL_AMT4 | BILL_AMT5 | BILL_AMT6 | PAY_AMT1 | PAY_AMT2 | PAY_AMT3 | PAY_AMT4 | PAY_AMT5 | PAY_AMT6 | Default |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0   | 0   | 0   | 1   | 0   | 0   | 1   | 0   | 0   | 0.01 | ... | 0.16 | 0.08 | 0.26 | 0.00e+00 | 4.09e-04 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 1   |
| 1   | 1   | 0   | 1   | 0   | 0   | 0   | 1   | 0   | 0.11 | ... | 0.16 | 0.08 | 0.26 | 0.00e+00 | 5.94e-04 | 1.12e-03 | 1.61e-03 | 0.00e+00 | 3.78e-03 | 1   |
| 2   | 0   | 0   | 1   | 0   | 0   | 0   | 1   | 0   | 0.08 | ... | 0.17 | 0.10 | 0.27 | 1.74e-03 | 8.91e-04 | 1.12e-03 | 1.61e-03 | 2.34e-03 | 9.46e-03 | 0   |
| 3   | 1   | 0   | 1   | 0   | 0   | 1   | 0   | 0   | 0.04 | ... | 0.19 | 0.11 | 0.28 | 2.29e-03 | 1.20e-03 | 1.34e-03 | 1.77e-03 | 2.51e-03 | 1.89e-03 | 0   |
| 4   | 0   | 0   | 1   | 0   | 0   | 1   | 0   | 0   | 0.04 | ... | 0.18 | 0.10 | 0.28 | 2.29e-03 | 2.18e-02 | 1.12e-02 | 1.45e-02 | 1.62e-03 | 1.28e-03 | 0   |

5 rows × 30 columns

### Escalado de variables (*Feature Scaling*)

Como se explicó anteriormente, ya se ha aplicado un `MinMaxScaler()` a todas las variables numéricas.

### División en conjunto de entrenamiento y prueba (*Train-Test Split*)

El conjunto de datos se divide en un **conjunto de entrenamiento** y un **conjunto de prueba**, utilizando una proporción de **3:1**.

Debido al desbalance presente en el conjunto de datos original, durante la división se aplicó una estrategia de **muestreo estratificado (*stratified sampling*)**, de modo que la proporción de clientes incumplidores (*defaulters*) y no incumplidores (*non-defaulters*) se mantuviera aproximadamente igual en ambos conjuntos, respetando la distribución original de la variable objetivo.

Este enfoque es especialmente recomendable cuando se trabaja con **_conjuntos de datos altamente desbalanceados_**, como ocurre en este caso.

La distribución final obtenida es la siguiente:

|     | Non-defaulter | Defaulters | Total |
| --- | --- | --- | --- |
| **Training set** | 17246 | 4954 | 22200 |
| **Test set** | 5750 | 1651 | 7401 |

_**Nota:** La división del conjunto de datos se generó de forma aleatoria, pero se mantuvo exactamente la misma para todos los métodos evaluados. Esto garantiza que las comparaciones entre los distintos clasificadores sean estadísticamente consistentes y que los resultados obtenidos sean comparables._

```python
random_state = 42
```

### Reducción de dimensionalidad

Como se mencionó anteriormente, en términos generales no siempre es conveniente utilizar todos los atributos del conjunto de datos para entrenar un modelo. Las principales razones para aplicar una reducción de dimensionalidad son las siguientes:

- Los datos de alta dimensionalidad pueden afectar negativamente el rendimiento de los algoritmos, incrementando el costo computacional.
- Un gran número de variables puede reducir la capacidad de generalización del modelo.
- La reducción de dimensionalidad facilita la interpretación y visualización de los datos.

Además, muchos algoritmos de aprendizaje automático sufren del problema conocido como la **maldición de la dimensionalidad (*curse of dimensionality*)**. Cuando el número de variables es muy elevado en relación con el número de observaciones, aumenta considerablemente el riesgo de **sobreajuste (*overfitting*)**.

Asimismo, al incrementar el número de dimensiones, las distancias entre las observaciones tienden a volverse muy similares. Esto representa un problema para los algoritmos basados en distancias, como **K-Nearest Neighbors (KNN)**, ya que si todas las distancias son aproximadamente iguales, las observaciones resultan indistinguibles entre sí, reduciendo significativamente la capacidad predictiva del algoritmo.

### Selección de variables (*Feature Selection*)

Además de la transformación de variables, la etapa de preprocesamiento suele incluir un proceso de **selección de variables**, en el que se analizan los atributos del conjunto de datos para identificar aquellos que aportan información relevante al modelo.

Como se observó previamente en la **matriz de correlación** y en los **gráficos de dispersión**, algunas variables de este conjunto de datos presentan una fuerte correlación lineal entre sí. Mantener todas estas variables no resulta conveniente, ya que contienen información redundante.

Por ello, se eliminaron aquellas variables cuyo coeficiente de correlación de Pearson cumplía la condición:

\[
\rho \geq 0.92
\]

En particular, se eliminaron las siguientes variables: `BILL_AMT2`, `BILL_AMT3`, `BILL_AMT4`, `BILL_AMT5`, `BILL_AMT6`.

### Análisis de Componentes Principales (*Principal Component Analysis, PCA*)

Otra forma de obtener un conjunto de datos de menor dimensionalidad cuando existe **multicolinealidad** entre las variables es mediante la aplicación del **Análisis de Componentes Principales (PCA)**.

El PCA es una técnica de **aprendizaje no supervisado** que realiza una transformación lineal del espacio de variables originales mediante una **matriz ortogonal**, con el objetivo de obtener un nuevo conjunto de componentes ordenados de acuerdo con la cantidad de varianza que explican, desde la mayor hasta la menor.

La intuición detrás de este método es que, si la dirección de máxima variabilidad de los datos no está representada directamente por las variables originales, es posible construir nuevas variables (componentes principales) que capturen esa variabilidad de manera más eficiente. En consecuencia, estos nuevos componentes suelen contener una mayor cantidad de información relevante que las variables originales consideradas de forma individual.

Para encontrar la dirección que maximiza la varianza de los datos, se resuelve el siguiente problema de optimización:
  
$$ X\in \mathbb{R}^{n\times d} , dataset\ centered\ in\ zero $$

$$ \Sigma :=\frac{X^{T} X}{n-1},\ sample\ covariance\ matrix $$ 

$$ find\ \ \overrightarrow{z_{1}} :=a_{1} \overrightarrow{e_{1}} +...+a_{d}\overrightarrow{e_{d}} $$ 

$$ s.t.: \ \ \overrightarrow{z_{1}} =\underset{\vec{z}{1}}{argmax} \ \ \vec{z_{1}}^{T} \ \Sigma \ \ \vec{z_{1}} , \ \ \ \Vert \vec{z_{1}}\Vert =1 $$

De forma recursiva, las demás dimensiones se calculan siguiendo el mismo procedimiento, imponiendo además la restricción de que cada nueva dimensión sea **ortogonal** a las previamente obtenidas. Estas nuevas variables se conocen como **componentes principales** (*Principal Components, PCs*).

Puede demostrarse que el problema de optimización descrito anteriormente es equivalente a calcular la **descomposición en valores propios (eigendecomposition)** de la matriz de covarianzas $\Sigma$ y seleccionar sus **vectores propios (eigenvectors)** como componentes principales, que es precisamente el procedimiento utilizado en la práctica para implementar el PCA.

En particular, dado que $\Sigma$ es una **matriz simétrica semidefinida positiva**, siempre es posible diagonalizarla mediante una matriz ortogonal $P$, cuyos vectores columna corresponden a los vectores propios de $\Sigma$. Como resultado, se obtiene la siguiente matriz diagonal $\Lambda$:
 
$$ \Lambda =\ \begin{pmatrix} \lambda_{1} & 0 & \cdots & 0\\\ 0 & \lambda_{2} & \ddots & \vdots \\\ \vdots & \ddots & \ddots & 0\\\ 0 & \cdots & 0 & \lambda_{d} \end{pmatrix} \ ,\ \lambda_{1} \geqslant \lambda_{2} \geqslant ...\geqslant \lambda_{d} \geqslant 0,\ \lambda_{i} \in \mathbb{R}^{+} $$

donde $\lambda_{i}$ son los valores propios (*eigenvalues*) de $\Sigma$, o de manera equivalente, las varianzas de las nuevas características obtenidas. Nótese que, dado que $\Sigma$ y $\Lambda$ son matrices semejantes, ambas tienen la misma traza, lo que significa que la varianza total inicial de las características del conjunto de datos no cambia, sino que simplemente se redistribuye sobre nuevos ejes (lo cual es de esperarse, ya que únicamente se está realizando una rotación rígida del espacio). En particular, podemos calcular la proporción de varianza explicada por cada nuevo componente principal con respecto a la varianza total del conjunto de datos.

Finalmente, observe que $\Lambda$ es ahora la matriz de covarianza en la nueva base obtenida y, al ser una matriz diagonal, todas las nuevas características resultan ser linealmente incorrelacionadas. Por ello, es posible seleccionar únicamente un subconjunto de los primeros componentes principales para reducir la dimensionalidad inicial del conjunto de datos con una pérdida mínima de información.

En este estudio, se aplicó el Análisis de Componentes Principales (PCA) al conjunto de datos de **Default de Tarjetas de Crédito** con el fin de abordar el problema de la multicolinealidad y reducir el número de dimensiones. La siguiente figura muestra cómo se ha redistribuido la varianza entre las nuevas características extraídas.

<p align = "center">
<img height="300" src="https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/blob/main/images/PCACumulative.svg">
</p>
<p align = "center">
Explained variance ratio of each principle component, together with the  
cumulative variance explained as more dimensions are considered.
</p>

El gráfico anterior muestra la **proporción de varianza explicada** por los componentes principales. La línea verde representa la proporción de varianza explicada por cada componente principal de forma individual, mientras que la línea naranja muestra la **proporción de varianza explicada acumulada**, es decir, la suma de la varianza explicada por los componentes principales anteriores.

Este gráfico resulta útil para decidir cuántos componentes principales conservar. La siguiente tabla presenta algunos valores de interés relacionados con el número de componentes seleccionados.

| Number of PCs | Cumulative Variance Explained |
| --- | --- |
| 4   | 89.7 % |
| 6   | 95.3 % |
| 8   | 97.1 % |
| 12  | 99.0 % |

Por ejemplo, los primeros **12 componentes principales (PCs)** son capaces de capturar aproximadamente el **99 % de la varianza total** del conjunto de datos. Por esta razón, se decidió conservar únicamente estos 12 componentes para el análisis posterior.

_**Nota:** El PCA se ajusta únicamente utilizando el conjunto de entrenamiento para evitar la fuga de información (*data leakage*) desde el conjunto de prueba hacia el proceso de entrenamiento del modelo._

``` python
from sklearn.decomposition import PCA

pca = PCA(n_components=len(X_train.columns))
pca.fit(X_train) # not on the whole dataset
```

|     | PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 | PC9 | PC10 | PC11 | PC12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0   | 1.15 | 0.53 | 0.42 | -0.20 | 0.05 | -0.06 | -0.03 | 8.00e-03 | -1.02e-01 | -0.12 | 0.13 | -9.53e-03 |
| 1   | 0.17 | -0.67 | -0.82 | -0.04 | 0.04 | 0.37 | -0.10 | 4.24e-02 | -4.01e-03 | -0.07 | -0.01 | -8.76e-03 |
| 2   | -0.87 | -0.29 | 0.56 | 0.85 | -0.28 | -0.02 | -0.27 | 1.47e-01 | 2.03e-01 | -0.08 | -0.11 | -2.49e-02 |
| 3   | -0.52 | -0.38 | 0.93 | -0.31 | 0.71 | -0.10 | 0.09 | -7.04e-02 | 2.04e-02 | -0.09 | -0.05 | 3.32e-03 |
| 4   | -1.07 | -0.25 | -0.30 | -0.23 | 0.24 | 0.39 | -0.03 | -9.61e-03 | 3.43e-02 | -0.20 | 0.11 | -1.48e-04 |

### Sobremuestreo y submuestreo

Cuando se trabaja con un **conjunto de datos desbalanceado** en la variable objetivo, resulta más difícil para la mayoría de los algoritmos de aprendizaje automático aprender de manera eficiente todas las clases. En efecto, el proceso de entrenamiento puede sesgarse hacia una determinada clase si la distribución del conjunto de datos está fuertemente desequilibrada.

En el caso específico del conjunto de datos de clientes de tarjetas de crédito, solo alrededor del **22,1 %** de las observaciones están etiquetadas como clientes en incumplimiento (*defaulters*, `y = 1`).

|     | Number of rows | Percentage |
| --- | --- | --- |
| Non-defaulters (class=0) | 17246 | 77.68 % |
| Defaulters (class=1) | 4954 | 22.32 % |

Si bien la solución más evidente y deseable sería recopilar más datos reales, las técnicas de **sobremuestreo (_oversampling_)** y **submuestreo (_undersampling_)** pueden ser de gran utilidad en este tipo de situaciones.

Para ambas técnicas existe un enfoque básico conocido como **sobremuestreo (o submuestreo) aleatorio (_random oversampling/undersampling_)**, en el cual el conjunto de entrenamiento se incrementa (o reduce) mediante copias aleatorias de las observaciones hasta obtener la misma proporción entre la clase minoritaria y la clase mayoritaria.

### Sobremuestreo: Técnica de Sobremuestreo Sintético de la Clase Minoritaria (SMOTE)

El algoritmo **SMOTE (Synthetic Minority Over-sampling Technique)** incrementa artificialmente la clase minoritaria mediante la generación de nuevas muestras sintéticas, ubicadas aleatoriamente entre una observación y sus *k* vecinos más cercanos.

En otras palabras, dado un conjunto limitado de observaciones pertenecientes a la clase que se desea aumentar, el algoritmo traza líneas en el espacio de características de alta dimensión que conectan dichas observaciones y genera nuevas muestras sintéticas a lo largo de esas líneas.

<p align = "center">
<img src="https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/blob/main/images/oversampling-smote.png?raw=true">
</p>
<p align = "center">
Example of SMOTE application on a small dataset
</p>
  
Por último, es importante señalar que una muestra sintética **nunca debe utilizarse como muestra de prueba (test)**, ya que técnicamente nunca ha existido en el mundo real y, por lo tanto, no pertenece formalmente a la distribución objetivo. Además, se considera que el conjunto de prueba estratificado representa correctamente la distribución del mundo real, por lo que aumentar artificialmente una parte de este conduciría a una evaluación sesgada y poco representativa.

``` python
from imblearn.over_sampling import SMOTE

clf = Classifier(**best_config)
#oversampling
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
```

### Submuestreo: Cluster Centroids

**Cluster Centroids** utiliza el algoritmo **K-means** para realizar el submuestreo de la clase mayoritaria. Tras identificar los centroides de los clústeres formados por las observaciones de dicha clase, el algoritmo selecciona las instancias pertenecientes a cada clúster (etiquetadas con la clase mayoritaria) que se encuentran más alejadas del centroide en el espacio de características. Estas observaciones se consideran las menos representativas o menos importantes.

Por el contrario, la instancia de la clase mayoritaria que se encuentra más próxima al centroide del clúster se considera la más representativa o importante. De esta manera, las instancias de la clase mayoritaria se eliminan en función de su importancia, conservando aquellas que mejor representan la distribución de los datos.

<p align = "center">
<img height="300" src="https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/blob/main/images/undersampling-cluster_centroids.png?raw=true">
</p>
<p align = "center">
Example of Cluster Centroids application on a trivial dataset
</p>

En particular, dadas dos clases con respectivamente $N$ y $M$ observaciones, donde $N < M$, el algoritmo entrena un modelo **K-Means** utilizando únicamente los puntos pertenecientes a la clase mayoritaria, con $k = N$, es decir, el número de observaciones de la clase minoritaria.

Posteriormente, para cada clúster, reemplaza los puntos de la clase mayoritaria por un nuevo punto cuyas coordenadas corresponden al centroide del clúster. De esta manera, se reduce el número de observaciones de la clase mayoritaria formando clústeres y sustituyéndolos por sus centroides.

## Algoritmos de clasificación

En esta sección se presenta una descripción de los algoritmos utilizados. Para cada uno de ellos, se ajustan diferentes hiperparámetros con el fin de encontrar la configuración que ofrece el mejor desempeño. Una vez identificada, el modelo entrenado con dicha configuración se utiliza para predecir las etiquetas de clase sobre el conjunto de prueba, combinando las distintas técnicas de preprocesamiento de datos presentadas anteriormente para analizar las diferencias en el rendimiento.

La mejor configuración se selecciona comparando diversas métricas de evaluación, principalmente el **F1-score**, ya que la exactitud (*accuracy*) en conjuntos de datos desbalanceados puede presentar valores elevados incluso cuando la clase minoritaria no se clasifica correctamente.

Cada algoritmo se entrena utilizando distintas combinaciones de técnicas de preprocesamiento de datos:

* Principal Component Analysis (PCA)
* PCA + SMOTE (oversampling)
* PCA + Cluster Centroids (undersampling)

Los algoritmos considerados son:

* Logistic Regression
* Random forest (and Decision Trees)
* SVM (both Linear and RBF)

## Métodos de evaluación

Aunque la teoría del aprendizaje garantiza que es posible obtener un modelo **probablemente aproximadamente correcto** (*Probably Approximately Correct, PAC*), en la práctica es necesario evaluar su desempeño para comprobar qué tan bien generaliza a nuevos datos.

Una forma común de hacerlo consiste en dividir el conjunto de datos original en dos partes: un **conjunto de entrenamiento** (*training set*) y un **conjunto de prueba** (*test set*). El modelo se entrena únicamente con los datos de entrenamiento y posteriormente se evalúa sobre datos que no ha visto previamente. Esta técnica se conoce como **_Holdout_**.

Por otro lado, la **_Validación Cruzada_** (*Cross Validation*) es un método de remuestreo que permite generar múltiples particiones del conjunto de entrenamiento con el fin de obtener una estimación más robusta del rendimiento del modelo.

Mediante la validación cruzada se busca verificar si el modelo es capaz de predecir correctamente las etiquetas de observaciones que no ha visto durante el entrenamiento. Para ello, el conjunto de datos se divide en $k$ subconjuntos (*folds*):

* Se utilizan $k-1$ *folds* para entrenar el modelo; en conjunto forman el **conjunto de entrenamiento**.
* El *fold* restante constituye el **conjunto de validación**, sobre el cual se evalúa el desempeño del modelo.

Este procedimiento se repite **$k$ veces**, de modo que en cada iteración cambia el subconjunto utilizado para la validación. Finalmente, el rendimiento del modelo se obtiene calculando el promedio de las **$k$ evaluaciones**, lo que reduce la variabilidad de la estimación y proporciona una medida más confiable de la capacidad de generalización del modelo.

<p align = "center">
<img height="300" src="https://upload.wikimedia.org/wikipedia/commons/c/c7/LOOCV.gif">
</p>
<p align = "center">
Ejemplo de Validación Cruzada (k=8), donde los cuadrados rojos y azules representan, respectivamente, los conjuntos de entrenamiento y validación.
</p>

El caso particular en el que $k$ es igual a $m$, el número total de observaciones, se conoce como **_Leave-One-Out Cross Validation (LOOCV)_**. Este procedimiento proporciona una muy buena estimación del error real del modelo; sin embargo, su principal desventaja es su elevado costo computacional, ya que requiere entrenar el modelo una vez por cada observación del conjunto de datos.

También es posible combinar las técnicas de **Holdout** y **Validación Cruzada** mediante el enfoque conocido como **_k-fold Cross Validation_**. En este caso, el conjunto de datos se divide inicialmente, de manera estratificada, en un **conjunto de entrenamiento** y un **conjunto de prueba**. Posteriormente, el conjunto de entrenamiento se vuelve a dividir en subconjuntos de entrenamiento y validación utilizando nuevamente una partición estratificada (`stratify=True`).

## Métricas de evaluación

En un problema de clasificación, el desempeño del modelo puede evaluarse mediante diferentes métricas. Para comprender correctamente estas métricas, es útil revisar primero algunos conceptos fundamentales.

|     | Predicted (1) | Predicted (0) |
| --- | --- | --- |
| **Actual (1)** | TP  | FN  |
| **Actual (0)** | FP  | TN  |

aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
* _Verdadero Positivo (TP)_: muestras para las cuales la predicción es positiva y la clase real también es positiva.
* _Falso Positivo (FP)_: muestras para las cuales la predicción es positiva, pero la clase real es negativa.
* _Verdadero Negativo (TN)_: muestras para las cuales la predicción es negativa y la clase real también es negativa.
* _Falso Negativo (FN)_: muestras para las cuales la predicción es negativa, pero la clase real es positiva.

Algunas de las métricas de evaluación más utilizadas son:

* **_Exactitud (Accuracy)_**: proporción de predicciones correctas respecto al número total de muestras clasificadas.
  <br>
  $$
  Accuracy = \frac{\text{Número de muestras correctamente clasificadas}}{\text{Número total de muestras evaluadas}}
  = \frac{TP + TN}{TP + FP + TN + FN}
  $$

* **_Precisión (Precision)_**: mide la proporción de instancias correctamente clasificadas como positivas entre todas las clasificadas como positivas. Es una métrica adecuada cuando el objetivo es minimizar los **falsos positivos**.
  <br>
  $$
  Precision(c) =
  \frac{\text{Número de muestras correctamente asignadas a la clase } c}
  {\text{Número de muestras asignadas a la clase } c}
  =
  \frac{TP}{TP + FP}
  $$

* **_Sensibilidad (Recall)_**: mide cuántos de los casos realmente positivos son identificados correctamente por el modelo. Es una métrica apropiada cuando el objetivo es minimizar los **falsos negativos**.
  <br>
  $$
  Recall(c) =
  \frac{\text{Número de muestras correctamente asignadas a la clase } c}
  {\text{Número de muestras que realmente pertenecen a la clase } c}
  =
  \frac{TP}{TP + FN}
  $$

* **_F1-score_**: corresponde a la media armónica entre la precisión y la sensibilidad (_Recall_).
  <br>
  $$
  F1\text{-}score(c) =
  \frac{2 \cdot Precision(c) \cdot Recall(c)}
  {Precision(c) + Recall(c)}
  $$

Las tres últimas métricas son **específicas por clase (_class-specific_)**, por lo que una métrica global puede obtenerse promediando sus valores entre todas las clases del conjunto de datos. En particular, estas métricas son más adecuadas cuando se trabaja con **conjuntos de datos desbalanceados**, ya que permiten evaluar el desempeño del modelo sobre cada clase de forma independiente. En estos casos, utilizar únicamente la exactitud (_Accuracy_) puede resultar engañoso, ya que es posible obtener un valor elevado incluso cuando la clase minoritaria es clasificada de manera deficiente.

En este proyecto, el objetivo principal es identificar a los clientes que podrían incurrir en **incumplimiento de pago (_default_)**, por lo que la **clase positiva** representa el mayor interés para el clasificador. Como se mencionó anteriormente, en las siguientes secciones se utilizará principalmente el **F1-score** para seleccionar la mejor configuración de cada modelo.

## Modelos

### Regresión Logística

La **Regresión Logística** es un algoritmo paramétrico y discriminativo para clasificación binaria. Su nombre se debe a que puede interpretarse como parte del **Modelo Lineal Generalizado _(Generalized Linear Model, GLM)_**, donde la variable respuesta sigue una distribución de Bernoulli. En particular, el modelo asume que los predictores están relacionados con la media de la variable respuesta \(p_i\) de la siguiente manera:

$$ observation\ ( x_{i} , y_{i}) \ ,\ y_{i} \ realization\ of\ Y_{i} \sim Bernoulli( p_{i}( x_{i}))$$

<p align = "center">
<img height="300" src="https://upload.wikimedia.org/wikipedia/commons/5/53/Sigmoid-function-2.svg">
</p>
<p align = "center">
Logits function
</p>

El algoritmo busca estimar la **Máxima Verosimilitud (Maximum Likelihood Estimation, MLE)** —o la **Máxima Probabilidad a Posteriori (MAP)** cuando se incorpora un término de regularización— de la media de las variables respuesta, ajustando los parámetros \(w\) y asumiendo que las muestras son **independientes e idénticamente distribuidas (i.i.d.)**. A diferencia de la regresión por mínimos cuadrados ordinarios, este problema de optimización no posee una solución en forma cerrada, por lo que se resuelve mediante métodos numéricos iterativos.

$$ \underset{w}{\max} \ \\prod ^{n_{i}} \ p_{i}( x_{i} |w)^{y_{i}} \ *\ ( 1-p_{i}( x_{i} |w))^{1-y_{i}} $$

Con frecuencia se incorpora un **término de regularización** para evitar que los coeficientes alcancen valores excesivamente grandes, lo que podría provocar **sobreajuste (_overfitting_)** sobre los datos de entrenamiento. El hiperparámetro \(C\) actúa sobre este término de penalización y controla de manera **inversa** la intensidad de la regularización en la función objetivo: valores pequeños de \(C\) implican una regularización más fuerte, mientras que valores grandes reducen su efecto.

```python
# C values tuned:
params = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10]}

#Best configuration found C=0.01
#Data prepocessing: PCA+SMOTE
```

Como resultado, un modelo de **Regresión Logística** produce como salida la **probabilidad** de que una observación pertenezca a la clase positiva (clase 1), proporcionando además una medida de confianza para cada predicción. Asimismo, los coeficientes estimados por el modelo permiten interpretar qué variables contribuyen en mayor medida a aumentar o disminuir la probabilidad de pertenecer a la clase positiva.

### Árbol de Decisión

Los **Árboles de Decisión** son uno de los modelos de aprendizaje automático más intuitivos e interpretables. Su objetivo es predecir la variable objetivo aprendiendo reglas de decisión simples inferidas a partir de los datos.

En cada división, el árbol selecciona la característica que mejor separa las observaciones entre las distintas clases utilizando la **impureza de Gini (_Gini Impurity_)**: <br><br>

$$ GINI(t) =\sum_{i=1}^{C} p_{t}(k)(1-p_{t}(k)) = 1-\sum_{i=1}^{C} p_{t}(k)^{2} $$

donde \(p_t(k)\) representa la frecuencia de la clase \(k\) en el nodo \(t\), y \(C\) es el número total de clases. Cuanto menor sea el valor de esta medida, menor será la impureza del nodo y, por tanto, más homogéneas serán las observaciones que contiene.

A partir de los datos de entrenamiento se construye un árbol completo, que posteriormente se utiliza para clasificar nuevas observaciones. Sin embargo, para evitar el **sobreajuste (_overfitting_)**, es habitual **podar (_prune_)** el árbol antes de que alcance su crecimiento máximo. La predicción final se realiza mediante **votación mayoritaria** entre las observaciones contenidas en cada nodo hoja.

Cabe destacar que los árboles de decisión **no pueden manejar valores faltantes de forma nativa**, aunque presentan la ventaja de ser **insensibles al escalado de las variables**, ya que cada característica se evalúa de manera independiente durante el proceso de partición.

```python
# Best configuration found 
# Data prepocessing: PCA+SMOTE
params = {'criterion': 'gini', 'max_depth': 5, 'min\_impurity\_decrease': 0.0, 'splitter': 'best'}
```

Decision trees are however considered weak learners in the majority of cases, and ensemble techniques such as _bagging_ (Random forests) or _boosting_ are in practice applied on them in order to reach higher performances and more robust solutions.

### Random Forest

Random Forests are an ensemble method made out of multiple decision trees. Each individual tree in the random forest makes a class prediction and the class with the most votes is assigned to the sample taken into account. The intuition behind this method is that a large number of relatively **uncorrelated** models (trees) operating as a committee will outperform any of the individual constituent model. Uncorrelation is the key of Random Forests better performances, and this is ensured through two methods:

1.  Bagging (boostrap aggregation) to promote variance between each internal model, each individual tree is trained on a sample of the same size of the training dataset randomly chosen with replacement;
2.  In addition, trees are further decorrelated by choosing only a subset of features at each split of each tree (usually $\sqrt{ Numfeatures }$).

Overall, each tree is then trained on a sample of the same size of the training dataset chosen with replacement, and their splits consider different features every time. This way, even if the decision trees are somewhat overfitting the data and are unstable (high variance), the overall prediction will be statistically more robust (low variance) because of Central Limit Theorem. The prediction is indeed performed by picking the majority class out of all trees.

* The choice of the imbalancing technique affect the performance of the Random forest. Indeed, the oversampling technique is preferable.
* Also, the hyperparameter that regulates the number of estimators seems to be irrelevant over the performance of the algorithms.

```python
# Best configuration found 
# Data prepocessing: PCA+SMOTE
params = {'criterion': 'gini', 'max_features': 'sqrt', 'n_estimators': 100}
```

Finally, tree-based model can be used as an alternative method for feature selection. They indeed provide a feature importance measure by means of how much the gini index has been affected on the various splits with the given feature. Compared to decision tree, random forests guarantee robustness and are less prone to overfitting.  

<p align = "center">
<img height="300" src="https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/blob/main/LaTex/featureRF.png">
</p>
<p align = "center">
Feature importance computed by the random forest on each of the attributes</p>

### Máquina de Vectores de Soporte (Support Vector Machine, SVM)

La **Máquina de Vectores de Soporte (Support Vector Machine, SVM)** es un algoritmo paramétrico de clasificación lineal cuyo objetivo es separar dos clases mediante un **hiperplano** en el espacio de características. Una vez definido dicho hiperplano, la clasificación de nuevas observaciones consiste simplemente en determinar en cuál de sus lados se encuentra cada punto.

Cuando existen varios hiperplanos capaces de separar correctamente las clases, el algoritmo selecciona aquel que **maximiza el margen**, es decir, la distancia entre el hiperplano y los puntos de entrenamiento más cercanos de cada clase.

Los puntos que se encuentran sobre los márgenes reciben el nombre de **vectores de soporte (_support vectors_)**, ya que son los que determinan completamente la posición del hiperplano óptimo.

<p align = "center">
<img height="300" src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/800px-SVM_margin.png">
</p>
<p align = "center">

**Hiperplano de máximo margen** y sus márgenes para una SVM entrenada con muestras de dos clases. Las observaciones ubicadas sobre los márgenes reciben el nombre de **vectores de soporte (_support vectors_)**.

De manera más formal, el algoritmo busca encontrar los parámetros \(\vec{w}\) y \(b\) de forma que el hiperplano $w^{T}x+b=0$ sea el **hiperplano de máximo margen**, es decir, aquel que maximiza la distancia entre el hiperplano y la observación más cercana \(\overrightarrow{x_i}\) de cualquiera de las dos clases.

Si se impone que todas las observaciones satisfagan la restricción $|w^{T}x+b|\geq1$ de acuerdo con su etiqueta de clase, entonces el problema de optimización puede reformularse como la búsqueda del vector de pesos con la **menor norma euclídea** posible, es decir, encontrar el menor valor de $\|w\|$:

#### Problema de Optimización Primal (Hard Margin)

$$ \underset{w,b}{min} \tfrac{1}{2}\Vert w\Vert ^{2} \\ \\\ s.t.\ y_{i}\left( w^{T} x_{i} +b\right) \geqslant 1,\ \forall i $$

La función de decisión para un problema de clasificación binaria con etiquetas \(\{-1,1\}\) se define como: $x \mapsto \operatorname{sign}\left(w^{T}x+b\right)$. Desde una perspectiva geométrica, el **hiperplano óptimo** queda completamente determinado por las observaciones \(\overrightarrow{x_i}\) que se encuentran exactamente sobre los límites del margen, es decir, aquellas que satisfacen: $|w^{T}x_i+b|=1$. Estas observaciones reciben el nombre de **vectores de soporte (_support vectors_)**, ya que son las únicas que determinan la posición del hiperplano de separación.

Este modelo se conoce como **SVM de margen duro (_Hard-margin SVM_)**, en el cual se asume que las clases son **linealmente separables**. Si esta condición no se cumple, el problema de optimización no tiene una solución factible.

Para extender el modelo a situaciones donde las clases no pueden separarse perfectamente mediante un hiperplano, se introduce la **función de pérdida Hinge (_Hinge Loss_)**, la cual penaliza las observaciones que quedan dentro del margen de separación:

$$ L_{hinge} := \max\left( 0, 1-y_{i}\left( w^{T} x_{i} +b\right)\right) $$

A partir de esta función de pérdida, el problema de optimización anterior se generaliza al caso de **margen suave (_Soft-margin SVM_)**, que puede interpretarse como un problema de **Minimización del Riesgo Empírico (Empirical Risk Minimization, ERM)** utilizando la pérdida *Hinge* junto con un término de **regularización L2**.

#### Problema de Optimización Primal (Soft Margin)

$$ \\underset{w,b,\xi}{min} \ \ \\tfrac{1}{2}\\Vert w\\Vert ^{2} \ +C\\sum_{i}^{n} \\xi_{i} $$

$$ s.t.:\\\ y_{i}\\left( w^{T} x_{i} +b\\right) \\geqslant 1-\\xi_{i} \ ,\ \\forall i\\\ $$

$$\\xi_{i} \\geqslant 0\ ,\ \\forall i\\\ $$

Donde $\xi$ es una **variable de holgura (*slack variable*)**, introducida para suavizar la restricción de clasificación, permitiendo que el modelo cometa un cierto número de errores y mantenga el margen lo más amplio posible.

La cantidad de clasificaciones erróneas permitidas está controlada por el hiperparámetro $C$, el cual regula la intensidad del término de **pérdida Hinge (*Hinge Loss*)** incorporado al modelo. Un valor elevado de $C$ conduce a una mayor minimización de la pérdida Hinge, obligando a que las observaciones de entrenamiento sean clasificadas correctamente. Por el contrario, un valor reducido de $C$ impone una regularización más fuerte, permitiendo un mayor número de errores de clasificación y dando lugar a un margen de mayor amplitud.

El **problema dual de Lagrange** se define de la siguiente manera:

**Problema dual de optimización (Soft Margin)**

$$ \underset{\alpha}{max} \ \ \sum_{i}^{n} \alpha_{i} -\tfrac{1}{2}\sum_{i,j} \alpha_{i} \alpha_{j} y_{i} y_{j}\left( x_{i}^{T} x_{j}\right) $$

$$ s.t.: \\ \sum \alpha_{i} y_{i} =0\ \ \land \ 0\leqslant \alpha_{i} \leqslant C,\ \forall i $$

Al resolver el problema dual (por ejemplo, mediante **Programación Cuadrática**), se obtiene $$w = \sum_{i}^{n} \alpha_{i} y_{i} x_{i}$$ es decir, el vector $w$ se expresa como una combinación lineal de los datos de entrenamiento. En particular, únicamente los puntos que se encuentran sobre el margen o dentro de él tendrán un valor $\alpha_i \neq 0$; es decir, solo los denominados **vectores de soporte (support vectors)** influyen en la función de decisión. Esta última también puede expresarse como $\operatorname{sign}\left(\sum \alpha_{i} y_{i} (x_i^{T}x) + b\right)$ donde $x$ representa una observación de prueba cualquiera. Cabe destacar que, en la versión de **margen suave (soft margin)**, el número de vectores de soporte encontrados suele ser considerablemente mayor.

A partir de estos fundamentos, el modelo **SVM** puede extenderse tanto a problemas de **clasificación multiclase** como a **clases no linealmente separables**. En este último caso, las clases pueden seguir un patrón no lineal, por lo que un hiperplano puede no ser la mejor alternativa para representar los datos. Sin embargo, el conjunto de datos puede llegar a ser linealmente separable al proyectarlo en un espacio de mayor dimensión mediante una función no lineal $\varphi$, de modo que las observaciones de entrenamiento se transforman como $$x_i \in \mathbb{R}^{d} \mapsto \varphi(x_i) \in \mathbb{R}^{d'}, \qquad d' > d$$.

Aunque es posible aplicar el algoritmo **SVM** directamente en este espacio de mayor dimensión utilizando una transformación de características (*feature map*), en la práctica se emplea un enfoque diferente que aprovecha una propiedad fundamental de las máquinas de vectores de soporte. En efecto, tanto el problema de optimización dual como la función de decisión dependen únicamente del **producto escalar** entre las observaciones. Siempre que esta propiedad se mantenga, no es necesario conocer explícitamente el espacio de mayor dimensión; basta con disponer de una función capaz de calcular el producto escalar en dicho espacio transformado.

Con este propósito, se introduce el concepto de **función kernel** $K$, definida como:

$$ K\ :\ \\mathcal{X} \\times \\mathcal{X}\\rightarrow \ \\mathbb{R}\\\ ( x,\ x^{\\prime} ) \\mapsto K( x,\ x^{\\prime} ):=\\varphi ( x)^{T} \\varphi ( x^{\\prime} ) $$

para alguna función de transformación de características $\phi$. De esta manera, es posible aplicar diferentes funciones **kernel** al algoritmo **SVM** sin conocer explícitamente cuál es la transformación de características utilizada.

La eficacia de este enfoque se fundamenta en el **Teorema de Mercer**, el cual proporciona una condición suficiente para que una función sea un **kernel** asociado a alguna transformación desconocida $\varphi(\cdot)$.

Las funciones **kernel** más utilizadas son:

* Linear Kernel: $K( x,x^{\\prime} ) =x^{T} x^{\\prime}$
* Gaussian RBF Kernel: $K(x,x^{\\prime} )\ =\ \\exp (-\\gamma \\| x-x^{\\prime} \\| ^{2} ),\ \\gamma >0$
* Polynomial Kernel: $K( x,x^{\\prime} ) \ =\ \\left( x^{T} x^{\\prime} +c\\right)^{d}$

En particular, para el **kernel Gaussiano RBF**, la función de decisión calcula una medida de similitud entre la observación de prueba y los vectores de soporte (ponderada por $\alpha_i$ y $y_i$), la cual es inversamente proporcional a la distancia euclidiana entre ambos. Como resultado, la etiqueta predicha está determinada por los vectores de soporte más cercanos. El nuevo hiperparámetro **gamma** puede ajustarse para regular cuántos vectores de soporte influyen significativamente en la función de decisión (generalmente se establece como **auto**: $1/n\_features$ o **scale**: $1/(n\_features \times X.var())$).

La función de decisión queda definida como:

$$
\operatorname{sign}\left(\sum \alpha_{i} y_{i} K(x_{i}, x) + b\right).
$$

Finalmente, cabe destacar que, aunque resulta conveniente utilizar una función kernel, el hiperplano en el espacio de mayor dimensión no se calcula explícitamente cuando se aplica el **truco del kernel (kernel trick)**; es decir, los coeficientes del vector $w$ permanecen desconocidos.

En este estudio se evaluaron tanto el **kernel lineal** como el **kernel Gaussiano RBF**. Asimismo, el hiperparámetro de regularización $C$ fue optimizado mediante ajuste de hiperparámetros para encontrar el modelo con el mejor desempeño sobre el conjunto de entrenamiento.

```python
#Best configuration found 
#polynomial kernel
params = {'C': 100, 'kernel': 'poly'}
#kernel RBF
params = {'C': 1, 'kernel': 'rbf'}
```

|     | Precision | Recall | F1-score |
| --- | --- | --- | --- |
| SVM POLY + SMOTE | 0.64 | 0.28 | 0.39 |
| SVM RBF + SMOTE | 0.54 | 0.52 | 0.53 |
| SVM POLY + Cluster Centroids | 0.68 | 0.18 | 0.29 |
| SVM RBF + Cluster Centroids | 0.58 | 0.46 | 0.51 |

_Puntuaciones obtenidas en el conjunto de validación para la mejor configuración encontrada con ambos kernels y ambas técnicas de preprocesamiento_

* Las métricas obtenidas son muy similares independientemente de la técnica de preprocesamiento utilizada, aunque **SMOTE** presenta un rendimiento ligeramente superior.
* En general, al utilizar un **kernel polinómico**, el **recall** de la clase positiva es inferior al obtenido en los demás casos.
* La versión de **SVM con kernel gaussiano (RBF)** supera en rendimiento a la versión con **kernel polinómico**.

En conclusión, también en este caso los resultados podrían mejorarse para obtener predicciones más precisas de los clientes que incumplirán con el pago de su tarjeta de crédito.

## Resultados

El siguiente gráfico de barras presenta un resumen de los resultados obtenidos durante la fase de entrenamiento y validación, en la que todos los algoritmos fueron entrenados utilizando sus mejores hiperparámetros (es decir, aquellos que maximizan el **F1-score** para la clase positiva), empleando ambas técnicas propuestas para abordar el problema del desbalance de clases.

![](https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/blob/main/images/results.png)

_Comparación del F1-score entre diferentes algoritmos_

### Discusión

* La **Regresión Logística** y las **Máquinas de Vectores de Soporte (Support Vector Machines, SVM)** mantienen el mismo rendimiento independientemente de la técnica utilizada para tratar el desbalance de clases. En cambio, esto no ocurre con **Random Forest**, probablemente porque los árboles de decisión requieren una gran cantidad de datos para lograr un buen desempeño, mientras que la técnica de submuestreo (*undersampling*) reduce la cantidad de datos disponibles para el entrenamiento.
* En términos generales, las técnicas de **sobremuestreo (*oversampling*)** obtienen un rendimiento ligeramente superior al de las técnicas de **submuestreo (*undersampling*)**.
* La **clase positiva** es la más difícil de clasificar y ninguno de los modelos seleccionados parece ser capaz de capturar completamente la complejidad del problema.
* De forma empírica, se observó que el **submuestreo (*undersampling*)** requiere, en general, más tiempo de procesamiento que el **sobremuestreo (*oversampling*)** durante la etapa de preprocesamiento. Esto probablemente se deba a que el primero ejecuta el algoritmo **k-means** sobre todo el conjunto de datos para calcular los centroides.

Todos los algoritmos mencionados anteriormente fueron entrenados y optimizados mediante una técnica de **validación cruzada k-fold** utilizando el conjunto de entrenamiento. Las configuraciones de hiperparámetros que obtuvieron el mejor desempeño durante esta etapa fueron seleccionadas para reconstruir los modelos y evaluarlos posteriormente sobre el conjunto de prueba.

Finalmente, en la siguiente tabla se comparan los resultados obtenidos por todos los clasificadores (utilizando su mejor configuración de hiperparámetros) tanto en el conjunto de validación como en el conjunto de prueba. Como puede observarse, las métricas obtenidas en el conjunto de prueba son muy similares a las alcanzadas durante la fase de validación.

| Algorithm | Validation Accuracy | Validation F1-score | Test Accuracy | Test F1-score |
| --- | --- | --- | --- | --- |
| LR + PCA | 0.75 | 0.45 | 0.73 | 0.44 |
| LR + PCA + OS | 0.81 | 0.50 | 0.81 | 0.51 |
| LR + PCA + US | 0.77 | 0.51 | 0.77 | 0.51 |
| RF + PCA | 0.74 | 0.47 | 0.74 | 0.47 |
| RF + PCA + OS | 0.75 | 0.46 | 0.75 | 0.48 |
| RF + PCA + US | 0.47 | 0.38 | 0.47 | 0.37 |
| SVM + PCA | 0.77 | 0.52 | 0.78 | 0.52 |
| SVM + PCA + OS | 0.78 | **0.53** | 0.78 | 0.52 |
| SVM + PCA + US | 0.77 | 0.51 | 0.77 | 0.51 |

_LR: Logistic Regression, RF: Random Forest, SVM: Support Vector Machines,  
OS: Oversampling (SMOTE), US: Undersampling (Cluster Centroids)_

_**Nota:** Las métricas obtenidas sobre el conjunto de prueba nunca deben utilizarse como información adicional para modificar el proceso de entrenamiento del modelo. Su único propósito es realizar una evaluación final e imparcial de su desempeño._

## Conclusiones

En este estudio se analizaron e implementaron diferentes algoritmos de aprendizaje supervisado, presentando además sus fundamentos matemáticos, para construir un modelo de clasificación capaz de predecir si un cliente incumplirá con el pago de su tarjeta de crédito el mes siguiente utilizando el conjunto de datos de UCI.

El preprocesamiento de los datos permitió mejorar ligeramente el rendimiento de los algoritmos en comparación con el uso de los datos originales. En particular, la aplicación de **PCA** produjo resultados muy similares a los obtenidos sin reducción de dimensionalidad, pero con un menor costo computacional.

Asimismo, se combinaron técnicas de sobremuestreo (*oversampling*) y submuestreo (*undersampling*) con PCA para abordar el problema del desbalance del conjunto de datos. Como se esperaba, las técnicas de sobremuestreo obtuvieron un desempeño ligeramente superior al submuestreo, probablemente porque permiten entrenar los modelos con una mayor cantidad de observaciones.

En general, todos los modelos implementados alcanzaron resultados comparables en términos de exactitud (*accuracy*).

### Referencias

\[1\] Default of credit card clients Data Set: UCI dataset [link](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  
  
\[2\] Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480. [link](https://bradzzz.gitbooks.io/ga-seattle-dsi/content/dsi/dsi_05_classification_databases/2.1-lesson/assets/datasets/DefaultCreditCardClients_yeh_2009.pdf)  
  
\[3\] Understanding Machine Learning: From Theory to Algorithms, S. Shalev-Shwartz, S. Ben-David, 2014 [link](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf)