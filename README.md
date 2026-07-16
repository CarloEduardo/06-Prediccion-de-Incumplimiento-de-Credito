# Machine learning para la predicción del incumplimiento de pago de clientes de tarjetas de crédito 💳

Clasificación del incumplimiento de pago de clientes de tarjetas de crédito utilizando el [conjunto de datos de la UCI](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) y técnicas de aprendizaje automático.

Para conocer en detalle las técnicas y metodologías empleadas, consulta el [**informe técnico**](https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/blob/main/Default_of_Credit_Card_Clients.ipynb) o la [**versión web del informe**](https://htmlpreview.github.io/?https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/blob/main/Default_of_Credit_Card_Clients.html).

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

Furthermore, through them it is possible to detect outliers presence (i.e. the single points), with the following rule: every sample located beyond $Q_1 − 1.5\cdot IQR$ and $Q_3 + 1.5\cdot IQR$ is considered an outlier.

_Note that for the purpose a normalization was applied on all the attributes to make them have the same scale, otherwise it would have been impossible to compare them._

### Min-Max scaling for numerical variables

As mentioned above, input variables may have different units so different scales and magnitude; for this reason before drawing a boxplot, a `MinMaxScaler()` is applied in order to scale the features between a range $(0, 1)$. The basic idea behind this rescaling technique is that for every feature, the minimum value of that feature gets transformed into a 0, the maximum value gets transformed into a 1, and every other value gets transformed into a decimal between 0 and 1.  
The transformation is given by the following formula:

$$ X_{scaled} = \frac{(X - X_{min})}{(X_{max} - X_{min})} $$

where $X_{min}$ is the minimum value on the column and $X_{max}$ is the maximum value on the column.

This transformation is applied on **numerical features** only as the categorical variables has been already transformed into one-hot vectors, that rescale the categorical variable in the range $(0,1)$.

### Standardization

Another possible scaling technique is referred to as **standardization**, which transforms each value $x$ of feature $X$ as follows, independently for each column:

$$ x^{\prime} =\frac{x-\mu_{X}}{\sigma_{X}} $$

where $\mu_{X}$ is the sample mean and $\\sigma_{X}$ is the sample standard deviation of feature $X$. This way, we force each attribute to have a new empirical mean value of 0 and variance 1.

_Note that the statistics (e.g. mean and standard deviation for the Standard Scaler) are computed on the training set only, then both datasets are transformed according to them. This is very important as it avoids leaking any information from the test set into the training process, which would be formally incorrect by definition._
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
Boxplots of numerical features scaled according min-max normalization (left) and standardization (right)
</p>

According to the method previously described from the boxplot we can identifty many outliers. It is possible to change the multiplicative coefficient to decide the threshold to overcome to be considered an outlier, but still dubling the coefficent the possible outliers are more than 4000.

Due to the lack of domain knowledge and the very high number of identifiable outliers, no samples have been discarded as outliers.

## Data Preprocessing

### One-hot encoding for categorical variables

An integer encoding may impose some ordinal relationship between categorical variables that does not exist, for this reason a **one-hot encoding** is performed. Categorical variable such as `SEX`, `MARRIAGE` and `EDUCATION` are turned into one-hot variables in order to remove any orders that in this case have no meaning.

A one hot encoding is a representation of categorical variables as binary vectors. This first requires that the categorical values be mapped to integer values. Then, each integer value is represented as a binary vector that is all zero values except the index of the integer, which is marked with a 1.

|     | SEX | EDUCATION_1 | EDUCATION_2 | EDUCATION_3 | EDUCATION_4 | MARRIAGE_1 | MARRIAGE_2 | MARRIAGE_3 | LIMIT_BAL | ... | BILL_AMT4 | BILL_AMT5 | BILL_AMT6 | PAY_AMT1 | PAY_AMT2 | PAY_AMT3 | PAY_AMT4 | PAY_AMT5 | PAY_AMT6 | Default |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0   | 0   | 0   | 1   | 0   | 0   | 1   | 0   | 0   | 0.01 | ... | 0.16 | 0.08 | 0.26 | 0.00e+00 | 4.09e-04 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 1   |
| 1   | 1   | 0   | 1   | 0   | 0   | 0   | 1   | 0   | 0.11 | ... | 0.16 | 0.08 | 0.26 | 0.00e+00 | 5.94e-04 | 1.12e-03 | 1.61e-03 | 0.00e+00 | 3.78e-03 | 1   |
| 2   | 0   | 0   | 1   | 0   | 0   | 0   | 1   | 0   | 0.08 | ... | 0.17 | 0.10 | 0.27 | 1.74e-03 | 8.91e-04 | 1.12e-03 | 1.61e-03 | 2.34e-03 | 9.46e-03 | 0   |
| 3   | 1   | 0   | 1   | 0   | 0   | 1   | 0   | 0   | 0.04 | ... | 0.19 | 0.11 | 0.28 | 2.29e-03 | 1.20e-03 | 1.34e-03 | 1.77e-03 | 2.51e-03 | 1.89e-03 | 0   |
| 4   | 0   | 0   | 1   | 0   | 0   | 1   | 0   | 0   | 0.04 | ... | 0.18 | 0.10 | 0.28 | 2.29e-03 | 2.18e-02 | 1.12e-02 | 1.45e-02 | 1.62e-03 | 1.28e-03 | 0   |

5 rows × 30 columns

### Feature Scaling

As previously explained, a `MinMaxScaler()` has been already applied to all the numerical features.

### Train-test split

The dataset is divided in training set and test set, with the proportion 3:1. Given the imbalance fashion in the original dataset a **stratified sampling** strategy has been applied during the split, so that the proportion of defaulters/non-defaulters clients in each of the final sets would approximately be the same as the initial one. This approach is desirable every time we have a highly _unbalanced dataset_ as in our case.  
The final split obtained is distributed as follows:

|     | Non-defaulter | Defaulters | Total |
| --- | --- | --- | --- |
| **Training set** | 17246 | 4954 | 22200 |
| **Test set** | 5750 | 1651 | 7401 |

_Note that the split has been randomly generated but kept the same for all the following methods, in order to to make evaluations statistically meaningful and provide consistent results among the different classifiers._
```python
random_state = 42
```

### Dimensionality reduction

As already mentioned above, generally speaking it is not always a good idea to propagate in the algorithms all the samples attributes. The main reasons for which the dimensionality reduction is required are the following:

* high dimensional data may negatively affect the algorithms perfomances (i.e. high computational costs);
* high dimensionality might lead to poor generalization;
* can be used for interpretability of the data (e.g. for illustraton purposes).

Moreover different machine learning algorithms suffer from the **_curse of dimensionality_** problem: if there are more features than observations the risk of massively overfitting the model becomes really high. Then, too many dimensions causes every observation in the dataset to appear equidistant from all the other, and this is effectively a problem when used distance-based algorithms (such as K-Nearest Neighbors), because if the distances are all approximately equal, then the observation appear equally alike (as well as equally different), making the algorithm perfomances meaningless.

### Feature Selection

On top of feature transformation, the data preprocessing phase often includes a feature selection step, where the attributes from the given dataset are carefully analysed and finally selected to feed the machine learning algorithm.  
As previously seen in the correlation matrix and with scatter plots, some features in this dataset are strongly linearly correlated, so it is not meaningful to keep all of them because they may contain redundant information. In this way, features with a Pearson coefficient $\rho \geqslant 0.92$ with other predictors should be discarded.

In particular, at this step the following features are removed: `BILL_AMT2`, `BILL_AMT3`, `BILL_AMT4`, `BILL_AMT5`, `BILL_AMT6`.

### Principal Component Analysis

Another way to obtain a lower dimensional dataset when dealing with multicollinearity among features, is by applying a Principal Component Analysis (PCA) to our data.

PCA is an **unsupervised learning** technique that performs a linear transformation (by means of an orthogonal matrix) of the original space such that the new basis found has dimensions (features) that are sorted from largest to smallest possible variance.

The intuition behind this approach lies in the fact that if the direction of maximum variance in the original space is not directly captured by the features in the dataset, then that direction might be used to construct a new feature that has a larger variance, hence probably encoding more information. Therefore, the following minimization problem is performed for finding the direction of maximum variance:  
  
$$ X\in \mathbb{R}^{n\times d} , dataset\ centered\ in\ zero $$

$$ \Sigma :=\frac{X^{T} X}{n-1},\ sample\ covariance\ matrix $$ 

$$ find\ \ \overrightarrow{z_{1}} :=a_{1} \overrightarrow{e_{1}} +...+a_{d}\overrightarrow{e_{d}} $$ 

$$ s.t.: \ \ \overrightarrow{z_{1}} =\underset{\vec{z}{1}}{argmax} \ \ \vec{z_{1}}^{T} \ \Sigma \ \ \vec{z_{1}} , \ \ \ \Vert \vec{z_{1}}\Vert =1 $$

Recursively, all other dimensions are then computed in the same way, additionally forcing them to be orthogonal to the previous dimensions found. The new features $z_{i}$ are denoted as **principal components** (PCs).

The above optimization problem can be easily proved to be equivalent to computing the eigendecomposition of $\Sigma$ and selecting its eigenvectors as the principal components, which is the way PCA is executed in practice. In particular, being $\Sigma$ a symmetric positive semidefinite matrix, it is always possible to diagonalize it by means of an orthogonal matrix $P$ (eigenvectors of $\Sigma$), and obtain the resulting similar matrix $\Lambda$:  
 
$$ \Lambda =\ \begin{pmatrix} \lambda_{1} & 0 & \cdots & 0\\\ 0 & \lambda_{2} & \ddots & \vdots \\\ \vdots & \ddots & \ddots & 0\\\ 0 & \cdots & 0 & \lambda_{d} \end{pmatrix} \ ,\ \lambda_{1} \geqslant \lambda_{2} \geqslant ...\geqslant \lambda_{d} \geqslant 0,\ \lambda_{i} \in \mathbb{R}^{+} $$

where $\lambda_{i}$ are the eigenvalues of $\Sigma$, or equivalently the variances of the new features found. Note that, since $\Sigma$ and $\Lambda$ are similar matrices, they have the same trace, meaning that the initial total variance among the features of the dataset is not changing but it is just getting redistributed on new axes (which is expected as all we are doing is just a rigid rotation of the space). In particular, we can compute the variance explained by each new principal component with respect to the total variance of the dataset.

Finally note that $\Lambda$ is now the covariance matrix in the new basis found, and since it’s diagonal all new features result to be linearly uncorrelated. We can then select only a subset of the first principal components in order to reduce the initial dimensionality of the dataset with minimal information loss.

In this study, PCA has been performed on the Credit card dataset to deal with the multicollinearity problem and reduce the number of dimensions. The following figure shows how the variance has been redistributed on the new features extracted.

<p align = "center">
<img height="300" src="https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/blob/main/images/PCACumulative.svg">
</p>
<p align = "center">
Explained variance ratio of each principle component, together with the  
cumulative variance explained as more dimensions are considered.
</p>

The graph above is relative to the **proportion of explained variance**: the green line represents the proportion of variance explained by each principal component. While the orange line instead is the cumulative amount of proportion of variance explained (i.e. the sum of the single variance explained by the principal components on the left).

This graph is useful to take decision regarding the number of components to keep. The following table report some interesting values for the number of components.

| Number of PCs | Cumulative Variance Explained |
| --- | --- |
| 4   | 89.7 % |
| 6   | 95.3 % |
| 8   | 97.1 % |
| 12  | 99.0 % |

For example, the first 12 PCs are able to capture almost the whole variance (99%) of the data points and for this reason we decided to keep these number.

_Note that PCA is applied based only on the training data in order to avoid any leaking the information of test data._
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

### Oversampling and undersampling

When dealing with significantly **unbalanced dataset** in the target label, it becomes harder for most machine learning algorithms to efficiently learn all classes. The training process might be indeed biased towards a certain class if the dataset distribution is poorly balanced.

In the specific case of the credit card clients, only about 22.1% of the data are labelled as defaulters (y=1).

|     | Number of rows | Percentage |
| --- | --- | --- |
| Non-defaulters (class=0) | 17246 | 77.68 % |
| Defaulters (class=1) | 4954 | 22.32 % |

While the obvious and most desirable solution would be to collect more real data, _oversampling_ and _undersampling_ are techniques that may still come in handy in these situations.  
For both techniques there is a naïve approach that is the **_random oversampling (undersampling)_** where training data is incremented (decremented) with multiple copies of the samples, until the same proportion is obtained on the minority (majority) classes.

### Oversampling: Synthetic Minority Over-sampling Technique (SMOTE)

With SMOTE algorithm the minority class is augmented artificially, by constructing new synthetic samples randomly positioned in between one point and its k-neighbors. In other words, given a limited set of data points that belong to the class that we wish to augment, we trace high-dimensional lines connecting the data points and we draw new samples from such lines.  

<p align = "center">
<img src="https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/blob/main/images/oversampling-smote.png?raw=true">
</p>
<p align = "center">
Example of SMOTE application on a small dataset
</p>
  
Finally, note that a synthetic sample should never be used as a test sample, since it has technically never appeared in the real world and hence not formally belonging to the target distribution. Also, the stratified test set is believed to be the correct representation of the real world, and augmenting a part of it would lead to a misleading evaluation.

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

### Undersampling: Cluster Centroids

Cluster Centroids make use of K-means algorithm to perform undersampling. After finding the cluster centroids on the majority class points, the algorithm selects the instances belonging to the cluster (labelled with the majority class), which are furthest from the cluster centroid in feature space. These data points are considered as the most unimportant instance. On the contrary, the instance belonging to the majority class, that is nearest to the cluster centroid in feature space, is considered to be the most important instance. In this way, instances belonging to the majority class are removed on the basis of their importance.  

<p align = "center">
<img height="300" src="https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/blob/main/images/undersampling-cluster_centroids.png?raw=true">
</p>
<p align = "center">
Example of Cluster Centroids application on a trivial dataset
</p>

In particular, given two class which has respectively $N$ and $M$ data points with $N \lt M$, the algoritm trains a K-Means on points labelled with the majority class label, with $k = N$, the cardinality of the set of minority data points.  
Then for each cluster, it replace the data points of the majority class with a new point whose coordinates correspond to the cluster centroid's coordinates. So we undersample majority class by forming clusters and replacing it with cluster centroids.

## Classification algorithms

In this section a description of the algorithms used is presented. For each of them, different hyperparameters are tuned in order to find the ones performing best, and, once found, the model trained with them is used to predict class labels on test data combining different data preprocessing techniques presented above in order to spot some performances differences. The best configuration is selected by comparing the different metrics, principally based on the _f1-score_ because accuracy on unbalance dataset may return an high values even if the minority class is not correcly classified.

Each algorithm is trained on dataset with different preprocessing techniques combined:

* Principal Component Analysis (PCA)
* PCA + SMOTE (oversampling)
* PCA + Cluster Centroids (undersampling)

The algorithms considered are:

* Logistic Regression
* Random forest (and Decision Trees)
* SVM (both Linear and RBF)

### Evaluation methods

Even though learning theory guarantees to reach a probably approximately correct model, in practice it is often needed to perform an actual evaluation of the model to check how well it’s performing. This is generally done by splitting the original dataset into two parts, a training set and a test set, so that the model can be trained on the training samples only and be evaluated on previously unseen data. This technique is referred to as **_Holdout_**.

Instead, **_Cross Validation_** is a resampling method that allows to repeteadly drawing samples from a training set, in order to have a better estimate of the evaluation score used.

With the cross-validation technique we aim to verify if the model is able predict labels for data points that it hasn't seen so far. The complete dataset is divided into $k$ subsets (folds):

* $k-1$ folds will be used to train the model, all togheter compose the _training set_;
* one fold composes the _validation set_, on which we evaluate the performance of the model.

This operation is repeated $k$ times in order to reduce the variability, and at every round the validation subset changes. At the end, the $k$ estimates of model's predictive performance are averaged.

<p align = "center">
<img height="300" src="https://upload.wikimedia.org/wikipedia/commons/c/c7/LOOCV.gif">
</p>
<p align = "center">
Example of Cross Validation (k=8) where the red and blue square is used respectively as training and validation set
</p>

The special case where $k$ is equal to $m$, the number of examples, is called _leave-one-out_. This procedures gives a very good estimate of the true error but, on the other side, it computationally expensive.

A combination of holdout and cross validation can be used when also dealing with a validation set (_k-fold cross validation_). So after having divided the dataset in a stratify way into training and test set, the training part is again splitted into training and validation set (even in this case with `stratify=True`).

### Evaluation metrics

In a classification scenario, the model can be evaluated by computing different metrics. In order to better understand these metrics could be useful to get some fundamentals:

|     | Predicted (1) | Predicted (0) |
| --- | --- | --- |
| **Actual (1)** | TP  | FN  |
| **Actual (0)** | FP  | TN  |

* _True Positive (TP)_: samples for which the prediction is positive and the true class is positive
* _False Positive (FP)_: samples for which the prediction is positive but the true class is negative
* _True Negative (TN)_: samples for which the prediction is negative and the true class is negative
* _False Negative (FN)_: samples for which the prediction is negative but the true class is positive

Some of the most popular metrics are:

* **_Accuracy_**: ratio of correct predictions over the total number of data points classified <br>$Accuracy = \frac{Number\ correctly\ classified\ samples}{total\ number\ of\ samples\ tested}=\frac{TP+TN}{TP+FP+TN+FN}$

* **_Precision_**: measures the fraction of correct classified instances among the ones classified as positive. Precision is an appropriate measure to use when the aim is to minimize _false positives_. <br>$Precision(c) = \frac{ Number\ samples\ correctly\ assigned\ to\ class\ c}{Number\ of\ samples\ assigned\ to\ class\ c}=\frac{TP}{TP+FP}$

* **_Recall_**: it measures how many of the actual positives a model capture through labelling it as True Positive. It is an appropriate score when the aim is to minimize false negatives <br>$Recall(c) = \frac{Number\ samples\ correctly\ assigned\ to\ class\ c}{Number\ of\ samples\ actually\ belonging\ to\ c}=\frac{TP}{TP+FN}$

* **_F1-score_**: is the harmonic mean of the precision and recall. <br>$F1-score(c) = \frac{2\cdot precision(c)\cdot recall(c)}{precision(c) +recall( c)}$

The last three measures are _class-specific_, and a global metric can be obtained by averaging their values across all classes in the dataset. In particular, class-specific metrics work better when dealing with significantly unbalanced dataset, as they take into account how every class is being learned. Thus, the accuracy measure alone might be misleading in these cases, since may return an high score even if the minority class is not correcly classified.

In this analysis we focus our attention in detecting which customer may be defaults clients, and the positive class captures the attention of the classifier. As already mentioned, in the following section the F1-score will be used to find the best configuration for each model.

## Models

### Logistic Regression

Logistic Regression is a parametric, discriminative binary classification algorithm. The name is given after the fact that the algorithm can be interpreted as part of the _Generalized Linear Model_, where the response variables are distributed according to a Bernoulli distribution. In particular, the model assumes the predictors to be linked to the mean of the response variables $p_{i}$ as:

$$ observation\ ( x_{i} , y_{i}) \ ,\ y_{i} \ realization\ of\ Y_{i} \sim Bernoulli( p_{i}( x_{i}))$$

$$ log\left(\frac{p_{i}}{1-p_{i}}\right) =w^{T} x_{i} \ \Longleftrightarrow \ p_{i} =\frac{1}{1+e^{-w^{T} x_{i}}} \ ,\ for\ all\ i $$

<p align = "center">
<img height="300" src="https://upload.wikimedia.org/wikipedia/commons/5/53/Sigmoid-function-2.svg">
</p>
<p align = "center">
Logits function
</p>

The algorithm then tries to find the MLE (or MAP if a regularization term is added) of the mean of the response variables, by acting on _w_, assuming i.i.d. samples. Unlike ordinary least squares the optimization problem does not have a closed-form solution, but it’s instead solved through numerical methods.

$$ \underset{w}{\max} \ \\prod ^{n_{i}} \ p_{i}( x_{i} |w)^{y_{i}} \ *\ ( 1-p_{i}( x_{i} |w))^{1-y_{i}} $$

A **_regularization term_** is often added to prevent the coefficients to reach a too high value which would overfit the training data. The hyperparameter $C$ is applied on the new penalty term and inversely regulates the strength of the regularization in the objective function.

```python
# C values tuned:
params = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10]}

#Best configuration found C=0.01
#Data prepocessing: PCA+SMOTE
```

As a result, the output of a Logistic Regression model is the probability of the input sample to be of class 1, hence a confidence measure is also returned when predictions are performed. On top of this, the coefficients returned by the algorithm may give interpretable insights on what attributes are contributing the most to a higher output value, and viceversa.

### Decision Tree

Decision Trees are the most intuitive and interpretable machine learning model, which predict the target label by learning simple decision rules inferred from the data. At each step, a Decision Tree picks the feature that best divides the space into different labels, by means of the GINI impurity measure: <br><br>

$$ GINI(t) =\sum_{i=1}^{C} p_{t}(k)(1-p_{t}(k)) = 1-\sum_{i=1}^{C} p_{t}(k)^{2} $$

where $p_{t}(k)$ is the frequency of class $k$ appearing in node $t$, with $C$ the total number of classes. The lower the measure, the less impure the node is.

A full tree is then constructed on the training data and will be used at classification time for predicting the target label. To avoid overfitting, decision trees are often pruned before reaching their full growth, and predictions are made according to a majority vote on the samples of each leaf. Note how trees cannot handle missing values, but do not suffer from feature scaling since each attribute is treated independently.

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

### SVM

Support vector machine (SVM) is a parametric linear classification algorithm that aims at separating two classes through a hyperplane in the data dimension. Once the hyperplane has been set, the prediction rule is simply based on whether the test point lays on one side or the other one of it. If more than one hyperplanes fit the purpose, the one with the higher margin (distance) between support vector (the closest points to the hyperplane, one for each class) is selected.

<p align = "center">
<img height="300" src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/800px-SVM_margin.png">
</p>
<p align = "center">
Maximum-margin hyperplane and margins for an SVM trained with samples from two classes.  
Samples on the margin are called the support vectors.</p>

More formally, it tries to define $\vec{w} ,b$ s.t. the hyperplane $w^{T}x+b=0$ is the “maximum-margin hyperplane”, i.e. such that the distance between the hyperplane and the nearest point $\overrightarrow{x_{i}}$ from either group is maximized. If we impose that all points must satisfy $|w^{T}x+b|\\geqslant1$ relatively to their label, then the optimization problem becomes equivalent to finding the smallest $\|w\|$:

#### Primal Optimization Problem (Hard Margin) 

$$ \underset{w,b}{min} \tfrac{1}{2}\Vert w\Vert ^{2} \\ \\\ s.t.\ y_{i}\left( w^{T} x_{i} +b\right) \geqslant 1,\ \forall i $$

The decision function on a binary label $\{-1,1\}$ is then applied as $x\mapsto sign\left( w^{T} x+b\right)$. Geometrically, the best hyperplane is completely determined by the points $\overrightarrow{x_{i}}$ the lie right on the boundary of the margin $|w^{T}x_{i}+b|=1$. These points are called **_support vectors_**.

This model is referred to as **Hard-margin SVM**, where data is supposed to be linearly separable otherwise no feasible solution would be found.  

To extend the model on classes that are not perfectly separable by means of a hyperplane, the hinge loss function is introduced, which penalizes points falling in between the margin:

$$ L_{hinge} := \max\left( 0, 1-y_{i}\left( w^{T} x_{i} +b\right)\right) $$

The previous opt. problem is then generalized to the **soft margin** version, which is equivalent to the ERM of the hinge loss with an L2 regularization term:

#### Primal Optimization Problem (Soft Margin)

$$ \\underset{w,b,\xi}{min} \ \ \\tfrac{1}{2}\\Vert w\\Vert ^{2} \ +C\\sum_{i}^{n} \\xi_{i} $$

$$ s.t.:\\\ y_{i}\\left( w^{T} x_{i} +b\\right) \\geqslant 1-\\xi_{i} \ ,\ \\forall i\\\ $$

$$\\xi_{i} \\geqslant 0\ ,\ \\forall i\\\ $$

Where $\xi$ is a slack variable which is introduced to in order to soften the misclassification constraint, allowing the model to make a certain number of mistakes and letting the margin to remain as wide as possible. The amount of misclassification allowed is controlled by the hyperparameter $C$ which regulates the strength of the Hinge Loss term introduced: a higher $C$ will lead to a stronger minimization of the hinge loss (forcing training points to be more correctly classified), while a lower $C$ imposes a harder regularization (allowing more missclassifications) leading to a larger margin.  
The Lagrangian Dual Problem is as follows:

**Dual Optimization Problem (Soft Margin)**

$$ \underset{\alpha}{max} \ \ \sum_{i}^{n} \alpha_{i} -\tfrac{1}{2}\sum_{i,j} \alpha_{i} \alpha_{j} y_{i} y_{j}\left( x_{i}^{T} x_{j}\right) $$

$$ s.t.: \\ \sum \alpha_{i} y_{i} =0\ \ \land \ 0\leqslant \alpha_{i} \leqslant C,\ \forall i $$

By solving the dual problem (e.g. through Quadratic Programming) we find $$w = \sum_{i}^{n} \alpha_{i} y_{i} x_{i}$$ as a linear combination of the training data. In particular, only the points that are in between (or exactly on) the margin will have an $\\alpha_{i}\\neq0$, i.e. only the so called support vectors are affecting the decision function. The latter can be then also denoted as $sign\(\sum \alpha_{i} y_{i} \( x^{T}_{i} x\)+b \)$, with $x$ a generic test observation. Note how in the soft margin version the number of support vectors found is generally much higher.

Starting from these assumptions, the SVM model can still be generalized to multiclass classification and **non-linearly separable classes**. Indeed, as for the second case, classes may follow a non-linear pattern and a hyperplane may not be the best fit for explaining our data. However, the given dataset might still be linearly separable when mapped onto a higher-dimensional space by a non-linear function phi, such that the training points are transformed as $x_{i} \\in \\mathbb{R}^{d} \\mapsto \\varphi ( x_{i}) \\in \\mathbb{R}^{d^{\\prime}} ,\ d^{\\prime}>d$.

While it is possible to compute the SVM algorithm in a higher space through a feature map , a different trick is instead practically used which exploits a nice property of support vector machines. Indeed, note that the dual optimization problem and the newly defined decision function solely depend on the scalar product between the various points at all times. Whenever this property is satisfied then no matter what the higher-dimensional space is, all that it’s needed is a function able to compute the scalar products in the new transformed space.

We then introduce the notion of a Kernel function $K$, defined as:

$$ K\ :\ \\mathcal{X} \\times \\mathcal{X}\\rightarrow \ \\mathbb{R}\\\ ( x,\ x^{\\prime} ) \\mapsto K( x,\ x^{\\prime} ):=\\varphi ( x)^{T} \\varphi ( x^{\\prime} ) $$

for some feature map $\\phi$. Then, we can simply apply different kernel functions to the SVM algorithm without directly knowing what the feature transformation is. The power of this approach relies on the Mercer’s Theorem, which gives a sufficient condition for a function to be a Kernel for some unknown $\\varphi ( \\cdot )$.

Common Kernel functions used are:

* Linear Kernel: $K( x,x^{\\prime} ) =x^{T} x^{\\prime}$
* Gaussian RBF Kernel: $K(x,x^{\\prime} )\ =\ \\exp (-\\gamma \\| x-x^{\\prime} \\| ^{2} ),\ \\gamma >0$
* Polynomial Kernel: $K( x,x^{\\prime} ) \ =\ \\left( x^{T} x^{\\prime} +c\\right)^{d}$

In particular, for the Gaussian RBF Kernel the decision function is computing a similarity measure between the test point and the support vectors (weigthed by $\\alpha\_i$ and $y\_i$), which is inversely proportional to the euclidean distance of the two, leading to an output label that is determined by the closest support vectors. The new hyperparameter gamma can be tuned to regulate how many support vectors are significantly affecting the decision function (usually set to auto: $1/n\\\_features$ or scale: $1/\(n\\\_features*X.var()\)$.

The decision function then becomes $sign\(\\sum \\alpha_{i} y_{i}K(x_{i},x)+b\)$. Finally note that while it is convenient to use a kernel function, the hyperplane in higher space is not actually computed when the kernel trick is applied, i.e. the coefficients of the vector $w$ are unknown.

In this study, both the linear and the gaussian RBF Kernels have been tried. The regularization parameter $C$ has been hypertuned to find the best performing model on the training set.

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

_Scores on validation for the best configuration found with both kernels and both preprocessing techniques_

* the score obtained are quite similar indipendently from the preprocessing technique (SMOTE is a little bit better);
* with a polinomial kernel in general the recall on positive class is lower than all the other cases;
* the gaussian version of the SVM outperforms the polinomial one.

In conclusion, also in this case the results should be improved in order to get more accurate prediction on default credit card clients.

## Results

The following barplot displays a summary of results in the training-validation phase where all the algorithms are trained with their best hyperparameters (i.e. the ones that maximize the f1-score on positive class), with both techniques presented to overcome class imbalancing problem. ![](https://github.com/MatteoM95/Default-of-Credit-Card-Clients-Dataset-Analisys/blob/main/images/results.png)

_Comparison of F1-score with different algorithms_

### Discussion

* Logistic Regression and Support Vector Machines maintain the same performance regardless the imbalancing techniques used, while this is not true for Random Forest, the reason might be that a decision tree to perform well needs lot of data but at the same time undersampling technique reduce the amount of data.
* Overall, oversampling slightly outperforms undersampling;
* The positive class is the most difficult to classify and none of the models chosen seems to be not able to get the complexity of the problem.
* Empirically has been observed that in general undersampling require more time with respect to oversampling as preprocessing step, this is likely since the first one compute a k-means over all data to find centroids.

All the above mentioned algorithms have been tried out and hypertuned on the training set with a k-fold cross validation technique. The best performing configurations of hyperparameters on the training set have been chosen for rebuilding the models and evaluating them on the test set.

Finally, the table below the results of all the classifiers (in their best configuration) on validation and test set are compared. As we can see, the score obtained on the test set are pretty close to the one obtained in validation.

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

_Note that the scores obtained on the test should never be treated as additional information to change how the training is performed, but only as a final evaluation of the model._

## Conclusions

In this study different supervised learning algorithms have been inspected and presented with their mathematical details, and finally used on the UCI dataset to build a classification model that is able to predict if a credit card clients will default in the next month. Data preprocessing makes algorithms perform slightly better than when trained with original data: in particular, PCA results are approximately the same, but the computational cost has been lowered. Oversampling and undersampling techniques has been combined with PCA to assess the dataset imbalance problem. Oversampling as mentioned performed slightly better w.r.t. the undersampling, this is likely because the model is trained on a large amount of data. However, all the models implemented achieved comparable results in terms of accuracy.

### References

\[1\] Default of credit card clients Data Set: UCI dataset [link](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  
  
\[2\] Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480. [link](https://bradzzz.gitbooks.io/ga-seattle-dsi/content/dsi/dsi_05_classification_databases/2.1-lesson/assets/datasets/DefaultCreditCardClients_yeh_2009.pdf)  
  
\[3\] Understanding Machine Learning: From Theory to Algorithms, S. Shalev-Shwartz, S. Ben-David, 2014 [link](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/understanding-machine-learning-theory-algorithms.pdf)