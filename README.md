# ✈️ Predicción del precio de vuelos

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Proyecto Data Science**  
**Nombre:** Kattya Contreras  
**Contenido:** Machine Learning

---

## 📌 Descripción

Este proyecto tiene como objetivo **predecir el precio de boletos de avión** (clases Business y Economy) para una agencia de viajes online, considerando variables como aerolínea, clase, destino, fechas y escalas.  

Se utilizan dos datasets: [`business.xlsx`](./business.xlsx) y [`economy.xlsx`](./economy.xlsx), que contienen información histórica de vuelos, precios y características del viaje.  

El flujo del proyecto abarca todas las etapas de un **pipeline de Machine Learning**:

- Análisis exploratorio de datos (EDA)  
- Limpieza y preprocesamiento  
- Ingeniería de variables  
- Modelamiento y optimización  
- Evaluación de modelos  
- Conclusiones y próximos pasos  

**Variable objetivo:** `price`  

---

## 🛠 Metodología

<details>
<summary>Ver metodología detallada</summary>

1. **Carga y revisión de datos**  
   - Inspección de nulos, duplicados y tipos de datos  

2. **Limpieza y preprocesamiento**  
   - Tratamiento de columnas de texto (`stop`)  
   - Conversión de duración de vuelos a minutos  
   - Imputación de nulos y eliminación de columnas irrelevantes  

3. **Feature Engineering**  
   - Variables derivadas de fechas (`month`, `day_of_week`, `is_weekend`)  
   - Creación de rutas (`from-to`)  
   - Codificación de aerolíneas top y rutas  

4. **Análisis exploratorio visual (EDA)**  
   - Distribución de precios por clase (Business vs Economy)  
   - Correlaciones entre variables numéricas y `price`  
   - Identificación de outliers y colas largas  

5. **Modelamiento**  
   - Modelos entrenados: Random Forest, XGBoost y Lasso  
   - Comparación de métricas: MAE, RMSE y R²  
   - Baseline con promedio de `price`  

6. **Optimización de modelo**  
   - Búsqueda de hiperparámetros con `GridSearchCV` para Random Forest  
   - Selección del mejor modelo para predicciones

</details>

---

## 📊 Resultados clave

<details>
<summary>Ver tabla de comparación de modelos</summary>

| Modelo       | RMSE     | MAE      | R²      | Comentario                                         |
|-------------|----------|----------|---------|---------------------------------------------------|
| Random Forest | 3604.44 | 1708.94 | 0.9232 | Mejor desempeño, captura relaciones no lineales y robusto frente a outliers |
| XGBoost       | 4759.55 | 2087.23 | 0.8660 | Buen desempeño, cercano a Random Forest, permite regularización y relaciones no lineales |
| Lasso         | 8803.48 | 5342.01 | 0.5417 | Modelo lineal simple, no adecuado para esta distribución de precios |
| Baseline      | 13004.97 | 9732.75 | -0.0001 | Predicción de la media de `price`, sirve como referencia |

**Conclusión:** Random Forest es el modelo más recomendado para predecir precios en Business, seguido de XGBoost. Las variables más relevantes son:

- `stop_num` (cantidad de escalas)  
- `time_taken_min` (duración del vuelo)  
- `month` y `day_of_week`  

</details>

---

## 🚀 Próximos pasos

<details>
<summary>Ver detalles de próximos pasos</summary>

- Tratar outliers y colas largas en la clase Business  
- Evaluar transformaciones logarítmicas de `price`  
- Ajustar hiperparámetros de XGBoost  
- Aplicar la misma metodología a la clase Economy para comparar resultados  

</details>

---

## 🧰 Librerías principales utilizadas

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

