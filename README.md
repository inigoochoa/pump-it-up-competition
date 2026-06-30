# Pump It Up: Data Mining the Water Table (DrivenData)

Bienvenido a este repositorio. Aquí se presenta mi solución para la competición **"Pump it Up: Data Mining the Water Table"** alojada en DrivenData.

El objetivo principal de este reto de Machine Learning consiste en predecir el estado operativo de los puntos de suministro de agua (bombas de agua) distribuidos en toda Tanzania. Se trata de un problema de clasificación multiclase complejo debido a la alta cantidad de valores nulos implícitos, variables categóricas de elevadísima cardinalidad y un desbalanceo de clases muy marcado.

---

# Rendimiento y Posición en la Competición

- **Métrica Objetivo:** Clasificación por Accuracy.
- **Resultado Final Obtenido:** **0.8195 (81.95%)**
- **Posición en el Leaderboard:** **2364 de 19,775 participantes** (aproximadamente en el **Top 12%** a nivel global).

---

# Arquitectura de la Solución y Metodología

La estrategia de modelado e ingeniería se basa en cuatro pilares fundamentales para exprimir la señal de los datos sin comprometer la robustez estadística.

## 1. Análisis Exploratorio de Datos (EDA) y Limpieza

- Detección y tratamiento de nulos ocultos como ceros en variables geográficas o temporales (`longitude`, `gps_height`, `construction_year`).
- Eliminación de variables redundantes (`payment_type` por ser idéntica a `payment`, o `quantity_group` por duplicar a `quantity`).
- Imputación controlada en variables estructurales como `scheme_management` e `installer`.

## 2. Ingeniería de Características Libre de Contaminación (Leakage-Free Feature Engineering)

- Creación de variables estacionales (mes) y temporales (antigüedad calculada desde la fecha de registro).
- **Target-Ratio Encoding avanzado** mediante un transformador personalizado (`TargetRatioFeatures`) que hereda de `BaseEstimator` y `TransformerMixin` de Scikit-Learn.
- **Prevención del Data Leakage:** las variables basadas en tasas de fallos (`buenas_reg`, `malos_funders` y `lga_top_func`) se calculan únicamente utilizando el conjunto de entrenamiento de cada *fold* durante la validación cruzada, evitando cualquier contaminación del conjunto de validación.

## 3. Estrategia de Modelado (Ensemble de Soft Voting)

Construcción de un **VotingClassifier** mediante votación blanda combinando tres modelos optimizados previamente mediante `GridSearchCV` y `RandomizedSearchCV`:

- **Random Forest** (Peso = 3)
  - Modelo individual con mayor capacidad predictiva sobre este conjunto de datos.

- **LightGBM** (Peso = 2)
  - Excelente manejo de variables categóricas complejas y alta velocidad de entrenamiento.

- **XGBoost** (Peso = 2)
  - Complementa al resto del ensemble capturando patrones distintos gracias a un sesgo de aprendizaje diferente.

## 4. Validación y Tracking

- Validación Cruzada Estratificada de **5 folds** (`StratifiedKFold`) para mantener la distribución de clases.
- Gestión del ciclo de vida del experimento mediante **MLflow**, almacenando métricas, parámetros y modelos entrenados.

---

# Instalación y Uso

El proyecto utiliza **rutas relativas**, permitiendo ejecutarlo tanto en un entorno local como en Google Colab sin modificar el código.

## 1. Clonar el repositorio

```bash
git clone https://github.com/inigoochoa/pump-it-up-competition.git
cd pump-it-up-competition
```

## 2. Instalar dependencias

Instala todas las librerías necesarias mediante:

```bash
pip install -r requirements.txt
```

## 3. Descargar y organizar los datos

Por las políticas de DrivenData, los archivos de datos **no están incluidos** en este repositorio.

Para reproducir los resultados:

1. Regístrate e inicia sesión en DrivenData.
2. Descarga los datasets oficiales de la competición.
3. Crea la siguiente estructura de carpetas:

```text
pump-it-up-competition/
├── data/
│   ├── X_train.csv
│   ├── y_train.csv
│   └── X_test.csv
├── notebooks/
│   └── pump_it_up_ensemble.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## 4. Google Colab

Si ejecutas el notebook desde Google Colab:

- Monta Google Drive.
- Ajusta el directorio de trabajo mediante `os.chdir()`.
- Mantén la carpeta `data/` en la misma estructura para que las rutas relativas funcionen correctamente.

---

# Tecnologías Utilizadas

- **Lenguaje:** Python 3
- **Manipulación de datos:** pandas, numpy
- **Visualización:** matplotlib, seaborn, plotly
- **Machine Learning:** scikit-learn, XGBoost, LightGBM
- **EDA Automatizado:** ydata-profiling
- **Experiment Tracking:** MLflow

---

# Autor

**Íñigo Ochoa**

---

