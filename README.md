# Pump It Up: Data Mining the Water Table

**DrivenData Competition #7** — Predicción del estado operativo de bombas de agua en Tanzania.

## Resultado

| Métrica | Valor |
|---|---|
| Accuracy (leaderboard) | **0.8186** |
| Posición | **2624 / 19 775** (~top 13%) |

## Problema

Clasificación multiclase con tres categorías:

- `functional` — la bomba funciona correctamente
- `functional needs repair` — funciona pero necesita reparación
- `non functional` — no funciona

La métrica oficial es **classification rate (accuracy)**. La clase *functional needs repair* está muy subrepresentada (~7% del dataset), lo que hace especialmente difícil su predicción.

## Enfoque

### Preprocesado
- Eliminación de columnas redundantes (`payment_type`, `quantity_group`) e irrelevantes (`recorded_by`, `num_private`)
- Imputación de nulos explícitos (`scheme_management`, `installer` → `'Desconocido'`; `permit` → `0`)
- Los missings implícitos (`longitude = 0`, `gps_height ≤ 0`) se mantienen: el ensemble los maneja mejor que cualquier imputación probada

### Feature Engineering

| Variable | Tipo | Descripción |
|---|---|---|
| `antiguedad` | Numérica | `date_recorded.year − construction_year` |
| `mes` | Numérica | Mes de la medición (estacionalidad) |
| `gps_height_600` | Binaria | 1 si altitud > 600 m |
| `buenas_reg` | Binaria | 1 si la región tiene < 33% de bombas no funcionales |
| `malos_funders` | Binaria | 1 si el financiador tiene ≥ 40% de bombas no funcionales |
| `lga_top_func` | Ordinal (0/1/2) | Clasificación del distrito por tasa de bombas funcionales |

Las tres últimas features dependen de la variable objetivo. Para evitar **data leakage**, se encapsulan en un `TransformerMixin` personalizado (`TargetRatioFeatures`) que se ajusta únicamente con los datos de entrenamiento de cada fold del cross-validation.

### Modelo

Ensemble de **Soft Voting** con tres clasificadores:

| Modelo | Peso | Librería |
|---|---|---|
| Random Forest | 3 | scikit-learn |
| LightGBM | 2 | lightgbm |
| XGBoost | 2 | xgboost |

Los pesos se ajustaron empíricamente comparando experimentos en MLflow. RF recibe mayor peso porque obtuvo mejor accuracy individual en validación cruzada.

### Validación

- `StratifiedKFold` con 5 folds (preserva la proporción de clases en cada fold)
- `TargetRatioFeatures` se ajusta dentro de cada fold → evaluación sin leakage
- Tracking completo de experimentos con **MLflow** (parámetros, métricas y modelo)

### Búsqueda de hiperparámetros

- Random Forest: `GridSearchCV`
- LightGBM: `RandomizedSearchCV` (n_iter=50), acelerado con GPU (Google Colab)
- XGBoost: `GridSearchCV`, acelerado con GPU (Google Colab)

## Estructura del repositorio

```
pump-it-up/
├── pump_it_up_ensemble.ipynb   # Notebook principal
├── README.md
└── data/                       # No incluido en el repositorio
    ├── X_train.csv
    ├── y_train.csv
    └── X_test.csv
```

Los datos se pueden descargar desde [DrivenData](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/).

## Requisitos

```
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
lightgbm
xgboost
mlflow
ydata-profiling
```

Instalación:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn lightgbm xgboost mlflow ydata-profiling
```

## Cómo ejecutar

1. Descargar los datos desde DrivenData y colocarlos en `data/`
2. Actualizar las rutas `PATH_TRAIN_X`, `PATH_TRAIN_Y` y `PATH_TEST` en la sección de configuración del notebook
3. Ejecutar todas las celdas en orden
4. Para repetir la búsqueda de hiperparámetros, cambiar `SEARCH = True` en la sección 9

## Decisiones clave

- **No imputar missings implícitos**: imputar longitude, gps_height y population por medianas regionales no mejoró el CV en ninguna variante probada
- **class_weight=None**: a pesar del desbalanceo, el uso de pesos balanceados redujo el accuracy en CV para RF y LGBM
- **FE documentado**: el notebook incluye todas las transformaciones probadas (activas y descartadas), con el motivo de cada decisión
