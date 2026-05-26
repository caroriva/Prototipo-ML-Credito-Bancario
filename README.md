# Sistema ML de Pre-evaluación Crediticia Bancaria

> Tecnicatura Universitaria en Ciencia de Datos e IA Aplicada · UPATECO · Salta · 2026  
> Materia: Modelado de Sistemas de IA Aplicada
> Ciclo técnico: Machine Learning Clásico Supervisado — Clasificador local

**Grupo 14:** Ariel Escalante · Carolina Rivarola · Federico Lemos · Natalia Peloc

---

## Descripción del Problema

Las entidades financieras reciben diariamente miles de solicitudes de crédito que
requieren análisis manual de múltiples variables económicas, laborales y personales.
Este proceso genera demoras en la respuesta al cliente, costos operativos elevados,
riesgo de subjetividad humana y saturación del equipo de análisis crediticio.

Este proyecto desarrolla un sistema de pre-evaluación crediticia automatizada mediante
un modelo de clasificación supervisada capaz de predecir si una solicitud de crédito
debe ser aprobada o rechazada, basándose en datos históricos de clientes.

---

## Variable Objetivo

| Valor          | Significado       |
| -------------- | ----------------- |
| `APPROVED = 1` | Crédito Aprobado  |
| `APPROVED = 0` | Crédito Rechazado |

**Distribución de clases:** 72,9% aprobados — 27,1% rechazados (desbalance moderado,
tratado con `scale_pos_weight` en XGBoost).

---

## Dataset

| Atributo      | Detalle                            |
| ------------- | ---------------------------------- |
| Registros     | 438.557                            |
| Variables     | 21 (20 features + 1 target)        |
| Valores nulos | Solo en `OCCUPATION_TYPE` (~30,6%) |
| Formato       | CSV delimitado por `;`             |

### Variables del dataset

**Datos personales:** `CODE_GENDER`, `DAYS_BIRTH`, `AGE`

**Perfil económico:** `AMT_INCOME_TOTAL`, `NAME_INCOME_TYPE`, `NAME_EDUCATION_TYPE`

**Perfil laboral:** `DAYS_EMPLOYED`, `YEARS_EMPLOYED`, `OCCUPATION_TYPE`

**Patrimonio:** `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`, `NAME_HOUSING_TYPE`

**Núcleo familiar:** `NAME_FAMILY_STATUS`, `CNT_CHILDREN`, `CNT_FAM_MEMBERS`

**Contactabilidad:** `FLAG_MOBIL`, `FLAG_WORK_PHONE`, `FLAG_PHONE`, `FLAG_EMAIL`

---

## Modelo Seleccionado: XGBoost

### ¿Por qué XGBoost?

En la evaluación comparativa contra Random Forest, XGBoost obtuvo mejores resultados:

| Métrica       | Random Forest |  XGBoost   |
| ------------- | :-----------: | :--------: |
| Accuracy      |    0.9351     | **0.9403** |
| F1-Macro      |    0.9234     | **0.9292** |
| ROC-AUC       |    0.9859     | **0.9887** |

Además, `scale_pos_weight` maneja el desbalance de clases nativamente sin necesidad
de remuestreo manual.

---

## Feature Engineering aplicado

- `FLAG_MOBIL` eliminada — valor constante (1 en el 100% de los registros)
- `IS_PENSIONER_EMP` — flag binario para `DAYS_EMPLOYED = 365243` (codifica sin empleo activo)
- `INCOME_PER_MEMBER` — ingreso anual dividido por integrantes del grupo familiar
- `OCCUPATION_TYPE` nulos → categoría explícita `'Sin datos'` (preserva la información de ausencia)

---

## Métricas de Evaluación

Dado el desbalance de clases, accuracy sola no es suficiente. Métricas principales:

- **F1-Macro** — equilibra precision y recall entre ambas clases
- **ROC-AUC** — capacidad discriminativa general del modelo
- **Matriz de confusión** — análisis de falsos positivos y negativos

---

## Estructura del Repositorio

```
credito-ml-upateco/
├── data/
│   └── credito-dataset.csv
├── notebooks/
│   └── 02_modelo_entrenamiento.py
├── models/
│   ├── modelo_credito.pkl        ← generado al ejecutar el notebook
│   ├── random_forest.pkl
│   └── feature_info.json
├── app/
│   └── app.py                    ← aplicación Streamlit
├── reports/
│   ├── evaluacion_modelos.png
│   ├── feature_importance.png
│   └── metricas.json
├── requirements.txt
└── README.md
```

---

## Instalación y Ejecución

### Requisitos

- Python 3.10 o superior
- pip

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/credito-ml-upateco.git
cd credito-ml-upateco
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Entrenar el modelo

```bash
cd notebooks
python 02_modelo_entrenamiento.py
```

### 4. Ejecutar la aplicación

```bash
cd ../app
streamlit run app.py
```

La app abre en `http://localhost:8501`

---

## Stack Tecnológico

| Herramienta          | Uso                                  |
| -------------------- | ------------------------------------ |
| Python               | Lenguaje principal                   |
| Google Colab         | Desarrollo y entrenamiento           |
| Pandas / NumPy       | Procesamiento de datos               |
| Matplotlib / Seaborn | Visualización (EDA)                  |
| scikit-learn         | Pipelines, métricas, Random Forest   |
| XGBoost              | Modelo principal                     |
| SHAP                 | Explicabilidad del modelo (Avance 4) |
| Streamlit            | Aplicación local de demo             |
| GitHub               | Control de versiones y entrega       |

---
