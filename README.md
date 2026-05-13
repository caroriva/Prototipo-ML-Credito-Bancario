# TP1 — Análisis Exploratorio de Datos: Dataset de Crédito

**Primera entrega del Proyecto Integrador**  
Materia: Ciencia de Datos

---

## Objetivo

Explorar un dataset de solicitudes de crédito bancario para identificar los factores socioeconómicos y demográficos que más influyen en la aprobación de un crédito, detectar problemas de calidad de datos y sentar las bases para un modelo predictivo futuro.

---

## Estructura del proyecto

```
TP1_Credito/
├── data/
│   └── credito-dataset.csv       # Dataset original (separador: ;)
├── notebook/
│   └── TP1_EDA_Credito.ipynb     # Notebook principal con todo el análisis
└── README.md                     # Este archivo
```

---

## Dataset

- **Fuente:** Dataset de solicitudes de crédito bancario
- **Registros:** 438.557
- **Variables:** 21 (20 predictoras + 1 variable objetivo `APPROVED`)
- **Separador:** punto y coma (`;`)

### Variables principales

| Variable              | Descripción                                    |
| --------------------- | ---------------------------------------------- |
| `AMT_INCOME_TOTAL`    | Ingreso anual del solicitante                  |
| `AGE`                 | Edad en años                                   |
| `YEARS_EMPLOYED`      | Antigüedad laboral en años                     |
| `NAME_EDUCATION_TYPE` | Nivel educativo                                |
| `CODE_GENDER`         | Género                                         |
| `APPROVED`            | Variable objetivo: 1 = aprobado, 0 = rechazado |

---

## Preguntas de investigación

1. ¿Cuál es la tasa de aprobación por género?
2. ¿El ingreso influye en la aprobación?
3. ¿La antigüedad laboral se relaciona con la aprobación?
4. ¿El nivel educativo afecta la tasa de aprobación?
5. ¿Existen anomalías o valores atípicos relevantes?

---

## Principales hallazgos

- La **antigüedad laboral** es el predictor más fuerte (correlación 0.54 con la aprobación).
- El **ingreso** y el **nivel educativo** también muestran relación positiva con la aprobación.
- La **edad** tiene relación negativa: solicitantes más jóvenes son aprobados con mayor frecuencia.
- Se detectó un **valor centinela** en `DAYS_EMPLOYED = 365243` que codifica pensionados (17.2% del dataset).
- `OCCUPATION_TYPE` presenta **30.6% de valores nulos**.

---
