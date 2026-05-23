# Instrucciones de instalación y ejecución

## Requisitos
- Python 3.10 o superior
- pip

## 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/credito-ml-upateco.git
cd credito-ml-upateco
```

## 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 3. Entrenar el modelo (Avance 2)

```bash
cd notebooks
python 02_modelo_entrenamiento.py
```

Esto genera en `models/`:
- `modelo_credito.pkl` — modelo activo (XGBoost)
- `random_forest.pkl`
- `feature_info.json`

Y en `reports/`:
- `evaluacion_modelos.png`
- `feature_importance.png`
- `metricas.json`

## 4. Ejecutar la aplicación (Avance 3)

```bash
cd app
streamlit run app.py
```

La app abre automáticamente en `http://localhost:8501`

## Uso en Google Colab

1. Subir `credito-dataset.csv` a la sesión de Colab
2. Clonar el repositorio o subir los archivos manualmente
3. Instalar dependencias:
   ```python
   !pip install xgboost shap streamlit
   ```
4. Ejecutar el notebook celda por celda

## Estructura del proyecto

```
credito-ml-upateco/
├── data/
│   └── credito-dataset.csv
├── notebooks/
│   └── 02_modelo_entrenamiento.py
├── models/
│   ├── modelo_credito.pkl
│   ├── random_forest.pkl
│   └── feature_info.json
├── app/
│   └── app.py
├── reports/
│   ├── evaluacion_modelos.png
│   ├── feature_importance.png
│   └── metricas.json
├── requirements.txt
└── README.md
```
