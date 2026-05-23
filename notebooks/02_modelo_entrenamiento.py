# =============================================================================
# AVANCE 2 — Núcleo Técnico Funcionando
# Sistema ML de Pre-evaluación Crediticia
# Modelado de Sistemas de IA Aplicada · UPATECO · 2026
# =============================================================================

# %%
# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost no disponible — se omite del comparador")

RANDOM_STATE = 42
DATA_PATH    = os.path.join(os.path.dirname(__file__), '..', 'data', 'credito-dataset.csv')
MODELS_DIR   = os.path.join(os.path.dirname(__file__), '..', 'models')
REPORTS_DIR  = os.path.join(os.path.dirname(__file__), '..', 'reports')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print("✔ Imports listos")

# %%
# ─────────────────────────────────────────────
# 1. CARGA Y LIMPIEZA INICIAL
# ─────────────────────────────────────────────
print("\n=== 1. CARGA DE DATOS ===")
df = pd.read_csv(DATA_PATH, sep=';')
print(f"Shape original: {df.shape}")

# 1.1 Eliminar columnas sin varianza informativa
#   FLAG_MOBIL = 1 en el 100% de los casos → no aporta información
df = df.drop(columns=['ID', 'FLAG_MOBIL'])
print(f"Tras eliminar ID y FLAG_MOBIL: {df.shape}")

# 1.2 Deduplicación
#   El dataset tiene solicitudes múltiples del mismo cliente (mismo perfil, distinto ID).
#   Nos quedamos con la primera ocurrencia para evitar data leakage en el split.
before = len(df)
df = df.drop_duplicates()
print(f"Tras deduplicar: {df.shape}  ({before - len(df):,} filas eliminadas)")

# %%
# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n=== 2. FEATURE ENGINEERING ===")

# 2.1 Tratar DAYS_EMPLOYED = 365243
#   Este valor codifica "pensionado / sin empleo activo" en el dataset original.
#   Lo reemplazamos por NaN y creamos un flag binario.
df['IS_PENSIONER_EMP'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
df['DAYS_EMPLOYED']    = df['DAYS_EMPLOYED'].replace(365243, np.nan)
df['YEARS_EMPLOYED']   = df['YEARS_EMPLOYED'].replace(-1001, np.nan)
print("✔ DAYS_EMPLOYED 365243 → NaN + flag IS_PENSIONER_EMP creado")

# 2.2 Convertir DAYS_BIRTH a positivo (venía negativo)
df['AGE'] = df['AGE'].abs()
df['DAYS_BIRTH'] = df['DAYS_BIRTH'].abs()
print("✔ AGE y DAYS_BIRTH convertidos a positivos")

# 2.3 OCCUPATION_TYPE nulos → categoría explícita "Sin datos"
df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('Sin datos')
print(f"✔ OCCUPATION_TYPE nulos imputados como 'Sin datos'")

# 2.4 Feature adicional: ratio ingreso / tamaño familiar
df['INCOME_PER_MEMBER'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'].replace(0, 1)
print("✔ INCOME_PER_MEMBER creado")

print(f"\nShape final tras feature engineering: {df.shape}")

# %%
# ─────────────────────────────────────────────
# 3. DEFINICIÓN DE FEATURES Y SPLIT
# ─────────────────────────────────────────────
print("\n=== 3. SPLIT TRAIN / TEST ===")

TARGET = 'APPROVED'
DROP   = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'YEARS_EMPLOYED']  # redundantes con AGE/YEARS_EMPLOYED limpio

# Separamos features y target
X = df.drop(columns=[TARGET] + DROP)
y = df[TARGET]

# Identificar tipos de columnas
CAT_COLS = X.select_dtypes(include=['object', 'string']).columns.tolist()
NUM_COLS = X.select_dtypes(include=['number']).columns.tolist()

print(f"Features numéricas ({len(NUM_COLS)}): {NUM_COLS}")
print(f"Features categóricas ({len(CAT_COLS)}): {CAT_COLS}")
print(f"\nDistribución target:\n{y.value_counts()}\n{y.value_counts(normalize=True).round(3)}")

# Stratified split para mantener proporción de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

# %%
# ─────────────────────────────────────────────
# 4. PIPELINE DE PREPROCESAMIENTO
# ─────────────────────────────────────────────
print("\n=== 4. PIPELINE PREPROCESAMIENTO ===")

# Numérico: imputar con mediana (robusto a outliers), escalar
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

# Categórico: imputar con el más frecuente, OHE
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe',     OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, NUM_COLS),
    ('cat', cat_pipeline, CAT_COLS)
])

print("✔ Pipeline de preprocesamiento definido")

# %%
# ─────────────────────────────────────────────
# 5. ENTRENAMIENTO — RANDOM FOREST (modelo principal)
# ─────────────────────────────────────────────
print("\n=== 5. RANDOM FOREST (modelo principal) ===")

rf_pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf',  RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=10,
        class_weight='balanced',   # compensa desbalance 73/27
        random_state=RANDOM_STATE,
        n_jobs=-1
    ))
])

print("Entrenando Random Forest (puede tardar ~1-2 min con 350k filas)...")
rf_pipeline.fit(X_train, y_train)
print("✔ Random Forest entrenado")

# %%
# ─────────────────────────────────────────────
# 6. ENTRENAMIENTO — XGBOOST (modelo comparador)
# ─────────────────────────────────────────────
print("\n=== 6. XGBOOST (modelo comparador) ===")

scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

if XGBOOST_AVAILABLE:
    xgb_pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf',  XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos,   # compensa desbalance
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        ))
    ])
    print("Entrenando XGBoost...")
    xgb_pipeline.fit(X_train, y_train)
    print("✔ XGBoost entrenado")
else:
    xgb_pipeline = None
    print("⚠ XGBoost omitido")

# %%
# ─────────────────────────────────────────────
# 7. EVALUACIÓN COMPARATIVA
# ─────────────────────────────────────────────
print("\n=== 7. EVALUACIÓN ===")

def evaluate_model(name, pipeline, X_test, y_test):
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = {
        'modelo':    name,
        'accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'f1_macro':  round(f1_score(y_test, y_pred, average='macro'), 4),
        'f1_clase0': round(f1_score(y_test, y_pred, pos_label=0), 4),
        'f1_clase1': round(f1_score(y_test, y_pred, pos_label=1), 4),
        'precision': round(precision_score(y_test, y_pred, average='macro'), 4),
        'recall':    round(recall_score(y_test, y_pred, average='macro'), 4),
        'roc_auc':   round(roc_auc_score(y_test, y_proba), 4),
        'cm':        confusion_matrix(y_test, y_pred).tolist(),
        'report':    classification_report(y_test, y_pred, target_names=['Rechazado','Aprobado'])
    }
    return metrics

results = {}
results['Random Forest'] = evaluate_model('Random Forest', rf_pipeline, X_test, y_test)
if xgb_pipeline:
    results['XGBoost'] = evaluate_model('XGBoost', xgb_pipeline, X_test, y_test)

# Imprimir resumen
print("\n{'='*60}")
for name, m in results.items():
    print(f"\n--- {name} ---")
    print(f"  Accuracy : {m['accuracy']}")
    print(f"  F1 Macro : {m['f1_macro']}")
    print(f"  F1 Clase0 (Rechazado): {m['f1_clase0']}")
    print(f"  F1 Clase1 (Aprobado) : {m['f1_clase1']}")
    print(f"  ROC-AUC  : {m['roc_auc']}")
    print(f"\n{m['report']}")

# %%
# ─────────────────────────────────────────────
# 8. VISUALIZACIONES
# ─────────────────────────────────────────────
print("\n=== 8. GENERANDO GRÁFICOS ===")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Evaluación del Modelo — Random Forest vs XGBoost', fontsize=14, fontweight='bold')

# 8.1 Matriz de confusión — Random Forest
cm_rf = np.array(results['Random Forest']['cm'])
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Rechazado','Aprobado'],
            yticklabels=['Rechazado','Aprobado'])
axes[0].set_title('Matriz de Confusión\nRandom Forest')
axes[0].set_ylabel('Real')
axes[0].set_xlabel('Predicho')

# 8.2 Matriz de confusión — XGBoost (si existe)
if 'XGBoost' in results:
    cm_xgb = np.array(results['XGBoost']['cm'])
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=['Rechazado','Aprobado'],
                yticklabels=['Rechazado','Aprobado'])
    axes[1].set_title('Matriz de Confusión\nXGBoost')
    axes[1].set_ylabel('Real')
    axes[1].set_xlabel('Predicho')
else:
    axes[1].set_visible(False)

# 8.3 Curva ROC comparativa
y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
axes[2].plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={results['Random Forest']['roc_auc']})", color='steelblue')

if xgb_pipeline:
    y_proba_xgb = xgb_pipeline.predict_proba(X_test)[:, 1]
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
    axes[2].plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={results['XGBoost']['roc_auc']})", color='darkorange')

axes[2].plot([0,1],[0,1], 'k--', alpha=0.5, label='Random')
axes[2].set_xlabel('Tasa Falsos Positivos')
axes[2].set_ylabel('Tasa Verdaderos Positivos')
axes[2].set_title('Curva ROC')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'evaluacion_modelos.png'), dpi=150, bbox_inches='tight')
print("✔ Gráfico guardado en reports/evaluacion_modelos.png")

# 8.4 Feature Importance — Random Forest
rf_clf   = rf_pipeline.named_steps['clf']
ohe_cols = rf_pipeline.named_steps['prep'].named_transformers_['cat']['ohe'].get_feature_names_out(CAT_COLS)
all_feat  = NUM_COLS + list(ohe_cols)
importances = pd.Series(rf_clf.feature_importances_, index=all_feat).sort_values(ascending=False)

fig2, ax2 = plt.subplots(figsize=(10, 6))
top20 = importances.head(20)
top20.plot(kind='barh', ax=ax2, color='steelblue', edgecolor='white')
ax2.invert_yaxis()
ax2.set_title('Top 20 Features por Importancia — Random Forest', fontweight='bold')
ax2.set_xlabel('Importancia (Gini)')
ax2.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
print("✔ Feature importance guardada en reports/feature_importance.png")

# %%
# ─────────────────────────────────────────────
# 9. GUARDAR MODELO Y MÉTRICAS
# ─────────────────────────────────────────────
print("\n=== 9. GUARDANDO ARTEFACTOS ===")

# Elegir mejor modelo según F1 macro
best_name = max(results, key=lambda k: results[k]['f1_macro'])
best_pipeline = rf_pipeline if best_name == 'Random Forest' else xgb_pipeline
print(f"Mejor modelo según F1 Macro: {best_name}")

joblib.dump(best_pipeline, os.path.join(MODELS_DIR, 'modelo_credito.pkl'))
joblib.dump(rf_pipeline,   os.path.join(MODELS_DIR, 'random_forest.pkl'))
print("✔ Modelos guardados en models/")

# Guardar columnas para uso en la app
feature_info = {
    'num_cols':  NUM_COLS,
    'cat_cols':  CAT_COLS,
    'all_input': list(X.columns),
    'best_model': best_name
}
with open(os.path.join(MODELS_DIR, 'feature_info.json'), 'w') as f:
    json.dump(feature_info, f, indent=2)

# Guardar métricas sin la CM (no serializable limpio)
metrics_clean = {k: {m: v for m, v in vals.items() if m not in ['cm','report']}
                 for k, vals in results.items()}
with open(os.path.join(REPORTS_DIR, 'metricas.json'), 'w') as f:
    json.dump(metrics_clean, f, indent=2)

print("✔ feature_info.json y metricas.json guardados")
print("\n✅ AVANCE 2 COMPLETO")
