"""
Entrenamiento completo + generación de artefactos para Avance 2, 3 y 4.
Sistema ML de Pre-evaluación Crediticia · UPATECO · 2026
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, json, warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import shap

RANDOM_STATE = 42
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)
os.makedirs(os.path.join(BASE, 'reports'), exist_ok=True)

# ── 1. Carga y limpieza ──────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE, 'data', 'credito-dataset.csv'), sep=';')
df = df.drop(columns=['ID', 'FLAG_MOBIL'])

# ── 2. Feature engineering ───────────────────────────────────────────
df['IS_PENSIONER_EMP'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
df['DAYS_EMPLOYED']    = df['DAYS_EMPLOYED'].replace(365243, np.nan)
df['YEARS_EMPLOYED']   = df['YEARS_EMPLOYED'].replace(-1001, np.nan)
df['AGE']              = df['AGE'].abs()
df['DAYS_BIRTH']       = df['DAYS_BIRTH'].abs()
df['OCCUPATION_TYPE']  = df['OCCUPATION_TYPE'].fillna('Sin datos')
df['INCOME_PER_MEMBER']= df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS'].replace(0, 1)

TARGET = 'APPROVED'
DROP   = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'YEARS_EMPLOYED']
X = df.drop(columns=[TARGET] + DROP)
y = df[TARGET]

CAT_COLS = X.select_dtypes(include=['object','string']).columns.tolist()
NUM_COLS = X.select_dtypes(include=['number']).columns.tolist()

# ── 3. Split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ── 4. Pipeline preprocesamiento ─────────────────────────────────────
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer([('num', num_pipeline, NUM_COLS), ('cat', cat_pipeline, CAT_COLS)])

# ── 5. Random Forest ─────────────────────────────────────────────────
rf_pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=10,
                                    class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1))
])
rf_pipeline.fit(X_train, y_train)
print("✔ Random Forest entrenado")

# ── 6. XGBoost ────────────────────────────────────────────────────────
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
xgb_pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf', XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                           scale_pos_weight=scale_pos, eval_metric='logloss',
                           random_state=RANDOM_STATE, n_jobs=-1, verbosity=0))
])
xgb_pipeline.fit(X_train, y_train)
print("✔ XGBoost entrenado")

# ── 7. Evaluación ─────────────────────────────────────────────────────
def evaluate(name, pipe):
    yp  = pipe.predict(X_test)
    ypr = pipe.predict_proba(X_test)[:, 1]
    return {
        'modelo':    name,
        'accuracy':  round(accuracy_score(y_test, yp), 4),
        'f1_macro':  round(f1_score(y_test, yp, average='macro'), 4),
        'f1_clase0': round(f1_score(y_test, yp, pos_label=0), 4),
        'f1_clase1': round(f1_score(y_test, yp, pos_label=1), 4),
        'roc_auc':   round(roc_auc_score(y_test, ypr), 4),
        'cm':        confusion_matrix(y_test, yp).tolist(),
    }

results = {'Random Forest': evaluate('Random Forest', rf_pipeline),
           'XGBoost': evaluate('XGBoost', xgb_pipeline)}
print("Métricas:", {k: {m: v for m, v in r.items() if m != 'cm'} for k, r in results.items()})

# ── 8. Guardar modelos y metadatos ───────────────────────────────────
joblib.dump(xgb_pipeline, os.path.join(BASE, 'models', 'modelo_credito.pkl'))
joblib.dump(rf_pipeline,  os.path.join(BASE, 'models', 'random_forest.pkl'))

feature_info = {'num_cols': NUM_COLS, 'cat_cols': CAT_COLS, 'all_input': list(X.columns), 'best_model': 'XGBoost'}
with open(os.path.join(BASE, 'models', 'feature_info.json'), 'w') as f:
    json.dump(feature_info, f, indent=2)

metrics_export = {k: {m: v for m, v in vals.items() if m != 'cm'} for k, vals in results.items()}
with open(os.path.join(BASE, 'reports', 'metricas.json'), 'w') as f:
    json.dump(metrics_export, f, indent=2)
print("✔ Modelos y métricas guardados")

# ── 9. Gráfico evaluación (matrices + ROC) ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Evaluación del Sistema de Pre-evaluación Crediticia', fontsize=14, fontweight='bold')

cm_rf = np.array(results['Random Forest']['cm'])
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Rechazado','Aprobado'], yticklabels=['Rechazado','Aprobado'])
axes[0].set_title(f"Matriz de Confusión\nRandom Forest (F1={results['Random Forest']['f1_macro']})")
axes[0].set_ylabel('Real'); axes[0].set_xlabel('Predicho')

cm_xgb = np.array(results['XGBoost']['cm'])
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
            xticklabels=['Rechazado','Aprobado'], yticklabels=['Rechazado','Aprobado'])
axes[1].set_title(f"Matriz de Confusión\nXGBoost (F1={results['XGBoost']['f1_macro']})")
axes[1].set_ylabel('Real'); axes[1].set_xlabel('Predicho')

fpr_rf,  tpr_rf,  _ = roc_curve(y_test, rf_pipeline.predict_proba(X_test)[:, 1])
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_pipeline.predict_proba(X_test)[:, 1])
axes[2].plot(fpr_rf,  tpr_rf,  label=f"RF (AUC={results['Random Forest']['roc_auc']})", color='steelblue', lw=2)
axes[2].plot(fpr_xgb, tpr_xgb, label=f"XGB (AUC={results['XGBoost']['roc_auc']})", color='darkorange', lw=2)
axes[2].plot([0,1],[0,1],'k--', alpha=0.4)
axes[2].set_xlabel('FPR'); axes[2].set_ylabel('TPR'); axes[2].set_title('Curva ROC')
axes[2].legend(); axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'reports', 'evaluacion_modelos.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✔ evaluacion_modelos.png")

# ── 10. Feature importance (Random Forest) ───────────────────────────
rf_clf    = rf_pipeline.named_steps['clf']
ohe_cols  = rf_pipeline.named_steps['prep'].named_transformers_['cat']['ohe'].get_feature_names_out(CAT_COLS)
feature_names_full = NUM_COLS + list(ohe_cols)
importances = pd.Series(rf_clf.feature_importances_, index=feature_names_full).sort_values(ascending=False)

fig2, ax2 = plt.subplots(figsize=(10, 6))
importances.head(15).plot(kind='barh', ax=ax2, color='steelblue', edgecolor='white')
ax2.invert_yaxis()
ax2.set_title('Top 15 Variables por Importancia — Random Forest', fontweight='bold', fontsize=13)
ax2.set_xlabel('Importancia (Gini)'); ax2.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'reports', 'feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✔ feature_importance.png")

# ── 11. SHAP — explicabilidad global ──────────────────────────────────
print("Calculando SHAP global (muestra 500 casos)...")
X_test_prep = xgb_pipeline.named_steps['prep'].transform(X_test)
X_test_df   = pd.DataFrame(X_test_prep, columns=feature_names_full)
sample      = X_test_df.sample(500, random_state=42)

explainer   = shap.TreeExplainer(xgb_pipeline.named_steps['clf'])
shap_values = explainer.shap_values(sample)

fig3 = plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, sample, feature_names=feature_names_full,
                   max_display=15, show=False, plot_type='dot')
plt.title('Explicabilidad SHAP — Impacto de Variables en la Predicción', fontweight='bold', fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'reports', 'shap_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✔ shap_summary.png")

fig4, ax4 = plt.subplots(figsize=(10, 6))
shap_importance = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names_full).sort_values(ascending=False)
shap_importance.head(15).plot(kind='barh', ax=ax4, color='darkorange', edgecolor='white')
ax4.invert_yaxis()
ax4.set_title('Importancia Media SHAP — Top 15 Variables', fontweight='bold', fontsize=13)
ax4.set_xlabel('|SHAP value| promedio'); ax4.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'reports', 'shap_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✔ shap_importance.png")

# ── 12. Guardar el explainer SHAP para uso en la app ──────────────────
# Esto permite que la app calcule explicaciones por caso individual sin
# tener que recalcular todo el TreeExplainer cada vez.
joblib.dump(explainer, os.path.join(BASE, 'models', 'shap_explainer.pkl'))
with open(os.path.join(BASE, 'models', 'shap_feature_names.json'), 'w') as f:
    json.dump(feature_names_full, f)
print("✔ shap_explainer.pkl guardado para uso en la app")

print("\n✅ TODOS LOS ARTEFACTOS GENERADOS")
