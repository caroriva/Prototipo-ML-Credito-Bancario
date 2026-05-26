"""
App de Pre-evaluación Crediticia
Avance 3 — Aplicación local funcionando
Modelado de Sistemas de IA Aplicada · UPATECO · 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Pre-evaluación Crediticia",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
REPORTS_DIR= os.path.join(BASE_DIR, '..', 'reports')

# ─────────────────────────────────────────────
# CARGA DE MODELO
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, 'modelo_credito.pkl'))
    with open(os.path.join(MODELS_DIR, 'feature_info.json')) as f:
        info = json.load(f)
    return model, info

@st.cache_data
def load_metrics():
    path = os.path.join(REPORTS_DIR, 'metricas.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

model, feat_info = load_model()
metrics = load_metrics()

# ─────────────────────────────────────────────
# SIDEBAR — NAVEGACIÓN
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/bank-building.png", width=80)
st.sidebar.title("Sistema Crediticio")
st.sidebar.markdown("**UPATECO · 2026**")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navegación",
    ["🔍 Evaluación Individual", "📊 Métricas del Modelo", "ℹ️ Sobre el Sistema"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Modelo activo:** {feat_info.get('best_model', 'N/A')}")

# ─────────────────────────────────────────────
# OPCIONES PARA SELECTBOXES
# ─────────────────────────────────────────────
GENDER_OPTS       = {'Masculino': 'M', 'Femenino': 'F'}
YESNO_OPTS        = {'Sí': 'Y', 'No': 'N'}
INCOME_TYPE_OPTS  = ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student']
INCOME_TYPE_ES    = {
    'Working': 'Empleado en relación de dependencia',
    'Commercial associate': 'Asociado comercial / autónomo',
    'Pensioner': 'Jubilado / pensionado',
    'State servant': 'Empleado estatal',
    'Student': 'Estudiante'
}
EDUCATION_OPTS    = ['Higher education', 'Secondary / secondary special',
                     'Incomplete higher', 'Lower secondary', 'Academic degree']
EDUCATION_ES      = {
    'Higher education':             'Universitario completo',
    'Secondary / secondary special':'Secundario completo',
    'Incomplete higher':            'Universitario incompleto',
    'Lower secondary':              'Secundario incompleto',
    'Academic degree':              'Posgrado / académico'
}
FAMILY_OPTS       = ['Married', 'Civil marriage', 'Single / not married', 'Separated', 'Widow']
FAMILY_ES         = {
    'Married': 'Casado/a',
    'Civil marriage': 'Unión civil',
    'Single / not married': 'Soltero/a',
    'Separated': 'Separado/a',
    'Widow': 'Viudo/a'
}
HOUSING_OPTS      = ['House / apartment', 'Rented apartment', 'With parents',
                     'Municipal apartment', 'Co-op apartment', 'Office apartment']
HOUSING_ES        = {
    'House / apartment':  'Casa / departamento propio',
    'Rented apartment':   'Departamento alquilado',
    'With parents':       'Con los padres',
    'Municipal apartment':'Departamento municipal',
    'Co-op apartment':    'Departamento cooperativo',
    'Office apartment':   'Apartamento de empresa'
}
OCCUPATION_OPTS   = ['Sin datos', 'Laborers', 'Core staff', 'Sales staff', 'Managers',
                     'Drivers', 'High skill tech staff', 'Accountants', 'Medicine staff',
                     'Cooking staff', 'Security staff', 'Cleaning staff',
                     'Private service staff', 'Low-skill Laborers', 'Secretaries',
                     'Waiters/barmen staff', 'Realty agents', 'HR staff', 'IT staff']
OCCUPATION_ES     = {
    'Sin datos': 'Sin información', 'Laborers': 'Obrero / operario',
    'Core staff': 'Personal de núcleo', 'Sales staff': 'Vendedor/a',
    'Managers': 'Gerente / directivo', 'Drivers': 'Conductor / chofer',
    'High skill tech staff': 'Técnico especializado', 'Accountants': 'Contador/a',
    'Medicine staff': 'Personal médico / salud', 'Cooking staff': 'Gastronomía',
    'Security staff': 'Seguridad', 'Cleaning staff': 'Limpieza',
    'Private service staff': 'Servicio doméstico', 'Low-skill Laborers': 'Obrero no calificado',
    'Secretaries': 'Secretario/a', 'Waiters/barmen staff': 'Gastronómico (mozo/barman)',
    'Realty agents': 'Agente inmobiliario', 'HR staff': 'Recursos humanos', 'IT staff': 'IT / sistemas'
}

# ─────────────────────────────────────────────
# FUNCIÓN DE PREDICCIÓN
# ─────────────────────────────────────────────
def build_input(vals: dict) -> pd.DataFrame:
    """Armar el DataFrame con las mismas columnas que vio el modelo."""
    row = {
        'CODE_GENDER':       vals['gender'],
        'FLAG_OWN_CAR':      vals['own_car'],
        'FLAG_OWN_REALTY':   vals['own_realty'],
        'CNT_CHILDREN':      vals['children'],
        'AMT_INCOME_TOTAL':  vals['income'],
        'NAME_INCOME_TYPE':  vals['income_type'],
        'NAME_EDUCATION_TYPE': vals['education'],
        'NAME_FAMILY_STATUS':  vals['family_status'],
        'NAME_HOUSING_TYPE':   vals['housing'],
        'AGE':               vals['age'],
        'FLAG_WORK_PHONE':   int(vals['work_phone']),
        'FLAG_PHONE':        int(vals['phone']),
        'FLAG_EMAIL':        int(vals['email']),
        'OCCUPATION_TYPE':   vals['occupation'],
        'CNT_FAM_MEMBERS':   vals['fam_members'],
        'IS_PENSIONER_EMP':  int(vals['income_type'] == 'Pensioner'),
        'INCOME_PER_MEMBER': vals['income'] / max(vals['fam_members'], 1),
    }
    return pd.DataFrame([row])

def predict(input_df):
    proba = model.predict_proba(input_df)[0]
    pred  = model.predict(input_df)[0]
    return pred, proba

# ═══════════════════════════════════════════════════════
# PÁGINA 1 — EVALUACIÓN INDIVIDUAL
# ═══════════════════════════════════════════════════════
if page == "🔍 Evaluación Individual":
    st.title("🏦 Pre-evaluación de Solicitud Crediticia")
    st.markdown(
        "Completá los datos del solicitante. El sistema analizará el perfil y "
        "estimará la probabilidad de aprobación del crédito."
    )
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Datos personales")
        gender_label   = st.selectbox("Género", list(GENDER_OPTS.keys()))
        age            = st.slider("Edad", 18, 75, 35)
        family_label   = st.selectbox("Estado civil", [FAMILY_ES[o] for o in FAMILY_OPTS])
        children       = st.number_input("Hijos", 0, 10, 0, step=1)
        fam_members    = st.number_input("Integrantes del grupo familiar", 1, 15, 2, step=1)

    with col2:
        st.subheader("💼 Perfil laboral y económico")
        income_label   = st.selectbox("Tipo de ingreso", [INCOME_TYPE_ES[o] for o in INCOME_TYPE_OPTS])
        income         = st.number_input("Ingreso anual total ($)", 10_000, 5_000_000, 200_000, step=10_000,
                                          help="Ingreso bruto anual en moneda local")
        occup_label    = st.selectbox("Ocupación", [OCCUPATION_ES[o] for o in OCCUPATION_OPTS])
        edu_label      = st.selectbox("Nivel educativo", [EDUCATION_ES[o] for o in EDUCATION_OPTS])

    with col3:
        st.subheader("🏠 Patrimonio y contacto")
        own_car_label    = st.selectbox("¿Tiene vehículo propio?", list(YESNO_OPTS.keys()))
        own_realty_label = st.selectbox("¿Tiene propiedad inmobiliaria?", list(YESNO_OPTS.keys()))
        housing_label    = st.selectbox("Tipo de vivienda", [HOUSING_ES[o] for o in HOUSING_OPTS])
        st.markdown("**Medios de contacto disponibles:**")
        work_phone = st.checkbox("Teléfono laboral", value=False)
        phone      = st.checkbox("Teléfono personal", value=True)
        email_chk  = st.checkbox("Email", value=False)

    # Mapear etiquetas en español a valores del modelo
    family_val  = FAMILY_OPTS[[FAMILY_ES[o] for o in FAMILY_OPTS].index(family_label)]
    income_val  = INCOME_TYPE_OPTS[[INCOME_TYPE_ES[o] for o in INCOME_TYPE_OPTS].index(income_label)]
    occup_val   = OCCUPATION_OPTS[[OCCUPATION_ES[o] for o in OCCUPATION_OPTS].index(occup_label)]
    edu_val     = EDUCATION_OPTS[[EDUCATION_ES[o] for o in EDUCATION_OPTS].index(edu_label)]
    housing_val = HOUSING_OPTS[[HOUSING_ES[o] for o in HOUSING_OPTS].index(housing_label)]

    st.markdown("---")
    if st.button("🔍 Evaluar solicitud", use_container_width=True, type="primary"):
        vals = dict(
            gender=GENDER_OPTS[gender_label], own_car=YESNO_OPTS[own_car_label],
            own_realty=YESNO_OPTS[own_realty_label], children=children, income=income,
            income_type=income_val, education=edu_val, family_status=family_val,
            housing=housing_val, age=age, work_phone=work_phone, phone=phone,
            email=email_chk, occupation=occup_val, fam_members=fam_members
        )
        input_df = build_input(vals)

        with st.spinner("Analizando perfil crediticio..."):
            pred, proba = predict(input_df)

        prob_aprobado  = proba[1]
        prob_rechazado = proba[0]

        st.markdown("---")
        st.subheader("📋 Resultado de la evaluación")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if pred == 1:
                st.success("### ✅ APROBADO")
                st.metric("Probabilidad de aprobación", f"{prob_aprobado:.1%}")
            else:
                st.error("### ❌ NO APROBADO")
                st.metric("Probabilidad de rechazo", f"{prob_rechazado:.1%}")

            st.markdown("**Confianza del modelo:**")
            nivel = "Alta" if max(prob_aprobado, prob_rechazado) > 0.85 else \
                    "Media" if max(prob_aprobado, prob_rechazado) > 0.70 else "Baja"
            st.info(f"Nivel de confianza: **{nivel}**")

        with res_col2:
            fig, ax = plt.subplots(figsize=(5, 2.5))
            bars = ax.barh(
                ['Rechazado', 'Aprobado'],
                [prob_rechazado, prob_aprobado],
                color=['#e74c3c', '#2ecc71'],
                edgecolor='white', height=0.5
            )
            for bar, pct in zip(bars, [prob_rechazado, prob_aprobado]):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{pct:.1%}', va='center', fontsize=11, fontweight='bold')
            ax.set_xlim(0, 1.15)
            ax.set_xlabel('Probabilidad')
            ax.set_title('Distribución de probabilidades', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            ax.spines[['top','right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Resumen del perfil evaluado
        with st.expander("📄 Ver resumen del perfil evaluado"):
            resumen = {
                'Género': gender_label, 'Edad': f'{age} años',
                'Estado civil': family_label, 'Hijos': children,
                'Grupo familiar': fam_members, 'Tipo de ingreso': income_label,
                'Ingreso anual': f'${income:,.0f}', 'Ocupación': occup_label,
                'Educación': edu_label, 'Vehículo propio': own_car_label,
                'Inmueble propio': own_realty_label, 'Tipo de vivienda': housing_label,
                'Ingreso per cápita': f'${income/max(fam_members,1):,.0f}'
            }
            st.table(pd.DataFrame.from_dict(resumen, orient='index', columns=['Valor']))

        st.caption(
            "⚠️ Este resultado es una estimación basada en datos históricos. "
            "No reemplaza la evaluación crediticia formal de la entidad financiera."
        )

# ═══════════════════════════════════════════════════════
# PÁGINA 2 — MÉTRICAS
# ═══════════════════════════════════════════════════════
elif page == "📊 Métricas del Modelo":
    st.title("📊 Métricas de Evaluación del Modelo")
    st.markdown(
        "Resultados obtenidos en el conjunto de test (20% de los datos, ~18.000 registros)."
    )

    if metrics:
        tab_names = list(metrics.keys())
        tabs = st.tabs(tab_names)
        for tab, name in zip(tabs, tab_names):
            m = metrics[name]
            with tab:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy",  f"{m['accuracy']:.2%}")
                c2.metric("F1 Macro",  f"{m['f1_macro']:.2%}")
                c3.metric("ROC-AUC",   f"{m['roc_auc']:.4f}")
                c4.metric("F1 Clase 0\n(Rechazado)", f"{m['f1_clase0']:.2%}")

    st.markdown("---")
    st.subheader("Comparación visual de modelos")
    img_eval = os.path.join(REPORTS_DIR, 'evaluacion_modelos.png')
    if os.path.exists(img_eval):
        st.image(img_eval, use_container_width=True)
    else:
        st.warning("Imagen no encontrada. Ejecutá primero el notebook de entrenamiento.")

    st.markdown("---")
    st.subheader("Importancia de variables — Random Forest")
    img_feat = os.path.join(REPORTS_DIR, 'feature_importance.png')
    if os.path.exists(img_feat):
        st.image(img_feat, use_container_width=True)

    st.markdown("---")
    with st.expander("📖 Interpretación de métricas"):
        st.markdown("""
**¿Por qué no usamos solo Accuracy?**

El dataset tiene desbalance de clases: 72,9% aprobados vs 27,1% rechazados.
Un modelo que prediga siempre "Aprobado" obtendría 72,9% de accuracy sin aprender nada.
Por eso priorizamos:

- **F1-Macro**: promedio entre F1 de aprobados y rechazados. Penaliza si el modelo ignora alguna clase.
- **ROC-AUC**: mide la capacidad discriminativa general del modelo (1.0 = perfecto, 0.5 = aleatorio).

**¿Cuándo puede fallar el modelo?**

- Perfiles muy poco frecuentes en los datos históricos: el modelo aprende de ejemplos, si nunca vio muchos casos similares, puede equivocarse.
- Jubilados con ingresos altos: hay pocos casos así en el dataset, por lo que el modelo tiene menos experiencia evaluando ese perfil.
- Clientes sin información de ocupación: el 30% del dataset original no tenía ese dato, lo que limita la precisión en esos casos.
        """)

# ═══════════════════════════════════════════════════════
# PÁGINA 3 — SOBRE EL SISTEMA
# ═══════════════════════════════════════════════════════
elif page == "ℹ️ Sobre el Sistema":
    st.title("ℹ️ Sobre el Sistema")

    st.markdown("""
## Sistema ML de Pre-evaluación Crediticia

Este sistema fue desarrollado como Trabajo Integrador Final (TIF) para la materia
**Modelado de Sistemas de IA Aplicada** de la Tecnicatura Universitaria en Ciencia de
Datos e IA Aplicada — **UPATECO, Salta, 2026**.

---

### El problema que resuelve

Las entidades financieras analizan manualmente miles de solicitudes de crédito,
generando demoras, costos operativos elevados y riesgo de subjetividad.
Este sistema automatiza la **pre-evaluación** usando un modelo predictivo entrenado
sobre 438.557 solicitudes históricas de clientes bancarios.

---

### Ciclo técnico recorrido

| Etapa | Descripción |
|-------|-------------|
| **Datos** | CSV con 438.557 registros y 21 variables |
| **EDA** | Análisis exploratorio: distribución de clases, nulos, outliers |
| **Preprocesamiento** | Tratamiento de datos faltantes, encoding, feature engineering |
| **Modelo** | Random Forest + XGBoost comparados con métricas completas |
| **Evaluación** | F1-Macro, ROC-AUC, Matriz de Confusión |
| **Aplicación** | Esta app Streamlit de acceso local |

---

### Modelo seleccionado: XGBoost

**¿Por qué XGBoost?** En la evaluación comparativa superó a Random Forest en F1-Macro
(0.929 vs 0.923) y ROC-AUC (0.9887 vs 0.9859), con manejo nativo del desbalance
de clases mediante `scale_pos_weight`.

**Limitaciones conocidas:**
- El modelo no tiene en cuenta historial crediticio previo (BCRA / Veraz).
- Variables como ocupación tienen ~30% de datos faltantes en el dataset original.
- Un resultado positivo **no garantiza** aprobación: es una pre-evaluación de apoyo.

---

### Stack tecnológico

`Python` · `scikit-learn` · `XGBoost` · `Pandas` · `Matplotlib/Seaborn` · `Streamlit` · `GitHub`

---

### Equipo

> Ariel Escalante · Carolina Rivarola · Federico Lemos · Natalia Peloc
    """)
