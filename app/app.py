"""
App de Pre-evaluación Crediticia — Entrega Final
Avances 2, 3 y 4 completos, incluyendo explicabilidad SHAP
Modelado de Sistemas de IA Aplicada · UPATECO · 2026
Equipo: Ariel Escalante · Carolina Rivarola · Federico Lemos · Natalia Peloc
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import shap
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

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, '..', 'models')
REPORTS_DIR = os.path.join(BASE_DIR, '..', 'reports')

# ─────────────────────────────────────────────
# CARGA DE MODELO Y EXPLAINER
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, 'modelo_credito.pkl'))
    with open(os.path.join(MODELS_DIR, 'feature_info.json')) as f:
        info = json.load(f)
    return model, info

@st.cache_resource
def load_explainer():
    path = os.path.join(MODELS_DIR, 'shap_explainer.pkl')
    names_path = os.path.join(MODELS_DIR, 'shap_feature_names.json')
    if os.path.exists(path) and os.path.exists(names_path):
        explainer = joblib.load(path)
        with open(names_path) as f:
            feature_names = json.load(f)
        return explainer, feature_names
    return None, None

@st.cache_data
def load_metrics():
    path = os.path.join(REPORTS_DIR, 'metricas.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

model, feat_info       = load_model()
explainer, shap_feats  = load_explainer()
metrics                = load_metrics()

# ─────────────────────────────────────────────
# SIDEBAR — NAVEGACIÓN
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/bank-building.png", width=80)
st.sidebar.title("Sistema Crediticio")
st.sidebar.markdown("**UPATECO · 2026**")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navegación",
    ["🔍 Evaluación Individual",
     "📋 Evaluación Completa y Comunicación de Resultados", "ℹ️ Sobre el Sistema"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Modelo activo:** {feat_info.get('best_model', 'N/A')}")
st.sidebar.markdown(f"**Explicabilidad:** {'✅ SHAP activo' if explainer else '⚠️ No disponible'}")

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

# Traducción de nombres de variables para mostrar al usuario en español
FEATURE_LABELS_ES = {
    'CNT_CHILDREN': 'Cantidad de hijos',
    'AMT_INCOME_TOTAL': 'Ingreso anual total',
    'AGE': 'Edad',
    'FLAG_WORK_PHONE': 'Teléfono laboral',
    'FLAG_PHONE': 'Teléfono personal',
    'FLAG_EMAIL': 'Email',
    'CNT_FAM_MEMBERS': 'Integrantes del grupo familiar',
    'IS_PENSIONER_EMP': 'Es jubilado/a',
    'INCOME_PER_MEMBER': 'Ingreso per cápita familiar',
}

def feature_label(raw_name):
    """Traduce el nombre técnico de una variable (incluyendo las generadas por OHE) a español legible."""
    if raw_name in FEATURE_LABELS_ES:
        return FEATURE_LABELS_ES[raw_name]
    # Variables categóricas tras One-Hot Encoding: ej "cat__NAME_INCOME_TYPE_Pensioner"
    for prefix, mapping, label in [
        ('CODE_GENDER_', GENDER_OPTS, 'Género'),
        ('FLAG_OWN_CAR_', YESNO_OPTS, 'Vehículo propio'),
        ('FLAG_OWN_REALTY_', YESNO_OPTS, 'Inmueble propio'),
        ('NAME_INCOME_TYPE_', INCOME_TYPE_ES, 'Tipo de ingreso'),
        ('NAME_EDUCATION_TYPE_', EDUCATION_ES, 'Educación'),
        ('NAME_FAMILY_STATUS_', FAMILY_ES, 'Estado civil'),
        ('NAME_HOUSING_TYPE_', HOUSING_ES, 'Vivienda'),
        ('OCCUPATION_TYPE_', OCCUPATION_ES, 'Ocupación'),
    ]:
        if raw_name.startswith(prefix):
            val = raw_name[len(prefix):]
            es_val = mapping.get(val, val)
            return f"{label}: {es_val}"
    return raw_name

# ─────────────────────────────────────────────
# FUNCIONES DE PREDICCIÓN Y EXPLICABILIDAD
# ─────────────────────────────────────────────
def build_input(vals: dict) -> pd.DataFrame:
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

def explain_prediction(input_df):
    """Calcula los valores SHAP para un caso individual usando el explainer pre-entrenado."""
    if explainer is None:
        return None, None
    input_prep = model.named_steps['prep'].transform(input_df)
    input_prep_df = pd.DataFrame(input_prep, columns=shap_feats)
    shap_vals = explainer.shap_values(input_prep_df)
    return shap_vals[0], input_prep_df.iloc[0]

# ═══════════════════════════════════════════════════════
# PÁGINA 1 — EVALUACIÓN INDIVIDUAL
# ═══════════════════════════════════════════════════════
if page == "🔍 Evaluación Individual":
    st.title("🏦 Pre-evaluación de Solicitud Crediticia")
    st.markdown(
        "Completá los datos del solicitante. El sistema analizará el perfil, estimará "
        "la probabilidad de aprobación y explicará qué variables influyeron en la decisión."
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
            shap_vals, input_row = explain_prediction(input_df)

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

        # ── EXPLICABILIDAD SHAP POR CASO INDIVIDUAL ──────────
        st.markdown("---")
        st.subheader("🧩 ¿Por qué el sistema tomó esta decisión?")

        if shap_vals is not None:
            shap_df = pd.DataFrame({
                'variable': shap_feats,
                'impacto': shap_vals
            })
            shap_df['variable_es'] = shap_df['variable'].apply(feature_label)
            shap_df = shap_df.reindex(shap_df['impacto'].abs().sort_values(ascending=False).index).head(8)
            shap_df = shap_df.sort_values('impacto')

            fig_shap, ax_shap = plt.subplots(figsize=(8, 4.5))
            colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in shap_df['impacto']]
            ax_shap.barh(shap_df['variable_es'], shap_df['impacto'], color=colors, edgecolor='white')
            ax_shap.axvline(0, color='#333333', linewidth=0.8)
            ax_shap.set_xlabel('Impacto en la predicción (SHAP)')
            ax_shap.set_title('Variables que más influyeron en este caso', fontweight='bold')
            ax_shap.grid(axis='x', alpha=0.3)
            ax_shap.spines[['top','right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close()

            st.caption(
                "🟢 Verde = empuja hacia **aprobado**.   🔴 Rojo = empuja hacia **rechazado**.  "
                "Cuanto más larga la barra, mayor el peso de esa variable en este caso particular."
            )

            top_factor = shap_df.iloc[-1]
            direccion = "a favor de la aprobación" if top_factor['impacto'] > 0 else "en contra de la aprobación"
            st.markdown(f"**Factor de mayor peso en este caso:** {top_factor['variable_es']} — actuó {direccion}.")
        else:
            st.warning("El módulo de explicabilidad SHAP no está disponible. Ejecutá el notebook de entrenamiento para generarlo.")

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
# PÁGINA 2 — AVANCE 4 (con explicabilidad)
# ═══════════════════════════════════════════════════════
elif page == "📋 Evaluación Completa y Comunicación de Resultados":
    st.title("📋 Evaluación Completa y Comunicación de Resultados")
    st.caption("Modelado de Sistemas de IA Aplicada · UPATECO · 2026")
    st.markdown("---")

    # ── 1. REPORTE DE EVALUACIÓN ──────────────────────────
    st.header("1. Reporte de Evaluación del Modelo")
    st.markdown("Resultados sobre el conjunto de test (20% de los datos, ~87.711 registros que el modelo nunca vio durante el entrenamiento).")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",        "94,84%", delta="vs RF: 94,58%")
    col2.metric("F1-Macro",        "0,9366",  delta="vs RF: 0,9335")
    col3.metric("F1 Rechazados",   "0,9093",  delta="vs RF: 0,9048")
    col4.metric("ROC-AUC",         "0,9931",  delta="vs RF: 0,9923")

    st.markdown("---")
    st.subheader("Tabla comparativa de modelos")
    tabla = pd.DataFrame({
        "Modelo":        ["Random Forest", "XGBoost ✓"],
        "Accuracy":      ["94,58%", "94,84%"],
        "F1-Macro":      ["0,9335", "0,9366"],
        "F1 Clase 0":    ["0,9048", "0,9093"],
        "F1 Clase 1":    ["0,9621", "0,9639"],
        "ROC-AUC":       ["0,9923", "0,9931"],
    })
    st.dataframe(tabla, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Visualizaciones")
    img_eval = os.path.join(REPORTS_DIR, 'evaluacion_modelos.png')
    if os.path.exists(img_eval):
        st.image(img_eval, use_container_width=True, caption="Matrices de confusión y curva ROC comparativa")

    with st.expander("📖 ¿Qué significa cada métrica?"):
        st.markdown("""
**Accuracy (94,84%):** de cada 100 solicitudes, el modelo acierta 95. No es suficiente sola porque el dataset tiene desbalance: 72,9% aprobados vs 27,1% rechazados.

**F1-Macro (0,9366):** promedia el desempeño en ambas clases por igual. Es la métrica principal porque penaliza si el modelo ignora alguna de las dos clases.

**F1 Clase 0 — Rechazados (0,9093):** qué tan bien detecta el modelo los rechazos. Crítico para la entidad financiera: un rechazo no detectado implica aprobar a alguien que no pagará.

**ROC-AUC (0,9931):** ante un aprobado y un rechazado elegidos al azar, el modelo identifica correctamente cuál es cuál en el 99,31% de los casos. Escala de 0,5 (sin información) a 1,0 (perfecto).
        """)

    st.markdown("---")

    # ── 2. EXPLICABILIDAD ─────────────────────────────────
    st.header("2. Explicabilidad del Modelo")
    st.markdown(
        "Se utilizaron dos enfoques complementarios de explicabilidad: la **importancia de variables** "
        "nativa del Random Forest (impacto global sobre todo el dataset) y **SHAP** (impacto individual "
        "por predicción, con dirección del efecto)."
    )

    st.subheader("2.1 Importancia de variables — Random Forest")
    img_feat = os.path.join(REPORTS_DIR, 'feature_importance.png')
    if os.path.exists(img_feat):
        st.image(img_feat, use_container_width=True, caption="Top 15 variables por importancia (criterio Gini)")

    st.subheader("2.2 SHAP — Impacto y dirección por variable")
    img_shap = os.path.join(REPORTS_DIR, 'shap_summary.png')
    if os.path.exists(img_shap):
        st.image(img_shap, use_container_width=True,
                 caption="Cada punto es un caso. Rojo = valor alto de la variable, azul = valor bajo. "
                         "La posición horizontal indica si esa variable empujó hacia aprobado o rechazado.")
    else:
        st.warning("Gráfico SHAP no encontrado. Ejecutá el notebook de entrenamiento para generarlo.")

    img_shap_imp = os.path.join(REPORTS_DIR, 'shap_importance.png')
    if os.path.exists(img_shap_imp):
        st.image(img_shap_imp, use_container_width=True, caption="Importancia media SHAP — Top 15 variables")

    st.markdown("""
**Principales hallazgos del análisis SHAP:**
- La condición de **jubilado** es el factor con mayor impacto negativo sobre la probabilidad de aprobación.
- La **edad** tiene efecto no lineal: edades medias (30-50 años) favorecen la aprobación; edades altas la perjudican.
- El **ingreso total** y el **ingreso per cápita familiar** tienen impacto positivo: a mayor ingreso, mayor probabilidad de aprobación.
- Las variables patrimoniales (inmueble, vehículo) tienen impacto positivo pero moderado, menor que las variables económicas.

💡 *Podés ver la explicación SHAP de un caso particular en la pestaña "Evaluación Individual", debajo del resultado de cada predicción.*
    """)

    st.markdown("---")

    # ── 3. RESUMEN PARA AUDIENCIA NO TÉCNICA ──────────────
    st.header("3. Resumen para Audiencia No Técnica")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("✅ Qué hace el sistema")
        st.markdown("""
Analiza los datos de una persona que solicita un crédito y predice si debería aprobarse o rechazarse.

Lo hace en menos de un segundo, basándose en patrones aprendidos de **438.557 casos históricos**.

Devuelve una recomendación con nivel de confianza y **explica qué variables pesaron más** en cada decisión particular.
        """)
    with col_b:
        st.subheader("⚠️ Qué no sabe hacer")
        st.markdown("""
- No consulta el historial crediticio previo (BCRA / Veraz).
- No garantiza una decisión correcta en todos los casos.
- No detecta errores en los datos ingresados por el analista.
- Requiere reentrenamiento si cambian las condiciones económicas del país.
        """)

    st.markdown("---")

    # ── 4. ANÁLISIS DE LÍMITES Y RIESGOS ──────────────────
    st.header("4. Análisis de Límites y Riesgos")
    st.markdown("Escenarios donde el sistema podría producir resultados incorrectos:")

    with st.expander("⚠️ Escenario 1 — Jubilado con ingresos altos y patrimonio consolidado"):
        st.markdown("""
**Situación:** persona de 65 años, jubilada, con ingreso de $800.000, casa y vehículo propios solicita un crédito pequeño.

**Riesgo:** el modelo podría predecir rechazo a pesar del sólido perfil económico.

**Motivo:** el dataset tiene pocos casos de jubilados con ingresos muy altos. El modelo tiene menos experiencia evaluando ese perfil y tiende a generalizar a partir del factor "jubilado", que el análisis SHAP confirma como la señal de riesgo más fuerte.
        """)

    with st.expander("⚠️ Escenario 2 — Solicitante sin información de ocupación"):
        st.markdown("""
**Situación:** el cliente no informa su ocupación.

**Riesgo:** el 30% del dataset original no tenía este dato. Aunque el modelo maneja la ausencia con la categoría "Sin datos", la falta de información reduce la precisión de la predicción.

**Motivo:** el modelo nunca puede ser más preciso que los datos que recibe.
        """)

    with st.expander("⚠️ Escenario 3 — Cambio en las condiciones económicas del país"):
        st.markdown("""
**Situación:** el contexto económico cambia significativamente después del entrenamiento del modelo.

**Riesgo:** los patrones históricos pueden dejar de ser válidos, llevando a predicciones incorrectas.

**Motivo:** el modelo no actualiza su conocimiento automáticamente. Requiere ser reentrenado con datos recientes.
        """)

    with st.expander("⚠️ Escenario 4 — Datos ingresados incorrectamente por el analista"):
        st.markdown("""
**Situación:** el analista ingresa por error $45.000 de ingreso en lugar de $450.000.

**Riesgo:** el modelo procesa los datos tal como los recibe, sin validar si son razonables. Un dato incorrecto puede cambiar completamente la predicción.

**Motivo:** el sistema no tiene mecanismos de detección de errores humanos en el ingreso de datos.
        """)

    st.markdown("---")
    st.caption("Evaluación Completa con Explicabilidad · Modelado de Sistemas de IA Aplicada · UPATECO · 2026")

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
| **Preprocesamiento** | Imputación, encoding, feature engineering |
| **Modelo** | Random Forest + XGBoost comparados con métricas completas |
| **Evaluación** | F1-Macro, ROC-AUC, Matriz de Confusión |
| **Explicabilidad** | Importancia de variables (Random Forest) + SHAP por caso individual |
| **Aplicación** | Esta app Streamlit de acceso local |

---

### Modelo seleccionado: XGBoost

**¿Por qué XGBoost?** En la evaluación comparativa superó a Random Forest en F1-Macro
(0,9366 vs 0,9335) y ROC-AUC (0,9931 vs 0,9923), con manejo nativo del desbalance
de clases mediante `scale_pos_weight`.

**Limitaciones conocidas:**
- El modelo no tiene en cuenta historial crediticio previo (BCRA / Veraz).
- Variables como ocupación tienen ~30% de datos faltantes en el dataset original.
- Un resultado positivo no garantiza aprobación: es una pre-evaluación de apoyo.

---

### Stack tecnológico

`Python` · `scikit-learn` · `XGBoost` · `SHAP` · `Pandas` · `Matplotlib/Seaborn` · `Streamlit` · `GitHub`

---

### Equipo

**Ariel Escalante · Carolina Rivarola · Federico Lemos · Natalia Peloc**

> Ciclo lectivo 2026 · Lic. Walter Gabriel Ramírez (formador)
    """)
