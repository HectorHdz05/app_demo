import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

st.set_page_config(page_title="LIME Dashboard", page_icon="🧠", layout="centered")

# --- 1. Título ---
st.title("🧠 LIME Dashboard: Explicación de costos en actividades")

# --- 2. Carga de datos: archivo del alumno o dataset por defecto ---
st.sidebar.header("📁 Datos")

uploaded_file = st.sidebar.file_uploader(
    "Sube tu archivo CSV con tus actividades",
    type="csv"
)

@st.cache_data
def cargar_datos_defecto():
    df = pd.read_csv("datos_actividades.csv")
    return df

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Usando el archivo que subiste ✅")
else:
    df = cargar_datos_defecto()
    st.sidebar.info("Usando el dataset de ejemplo por defecto.")

# Limpiamos datos básicos
df = df.dropna()

# Validación mínima de columnas necesarias
columnas_necesarias = [
    "Name of the activity",
    "Cost",
    "Budget",
    "Time invested",
    "Type",
    "Moment",
    "No. of people"
]

faltantes = [c for c in columnas_necesarias if c not in df.columns]
if faltantes:
    st.error(f"Faltan estas columnas en tu CSV: {faltantes}")
    st.stop()

# --- 3. Seleccionar actividad (selectbox en lugar de slider) ---
st.sidebar.subheader("📌 Selección de actividad")

opciones = [
    f"{i} - {row['Name of the activity']}"
    for i, row in df.iterrows()
]

opcion_sel = st.sidebar.selectbox(
    "Elige una actividad por nombre",
    opciones
)

# Recuperar índice
idx = int(opcion_sel.split(" - ")[0])

# --- 4. Preparar datos para el modelo ---
X = df[['Budget', 'Time invested', 'Type', 'Moment', 'No. of people']]
y = df['Cost']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42
)
model = LinearRegression().fit(X_train, y_train)

# --- 5. Mostrar datos de la instancia seleccionada ---
actividad = df.iloc[idx]
x_instance = X.iloc[idx].values.reshape(1, -1)

st.subheader("📋 Detalles de la actividad seleccionada")
st.write(
    actividad[
        [
            'Name of the activity',
            'Cost',
            'Budget',
            'Time invested',
            'Type',
            'Moment',
            'No. of people'
        ]
    ]
)

pred = model.predict(x_instance)[0]
st.metric("💸 Costo predicho", f"${pred:.2f}")
st.metric("📊 Costo real", f"${actividad['Cost']:.2f}")

# --- 6. Explicación con LIME ---
explainer = LimeTabularExplainer(
    training_data=X.values,
    feature_names=X.columns.tolist(),
    mode='regression'
)

exp = explainer.explain_instance(
    X.iloc[idx].values,
    model.predict,
    num_features=5
)

st.subheader("🔍 Explicación local (LIME)")
exp_list = exp.as_list()
for feature, weight in exp_list:
    signo = "⬆️" if weight > 0 else "⬇️"
    st.write(f"{signo} **{feature}** → {weight:.2f}")

# --- 7. Gráfico ---
fig = exp.as_pyplot_figure()
st.pyplot(fig)

# --- 8. “Recomendaciones” automáticas basadas en LIME ---
st.subheader("🤖 Recomendaciones automáticas (beta)")

# Tomamos las 2 variables con mayor impacto absoluto
exp_sorted = sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)
top_exp = exp_sorted[:2]

reco_textos = []
for feature, weight in top_exp:
    impacto = "aumenta" if weight > 0 else "disminuye"
    reco_textos.append(
        f"- La condición **{feature}** {impacto} el costo estimado."
    )

st.write("Con base en la explicación de LIME para esta actividad:")
for linea in reco_textos:
    st.write(linea)

st.info(
    "Puedes usar estas observaciones para reflexionar sobre qué variables "
    "tienen mayor impacto en el costo de tus actividades."
)

# --- 9. Reflexión guiada ---
with st.expander("🧠 Reflexión personal (opcional)"):
    st.write("¿Qué aprendiste de esta explicación y de las recomendaciones?")
    st.text_area("Escribe aquí tu reflexión:", "")
