import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Netflix AI Predictor",
    page_icon="🎬",
    layout="centered"
)

# --- 2. CARGA DE MODELOS (Cerebros) ---
@st.cache_resource # Esta función hace que la app sea súper rápida
def cargar_modelos():
    modelo = joblib.load('modelo_netflix_pro.pkl')
    columnas = joblib.load('nombres_columnas.pkl')
    return modelo, columnas

modelo, columnas = cargar_modelos()

# --- 3. INTERFAZ DE USUARIO (UI) ---
st.title("🎬 Netflix AI Pro - Sebastián")
st.markdown("---")
st.write("Ingresá los datos para que la Inteligencia Artificial prediga la duración de la película.")

# Columnas para que los controles se vean ordenados
col1, col2 = st.columns(2)

with col1:
    anio = st.number_input("Año de Estreno", min_value=1940, max_value=2030, value=2024)
    genero = st.selectbox("Género Principal", ["Dramas", "Comedies", "Action", "Documentaries", "International Movies"])

with col2:
    pais = st.selectbox("País de Origen", ["United States", "Argentina", "Spain", "Mexico", "United Kingdom"])
    estacionalidad = st.slider("Mes de estreno (Estacionalidad)", 1, 12, 6)

# --- 4. LÓGICA DE PREDICCIÓN ---
if st.button("🚀 Calcular Predicción"):
    # Creamos un DataFrame con ceros (como hacíamos en Tkinter)
    input_data = pd.DataFrame(np.zeros((1, len(columnas))), columns=columnas)
    
    # Asignamos los valores ingresados
    input_data['release_year'] = anio
    
    # Activamos las columnas "dummies" (Hot Encoding)
    gen_col = f'listed_in_{genero}'
    pais_col = f'country_{pais}'
    
    if gen_col in input_data.columns:
        input_data[gen_col] = 1
    if pais_col in input_data.columns:
        input_data[pais_col] = 1
        
    # Realizamos la predicción
    prediccion = modelo.predict(input_data)[0]
    
    # Resultado con estilo web
    st.balloons() # ¡Efecto de festejo!
    st.success(f"### Duración Estimada: {prediccion:.2f} minutos")
    st.info(f"Análisis realizado para una producción de {genero} en {pais}.")

# --- 5. PIE DE PÁGINA ---
st.markdown("---")
st.caption("Proyecto desarrollado por Sebastián - Data Science & IT Professional")