import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Netflix Data Pro", layout="wide")

# Título con Emoji y Color
st.markdown("<h1 style='text-align: center; color: #E50914;'>NETFLIX DATA ANALYTICS</h1>", unsafe_allow_html=True)

# 1. Título de la Web
st.title("🎬 Netflix Data Science Dashboard")

# --- 7. BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/0/08/Netflix_2015_logo.svg", width=200)
    st.title("Panel de Control")
    st.write("Bienvenido a tu herramienta de Inteligencia de Datos.")
    st.info("Usá el buscador central para filtrar películas y la IA de abajo para clasificar nuevas tramas.")
    
    # Podemos mover los filtros aquí si querés
    st.markdown("---")
    st.write("⚙️ *Configuración:*")
    st.checkbox("Mostrar datos crudos", value=False)




# 2. Carga y Limpieza Atómica (Optimización)
@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df.fillna({'director': 'Sin Datos', 'country': 'Desconocido'}, inplace=True)
    return df

df = load_data()

# 3. Sidebar (Filtros a la izquierda)
st.sidebar.header("Filtros")
tipo = st.sidebar.selectbox("Seleccioná el tipo:", df['type'].unique())

# Filtramos los datos según la elección
df_filtrado = df[df['type'] == tipo]

# 4. Mostrar Resultados

# --- MEJORA ESTÉTICA: GRÁFICO ROJO NETFLIX ---
st.subheader("Top 10 Países con más Contenido")

# Creamos el gráfico con el color oficial de Netflix (#E50914)
fig, ax = plt.subplots()
df['country'].value_counts().head(10).plot(kind='bar', color='#E50914', ax=ax)

# Estilizamos el fondo del gráfico para que combine
ax.set_facecolor('#000000') # Fondo negro para el gráfico
fig.patch.set_facecolor('#000000') # Fondo negro para el marco
ax.tick_params(colors='white') # Letras blancas
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')

st.pyplot(fig)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# --- 6. IA CLASIFICADORA PROFESIONAL (ENTRENAMIENTO COMPLETO) ---
st.markdown("---")
st.header("🧠 IA de Alta Precisión")

# 1. Usamos TODO el dataset para que la IA sea más inteligente
# Quitamos el .head(1000) para usar las 8000+ filas
df_ia = df[['description', 'type']].dropna() 

# 2. Vectorización Avanzada
# ngram_range(1,2) permite que la IA entienda frases de dos palabras (ej: "Special Agent")
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
X = vectorizer.fit_transform(df_ia['description'])
y = df_ia['type']

# 3. Entrenamos el Modelo
modelo_ia = MultinomialNB()
modelo_ia.fit(X, y)

import pickle

# Guardamos el cerebro y el vectorizador en archivos reales
with open('modelo_netflix.pkl', 'wb') as f:
    pickle.dump(modelo_ia, f)
with open('vectorizador_netflix.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)



# --- INTERFAZ ---
usuario_input = st.text_area("Poné a prueba a la IA (Entrenada con 8,000+ títulos):")

if usuario_input:
    input_vector = vectorizer.transform([usuario_input])
    prediccion = modelo_ia.predict(input_vector)[0]
    probabilidades = modelo_ia.predict_proba(input_vector)
    confianza = max(probabilidades[0]) * 100

    # Diseño de respuesta profesional
    col1, col2 = st.columns(2)
    with col1:
        if prediccion == 'Movie':
            st.success(f"🎯 Resultado: *PELÍCULA*")
        else:
            st.info(f"📺 Resultado: *SERIE*")
    with col2:
        st.metric("Nivel de Confianza", f"{confianza:.1f}%")