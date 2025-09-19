import streamlit as st
import pandas as pd

# Cargar tu base de datos
df = pd.read_csv("final_database.csv")  # O el nombre de tu archivo

st.title("Explorador de Reviews por Tópico")

# Crear lista de tópicos únicos
topics = df['topic_label'].unique().tolist()

# Selectbox para que el usuario elija un tópico
selected_topic = st.selectbox("Selecciona un tópico", topics)

# Número de hoteles/reviews a mostrar
n_reviews = st.slider("Número de reviews a mostrar", min_value=1, max_value=20, value=5)

# Filtrar el dataframe por el tópico seleccionado
filtered_df = df[df['topic_label'] == selected_topic].head(n_reviews)

# Mostrar resultados
for idx, row in filtered_df.iterrows():
    st.subheader(row['name'])
    st.write("Rating:", row['ratings'])
    st.write("Review:", row['text'])
    st.write("---")
