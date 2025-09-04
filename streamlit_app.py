import streamlit as st
import pandas as pd
import os

st.title("Proyecto BI")
st.write("Let's start building!")

# Asegurar ruta correcta
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "Data.xlsx")

# Leer archivo Excel
df = pd.read_excel(file_path)

# Mostrar en la app
st.dataframe(df.head())

# Chatbot simple
st.header("Chatbot basado en Excel")
user_question = st.text_input("Haz una pregunta sobre los datos:")

if user_question:
    # Buscar coincidencias en todas las columnas
    mask = df.apply(lambda row: row.astype(str).str.contains(user_question, case=False).any(), axis=1)
    results = df[mask]
    if not results.empty:
        st.write("Resultados encontrados:")
        st.dataframe(results)
    else:
        st.write("No se encontraron coincidencias para tu pregunta.")