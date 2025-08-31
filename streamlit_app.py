import streamlit as st
import pandas as pd
import os

st.title("Proyecto BI")
st.write(
    "Let's start building!"
)

# Asegurar ruta correcta
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "Data.xlsx")

# Leer archivo Excel
df = pd.read_excel(file_path)

# Mostrar en la app
st.dataframe(df.head())
