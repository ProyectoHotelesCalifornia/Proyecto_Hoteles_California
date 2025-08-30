import streamlit as st
import pandas as pd

st.title("Proyecto BI")
st.write(
    "Let's start building!"
)

df = pd.read_excel("Datos/Data.xlsx")
df.head()