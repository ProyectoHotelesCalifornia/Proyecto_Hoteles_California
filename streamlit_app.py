import streamlit as st
import pandas as pd
import openpyxl as op

st.title("Proyecto BI")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

df = pd.read_excel("Data.xlsx")
df.head()