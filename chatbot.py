import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar dataset
df = pd.read_csv("https://github.com/melody-10/Proyecto_Hoteles_California/blob/main/final_database.csv?raw=true")
texts = df["text"].tolist()

# Vectorizar con TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

st.title("Chatbot rápido con TF-IDF")

query = st.text_input("Pregunta:")

if query:
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, X).flatten()
    top_idx = sims.argsort()[-3:][::-1]  # top 3
    for idx in top_idx:
        st.write("👉", texts[idx])
