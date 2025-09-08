# import streamlit as st
# import pandas as pd
# import os

# st.title("Proyecto BI")
# st.write("Let's start building!")

# # Asegurar ruta correcta
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# file_path = os.path.join(BASE_DIR, "Data.xlsx")

# # Leer archivo Excel
# df = pd.read_excel(file_path)

# # Mostrar en la app
# st.dataframe(df.head())

# # Chatbot simple
# st.header("Chatbot basado en Excel")
# user_question = st.text_input("Haz una pregunta sobre los datos:")

# if user_question:
#     # Buscar coincidencias en todas las columnas
#     mask = df.apply(lambda row: row.astype(str).str.contains(user_question, case=False).any(), axis=1)
#     results = df[mask]
#     if not results.empty:
#         st.write("Resultados encontrados:")
#         st.dataframe(results)
#     else:
#         st.write("No se encontraron coincidencias para tu pregunta.")

# archivo: hotel_search.py

# archivo: hotel_search.py

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# ---------------------------
# 1. Descargar stopwords en español (solo la primera vez)
# ---------------------------
nltk.download("stopwords")
stop_words = stopwords.words("spanish")

# ---------------------------
# 2. Cargar dataset de ejemplo
# ---------------------------
hoteles = pd.DataFrame({
    "id": [1, 2, 3],
    "nombre": ["Hotel Pacific Breeze", "Sunset Inn", "Quiet Garden Hotel"],
    "ciudad": ["Los Angeles", "San Diego", "Los Angeles"],
    "estrellas": [4, 3, 5]
})

reseñas = pd.DataFrame({
    "id_hotel": [1, 2, 3],
    "reseña": [
        "Muy tranquilo y cerca de la playa.",
        "Económico pero ruidoso, ubicado en el centro.",
        "Un lugar muy silencioso con jardines hermosos."
    ]
})

# ---------------------------
# 3. Preparar TF-IDF con stopwords en español
# ---------------------------
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(reseñas["reseña"])

# ---------------------------
# 4. Interfaz en Streamlit
# ---------------------------
st.title("Buscador de Hoteles en California 🏨")

# Filtros
ciudad_sel = st.selectbox("Selecciona ciudad:", ["Todas"] + sorted(hoteles["ciudad"].unique().tolist()))
estrellas_sel = st.selectbox("Filtrar por estrellas:", ["Todas", 3, 4, 5])

consulta = st.text_input("¿Qué buscas en el hotel? (ej. tranquilo, cerca de la playa, económico)")

if st.button("Buscar"):
    if consulta.strip() == "":
        st.warning("Por favor escribe algo en la consulta.")
    else:
        # Vectorizamos la consulta
        q_vec = vectorizer.transform([consulta])
        sims = cosine_similarity(q_vec, X).flatten()

        # Ordenamos reseñas según similitud
        reseñas["similaridad"] = sims
        resultados = reseñas.sort_values(by="similaridad", ascending=False)

        # Unimos con hoteles
        resultados = resultados.merge(hoteles, left_on="id_hotel", right_on="id")

        # Aplicamos filtros
        if ciudad_sel != "Todas":
            resultados = resultados[resultados["ciudad"] == ciudad_sel]
        if estrellas_sel != "Todas":
            resultados = resultados[resultados["estrellas"] == estrellas_sel]

        # Mostramos top 5
        if resultados.empty:
            st.error("No se encontraron hoteles con esos criterios.")
        else:
            st.subheader("Resultados:")
            for _, row in resultados.head(5).iterrows():
                st.write(f"🏨 **{row['nombre']}** ({row['estrellas']}⭐, {row['ciudad']})")
                st.write(f"Reseña destacada: {row['reseña']}")
                st.write("---")
