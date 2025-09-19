import streamlit as st
import pandas as pd

# Cargar la base de datos
df = pd.read_csv("reviews_with_topics.csv")  # Cambia por tu archivo

st.title("Explorador de Reviews por Tópico y Hotel")

# --- Filtro por tópico ---
topics = df['topic_label'].unique().tolist()
selected_topic = st.selectbox("Selecciona un tópico", topics)

# --- Filtro por hotel ---
hotel_options = ['Todos'] + sorted(df['name'].unique().tolist())
selected_hotel = st.selectbox("Selecciona un hotel", hotel_options)

# Número máximo de reviews a mostrar
n_reviews = st.slider("Número máximo de reviews a mostrar", min_value=1, max_value=20, value=5)

# --- Filtrado por tópico ---
filtered_df = df[df['topic_label'] == selected_topic]

# --- Filtrado por hotel ---
if selected_hotel != 'Todos':
    filtered_df = filtered_df[filtered_df['name'] == selected_hotel]
else:
    # Mostrar solo una review representativa por hotel
    filtered_df = filtered_df.groupby('name').head(1).reset_index(drop=True)

# Limitar al número máximo de reviews seleccionado
filtered_df = filtered_df.head(n_reviews)

# --- Mostrar resultados ---
for idx, row in filtered_df.iterrows():
    st.subheader(row['name'])
    st.write("Rating:", row['ratings'])
    st.write("Review:", row['text'])
    st.write("---")
    