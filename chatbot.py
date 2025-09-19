import streamlit as st
import pandas as pd
import ast
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

st.title("Chatbot RAG sobre CSV de Reviews")

# 1️⃣ Cargar CSV
df = pd.read_csv("https://github.com/melody-10/Proyecto_Hoteles_California/blob/main/final_database.csv?raw=true")

# 2️⃣ Convertir ratings a texto legible
def ratings_to_text(ratings_str):
    try:
        ratings_dict = ast.literal_eval(ratings_str)
        return ", ".join([f"{k}: {v}" for k, v in ratings_dict.items()])
    except:
        return ratings_str

# 3️⃣ Crear documentos listos para embeddings
documents = []
for _, row in df.iterrows():
    text_block = f"Name: {row['name']}, County: {row['county']}, Ratings: {ratings_to_text(row['ratings'])}, Review: {row['text']}"
    documents.append(text_block)

# 4️⃣ Obtener API key de secrets.toml
api_key = st.secrets["openai"]["api_key"]

# 5️⃣ Crear embeddings y vectorstore
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.from_texts(documents, embeddings)

# 6️⃣ Crear chatbot RAG
retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=api_key),
    retriever=retriever
)

# 7️⃣ Interfaz de usuario
query = st.text_input("Escribe tu pregunta sobre los datos:")
if query:
    with st.spinner("Buscando respuesta..."):
        response = qa.run(query)
        st.markdown(f"**Respuesta:** {response}")
