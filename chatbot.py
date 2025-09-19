import streamlit as st
import pandas as pd
import ast
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

import time
import openai
from openai.error import RateLimitError


from langchain_community.vectorstores import Chroma

from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv

def safe_embed(texts, embeddings, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return embeddings.embed_documents(texts)
        except RateLimitError:
            if attempt < retries - 1:
                st.warning("⚠️ Se alcanzó el límite de velocidad. Reintentando...")
                time.sleep(delay)
            else:
                st.error("❌ Límite de velocidad alcanzado. Intenta de nuevo más tarde.")
                return []


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

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=api_key
)
docs_embeddings = safe_embed(documents, embeddings)
vectorstore = FAISS.from_embeddings(docs_embeddings, documents)

#vectorstore = FAISS.from_texts(documents, embeddings)


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
