import streamlit as st
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1️⃣ Configuración en la barra lateral
st.sidebar.title("🔑 Configuración")
openai_api_key = st.sidebar.text_input("Introduce tu OpenAI API Key:", type="password")

if not openai_api_key:
    st.warning("Por favor ingresa tu API key en la barra lateral.")
    st.stop()

# 2️⃣ Cargar tu CSV precargado (ajusta la ruta)
df = pd.read_csv("data/database.csv")

if "text" not in df.columns:
    st.error("El CSV no contiene una columna llamada 'text'.")
    st.stop()

# 3️⃣ Preparar documentos desde la columna "text"
documents = df["text"].dropna().astype(str).tolist()

# 4️⃣ Crear embeddings y vectorstore
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key
)
vectorstore = FAISS.from_texts(documents, embeddings)

# 5️⃣ Crear modelo y cadena QA
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o-mini")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# 6️⃣ Interfaz de chat
st.title("🤖 Chatbot sobre tu base de datos")
user_q = st.text_input("Haz tu pregunta:")

if user_q:
    answer = qa_chain.run(user_q)
    st.write("**Respuesta:**", answer)
