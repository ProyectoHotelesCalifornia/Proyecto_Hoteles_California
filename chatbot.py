import streamlit as st
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

# 1️⃣ Cargar el CSV existente
@st.cache_data
def load_data():
    df = pd.read_csv("https://github.com/melody-10/Proyecto_Hoteles_California/blob/main/final_database.csv?raw=true")  # tu archivo precargado
    return df

df = load_data()

# 2️⃣ Crear embeddings con HuggingFace (modelo ligero y rápido)
@st.cache_resource
def create_vectorstore(texts):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore

documents = df["text"].astype(str).tolist()
vectorstore = create_vectorstore(documents)

# 3️⃣ Crear el LLM desde HuggingFace (gratis, pero más lento que OpenAI)
#    Puedes cambiar "google/flan-t5-small" por uno más grande si tu hosting lo permite
llm = HuggingFaceHub(
    repo_id="google/flan-t5-small",
    model_kwargs={"temperature":0.1, "max_length":256}
)

# 4️⃣ Configurar el QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# 5️⃣ Interfaz en Streamlit
st.title("🔍 Chatbot de Reviews de Hoteles (HuggingFace)")

user_query = st.text_input("Escribe tu pregunta:")
if user_query:
    with st.spinner("Buscando respuesta..."):
        response = qa.run(user_query)
        st.write("**Respuesta:**", response)
