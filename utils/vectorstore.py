import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from utils.loader import load_data


DB_PATH = "vectorstore/faiss_index"


def create_vectorstore():

    docs = load_data()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(docs, embeddings)

    os.makedirs(DB_PATH, exist_ok=True)

    db.save_local(DB_PATH)

    return db


def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )