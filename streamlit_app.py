import streamlit as st
import os

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("streamlit").setLevel(logging.ERROR)


from dotenv import load_dotenv

from utils.vectorstore import (
    create_vectorstore,
    load_vectorstore
)

from utils.rag_chain import build_chain


# Load environment variables
load_dotenv()


st.set_page_config(
    page_title="AI Health Symptom Checker",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI Health Symptom Checker")

st.markdown(
    "Enter your symptoms to get AI-based medical suggestions."
)

st.warning(
    "⚠️ This is NOT a medical diagnosis. Always consult a doctor."
)


@st.cache_resource
def get_vectorstore():

    if not os.path.exists(
        "vectorstore/faiss_index/index.faiss"
    ):

        return create_vectorstore()

    return load_vectorstore()


vectorstore = get_vectorstore()

qa_chain = build_chain(vectorstore)

user_input = st.text_input(
    "Enter symptoms (e.g., fever, cough, headache):"
)


if st.button("Analyze"):

    if user_input:

        with st.spinner("🧠 Analyzing symptoms..."):

            response = qa_chain(user_input)

        st.subheader("📝 AI Analysis")

        st.success("Analysis Complete ✅")

        st.markdown(response)