import streamlit as st
import os

from dotenv import load_dotenv

from utils.vectorstore import (
    create_vectorstore,
    load_vectorstore
)

from utils.rag_chain import build_chain

from utils.stt import speech_to_text


# Load env
load_dotenv()

st.set_page_config(
    page_title="AI Health Symptom Checker"
)

st.title("🩺 AI Health Symptom Checker")

st.markdown(
    "Voice-enabled AI healthcare assistant"
)

st.warning(
    "⚠️ This is NOT a medical diagnosis. Always consult a doctor."
)


# Session state
if "user_input" not in st.session_state:
    st.session_state.user_input = ""


# Load vector DB
if not os.path.exists(
    "vectorstore/faiss_index/index.faiss"
):

    with st.spinner(
        "🔄 Creating vector database..."
    ):

        vectorstore = create_vectorstore()

else:

    vectorstore = load_vectorstore()


# Build RAG chain
qa_chain = build_chain(vectorstore)


# TEXT INPUT
st.subheader("⌨️ Text Input")

text_input = st.text_input(
    "Enter symptoms:",
    value=st.session_state.user_input,
    placeholder="fever, cough, headache"
)

if text_input:
    st.session_state.user_input = text_input


# VOICE INPUT
st.subheader("🎤 Voice Input")

audio_value = st.audio_input(
    "Record your symptoms"
)


if audio_value:

    os.makedirs("audio", exist_ok=True)

    audio_path = "audio/input.wav"

    with open(audio_path, "wb") as f:
        f.write(audio_value.read())

    #st.audio(audio_path)

    with st.spinner(
        "🧠 Converting speech to text..."
    ):

        recognized_text = speech_to_text(
            audio_path
        )

        st.session_state.user_input = (
            recognized_text
        )

    st.write("### Recognized Speech:")

    st.write(st.session_state.user_input)


# ANALYZE
if st.button("Analyze"):

    if st.session_state.user_input:

        with st.spinner(
            "🧠 Analyzing symptoms..."
        ):

            response = qa_chain(
                st.session_state.user_input
            )

        st.subheader("📝 AI Analysis")

        st.success("Analysis Complete ✅")

        st.markdown(response)

    else:

        st.error(
            "Please enter or record symptoms."
        )