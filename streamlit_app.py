import streamlit as st
import os
from dotenv import load_dotenv

# =========================
# INITIAL SETUP
# =========================

print("APP STARTED")

load_dotenv()

os.makedirs("audio", exist_ok=True)

st.set_page_config(
    page_title="AI Health Symptom Checker",
    layout="wide"
)

print("PAGE CONFIG DONE")

# =========================
# UI
# =========================

st.title("🩺 AI Health Symptom Checker")

st.markdown(
    "Voice-enabled AI healthcare assistant"
)

st.warning(
    "⚠️ This is NOT a medical diagnosis. Always consult a doctor."
)

print("UI LOADED")

# =========================
# SESSION STATE
# =========================

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

print("SESSION STATE READY")

# =========================
# TEXT INPUT
# =========================

st.subheader("⌨️ Text Input")

text_input = st.text_input(
    "Enter symptoms:",
    value=st.session_state.user_input,
    placeholder="fever, cough, headache"
)

if text_input:
    st.session_state.user_input = text_input

print("TEXT INPUT READY")

# =========================
# VOICE INPUT
# =========================

st.subheader("🎤 Voice Input")

audio_value = st.audio_input(
    "Record your symptoms"
)

print("AUDIO INPUT READY")

if audio_value:

    audio_path = "audio/input.wav"

    with open(audio_path, "wb") as f:
        f.write(audio_value.read())

    with st.spinner(
        "🧠 Converting speech to text..."
    ):

        print("STARTING STT")

        from utils.stt import speech_to_text

        recognized_text = speech_to_text(
            audio_path
        )

        print("STT COMPLETE")

        st.session_state.user_input = (
            recognized_text
        )

    st.write("### Recognized Speech:")

    st.write(st.session_state.user_input)

# =========================
# ANALYZE BUTTON
# =========================

if st.button("Analyze"):

    if st.session_state.user_input:

        try:

            with st.spinner(
                "🔄 Loading AI system..."
            ):

                print("IMPORTING VECTORSTORE")

                from utils.vectorstore import (
                    create_vectorstore,
                    load_vectorstore
                )

                print("VECTORSTORE IMPORTED")

                from utils.rag_chain import (
                    build_chain
                )

                print("CHAIN IMPORTED")

                # =========================
                # LOAD VECTORSTORE
                # =========================

                if not os.path.exists(
                    "vectorstore/faiss_index/index.faiss"
                ):

                    print("CREATING VECTORSTORE")

                    vectorstore = create_vectorstore()

                    print("VECTORSTORE CREATED")

                else:

                    print("LOADING VECTORSTORE")

                    vectorstore = load_vectorstore()

                    print("VECTORSTORE LOADED")

                # =========================
                # BUILD CHAIN
                # =========================

                print("BUILDING QA CHAIN")

                qa_chain = build_chain(
                    vectorstore
                )

                print("QA CHAIN READY")

            # =========================
            # RUN ANALYSIS
            # =========================

            with st.spinner(
                "🧠 Analyzing symptoms..."
            ):

                print("RUNNING ANALYSIS")

                response = qa_chain(
                    st.session_state.user_input
                )

                print("ANALYSIS COMPLETE")

            st.subheader("📝 AI Analysis")

            st.success(
                "Analysis Complete ✅"
            )

            st.markdown(response)

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )

            print("ERROR OCCURRED")
            print(e)

    else:

        st.error(
            "Please enter or record symptoms."
        )

print("APP FULLY LOADED")