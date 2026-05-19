import whisper
import streamlit as st


@st.cache_resource
def load_whisper_model():

    return whisper.load_model("medium")


model = load_whisper_model()


def speech_to_text(audio_path):

    result = model.transcribe(
        audio_path,

        # Force English
        language="en",

        # Standard transcription
        task="transcribe",

        # CPU mode
        fp16=False,

        # Deterministic decoding
        temperature=0,

        # Better accuracy
        beam_size=5,

        # Better candidate selection
        best_of=5
    )

    text = result["text"].strip()

    return text