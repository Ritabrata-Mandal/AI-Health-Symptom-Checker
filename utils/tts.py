import torch
import soundfile as sf
import streamlit as st

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_tts_model():

    model_name = "ai4bharat/indic-parler-tts"

    model = (
        ParlerTTSForConditionalGeneration
        .from_pretrained(model_name)
        .to(device)
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


model, tokenizer = load_tts_model()


def text_to_speech(
    text,
    output_path="audio/response.wav"
):

    description = """
    A calm Indian English female voice speaking clearly.
    """

    input_ids = tokenizer(
        description,
        return_tensors="pt"
    ).input_ids.to(device)

    prompt_input_ids = tokenizer(
        text,
        return_tensors="pt"
    ).input_ids.to(device)

    generation = model.generate(
        input_ids=input_ids,
        prompt_input_ids=prompt_input_ids
    )

    audio_arr = generation.cpu().numpy().squeeze()

    sf.write(
        output_path,
        audio_arr,
        model.config.sampling_rate
    )

    return output_path