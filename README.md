---
title: AI Health Symptom Checker
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.35.0"
python_version: "3.10"
app_file: streamlit_app.py
pinned: false
---

AI-powered health symptom checker with voice input support using Speech-to-Text (STT), Retrieval-Augmented Generation (RAG), and Large Language Models.

Built with Streamlit, FAISS, Whisper, LangChain, and Groq Llama models.

---

## ✨ Features

- 🧠 AI-powered symptom analysis
- 🎤 Voice-based symptom input
- 🗣️ Speech-to-Text using OpenAI Whisper
- 🔎 Retrieval-Augmented Generation (RAG)
- 📚 FAISS vector database for medical context retrieval
- 💻 Interactive Streamlit web interface
- ⚠️ Medical severity estimation
- 📋 Advice and consultation guidance

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Whisper
- Groq API
- Sentence Transformers
- HuggingFace Embeddings

---

## 📁 Project Structure

```bash
AI-Health-Symptom-checker/
│
├── app.py
├── requirements.txt
│
├── utils/
│   ├── loader.py
│   ├── rag_chain.py
│   ├── stt.py
│   ├── tts.py
│   └── vectorstore.py
│
├── vectorstore/
│   ├── index.faiss
│   └── index.pkl
│
├── Data/
│   └── symptoms_dataset.csv
│
└── audio/