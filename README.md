---
title: AI Health Symptom Checker
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.35.0"
python_version: "3.11"
app_file: streamlit_app.py
pinned: false
---

# 🩺 AI Health Symptom Checker

🌐 **Live Demo:**  
https://huggingface.co/spaces/RitabrataMandalCSE/AI-HEALTH-SYMPTOM-CHECKER


AI-powered medical symptom checker using:

- Retrieval-Augmented Generation (RAG)
- FAISS Vector Database
- LangChain
- Groq LLMs
- Streamlit

The app analyzes user symptoms and provides:
- possible diseases
- severity estimation
- medical advice
- recommended specialists

---

# ✨ Features

- 🧠 AI-powered symptom analysis
- 🔎 Retrieval-Augmented Generation (RAG)
- 📚 FAISS vector database
- ⚠️ Severity estimation
- 👨‍⚕️ Specialist recommendation
- 💻 Interactive Streamlit UI
- 🚀 Hugging Face deploy-ready

---

# 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Groq API
- Sentence Transformers
- HuggingFace Embeddings

---

# 📁 Project Structure

```bash
AI-Health-Symptom-Checker/
│
├── .streamlit/
│   └── config.toml
│
├── Data/
│   └── symptoms_dataset.csv
│
├── utils/
│   ├── loader.py
│   ├── rag_chain.py
│   └── vectorstore.py
│
├── .gitignore
├── requirements.txt
├── README.md
└── streamlit_app.py

```