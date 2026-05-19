from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os


def build_chain(vectorstore):

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt_template = """
    You are a medical assistant AI.

    Given the user's symptoms and the retrieved medical data, provide:

    1. Possible diseases (top 2-3)
    2. Explanation
    3. Severity level (Low / Medium / High)
    4. Advice
    5. When to consult a doctor

    User Symptoms:
    {question}

    Medical Context:
    {context}

    Answer clearly and safely.
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0.1,
    )

    #  Manual RAG function
    def run_chain(user_query):

        # Retrieve docs
        docs = retriever.invoke(user_query)

        # Build context safely
        if not docs:
            context = "No relevant medical data found."
        else:
            context = "\n\n".join([doc.page_content[:80] for doc in docs])

        # 3. Format prompt
        final_prompt = PROMPT.format(
            context=context,
            question=user_query
        )

        # 4. Call LLM
        try:
            response = llm.invoke(final_prompt)
            return response.content
        except Exception as e:
            return f"⚠️ Error: {str(e)}"

        return response.content

    return run_chain