from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq


def build_chain(vectorstore):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 2}
    )

    prompt_template = """
You are an AI medical assistant.

Using the user's symptoms and the retrieved medical context,
provide a clear medical analysis.

Include:

1. Possible diseases (Top 2-3)
2. Brief explanation
3. Severity level (Low / Medium / High / Critical)
4. Advice and precautions
5. Recommended specialist doctor
6. Whether immediate medical attention is needed

User Symptoms:
{question}

Medical Context:
{context}

Answer in a clean and structured format.
"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1
    )

    def run_chain(user_query):

        docs = retriever.invoke(user_query)

        if not docs:
            context = "No relevant medical data found."

        else:
            context = "\n\n".join(
                [doc.page_content[:200] for doc in docs]
            )

        final_prompt = PROMPT.format(
            context=context,
            question=user_query
        )

        try:

            response = llm.invoke(final_prompt)

            return response.content

        except Exception as e:

            return f"⚠️ Error: {str(e)}"

    return run_chain