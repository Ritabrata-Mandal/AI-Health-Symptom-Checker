import pandas as pd
from langchain_core.documents import Document


def load_data(path="Data/symptoms_dataset.csv"):

    df = pd.read_csv(path)

    docs = []

    for _, row in df.iterrows():

        content = f"""
        Symptoms: {row['symptoms']}
        Disease: {row['disease']}
        Advice: {row['advice']}
        Specialist: {row['specialist']}
        """

        docs.append(
            Document(page_content=content.strip())
        )

    return docs