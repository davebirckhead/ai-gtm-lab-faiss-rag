
import os
import pandas as pd
import json
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # Load OpenAI API key

# 🔁 Shared retriever defined here so it's accessible outside
retriever = None

def build_chain():
    global retriever  # so we can export it

    # Set up paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data")

    # Load product catalog and reviews
    catalog = pd.read_csv(os.path.join(DATA_PATH, "catalog.csv"))
    with open(os.path.join(DATA_PATH, "reviews.json")) as f:
        reviews = json.load(f)

    # Combine catalog + reviews
    def combine_product_info(row):
        product_reviews = [r["review"] for r in reviews if r["product_id"] == row["id"]]
        return (
            f"Product: {row['name']} by {row['brand']}.\n"
            f"Category: {row['category']}.\n"
            f"Volume: {row['volume_liters']}L, Color: {row['color']}, "
            f"Weight: {row['weight_kg']}kg, Price: ${row['price_usd']}.\n"
            f"Laptop Sleeve: {'Yes' if row['laptop_sleeve'] else 'No'}.\n"
            f"Reviews: {' | '.join(product_reviews)}"
        )

    documents = catalog.apply(combine_product_info, axis=1).tolist()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents(documents)

    # Build vectorstore and retriever
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", k=4)

    # Define prompt and LLM
    prompt = PromptTemplate.from_template("""
You are a helpful shopping assistant. Based on the following context, recommend the best backpack for the user question.
Respond in this JSON format:
{{
  "product_id": "recommended_id",
  "name": "product_name",
  "rationale": "explanation"
}}

Context:
{context}

User Question:
{question}
""")

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    output_parser = StrOutputParser()

    # Assemble chain
    chain = RunnableMap({
    "context": lambda x: x["question"],  # pass only the question string
    "question": lambda x: x["question"]
}) | prompt | llm | output_parser

    return chain

# Exportables
__all__ = ["build_chain", "retriever"]

# Optional interactive test
if __name__ == "__main__":
    chain = build_chain()
    result = chain.invoke({"question": "What’s the best backpack for hiking and laptops?"})
    print(result)