import os
import json
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap
from langchain_core.documents import Document

# Loads the OpenAI API key and other env variables from .env.
load_dotenv()

# === Helper: Load documents from the local data/ folder ===
def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r") as f:
                content = f.read()
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": filename, "doc_id": filename.replace(".txt", "")}
                    )
                )
    return documents

# === Global Variables for Shared Use ===
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data")

documents = load_documents_from_folder(data_path)
splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=50)
docs = splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", k=5)

# === RAG Chain Builder ===
def build_chain():
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = PromptTemplate.from_template("""
You are a helpful sales engineering assistant for a Customer Data Platform company. Use the context below to answer the user's question clearly, including specific details from the documents.

When you cite or use facts from a document, **always include the document ID in parentheses** at the end of the relevant sentence. For example: "... and achieved 25% ROI (doc_id: 1234abcd)".

Respond in this JSON format:
{{
  "materials": ["list of documents referenced by name or ID"],
  "rationale": "detailed explanation that uses facts from the context, citing doc IDs"
}}

Context:
{context}

User Question:
{question}
""")

    output_parser = StrOutputParser()

    chain = RunnableMap({
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"]
    }) | prompt | llm | output_parser

    return chain

# Allows the script to be run standalone to test the assistant.
if __name__ == "__main__":
    chain = build_chain()
    result = chain.invoke({
       "question": "What materials should I send a Fortune 500 retail company evaluating our CDP?"
    })
    print(result)
    