import os
import json
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap

# Loads the OpenAI API key and other env variables from .env.
load_dotenv()

# Ingests all .txt files in the data directory (e.g., case_study_1.txt, contract_1.txt).
def load_documents_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r") as f:
                content = f.read()
                documents.append(content)
    return documents

def build_chain():
    # Defines the path to local data files.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data")

    # Load and split documents; Splits each document into overlapping text chunks suitable for embedding
    documents = load_documents_from_folder(data_path)
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    docs = splitter.create_documents(documents)

    # Embed and store in FAISS; Enables semantic search based on user questions.
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", k=5)

    # LLM and prompt; Instructs the LLM to act as a helpful assistant, answering questions using retrieved context.
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = PromptTemplate.from_template("""
You are a sales engineer helper for a Customer Data Platform. Based on the provided context, generate a tailored enablement packet for the target account.
The format should be:
{{
  "summary": "...",
  "key_features": ["...", "..."],
  "relevant_case_studies": ["..."],
  "next_steps": "..."
}}

Context:
{context}

Target Account Question:
{question}
""")

    output_parser = StrOutputParser()

    # Defines the LangChain RAG pipeline: retrieve relevant docs → insert into prompt → get response from LLM → parse output.
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