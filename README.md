# Track A - B2B Sales Engineer Assistant

This project is a prototype AI-powered Sales Engineer Assistant for B2B go-to-market teams. It is designed to help sales engineers, account executives, or solution consultants quickly assemble enablement packets tailored to target accounts.

The assistant uses a Retrieval-Augmented Generation (RAG) architecture to ingest and reason over:
- Case studies
- Security documentation
- Example contracts
- Product FAQs
- Sales collateral

## Use Case
The assistant is optimized for sales conversations around enterprise Customer Data Platforms (CDPs) like mParticle or Uniphore. It supports target personas such as:
- Heads of Marketing Technology
- IT Managers
- Procurement and Security leads

## Folder Structure
apps/
└── track_a_b2b_sales_eng_assistant/
├── data/                  # Ingested content (case studies, security docs, etc.)
├── main.py                # Main RAG application
└── README.md              # Project description

## How to Run

1. Set your OpenAI key in a `.env` file:
OPENAI_API_KEY=your_key_here

2. Install dependencies:
pip install -r requirements.txt

3.	Run the assistant:
python3 main.py

## Stack
	•	LangChain
	•	OpenAI Embeddings + Chat Model
	•	FAISS for vector search
	•	Local JSON/Markdown content ingestion

## Status

Functional
🚧 In early prototype stage
✨ Future enhancements may include web UI, support for multiple personas, and integrations with CRM

© 2025 Dave Birckhead. See LICENSE for details.