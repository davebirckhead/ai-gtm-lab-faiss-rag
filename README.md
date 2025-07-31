# AI GTM Lab with FAISS, Langchain, and RAG

Sabbatical projects exploring AI for Marketing, Growth and GTM use cases, with FAISS, Langchain, and RAG support.

AI GTM Lab: Exploring how generative AI and retrieval-based systems can transform marketing, growth, and go-to-market strategy.

This is a sabbatical project space focused on building and evaluating AI-native workflows for B2C and B2B use cases. It combines fast vector search (FAISS), Retrieval-Augmented Generation (RAG), and modern LLM frameworks (like LangChain) to prototype and test intelligent systems that can:
	•	Recommend the right product or service based on customer intent and business rules
	•	Assist sales and marketing teams with grounded, agentic decision support
	•	Evaluate LLM output for structure, truthfulness, grounding, and business logic
	•	Demonstrate how “Super ICs” and cross-functional teams can ship faster with AI

⸻

 What’s Inside
	•	Monorepo structure for multiple apps, packages, eval tools, and experiments
	•	FAISS + LangChain for local semantic search and context-aware generation
	•	LLM & agentic workflows (OpenAI, LangChain, JSON output, etc.)
	•	RAG eval framework for structured validation and scoring
	•	CI-ready with Makefile, pre-commit hooks, GitHub Actions support

⸻

Sample Use Cases
	•	Track A – B2C Advisor: Given a product catalog and user query, return JSON-formatted product recommendations with rationale
	•	Track B – B2B Assistant: Given account metadata and firmographic data, suggest outreach timing or LTV-based segmentation
	•	Eval Harness: Score groundedness, structural validity, and constraint adherence of LLM output
	•	Chunking + Retrieval Library: Reusable logic to transform unstructured data into vectorized, searchable memory

⸻

Stack Highlights
	•	Python, FastAPI, FAISS, DuckDB, LangChain, OpenAI
	•	pyproject-based dev environment
	•	Pre-commit, Ruff, Black, pytest
	•	GitHub Actions CI workflow

⸻

Purpose

This project is part of a 6-month sabbatical to go deep on how AI is transforming marketing and growth strategy. Inspired by leaders like Joachim Candela, it embraces learning, exploration, and hands-on craft over titles—an effort to do work that is both great and good.