import json
import os
import sys

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from apps.track_a_b2b_sales_eng_assistant.main import build_chain, retriever

# Load eval set
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
eval_set_path = os.path.join(BASE_DIR, "eval_set.json")

with open(eval_set_path) as f:
    eval_set = json.load(f)

chain = build_chain()

print("\n================ B2B Sales Engineer Assistant Evaluation ================\n")

score = 0

for i, example in enumerate(eval_set):
    question = example["question"]
    expected_keywords = example.get("keywords", [])
    expected_docs = example.get("expected_docs", [])

    print(f"\U0001F50D Question: {question}")

    # 🔍 Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    retrieved_doc_ids = [doc.metadata.get("id", "unknown") for doc in retrieved_docs]

    # 🧠 Run chain with context
    result = chain.invoke({
        "question": question,
        "context": context
    })

    try:
        parsed = json.loads(result)
    except Exception as e:
        print("\u26a0\ufe0f Failed to parse output:", str(e))
        print("Raw output:", result)
        continue

    rationale = parsed.get("rationale", "")
    doc_hits = [doc_id for doc_id in expected_docs if doc_id in rationale or doc_id in retrieved_doc_ids]
    keyword_hits = [kw for kw in expected_keywords if kw.lower() in rationale.lower()]

    print(f"\U0001F4DD Rationale: {rationale}")
    print(f"✅ Expected Keywords: {expected_keywords}")
    print(f"📊 Keyword Match: {'✅' if keyword_hits else '❌'} ({len(keyword_hits)}/{len(expected_keywords)})")
    print(f"📄 Expected Docs Mentioned: {'✅' if doc_hits else '❌'} ({len(doc_hits)}/{len(expected_docs)})")
    print("📚 Retrieved Doc IDs:", retrieved_doc_ids)

    if keyword_hits and (not expected_docs or doc_hits):
        score += 1

    print("\n---\n")

print(f"\n\U0001F52E Final Score: {score}/{len(eval_set)}\n")