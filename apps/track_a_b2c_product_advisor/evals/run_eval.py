
import json
from difflib import SequenceMatcher

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from apps.track_a_b2c_product_advisor.main import build_chain

chain = build_chain()

# Define or import the retriever object
from apps.track_a_b2c_product_advisor.main import retriever

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "eval_set.json")) as f:
    eval_set = json.load(f)

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

for item in eval_set:
    question = item["question"]
    expected = item["expected_product_id"]
    keywords = item["keywords"]

    print(f"\n🔍 Question: {question}")
    result = chain.invoke({"question": question})

    try:
        parsed = json.loads(result)
        product_id = parsed["product_id"]
        rationale = parsed["rationale"]

        print(f"🧠 Predicted: {product_id}")
        print(f"✅ Expected: {expected}")
        print(f"📝 Rationale: {rationale}")

        id_match = product_id.strip() == expected
        keyword_match = all(k.lower() in rationale.lower() for k in keywords)

        print("🎯 ID Match:", "✅" if id_match else "❌")
        print("🔍 Keywords in rationale:", "✅" if keyword_match else "❌")

    except Exception as e:
        print("⚠️ Failed to parse output:", e)
        print("Raw output:", result.content)


retrieved_docs = retriever.invoke(question)
print("📚 Retrieved Context:", retrieved_docs)

score = 0
if id_match:
    score += 1
if keyword_match:
    score += 1
print(f"🧪 Score: {score}/2")

import csv

with open("eval_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["question", "expected", "predicted", "id_match", "keyword_match", "rationale"])
    for row in result:
        writer.writerow(row)