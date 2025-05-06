import os
import json
import faiss
import string
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
import transformers
from sentence_transformers import SentenceTransformer, CrossEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


company_name = "cirrascale"


def chunk_document(text, chunk_size=10):
    """Split document into sentence chunks"""
    sentences = sent_tokenize(text)
    return [
        " ".join(sentences[i : i + chunk_size])
        for i in range(0, len(sentences), chunk_size)
    ]


corpus_chunks = []
sentences_file = f"{company_name}_sentences.json"
generate_context_file = f"{company_name}_generated_context.json"


with open(os.path.join("extracted_data/", sentences_file), "r") as f:
    sentences = json.load(f)  # Load the list of sentences
with open(os.path.join("generated_context/", generate_context_file), "r") as f:
    generated_context = json.load(f)

generated_context_chunks = [{"text": chunk["response"]} for chunk in generated_context]

# Chunk the sentences using the existing chunk_document function
corpus_chunks = [{"text": chunk} for chunk in chunk_document(" ".join(sentences))]

corpus_chunks.extend(generated_context_chunks)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(
    [chunk["text"] for chunk in corpus_chunks], show_progress_bar=True
)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings)
index.add(embeddings)

question_filename = os.path.join("questions/", f"{company_name}_questions.json")

with open(question_filename, "r") as f:
    questions = json.load(f)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-6")
THRESHOLD = 0.45

results = []
for question in questions:
    # Semantic search
    question_embedding = model.encode([question])
    faiss.normalize_L2(question_embedding)
    D, I = index.search(question_embedding, 5)

    # Re-rank with cross-encoder
    hits = [corpus_chunks[i] for i in I[0]]
    pairs = [
        [str(question), str(hit["text"])] for hit in hits
    ]  # Ensure both elements are strings

    # Predict scores
    scores = cross_encoder.predict(pairs)

    # Process results
    best_score = max(scores)
    best_match = hits[np.argmax(scores)]["text"]
    if best_score < THRESHOLD:
        results.append(
            {
                "question": question,
                "status": "Gap",
                "sources": [hit["text"] for hit in hits],
                "best_match": best_match,
                "score": best_score,
            }
        )
    else:
        results.append(
            {
                "question": question,
                "status": "Covered",
                "sources": [hit["text"] for hit in hits],
                "best_match": best_match,
                "score": best_score,
            }
        )


def extract_keywords(text):
    tokens = word_tokenize(text.lower())
    return set(
        [
            t
            for t in tokens
            if t not in stopwords.words("english")
            and t not in string.punctuation
            and len(t) > 2
        ]
    )


# Get all corpus keywords
corpus_keywords = set()
for chunk in corpus_chunks:
    corpus_keywords.update(extract_keywords(chunk["text"]))


# Create a DataFrame from the results
df = pd.DataFrame(results)

# Add missing_keywords and best_match fields
df["missing_keywords"] = None

# Fill gap analysis data
for idx, row in df[df["status"] == "Gap"].iterrows():
    question_keywords = extract_keywords(row["question"])
    missing_keywords = question_keywords - corpus_keywords
    df.at[idx, "missing_keywords"] = (
        list(missing_keywords) if missing_keywords else None
    )

# Safe visualization
print("Coverage Summary:")
print(df["status"].value_counts())

if "missing_keywords" in df.columns:
    gap_rows = df[df["status"] == "Gap"]
    if not gap_rows.empty:
        missing_kws = gap_rows["missing_keywords"].dropna().explode().value_counts()
        if not missing_kws.empty:
            print("\nTop Missing Keywords:")
            print(missing_kws.head(10))

# Export the DataFrame to an Excel file
df.to_excel("documentation_coverage_report.xlsx", index=False)
