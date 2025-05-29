import json
import openai
import faiss
import numpy as np
import os

# ------------------ CONFIG ------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Set your OpenAI API key using the OPENAI_API_KEY environment variable.")

openai.api_key = OPENAI_API_KEY

ARTICLE_FILES = [
    "team_conflict_resolution_articles.json",
    "team_conflict_resolution_articles1.json",
    "team_conflict_resolution_articles2.json"
]

EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_DIM = 1536  # Dimension for 'text-embedding-ada-002'

# ------------------ STEP 1: Load Articles ------------------

def load_articles(file_paths):
    texts = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            articles = json.load(f)
            for art in articles:
                if art.get("summary"):
                    text = f"{art.get('title', '')}\n\n{art['summary']}"
                    texts.append(text)
    return texts

# ------------------ STEP 2: Generate Embeddings ------------------

def get_embedding(text, model=EMBEDDING_MODEL):
    try:
        response = openai.Embedding.create(input=[text], model=model)
        return response["data"][0]["embedding"]
    except Exception as e:
        print(f"Error embedding text: {e}")
        return None

def build_faiss_index(texts):
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    valid_embeddings = []
    valid_texts = []

    for text in texts:
        embedding = get_embedding(text)
        if embedding:
            valid_embeddings.append(embedding)
            valid_texts.append(text)

    index.add(np.array(valid_embeddings).astype("float32"))
    return index, valid_texts

# ------------------ STEP 3: Query Search ------------------

def search_similar(query, index, texts, k=3):
    query_embedding = get_embedding(query)
    if not query_embedding:
        return []

    D, I = index.search(np.array([query_embedding]).astype("float32"), k)
    return [texts[i] for i in I[0]]

# ------------------ STEP 4: GPT Response ------------------

def answer_query_with_context(query, relevant_texts):
    context = "\n\n".join(relevant_texts)
    prompt = f"""You are a corporate conflict resolution expert.

Use the following context to answer the user's question.

Context:
{context}

Question: {query}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You help resolve corporate and team conflicts using provided research."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error from OpenAI: {e}")
        return "An error occurred while generating the answer."

# ------------------ MAIN FUNCTION ------------------

def main():
    print("🔄 Loading articles and building vector index...")
    articles = load_articles(ARTICLE_FILES)
    index, texts = build_faiss_index(articles)

    print("✅ Agent is ready. Ask your conflict resolution question!")
    while True:
        query = input("\n🧠 Your question (or type 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting. Goodbye!")
            break

        print("🔍 Searching relevant content...")
        relevant_docs = search_similar(query, index, texts, k=3)

        print("🤖 Generating expert response...")
        answer = answer_query_with_context(query, relevant_docs)

        print("\n📘 Answer:\n")
        print(answer)

# ------------------ RUN ------------------

if __name__ == "__main__":
    main()
