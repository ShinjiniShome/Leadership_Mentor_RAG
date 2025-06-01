import os
import json
import openai
import faiss
import numpy as np

# Load OpenAI API Key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OpenAI API key is missing. Set the OPENAI_API_KEY environment variable.")
    exit(1)
openai.api_key = api_key

# Load FAISS index and metadata
print("📂 Loading FAISS index and metadata...")
faiss_index = faiss.read_index('faiss_productivity_index.index')
with open('faiss_productivity_index_metadata.json', 'r') as f:
    metadata = json.load(f)  # metadata is a list of dicts

# Helper function to embed the user query
def embed_query(text):
    try:
        response = openai.Embedding.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return np.array(response['data'][0]['embedding'], dtype=np.float32)
    except Exception as e:
        print(f"Error embedding query: {e}")
        return None

# Function to query the knowledge base
def query_knowledge_base(user_query, top_k=5):
    query_embedding = embed_query(user_query)
    if query_embedding is None:
        return "Query embedding failed."
    
    D, I = faiss_index.search(np.array([query_embedding]), top_k)
    results = []
    for idx in I[0]:
        # Access metadata by integer index (list)
        if idx < len(metadata):
            record = metadata[idx]
            results.append(record)
    return results

# (Optional) Generate GPT summary from results
def generate_gpt_summary(user_query, results):
    context = "\n\n".join(
        [f"Day: {rec['day']}, Name: {rec['name']}\n" + 
         "\n".join([f"- {m['metric']}: {m['value']} ({m['status']})" for m in rec.get('evaluations', [])]) +
         ("\nNotes: " + "; ".join(rec.get('notes', [])) if rec.get('notes') else "") for rec in results]
    )
    prompt = (
        f"Context from productivity evaluations:\n{context}\n\n"
        f"User Query: {user_query}\n\n"
        f"Please provide a detailed analysis integrating the productivity data."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a productivity analysis assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error generating GPT response: {e}")
        return "GPT response generation failed."

if __name__ == "__main__":
    print("🔎 Welcome to the Productivity Knowledge Base Query System")
    while True:
        query = input("Enter your query (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break
        results = query_knowledge_base(query)
        print("\nTop matches from the knowledge base:\n")
        for rec in results:
            print(json.dumps(rec, indent=2))
        summary = generate_gpt_summary(query, results)
        print("\n💡 GPT Analysis:\n", summary, "\n")
