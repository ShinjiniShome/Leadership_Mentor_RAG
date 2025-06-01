import json
import os
import openai
import faiss
import numpy as np

# Load OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OpenAI API key is missing. Set the OPENAI_API_KEY environment variable.")
    exit(1)
openai.api_key = api_key

# Load Papers
with open('employee_development_papers.json', 'r') as f:
    papers = json.load(f)

# Load Employee Data
with open('employeeData.json', 'r') as f:
    employees = json.load(f)
employee_lookup = {emp['name'].lower(): emp for emp in employees}

# Prepare FAISS Index
embedding_dim = 1536  # For text-embedding-ada-002
index = faiss.IndexFlatL2(embedding_dim)
paper_texts = []
paper_ids = []

print("🔍 Generating embeddings for papers...")
for idx, paper in enumerate(papers):
    text = paper.get('abstract', '') + " " + paper.get('summary', '')
    paper_texts.append(text)
    paper_ids.append(idx)
    
    try:
        response = openai.Embedding.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        embedding = np.array(response['data'][0]['embedding'], dtype=np.float32)
        index.add(np.array([embedding]))
    except Exception as e:
        print(f"Error embedding paper {idx}: {e}")

print(f"✅ FAISS index built with {index.ntotal} vectors.")

def get_employee_by_name(name):
    return employee_lookup.get(name.lower(), None)

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

def query_rag_system(query_text, employee_name=None, top_k=3):
    employee = None
    if employee_name:
        employee = get_employee_by_name(employee_name)
        if not employee:
            print(f"⚠️ Employee '{employee_name}' not found. Proceeding without employee data.")

    query_embedding = embed_query(query_text)
    if query_embedding is None:
        return "Query embedding failed."

    D, I = index.search(np.array([query_embedding]), top_k)
    retrieved_contexts = []
    for idx in I[0]:
        paper = papers[paper_ids[idx]]
        retrieved_contexts.append(f"Title: {paper['title']}\nSummary: {paper['summary']}\nURL: {paper['url']}")

    context = "\n\n".join(retrieved_contexts)

    if employee:
        employee_context = f"Employee: {employee['name']}\nSkills: {', '.join(employee['skills'])}\nWeaknesses: {', '.join(employee['weaknesses'])}\nEngagement Score: {employee['engagement_score']}\nLeadership Score: {employee['leadership_score']}\nBurnout Risk: {employee['burnout_risk']}\nTraining History: {', '.join([t['title'] for t in employee['training_history']])}"
    else:
        employee_context = "No specific employee data provided."

    prompt = (
        f"Context from research papers:\n{context}\n\n"
        f"Employee Information:\n{employee_context}\n\n"
        f"User Query: {query_text}\n\n"
        f"Please provide a detailed suggestion integrating the research papers and (if available) the employee data."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant generating employee development recommendations."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error generating GPT response: {e}")
        return "GPT response generation failed."

if __name__ == "__main__":
    while True:
        print("\n=== RAG SYSTEM ===")
        employee_name = input("Enter employee name (or press Enter to skip): ").strip()
        if employee_name.lower() == 'exit':
            break
        if employee_name == "":
            employee_name = None  # Allow skipping
        query_text = input("Enter your query: ").strip()
        response = query_rag_system(query_text, employee_name)
        print("\n💡 GPT Response:\n", response, "\n")
        print("\n Enter exit if there is no more query")
