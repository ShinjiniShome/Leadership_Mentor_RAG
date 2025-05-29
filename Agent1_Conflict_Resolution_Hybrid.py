import pandas as pd
import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

openai.api_key = api_key

# Initialize the model for generating sentence embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to generate recommendations using GPT-3.5
def generate_recommendation(query, retrieved_data, source_type):
    context = f"Query: {query}\n"
    
    if source_type == "csv":
        context += "Here are some conflict resolution examples from the dataset:\n"
        for data in retrieved_data:
            context += (f"Conflict: {data.get('Conflict_Type', 'N/A')}, "
                        f"Resolution: {data.get('Resolution_Method', 'N/A')}, "
                        f"Outcome: {data.get('Outcome', 'N/A')}\n")
    elif source_type == "json":
        context += "Here are some conflict resolution papers:\n"
        for data in retrieved_data:
            context += (f"Title: {data.get('title', 'N/A')}, "
                        f"Summary: {data.get('summary', 'N/A')}\n")
    context += "\nBased on this, what is the best solution for this type of conflict?"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": context}],
        max_tokens=300
    )
    
    return response['choices'][0]['message']['content'].strip()

# Function to retrieve similar conflict scenarios from CSV
def retrieve_similar_conflicts_from_csv(query, df, top_k=3):
    conflict_texts = df['Conflict_Type'] + ' ' + df['Strategy'] + ' ' + df['Description']
    embeddings = model.encode(conflict_texts.tolist())
    embedding_matrix = np.array(embeddings, dtype=np.float32)
    
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)
    
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i in indices[0]:
        if i < len(df):
            results.append(df.iloc[i].to_dict())
    return results

# Function to load JSON files and retrieve similar research papers
def retrieve_similar_papers_from_json(query, json_files, top_k=3):
    all_papers = []
    
    for file in json_files:
        with open(file, "r") as f:
            papers = json.load(f)
            for paper in papers:
                text = paper.get('abstract', '') + ' ' + paper.get('conclusion', '')
                all_papers.append((text, paper))
    
    # Generate embeddings for the papers and query
    paper_texts = [text for text, _ in all_papers]
    paper_embeddings = model.encode(paper_texts)
    query_embedding = model.encode([query])
    
    paper_matrix = np.array(paper_embeddings, dtype=np.float32)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    
    # Create FAISS index and search
    index = faiss.IndexFlatL2(paper_matrix.shape[1])
    index.add(paper_matrix)
    
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i in indices[0]:
        if i < len(all_papers):
            results.append(all_papers[i][1])
    return results

# Main function for conflict resolution query
def resolve_conflict(query):
    # Load CSV data
    df = pd.read_csv('conflict_resolution_data_gpt3.5_100.csv')

    # Load research papers (hardcoded JSON files)
    json_files = ['team_conflict_resolution_articles.json', 'team_conflict_resolution_articles1.json', 'team_conflict_resolution_articles2.json']

    # Retrieve similar conflicts from CSV
    csv_results = retrieve_similar_conflicts_from_csv(query, df)
    # Retrieve similar research papers from JSON
    json_results = retrieve_similar_papers_from_json(query, json_files)

    # If no relevant CSV context is found, use LLM to refine the response
    if not csv_results and not json_results:
        print(f"No relevant results found for query: {query}")
        fallback_prompt = f"No relevant examples or research found. Based on your knowledge, what would be a good way to resolve this conflict?\nQuery: {query}"
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": fallback_prompt}],
        )
        print(response['choices'][0]['message']['content'].strip())
        return
    
    # Otherwise, combine the retrieved data from CSV and JSON and generate recommendations
    all_results = csv_results + json_results
    all_sources = []
    if csv_results:
        all_sources.append("csv")
    if json_results:
        all_sources.append("json")
    
    recommendation = generate_recommendation(query, all_results, all_sources[0] if len(all_sources) == 1 else "both")
    print("\nRecommendation for resolving the conflict:\n", recommendation)

# User input for query
query = input("Enter a conflict resolution query: ")
resolve_conflict(query)
