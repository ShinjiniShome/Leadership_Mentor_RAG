import pandas as pd
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

openai.api_key = api_key

# Load conflict data
df = pd.read_csv('conflict_resolution_data_gpt3.5_100.csv')

# Initialize the model for generating sentence embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Combine relevant columns to generate a textual representation of each conflict scenario
conflict_texts = df['Conflict_Type'] + ' ' + df['Strategy'] + ' ' + df['Description']

# Generate sentence embeddings for each conflict
embeddings = model.encode(conflict_texts.tolist())

# Convert embeddings to numpy array with correct dtype
embedding_matrix = np.array(embeddings, dtype=np.float32)

# Create a FAISS index using the L2 distance metric
index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # L2 distance (Euclidean)
index.add(embedding_matrix)  # Add the embeddings to the index

# Function to retrieve the most similar conflict scenarios based on a query
def retrieve_similar_conflicts(query, top_k=3):
    query_embedding = model.encode([query])  # Generate embedding for the query
    query_embedding = np.array(query_embedding, dtype=np.float32)  # Convert to numpy array
    
    # Search the FAISS index for the top_k most similar conflicts
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve the conflict data based on the indices
    results = []
    for i in indices[0]:
        if i < len(df):  # Ensure index is within bounds
            results.append(df.iloc[i].to_dict())  # Convert to dictionary
    return results

# Function to generate recommendations using GPT-3.5
def generate_recommendation(query, retrieved_data):
    context = f"Query Type: {query}\nHere are some conflict resolution examples:\n"
    for data in retrieved_data:
        context += (f"Conflict: {data.get('Conflict_Type', 'N/A')}, "
                    f"Resolution: {data.get('Resolution_Method', 'N/A')}, "
                    f"Outcome: {data.get('Outcome', 'N/A')}\n")
    context += "\nBased on this, what is the best solution for this type of conflict?"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": context}],
        max_tokens=300
    )
    
    return response['choices'][0]['message']['content'].strip()

# User input for query
query = input("Enter a conflict resolution query: ")

# Retrieve similar conflicts
results = retrieve_similar_conflicts(query)

# Generate a recommendation based on retrieved conflicts
if results:
    recommendation = generate_recommendation(query, results)
    print("\nRecommendation for resolving the conflict:\n", recommendation)
else:
    print("\nNo relevant conflicts found.")
