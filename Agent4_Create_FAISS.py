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

# Load productivity evaluation data
with open('productivity_evaluation.json', 'r') as f:
    evaluations = json.load(f)

embedding_dim = 1536  # For text-embedding-ada-002
index = faiss.IndexFlatL2(embedding_dim)

texts_for_embedding = []
structured_metadata = []

print("🔍 Generating embeddings for productivity evaluation data...")

for idx, record in enumerate(evaluations):
    # Convert JSON record to readable text for embedding
    lines = []
    lines.append(f"Day: {record['day']}, Name: {record['name']}")
    for ev in record.get("evaluations", []):
        lines.append(f"- {ev['metric']}: {ev['value']} ({ev['status']})")
    if record.get("notes"):
        for note in record["notes"]:
            lines.append(f"Note: {note}")
    text = "\n".join(lines)
    
    # Save text for embedding
    texts_for_embedding.append(text)

    # Save full structured record as metadata
    structured_metadata.append(record)

    # Get embedding and add to FAISS index
    try:
        response = openai.Embedding.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        embedding = np.array(response['data'][0]['embedding'], dtype=np.float32)
        index.add(np.array([embedding]))
    except Exception as e:
        print(f"⚠️ Error embedding record {idx}: {e}")

print(f"✅ FAISS index built with {index.ntotal} vectors.")

# Save the index and full structured metadata
index_path = "faiss_productivity_index"
faiss.write_index(index, f"{index_path}.index")
with open(f"{index_path}_metadata.json", 'w') as f:
    json.dump(structured_metadata, f, indent=2)
print(f"📦 Saved FAISS index to {index_path}.index and metadata to {index_path}_metadata.json.")
