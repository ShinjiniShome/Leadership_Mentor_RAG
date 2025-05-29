import os
import openai
import numpy as np
import faiss
import json

from communication_data_analysis import (
    analyze_calendar,
    analyze_emails,
    analyze_slack,
    analyze_tasks,
    compose_summary,
)
from config import team, NUM_DAYS
from overload_threshold_config import (
    meeting_threshold,
    email_sent_threshold,
    email_received_threshold,
    slack_message_threshold,
    task_count_threshold
)

# --- Load JSON ---
def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

# --- Embedding Helper ---
def get_embedding(text, model="text-embedding-3-large"):
    try:
        response = openai.Embedding.create(input=[text], model=model)
        return np.array(response['data'][0]['embedding'], dtype=np.float32)
    except Exception as e:
        print(f"Embedding API error: {e}")
        return None

# --- Build FAISS index on papers ---
def build_faiss_index(papers):
    embeddings = []
    valid_papers = []
    for paper in papers:
        text = paper.get("summary") or paper.get("abstract") or paper.get("title") or ""
        emb = get_embedding(text)
        if emb is not None:
            embeddings.append(emb)
            valid_papers.append(paper)
    if not embeddings:
        raise RuntimeError("No embeddings generated for any paper.")
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.vstack(embeddings))
    return index, valid_papers, embeddings

# --- Search FAISS ---
def search_faiss_index(index, papers, embeddings, query, top_k=3):
    query_emb = get_embedding(query)
    if query_emb is None:
        return []
    D, I = index.search(np.array([query_emb]), top_k)
    results = []
    for idx in I[0]:
        results.append(papers[idx])
    return results

# --- Get GPT recommendations ---
def get_recommendations(prompt):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Error: OPENAI_API_KEY environment variable not set."
    openai.api_key = api_key
    messages = [
        {"role": "system", "content": "You are a helpful AI leadership mentor assistant."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=600,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error calling OpenAI API: {e}"

def main():
    # Load synthetic data
    calendar_data = load_json("fake_calendar_data.json")
    email_data = load_json("synthetic_emails.json")
    slack_data = load_json("synthetic_slack_logs.json")
    task_data = load_json("synthetic_task_board.json")
    papers = load_json("communication_overload_papers.json")

    # Analyze data with thresholds
    calendar_analysis = analyze_calendar(calendar_data)
    emails_analysis = analyze_emails(email_data)
    slack_analysis = analyze_slack(slack_data)
    task_analysis = analyze_tasks(task_data)

    # Compose detailed summary for prompt
    summary = compose_summary(calendar_analysis, emails_analysis, slack_analysis, task_analysis)

    # Build FAISS index for papers
    print("🔍 Building FAISS index on research papers...")
    try:
        index, valid_papers, embeddings = build_faiss_index(papers)
    except Exception as e:
        print(f"Error building FAISS index: {e}")
        valid_papers, index, embeddings = [], None, None

    # User query input
    user_query = input("\nEnter your query about communication overload, meeting fatigue, or team motivation:\n> ")

    # Search top relevant papers by query
    if index:
        top_papers = search_faiss_index(index, valid_papers, embeddings, user_query, top_k=3)
    else:
        top_papers = []

    # Prepare prompt
    if top_papers:
        papers_text = "\n\n".join([f"Title: {p.get('title', 'N/A')}\nSummary: {p.get('summary', 'No summary')}" for p in top_papers])
        full_prompt = (
            f"Synthetic Data Analysis Summary:\n{summary}\n\n"
            f"User Query:\n{user_query}\n\n"
            f"Relevant Research Insights:\n{papers_text}\n\n"
            "Based on the above analysis and research, provide actionable recommendations for the manager."
        )
    else:
        full_prompt = (
            f"Synthetic Data Analysis Summary:\n{summary}\n\n"
            f"User Query:\n{user_query}\n\n"
            "No relevant research papers found.\n\n"
            "Based on the above analysis, provide actionable recommendations for the manager."
        )

    print("\n=== Prompt sent to GPT ===\n")
    print(full_prompt[:1500] + ("..." if len(full_prompt) > 1500 else ""))

    # Get GPT recommendations
    recommendations = get_recommendations(full_prompt)

    print("\n=== Recommendations from GPT ===\n")
    print(recommendations)


if __name__ == "__main__":
    main()
