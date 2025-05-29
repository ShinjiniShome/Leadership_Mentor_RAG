import json
import os
import openai
import faiss
import numpy as np

# --- Load JSON ---
def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

# --- Synthetic Data Analysis (same as before) ---
def analyze_calendar(calendar_data):
    total_meetings = 0
    meeting_counts_per_employee = {}
    total_duration_minutes = 0
    for day in calendar_data:
        for schedule in day["schedules"]:
            employee = schedule["employee"]
            meetings = schedule["meetings"]
            meeting_counts_per_employee[employee] = meeting_counts_per_employee.get(employee, 0) + len(meetings)
            total_meetings += len(meetings)
            for m in meetings:
                dur_str = m["duration"]
                try:
                    dur_min = int(dur_str.split()[0])
                except Exception:
                    dur_min = 0
                total_duration_minutes += dur_min
    avg_meetings_per_day = total_meetings / len(calendar_data) if calendar_data else 0
    avg_duration_per_meeting = total_duration_minutes / total_meetings if total_meetings > 0 else 0
    return {
        "total_meetings": total_meetings,
        "avg_meetings_per_day": avg_meetings_per_day,
        "avg_duration_per_meeting": avg_duration_per_meeting,
        "meeting_counts_per_employee": meeting_counts_per_employee
    }

def analyze_emails(email_data):
    emails_sent = {}
    emails_received = {}
    for email in email_data:
        sender = email["from"]
        receiver = email["to"]
        emails_sent[sender] = emails_sent.get(sender, 0) + 1
        emails_received[receiver] = emails_received.get(receiver, 0) + 1
    return emails_sent, emails_received

def analyze_slack(slack_data):
    messages_per_user = {}
    for day in slack_data:
        for log_entry in day["log"]:
            try:
                parts = log_entry.split("] ")
                user_message = parts[1]
                user = user_message.split(":")[0]
                messages_per_user[user] = messages_per_user.get(user, 0) + 1
            except Exception:
                continue
    return messages_per_user

def analyze_tasks(task_data):
    task_status_counts = {}
    tasks_per_assignee = {}
    for day in task_data:
        for task in day["tasks"]:
            status = task["status"]
            assignee = task["assignee"]
            task_status_counts[status] = task_status_counts.get(status, 0) + 1
            tasks_per_assignee[assignee] = tasks_per_assignee.get(assignee, 0) + 1
    return task_status_counts, tasks_per_assignee

# --- Compose summary ---
def compose_summary(calendar_analysis, emails_sent, emails_received, slack_analysis, task_status_counts, tasks_per_assignee):
    summary = (
        f"Communication Analysis Summary:\n"
        f"- Total meetings scheduled: {calendar_analysis['total_meetings']} (avg per day: {calendar_analysis['avg_meetings_per_day']:.2f})\n"
        f"- Average meeting duration: {calendar_analysis['avg_duration_per_meeting']:.1f} mins\n"
        f"- Meetings per employee: {calendar_analysis['meeting_counts_per_employee']}\n\n"
        f"- Emails sent per employee: {emails_sent}\n"
        f"- Emails received per employee: {emails_received}\n\n"
        f"- Slack messages per user: {slack_analysis}\n\n"
        f"- Task statuses count: {task_status_counts}\n"
        f"- Tasks assigned per employee: {tasks_per_assignee}\n"
    )
    return summary

# --- OpenAI embedding helper ---
def get_embedding(text, model="text-embedding-3-large"):
    # Adjust model if you want, "text-embedding-3-large" is latest as of May 2025
    try:
        response = openai.Embedding.create(input=[text], model=model)
        return np.array(response['data'][0]['embedding'], dtype=np.float32)
    except Exception as e:
        print(f"Embedding API error: {e}")
        return None

# --- Build FAISS index on paper summaries ---
def build_faiss_index(papers):
    embeddings = []
    valid_papers = []
    for paper in papers:
        text = paper.get("summary") or paper.get("abstract") or paper.get("title")
        emb = get_embedding(text)
        if emb is not None:
            embeddings.append(emb)
            valid_papers.append(paper)
    if not embeddings:
        raise RuntimeError("No embeddings could be generated.")
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

# --- Query GPT-3.5-turbo ---
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
            max_tokens=500,
            temperature=0.7
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error calling OpenAI API: {e}"

# --- Main orchestration ---
def main():
    # Load all your data and analyze synthetic data...
    calendar_data = load_json("fake_calendar_data.json")
    email_data = load_json("synthetic_emails.json")
    slack_data = load_json("synthetic_slack_logs.json")
    task_data = load_json("synthetic_task_board.json")
    papers = load_json("communication_overload_papers.json")

    calendar_analysis = analyze_calendar(calendar_data)
    emails_sent, emails_received = analyze_emails(email_data)
    slack_analysis = analyze_slack(slack_data)
    task_status_counts, tasks_per_assignee = analyze_tasks(task_data)

    summary = compose_summary(calendar_analysis, emails_sent, emails_received, slack_analysis, task_status_counts, tasks_per_assignee)

    # Build FAISS index
    print("🔍 Building FAISS index on research papers...")
    try:
        index, valid_papers, embeddings = build_faiss_index(papers)
    except Exception as e:
        print(f"Error building FAISS index: {e}")
        valid_papers, index, embeddings = [], None, None

    # Take user query input
    user_query = input("\nEnter your query or question about communication overload, meeting fatigue, or team motivation:\n> ")

    # Retrieve top relevant papers based on user query
    if index:
        top_papers = search_faiss_index(index, valid_papers, embeddings, user_query, top_k=3)
    else:
        top_papers = []

    # Compose prompt to GPT
    if top_papers:
        papers_text = "\n\n".join([f"Title: {p['title']}\nSummary: {p['summary']}" for p in top_papers])
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

    recommendations = get_recommendations(full_prompt)

    print("\n=== Recommendations from GPT ===\n")
    print(recommendations)


if __name__ == "__main__":
    main()
