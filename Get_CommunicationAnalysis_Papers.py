import requests
import json
import openai
import time
import os

def generate_gpt_summary(abstract, conclusion):
    """Generates a short summary using GPT-3.5."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key is missing. Set the OPENAI_API_KEY environment variable.")
        return "Summary could not be generated due to missing API key."
    
    openai.api_key = api_key
    
    prompt = f"Generate a concise summary based on the abstract: {abstract} and conclusion: {conclusion}."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI that generates summaries for research papers."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return "Summary could not be generated due to an error."

def fetch_semantic_scholar_articles(total_papers=10, batch_size=10):
    """Fetches open-access research papers from Semantic Scholar without pagination."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    all_articles = []
    
    # Updated queries for Communication Overload & Productivity
    queries = queries = [
    # Digital communication & meeting overload
    "information overload in digital collaboration tools like Slack or Microsoft Teams",
    "impact of meeting overload on employee productivity and well-being",
    "digital communication overload and employee performance",
    "meeting fatigue and its impact on work efficiency",
    "asynchronous vs synchronous communication impact on productivity",
    "strategies to reduce communication overload in organizations",
    "employee burnout due to communication volume",
    "collaboration tool overload and team performance",
    "best practices for reducing digital communication stress",
    "meeting science and overload management",
    
    # Work management & collaboration methods
    "implementing kanban boards to improve task visibility and reduce overload",
    "kanban boards for managing communication and task flow",
    "agile workflows and their impact on reducing meeting overload",
    "scrum methodology and digital collaboration overload",
    "task prioritization methods in digital workplaces",
    "impact of task management tools on productivity and overload",
    "hybrid work environments and their effect on communication overload",
    "strategies for effective collaboration in hybrid work models",
    "digital workplace optimization with kanban and agile methods",
    "reducing overload through work visualization tools like Trello and Jira",
    
    # Well-being and resilience
    "employee well-being and resilience in digital workplaces",
    "psychological safety in high-communication environments",
    "impact of collaboration tool usage on stress levels",
    "mindful work practices to reduce digital fatigue"
]

    
    for query in queries:
        params = {
            "query": query,
            "fields": "title,abstract,url,isOpenAccess",
            "limit": batch_size
        }
        try:
            response = requests.get(url, params=params)
            
            # Debugging: Print the raw response text
            print(f"Response for query '{query}': {response.text}")
            
            response.raise_for_status()
            if response.text.strip() == "":
                print(f"Empty response received for query '{query}'")
                continue
            
            papers = response.json().get("data", [])
            
            for paper in papers:
                if not paper.get("isOpenAccess", False):
                    continue  # Skip non-open-access papers
                
                title = paper.get("title", "No title available")
                abstract = paper.get("abstract", "")
                url = paper.get("url", "No URL available")
                
                conclusion = "Conclusion not available in metadata."
                summary = generate_gpt_summary(abstract, conclusion)
                
                all_articles.append({
                    "title": title,
                    "abstract": abstract,
                    "conclusion": conclusion,
                    "summary": summary,
                    "url": url
                })
                
                if len(all_articles) >= total_papers:
                    break  # Stop if we reach the desired count
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for query '{query}': {e}")
            continue
        
        if len(all_articles) >= total_papers:
            break

    return all_articles

def main():
    num_articles = 10  # Target paper count for Communication Analysis
    articles = fetch_semantic_scholar_articles(total_papers=num_articles)
    
    if articles:
        with open("communication_overload_papers.json", "w") as f:
            json.dump(articles, f, indent=4)
        print(f"✅ Saved {len(articles)} open-access articles to communication_overload_papers.json")
    else:
        print("⚠️ No articles were fetched.")

if __name__ == "__main__":
    main()
