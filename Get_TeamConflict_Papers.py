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
    queries = [
        "team conflict resolution",
        "corporate team conflict resolution",
        "workplace conflict mediation",
        "organizational dispute management",
        "Conflict resolution in corporate teams",
        "Strategies for resolving team conflicts in organizations",
        "Team conflict management in the workplace",
        "Corporate team conflict resolution methods",
        "Effective conflict resolution strategies for teams",
        "Leadership and team conflict resolution in business",
        "Interpersonal conflict in teams and resolution strategies",
        "Communication and conflict resolution in corporate teams",
        "Conflict resolution in cross-functional teams",
        "Mediation in team conflict resolution",
        "Psychological aspects of conflict resolution in teams",
        "Case studies on team conflict resolution in organizations",
        "Best practices for managing team conflict",
        "Team conflict and collaboration in the workplace",
        "Resolving conflicts in team dynamics",
        "Resolving conflicts n teams"
    ]
    
    for query in queries:
        params = {
            "query": query,
            "fields": "title,abstract,url,isOpenAccess",
            "limit": batch_size  # Fetch only the desired number of papers
        }
        try:
            response = requests.get(url, params=params)
            
            # Debugging: Print the raw response text
            print(f"Response for query '{query}': {response.text}")
            
            response.raise_for_status()  # Raise an HTTPError for bad responses
            if response.text.strip() == "":
                print(f"Empty response received for query '{query}'")
                continue  # Skip to the next query if response is empty
            
            papers = response.json().get("data", [])
            
            for paper in papers:
                if not paper.get("isOpenAccess", False):
                    continue  # Skip non-open-access papers
                
                title = paper.get("title", "No title available")
                abstract = paper.get("abstract", "")
                url = paper.get("url", "No URL available")
                
                # Assume conclusion is unavailable, generate summary from abstract
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
            continue  # Skip this query and move on to the next one
    
        if len(all_articles) >= total_papers:
            break  # Stop if we've reached the desired paper count

    return all_articles

def main():
    num_articles = 10  # Target paper count is now 10
    articles = fetch_semantic_scholar_articles(total_papers=num_articles)
    
    if articles:
        with open("team_conflict_resolution_articles1.json", "w") as f:
            json.dump(articles, f, indent=4)
        print(f"Saved {len(articles)} open-access articles to team_conflict_resolution_articles1.json")
    else:
        print("No articles were fetched.")

if __name__ == "__main__":
    main()