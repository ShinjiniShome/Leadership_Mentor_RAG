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

def fetch_employee_development_papers(total_papers=10, batch_size=10):
    """Fetches open-access research papers about employee development from Semantic Scholar."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    all_articles = []
    
    queries = [
        "employee development and upskilling best practices",
        "leadership and mentorship programs in organizations",
        "effective employee engagement strategies",
        "career pathing and growth in corporate environments",
        "employee training effectiveness research",
        "motivating disengaged employees through development",
        "designing impactful training programs for employee growth",
        "burnout prevention through development and support",
        "digital learning solutions for employee upskilling",
        "reskilling and continuous learning in modern workplaces"
    ]
    
    for query in queries:
        params = {
            "query": query,
            "fields": "title,abstract,url,isOpenAccess",
            "limit": batch_size
        }
        try:
            response = requests.get(url, params=params)
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
                    break
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for query '{query}': {e}")
            continue
        
        if len(all_articles) >= total_papers:
            break

    return all_articles

def main():
    num_articles = 10  # Target paper count for Employee Development
    articles = fetch_employee_development_papers(total_papers=num_articles)
    
    if articles:
        with open("employee_development_papers.json", "w") as f:
            json.dump(articles, f, indent=4)
        print(f"✅ Saved {len(articles)} open-access articles to employee_development_papers.json")
    else:
        print("⚠️ No articles were fetched.")

if __name__ == "__main__":
    main()
