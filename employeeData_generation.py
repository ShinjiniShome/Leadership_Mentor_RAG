import json
import random
from collections import defaultdict
from faker import Faker

faker = Faker()

# Load data
with open('peer_reviews.json', 'r') as f:
    peer_reviews = json.load(f)
with open('survey_results.json', 'r') as f:
    survey_results = json.load(f)
with open('trainings.json', 'r') as f:
    trainings = json.load(f)

# Keyword heuristics
POSITIVE_KEYWORDS = {
    "strong": "Technical Strength",
    "great": "Leadership",
    "excellent": "Leadership",
    "creative": "Creativity",
    "innovative": "Innovation",
    "reliable": "Reliability",
    "supports": "Team Support",
    "thorough": "Attention to Detail",
    "fast": "Speed"
}
NEGATIVE_KEYWORDS = {
    "needs": "Improvement Needed",
    "sometimes": "Inconsistent",
    "misses": "Misses Deadlines",
    "overlooks": "Overlooked Details",
    "critical": "Overly Critical",
    "quiet": "Communication Gap",
    "overwhelmed": "Overwhelmed",
    "micromanages": "Micromanagement"
}

# Survey lookup
survey_lookup = {item['name']: item for item in survey_results}

# Generate synthetic employees
employeeData = []

for employee in peer_reviews:
    name = employee['name']
    reviews = employee['reviews']
    skills = set()
    weaknesses = set()

    for review in reviews:
        comment = review['comment'].lower()
        for keyword, category in POSITIVE_KEYWORDS.items():
            if keyword in comment:
                skills.add(category)
        for keyword, category in NEGATIVE_KEYWORDS.items():
            if keyword in comment:
                weaknesses.add(category)

    survey_data = survey_lookup.get(name, {})

    # Generate random training history
    num_trainings = random.randint(1, 2)
    training_history = []
    selected_trainings = random.sample(trainings, num_trainings)
    for training in selected_trainings:
        training_history.append({
            "title": training["title"],
            "topic": training["topic"],
            "format": training["format"],
            "duration": training.get("duration", "Unknown"),
            "completion_date": faker.date_between(start_date="-2y", end_date="today").isoformat()
        })

    employeeData.append({
        "name": name,
        "skills": list(skills) if skills else ["Generalist"],
        "weaknesses": list(weaknesses) if weaknesses else ["None Reported"],
        "engagement_score": survey_data.get("engagement_score", 5),
        "leadership_score": survey_data.get("leadership_score", 5),
        "burnout_risk": survey_data.get("burnout_risk", "Unknown"),
        "training_history": training_history
    })

# Save output
with open('employeeData.json', 'w') as f:
    json.dump(employeeData, f, indent=2)

print("✅ employeeData.json generated")
