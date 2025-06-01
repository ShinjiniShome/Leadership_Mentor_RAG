import random
import json
from config import team, NUM_DAYS

def generate_survey_data():
    data = []
    for day in range(NUM_DAYS):
        date = f"day {day+1}"
        for member in team:
            daily_motivation = random.randint(2, 5)
            stress_level = 6 - daily_motivation if daily_motivation < 4 else random.randint(1,3)
            workload = random.choice(["Light", "Normal", "Heavy"])
            role_clarity = random.choice(["Yes", "Maybe", "No"])
            contribution = random.choice(["High", "Medium", "Low"])
            leadership_score = random.randint(1,5)
            data.append({
                "day": date,
                "name": member["name"],
                "daily_motivation": daily_motivation,
                "stress_level": stress_level,
                "perceived_workload": workload,
                "satisfaction_with_role_clarity": role_clarity,
                "feeling_of_contribution": contribution,
                "leadership_score": leadership_score
            })
    return data

def main():
    survey = generate_survey_data()
    with open('survey_data.json', 'w') as f:
        json.dump(survey, f, indent=2)
    print("Generated survey_data.json")

if __name__ == "__main__":
    main()
