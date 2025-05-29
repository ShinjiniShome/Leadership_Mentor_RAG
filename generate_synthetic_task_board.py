import json
import random
from faker import Faker
from config import team, NUM_DAYS

fake = Faker()
task_data = []

statuses = ["To Do", "In Progress", "Done", "Blocked"]
categories = ["Frontend", "Backend", "QA", "UX", "Docs"]

for day in range(1, NUM_DAYS + 1):
    tasks = []
    for member in team:
        for _ in range(random.randint(2, 4)):
            tasks.append({
                "assignee": member["name"],
                "title": f"{random.choice(categories)}: {fake.bs().capitalize()}",
                "status": random.choice(statuses),
                "priority": random.choice(["Low", "Medium", "High"]),
            })
    task_data.append({"day": day, "tasks": tasks})

with open("synthetic_task_board.json", "w") as f:
    json.dump(task_data, f, indent=2)

print(f"✅ Task board data generated for {NUM_DAYS} days.")
