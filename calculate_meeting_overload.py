# calculate_meeting_overload.py
import json

with open("parsed_meeting_mentions.json") as f:
    meeting_mentions = json.load(f)

with open("fake_calendar_data.json") as f:
    calendar_data = json.load(f)

overload_scores = []

for day in range(len(calendar_data)):
    slack_count = len(meeting_mentions[day]["meeting_mentions"])
    meeting_count = sum(len(emp["meetings"]) for emp in calendar_data[day]["schedules"])
    overload = min(100, (slack_count * 3 + meeting_count * 5))  # simple scoring
    overload_scores.append({"day": day + 1, "meeting_overload_index": overload})

with open("meeting_overload_scores.json", "w") as f:
    json.dump(overload_scores, f, indent=2)

print("✅ Meeting overload scores calculated.")
