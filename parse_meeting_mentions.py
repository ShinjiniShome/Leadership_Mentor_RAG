# parse_meeting_mentions.py
import json
import re

with open("synthetic_slack_logs.json") as f:
    slack_logs = json.load(f)

meeting_keywords = ["meeting", "sync", "call", "check-in", "standup", "1-on-1"]
parsed = []

for log in slack_logs:
    day = log["day"]
    mentions = []
    for line in log["log"]:
        if any(word in line.lower() for word in meeting_keywords):
            mentions.append(line)
    parsed.append({"day": day, "meeting_mentions": mentions})

with open("parsed_meeting_mentions.json", "w") as f:
    json.dump(parsed, f, indent=2)

print("✅ Parsed meeting mentions from Slack.")
