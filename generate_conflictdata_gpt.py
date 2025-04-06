import openai
import pandas as pd
import os
import csv
import re
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file.")

openai.api_key = api_key

def clean_conflict_type(conflict_type):
    """Cleans the conflict type by removing leading numbers, special characters, and extra whitespace."""
    return re.sub(r'^\s*\d+\.\s*', '', conflict_type).strip()

def generate_conflict_data(request_num):
    prompt = (
        f"Generate a list of exactly {request_num} unique workplace conflict types, their corresponding resolution strategies, "
        "and a brief description for each in CSV format with headers: Conflict_Type, Strategy, Description.\n\n"
        "Example:\n"
        "Conflict_Type,Strategy,Description\n"
        "\"Task Distribution\",\"Mediation\",\"Organize a one-on-one meeting to clarify roles and responsibilities.\"\n\n"
        "Now generate the full list strictly in this format, ensuring all values are properly quoted to maintain CSV structure:"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are an expert in workplace conflict resolution."},
                      {"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4096
        )

        # Extract generated text
        generated_text = response['choices'][0]['message']['content'].strip()

        # Debugging: Print response for troubleshooting
        #print("Generated Text:\n", generated_text)

        # Parse CSV output
        lines = generated_text.strip().split("\n")
        reader = csv.reader(lines)
        
        # Extract data while ensuring exactly 3 columns per row
        data = [row[:3] for row in reader if len(row) >= 3]
        
        # Clean conflict type column
        for row in data[1:]:  # Skip header row
            row[0] = clean_conflict_type(row[0])
        
        # Ensure we have at least 1 header + 1 data row
        if len(data) < 2:
            raise ValueError("OpenAI response did not return valid CSV data.")

        # Convert to DataFrame
        df = pd.DataFrame(data[1:], columns=data[0])  # First row as header
        return df

    except openai.OpenAIError as e:  # Catches API errors
        print(f"OpenAI API Error: {e}")
    except ValueError as e:  # Catches formatting errors
        print(f"Data Formatting Error: {e}")
    except Exception as e:  # Catches any other unexpected errors
        print(f"Unexpected Error: {e}")

    return None  # Return None if an error occurs

# Generate first 50 rows
df1 = generate_conflict_data(50)

# Generate another 50 rows
df2 = generate_conflict_data(50)

# Step 3: Combine and save
if df1 is not None and df2 is not None:
    df2 = df2.iloc[1:].reset_index(drop=True)  # Remove duplicate header from second dataset
    df_final = pd.concat([df1, df2], ignore_index=True)

    # Ensure no duplicate headers by checking column names
    if list(df_final.columns) != ["Conflict_Type", "Strategy", "Description"]:
        raise ValueError("Column headers are incorrect after concatenation.")

    csv_file = "conflict_resolution_data_gpt3.5_100.csv"
    df_final.to_csv(csv_file, index=False)
    print(f"CSV file saved: {csv_file}")
else:
    print("Data generation failed. No CSV file created.")
