import pandas as pd

# Read the CSV file
df = pd.read_csv('conflict_resolution_data_gpt3.5_100.csv', header=0)

# Column to check
column_name = 'Conflict_Type'

# Count occurrences of each unique value
value_counts = df[column_name].value_counts()

# Filter out values that only appear once (keep only duplicates)
duplicates_only = value_counts[value_counts > 1]

# Print results
if not duplicates_only.empty:
    print(f"Duplicate values in column '{column_name}' with their occurrences:")
    for value, count in duplicates_only.items():
        print(f"{value}: {count} times")
else:
    print(f"No duplicates found in column '{column_name}'.")

    print("\n" + "="*50 + "\n")  # Separator for better readability

# Check for entire duplicate rows
# Find duplicate rows in the entire DataFrame
duplicate_rows = df[df.duplicated(keep=False)]  # `keep=False` retains all occurrences

# Print duplicate rows if found
if not duplicate_rows.empty:
    print(f"Duplicate rows found in the dataset (showing first 5 rows):")
    print(duplicate_rows.head())  # Display first 5 duplicate rows for preview
    print(f"\nTotal duplicate rows: {duplicate_rows.shape[0]}")
else:
    print("No duplicate rows found in the dataset.")