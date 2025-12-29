import pandas as pd
import os

# Path to the submission file
file_path = "results_zero_plus/submission.csv"

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit(1)

# Read the CSV
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

# Inspect the 'choice' column
print(f"Total rows: {len(df)}")

# Check for empty or None values
empty_choices = df[df['choice'].isna() | (df['choice'] == "") | (df['choice'] == "NOTAVALUE")]
print(f"Rows with empty/NOTAVALUE choice: {len(empty_choices)}")

# Check for valid options (A, B, C, D)
valid_options = ['A', 'B', 'C', 'D']
invalid_choices = df[~df['choice'].isin(valid_options) & ~df['choice'].isin(["NOTAVALUE", "nan"])]
print(f"Rows with invalid choice (not A-D): {len(invalid_choices)}")

if len(invalid_choices) > 0:
    print("\nSample invalid choices:")
    print(invalid_choices[['id', 'choice']].head())

# Detailed breakdown
print("\nValue counts for 'choice':")
print(df['choice'].value_counts(dropna=False))
