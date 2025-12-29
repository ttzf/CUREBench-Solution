import pandas as pd
import json
import os

# Path to the submission file and dataset
file_path = "results_few_all_1/submission.csv"
dataset_path = "data/curebench_valset_pharse1.jsonl"

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit(1)

# Load Question Types from Dataset
question_types = {}
with open(dataset_path, 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            question_types[data['id']] = data.get('question_type', 'unknown')
        except:
            continue

# Read the CSV
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

# Inspect the 'choice' column
print(f"Total rows in submission: {len(df)}")

# Filter out pure 'open_ended' questions (they are allowed to have empty choice)
# But 'open_ended_multi_choice' MUST have a choice
df['question_type'] = df['id'].map(question_types)
target_df = df[df['question_type'] != 'open_ended']

print(f"Rows to check (excluding 'open_ended'): {len(target_df)}")

# Check for empty or None values in the filtered set
empty_choices = target_df[target_df['choice'].isna() | (target_df['choice'] == "") | (target_df['choice'] == "NOTAVALUE")]
print(f"Rows with empty/NOTAVALUE choice (excluding open_ended): {len(empty_choices)}")

if len(empty_choices) > 0:
    print("\nBreakdown of problematic empty choices by type:")
    print(empty_choices['question_type'].value_counts())
    print("\nSample IDs:")
    print(empty_choices['id'].head().tolist())

# Check for valid options (A, B, C, D)
valid_options = ['A', 'B', 'C', 'D']
invalid_choices = target_df[~target_df['choice'].isin(valid_options) & ~target_df['choice'].isin(["NOTAVALUE", "nan"])]
print(f"Rows with invalid choice (not A-D): {len(invalid_choices)}")
