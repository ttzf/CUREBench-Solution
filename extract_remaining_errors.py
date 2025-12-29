import pandas as pd
import json
import os

# Paths
dataset_path = "data/curebench_valset_pharse1_errors.jsonl"
submission_path = "results_zero_plus/submission.csv"
output_dataset_path = "data/curebench_valset_pharse1_errors_remaining.jsonl"
output_csv_path = "results_zero_plus/errors_remaining_analysis.csv"

# Load Ground Truth
truth = {}
question_data = {}
with open(dataset_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        truth[data['id']] = data.get('correct_answer')
        question_data[data['id']] = data

# Load Submission
if not os.path.exists(submission_path):
    print(f"Error: {submission_path} not found.")
    exit(1)

df = pd.read_csv(submission_path)

# Filter Errors
remaining_errors = []
error_analysis = []

for _, row in df.iterrows():
    uid = row['id']
    if uid not in question_data:
        continue
        
    pred = str(row['choice']).strip()
    if pred in ['nan', 'NOTAVALUE', 'None']:
        pred = ""
        
    correct = str(truth.get(uid, "")).strip()
    
    if pred != correct:
        # Save to JSONL dataset list
        remaining_errors.append(question_data[uid])
        
        # Save to CSV for analysis
        error_analysis.append({
            'id': uid,
            'question_type': question_data[uid].get('question_type', ''),
            'question': question_data[uid].get('question', ''),
            'correct_answer': correct,
            'predicted_choice': pred,
            'reasoning': row['reasoning']
        })

# Write JSONL
with open(output_dataset_path, 'w') as f:
    for item in remaining_errors:
        f.write(json.dumps(item) + '\n')

# Write CSV Analysis
if error_analysis:
    pd.DataFrame(error_analysis).to_csv(output_csv_path, index=False)

print(f"✅ Extracted {len(remaining_errors)} remaining errors.")
print(f"📂 Dataset saved to: {output_dataset_path}")
print(f"📊 Analysis saved to: {output_csv_path}")

# Print breakdown
if error_analysis:
    df_err = pd.DataFrame(error_analysis)
    print("\nBreakdown by Question Type:")
    print(df_err['question_type'].value_counts())
