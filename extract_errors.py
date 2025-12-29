import pandas as pd
import json
import os

# Paths
dataset_path = "data/curebench_valset_pharse1.jsonl"
submission_path = "results_zero/submission.csv"
output_path = "results_zero/errors_filtered.csv"

# Ensure output dir exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load Ground Truth and Question Types
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

# Filter and Extract Errors
extracted_errors = []

print(f"Total predictions: {len(df)}")

for _, row in df.iterrows():
    uid = row['id']
    
    # Skip if ID not found in dataset (should not happen usually)
    if uid not in question_data:
        continue
        
    q_type = question_data[uid].get('question_type', '')
    
    # Filter out 'open_ended' type
    if q_type == 'open_ended':
        continue
        
    # Get prediction and truth
    pred = str(row['choice']).strip()
    # Normalize prediction (sometimes it might be float nan or 'NOTAVALUE')
    if pred in ['nan', 'NOTAVALUE', 'None']:
        pred = ""
        
    correct = str(truth.get(uid, "")).strip()
    
    # Check correctness
    if pred != correct:
        error_item = {
            'id': uid,
            'question_type': q_type,
            'question': question_data[uid].get('question', ''),
            'options': json.dumps(question_data[uid].get('options', {})),
            'correct_answer': correct,
            'predicted_choice': pred,
            'reasoning': row['reasoning'],
            'full_prediction': row['prediction']
        }
        extracted_errors.append(error_item)

# Save to CSV
if extracted_errors:
    error_df = pd.DataFrame(extracted_errors)
    error_df.to_csv(output_path, index=False)
    print(f"✅ Extracted {len(extracted_errors)} non-open-ended errors to: {output_path}")
    
    # Print summary
    print(f"\nBreakdown by Question Type:")
    print(error_df['question_type'].value_counts())
    
    print(f"\nFirst 3 Examples:")
    for i, err in enumerate(extracted_errors[:3]):
        print(f"\n--- Error {i+1} ({err['question_type']}) ---")
        print(f"Q: {err['question']}")
        print(f"Correct: {err['correct_answer']} | Predicted: {err['predicted_choice']}")
else:
    print("No errors found (excluding open-ended questions).")
