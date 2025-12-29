import pandas as pd
import json
from collections import Counter
import re

# Paths
dataset_path = "data/curebench_valset_pharse1.jsonl"
submission_path = "results_zero/submission.csv"

# Load Ground Truth
truth = {}
questions = {}
with open(dataset_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        truth[data['id']] = data['correct_answer']
        questions[data['id']] = data

# Load Submission
try:
    df = pd.read_csv(submission_path)
except FileNotFoundError:
    print(f"Error: {submission_path} not found.")
    exit()

# Analyze
errors = []
keywords = {
    "Negative Constraint (NOT/EXCEPT)": ["NOT", "EXCEPT", "FALSE", "INCORRECT", "LEAST"],
    "Protocol/Timing (Next Step/Immediate)": ["NEXT STEP", "IMMEDIATE", "FIRST", "PRIORITY", "INITIAL", "ACTION"],
    "Contraindication/Side Effect": ["CONTRAINDICATION", "SIDE EFFECT", "ADVERSE", "REACTION", "RISK"],
    "Drug/Mechanism": ["MECHANISM", "CLASS", "INHIBITOR", "RECEPTOR", "ANTAGONIST"]
}

error_categories = Counter()

for _, row in df.iterrows():
    uid = row['id']
    pred = str(row['choice']).strip()
    if uid in truth:
        correct = truth[uid]
        if pred != correct:
            q_text = questions[uid]['question'].upper()
            
            # Categorize
            matched_cats = []
            for cat, words in keywords.items():
                if any(w in q_text for w in words):
                    matched_cats.append(cat)
            
            if not matched_cats:
                matched_cats.append("Other")
            
            for cat in matched_cats:
                error_categories[cat] += 1

            errors.append({
                'id': uid,
                'question': questions[uid]['question'],
                'correct': correct,
                'predicted': pred,
                'categories': matched_cats
            })

print(f"Analyzed {len(errors)} errors.")
print("\nError Patterns by Keyword Category:")
for cat, count in error_categories.most_common():
    print(f"  - {cat}: {count} errors")

print("\n--- Sample of 'Other' Errors (No obvious keyword) ---")
other_errors = [e for e in errors if "Other" in e['categories']]
for i, err in enumerate(other_errors[:5]):
    print(f"\n[Other Error {i+1}]")
    print(f"Q: {err['question']}")
    print(f"Pred: {err['predicted']} | Correct: {err['correct']}")
