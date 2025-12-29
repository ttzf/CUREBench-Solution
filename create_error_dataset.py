import pandas as pd
import json
import os

# Paths
error_csv_path = "results_zero/errors_filtered.csv"
source_dataset_path = "data/curebench_valset_pharse1.jsonl"
output_dataset_path = "data/curebench_valset_pharse1_errors.jsonl"

# 1. Get IDs of the error questions
if not os.path.exists(error_csv_path):
    print(f"Error: {error_csv_path} not found.")
    exit(1)

df = pd.read_csv(error_csv_path)
error_ids = set(df['id'].astype(str).tolist())
print(f"Found {len(error_ids)} unique error IDs in CSV.")

# 2. Filter the source dataset
filtered_count = 0
with open(source_dataset_path, 'r') as f_in, open(output_dataset_path, 'w') as f_out:
    for line in f_in:
        try:
            data = json.loads(line)
            if str(data.get('id')) in error_ids:
                f_out.write(line)
                filtered_count += 1
        except json.JSONDecodeError:
            continue

print(f"✅ Successfully created new dataset with {filtered_count} items.")
print(f"📂 Saved to: {output_dataset_path}")
