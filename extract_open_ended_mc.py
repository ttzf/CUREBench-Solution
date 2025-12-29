import json
import os

input_file = "data/curebench_valset_pharse1_errors.jsonl"
output_file = "data/curebench_valset_pharse1_errors_open_ended_mc.jsonl"

count = 0

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        try:
            data = json.loads(line)
            if data.get('question_type') == 'open_ended_multi_choice':
                f_out.write(line)
                count += 1
        except json.JSONDecodeError:
            continue

print(f"✅ Extracted {count} 'open_ended_multi_choice' questions.")
print(f"📂 Saved to: {output_file}")
