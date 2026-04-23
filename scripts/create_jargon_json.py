import pandas as pd
import json
import os

# Path to your CSV file (relative to project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "data", "readme_exp.csv")
df = pd.read_csv(csv_path)

# Convert to a simple dictionary
jargon_dict = dict(zip(df['ann_text'], df['split_print']))

# Save as JSON file
output_path = os.path.join(BASE_DIR, "medical_jargon.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(jargon_dict, f, indent=4, ensure_ascii=False)

print(f"✅ Saved medical jargon dictionary at: {output_path}")
print(f"Total entries: {len(jargon_dict)}")
