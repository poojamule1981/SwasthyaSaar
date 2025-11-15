import pandas as pd
import json

# Path to your CSV file
df = pd.read_csv(r"C:\Users\Pooja\nlp project\data\readme_exp.csv")

# Convert to a simple dictionary
jargon_dict = dict(zip(df['ann_text'], df['split_print']))

# Save as JSON file
output_path = r"C:\Users\Pooja\nlp project\medical_jargon.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(jargon_dict, f, indent=4, ensure_ascii=False)

print(f"✅ Saved medical jargon dictionary at: {output_path}")
print(f"Total entries: {len(jargon_dict)}")
