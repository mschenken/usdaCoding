import pandas as pd
import json

# Load your merged file
df = pd.read_csv('merged_food_data.csv')

# Drop the 'nutrients' column if present
if 'nutrients' in df.columns:
    df = df.drop(columns=['nutrients'])

# Add an empty metadata column (or customize as needed)
df['metadata'] = '{}'

# Combine all columns except fdc_id and metadata into a single JSON string for 'content'
def row_to_content(row):
    d = row.drop(['fdc_id', 'metadata']).to_dict()
    # Remove NaNs for cleaner JSON
    d = {k: v for k, v in d.items() if pd.notnull(v)}
    return json.dumps(d, ensure_ascii=False)

df['content'] = df.apply(row_to_content, axis=1)

# Keep only the required columns and rename fdc_id to id
df_out = df[['fdc_id', 'content', 'metadata']].rename(columns={'fdc_id': 'id'})

# Save to CSV
df_out.to_csv('fooddata_vectors_ready.csv', index=False)