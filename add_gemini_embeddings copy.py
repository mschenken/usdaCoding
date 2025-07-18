import pandas as pd
import requests
import json
import time
import tempfile
import os

API_KEY = ''
EMBEDDING_URL = 'https://generativelanguage.googleapis.com/v1/models/embedding-001:embedContent?key=' + API_KEY

def get_gemini_embedding(text):
    headers = {'Content-Type': 'application/json'}
    data = {
        "content": {"parts": [{"text": text}]}
    }
    while True:
        try:
            response = requests.post(EMBEDDING_URL, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                # Gemini returns {'embedding': {'values': [...]}}
                return response.json()['embedding']['values']
            else:
                print(f"Error: {response.status_code}, {response.text}")
                time.sleep(5)
        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(5)

# Read your file
df = pd.read_csv('fooddata_vectors_ready.csv')

# Generate embeddings
embeddings = []
for idx, row in df.iterrows():
    emb = get_gemini_embedding(row['content'])
    embeddings.append(emb)
    if idx % 100 == 0:
        print(f"Processed {idx} rows")

df['embedding'] = embeddings

# Save in chunks under 100MB
MAX_MB = 100
MAX_BYTES = MAX_MB * 1024 * 1024
batch_size = 5000  # Initial batch size
batch_num = 1
start_idx = 0
while start_idx < len(df):
    curr_batch_size = batch_size
    while curr_batch_size > 0:
        batch = df.iloc[start_idx:start_idx+curr_batch_size]
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmpfile:
            batch.to_csv(tmpfile.name, index=False)
            tmpfile.flush()
            size = os.path.getsize(tmpfile.name)
        if size < MAX_BYTES:
            # Save as final batch
            final_name = f'fooddata_vectors_with_embeddings_batch_{batch_num}.csv'
            os.replace(tmpfile.name, final_name)
            print(f"Saved {final_name} with {len(batch)} rows, size: {size/1024/1024:.2f} MB")
            batch_num += 1
            start_idx += curr_batch_size
            break
        else:
            # Too big, halve batch size and retry
            os.remove(tmpfile.name)
            curr_batch_size = curr_batch_size // 2
            if curr_batch_size == 0:
                raise Exception("Cannot create a batch under 100MB. Try reducing the number of columns or data size.") 