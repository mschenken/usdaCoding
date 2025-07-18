import pandas as pd
import asyncio
import httpx
import json
import ast
import math
import os

BATCH_SIZE = 100  # Qdrant and Gemini batch limit

QDRANT_URL = "http://172.16.0.11:6333/collections/usdalib/points?wait=true"
QDRANT_KEY = os.getenv("QDRANT_KEY", "")
GEMINI_API_KEY = ''
EMBEDDING_URL = f'https://generativelanguage.googleapis.com/v1/models/embedding-001:embedContent?key={GEMINI_API_KEY}'

# Async batch embedding function using Gemini
async def get_gemini_batch_embeddings_async(client, contents):
    headers = {'Content-Type': 'application/json'}
    data = {
        "requests": [
            {"model": "models/embedding-001", "content": {"parts": [{"text": json.dumps(content)}]}}
            for content in contents
        ]
    }
    while True:
        try:
            response = await client.post(
                'https://generativelanguage.googleapis.com/v1beta/models/embedding-001:batchEmbedContents?key=' + GEMINI_API_KEY,
                headers=headers,
                data=json.dumps(data)
            )
            if response.status_code == 200:
                embeddings = [emb['values'] for emb in response.json()['embeddings']]
                print(f"Batch vector sizes: {[len(e) for e in embeddings]}")
                return embeddings
            else:
                print(f"Batch embedding error: {response.status_code}, {response.text}")
                await asyncio.sleep(5)
        except Exception as e:
            print(f"Batch embedding exception: {e}")
            await asyncio.sleep(5)

def stringify_keys(d):
    if isinstance(d, dict):
        return {str(k): stringify_keys(v) for k, v in d.items()}
    return d

# Async batch insert to Qdrant
async def insert_rows_batch_qdrant(client, rows):
    payload = {
        "points": [
            {
                "id": row["id"],
                "vector": (row["embedding"]),
                "payload": {
                    **stringify_keys(row["content"]),
                    **(stringify_keys(row["metadata"]) if isinstance(row["metadata"], dict) else {})
                }
            }
            for row in rows
        ]
    }
    headers = {
        "api-key": QDRANT_KEY,
        "Content-Type": "application/json"
    }
    print("Full payload to Qdrant:\n", json.dumps(payload, indent=2))
    try:
        json.dumps(payload)
        print(f"\nInserting batch of {len(rows)} rows into Qdrant")
        print("ACTUAL JSON SENT TO QDRANT:\n", json.dumps(payload))
        response = await client.put(
            QDRANT_URL,
            headers=headers,
            json=payload  # Let httpx handle serialization
        )
        print(f"Qdrant batch insert response: {response.status_code} - {response.text}")
        if response.status_code not in (200, 201):
            print(f"Qdrant batch insert failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"HTTP exception for Qdrant batch insert: {e}")

# Main function
async def main():
    # Read last processed chunk index from progress.txt
    try:
        with open("progress.txt", "r") as f:
            last_processed_chunk = int(f.read().strip())
    except FileNotFoundError:
        last_processed_chunk = -1

    chunk_iter = pd.read_csv("fooddata_vectors_ready.csv", chunksize=10000)
    async with httpx.AsyncClient() as client:
        for chunk_idx, chunk in enumerate(chunk_iter):
            if chunk_idx <= last_processed_chunk:
                continue  # Skip already processed chunks
            chunk["content"] = chunk["content"].apply(json.loads)
            chunk["metadata"] = chunk["metadata"].apply(lambda x: {} if x == '{}' else x)
            chunk["id"] = chunk["id"].astype(int)
            num_batches = math.ceil(len(chunk) / BATCH_SIZE)
            for i in range(num_batches):
                batch_df = chunk.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                contents = list(batch_df["content"])
                embeddings = await get_gemini_batch_embeddings_async(client, contents)
                rows = []
                for idx, row in batch_df.iterrows():
                    rows.append({
                        "id": row["id"],
                        "content": row["content"],
                        "metadata": row["metadata"] if isinstance(row["metadata"], dict) else {},
                        "embedding": embeddings[idx - batch_df.index[0]]
                    })
                await insert_rows_batch_qdrant(client, rows)
            # After successful chunk processing, update progress.txt
            with open("progress.txt", "w") as f:
                f.write(str(chunk_idx))

if __name__ == "__main__":
    asyncio.run(main())