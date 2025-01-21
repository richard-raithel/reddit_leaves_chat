import os
import json
import pinecone
from dotenv import load_dotenv

def main():
    # 1. Load environment variables from .env
    load_dotenv()  # Reads variables from .env into os.environ
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

    # 2. Load data that already contains OpenAI embeddings
    #    (Assuming text-embedding-ada-002 (dimension=1536))
    data_file = "../data/leaves_combined_embedded_openai.json"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 3. Validate data
    if len(data) == 0:
        raise ValueError("No data to upload. The JSON file is empty.")

    first_embedding = data[0].get("embedding", None)
    if not first_embedding:
        raise ValueError("The first entry has no 'embedding' field. Check your JSON.")

    # The dimension for text-embedding-ada-002 is typically 1536,
    # but let's auto-detect to stay flexible if you switch models.
    embedding_dimension = len(first_embedding)
    print(f"Detected embedding dimension: {embedding_dimension}")

    # 4. Initialize Pinecone using environment variables
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

    index_name = "reddit-topic-index"
    # 5. Create or use existing index
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=embedding_dimension  # This should match the OpenAI model (1536 for ada-002)
        )
        print(f"Created new Pinecone index: {index_name}")
    else:
        print(f"Using existing Pinecone index: {index_name}")

    # 6. Connect to the index
    index = pinecone.Index(index_name)

    # 7. Prepare records for upsert
    records_to_upsert = []
    for entry in data:
        embedding = entry.get("embedding")
        if not embedding:
            continue  # Skip entries without embeddings

        # Convert embedding to a tuple for Pinecone
        metadata = {
            "submission_id": entry.get("submission_id"),
            "subreddit": entry.get("subreddit"),
            "title": entry.get("title"),
            "author": entry.get("author"),
            "score": entry.get("score")
        }

        # Use submission_id as the unique Pinecone vector ID
        record = (metadata["submission_id"], embedding, metadata)
        records_to_upsert.append(record)

    # 8. Upsert records into Pinecone
    print(f"Upserting {len(records_to_upsert)} records into the '{index_name}' index...")
    index.upsert(vectors=records_to_upsert)
    print("Upsert complete!")

if __name__ == "__main__":
    main()
