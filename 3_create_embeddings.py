import json
import os
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
import tiktoken
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv()

def count_tokens(text, tokenizer):
    """Count tokens in a given text using the specified tokenizer."""
    return len(tokenizer.encode(text))

def create_safe_batches(texts, tokenizer, max_tokens_per_request):
    """Create batches of texts that stay within the token limit."""
    batches = []
    current_batch = []
    current_tokens = 0

    for text in texts:
        text_tokens = count_tokens(text, tokenizer)
        if current_tokens + text_tokens > max_tokens_per_request:
            batches.append(current_batch)
            current_batch = [text]
            current_tokens = text_tokens
        else:
            current_batch.append(text)
            current_tokens += text_tokens

    if current_batch:
        batches.append(current_batch)

    return batches

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def embed_batch(batch_texts, embeddings):
    """Embed a batch of texts with retry logic."""
    return embeddings.embed_documents(batch_texts)

def embed_data(json_file_path, faiss_index_path="faiss_index", max_tokens_per_request=8191):
    """
    Reads JSON data (which combines submissions and comments),
    creates embeddings using OpenAI (via langchain_openai), and
    stores them in a FAISS index on disk -- all with a tqdm progress bar.
    """

    # 1. Load the combined JSON file
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Data loaded successfully.")

    # 2. Convert each submission + comments into Documents
    documents = []
    for submission in data:
        # Build text for the main submission
        submission_text = (
            f"Title: {submission.get('title', '')}\n\n"
            f"Body: {submission.get('body', '')}\n\n"
            f"Author: {submission.get('author', '')}\n"
            f"Subreddit: {submission.get('subreddit', '')}\n"
            f"Created (UTC): {submission.get('created_utc', '')}\n"
            f"Score: {submission.get('score', '')}"
        )
        documents.append(
            Document(
                page_content=submission_text,
                metadata={
                    "submission_id": submission.get("submission_id"),
                    "type": "submission",
                    "author": submission.get("author"),
                    "subreddit": submission.get("subreddit"),
                    "created_utc": submission.get("created_utc"),
                    "score": submission.get("score"),
                }
            )
        )

        # Build a Document for each comment
        for comment in submission.get("comments", []):
            comment_text = (
                f"Comment by {comment.get('author', '')}:\n"
                f"{comment.get('body', '')}\n\n"
                f"Created (UTC): {comment.get('created_utc', '')}\n"
                f"Score: {comment.get('score', '')}"
            )
            documents.append(
                Document(
                    page_content=comment_text,
                    metadata={
                        "submission_id": submission.get("submission_id"),
                        "comment_id": comment.get("comment_id"),
                        "type": "comment",
                        "author": comment.get("author"),
                        "created_utc": comment.get("created_utc"),
                        "score": comment.get("score"),
                    }
                )
            )
    print("Documents created successfully.")

    # 3. Prepare to embed with OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))
    print("OpenAI Embeddings initialized.")

    # 4. Tokenizer for counting tokens
    tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
    print("Tokenizer initialized.")

    # 5. Prepare safe batches
    all_texts = [doc.page_content for doc in documents]
    all_metadatas = [doc.metadata for doc in documents]
    safe_batches = create_safe_batches(all_texts, tokenizer, max_tokens_per_request)

    all_embeddings = []
    all_metadata = []
    print(f"Total batches: {len(safe_batches)}")

    # 6. Embed in batches, displaying progress
    for batch_texts in tqdm(safe_batches, desc="Embedding batches"):
        batch_embeddings = embed_batch(batch_texts, embeddings)
        all_embeddings.extend(batch_embeddings)
        all_metadata.extend(batch_texts)

    # 7. Build the FAISS index from these precomputed embeddings
    vectorstore = FAISS.from_embeddings(
        embeddings=all_embeddings,
        texts=all_texts,
        embedding=embeddings,  # pass the embedding instance for future similarity queries
        metadatas=all_metadatas
    )

    # 8. Save the FAISS index locally
    vectorstore.save_local(faiss_index_path)
    print(f"\nEmbedding complete. FAISS index saved to: {faiss_index_path}")


if __name__ == "__main__":
    # Example usage
    embed_data(
        json_file_path="../data/leaves_combined.json",
        faiss_index_path="faiss_index"
    )
