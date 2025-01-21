import os
import openai
import pinecone
import time
import textwrap

#############################################
# Constants and Utility Functions
#############################################

MAX_CHARS_RETRIEVED = 500  # Max characters to pull from each doc's text body
MAX_RETRIES = 3            # Number of times to retry an OpenAI call on rate-limit

def summarize_text(text: str, max_chars=MAX_CHARS_RETRIEVED) -> str:
    """
    Safely truncates text to max_chars.
    For a more advanced approach, you could call GPT again to summarize.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"

def openai_chat_completion(
    messages,
    model="gpt-4",
    temperature=0.7,
    max_tokens=500,
):
    """
    Helper to call openai.chat.completions.create with basic retry logic.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response
        except openai.RateLimitError as e:
            print(f"Rate limit error: {str(e)}. Retrying ({attempt+1}/{MAX_RETRIES})...")
            time.sleep(2 ** attempt)  # exponential backoff
        except openai.APIConnectionError as e:
            print(f"Connection error: {str(e)}. Retrying ({attempt+1}/{MAX_RETRIES})...")
            time.sleep(2 ** attempt)

    # If we exhaust retries, raise the last error
    raise RuntimeError("OpenAI API request failed after max retries.")

#############################################
# Main Chatbot Logic
#############################################

def init_openai():
    # Use environment variable or .env file for OPENAI_API_KEY
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("Missing OPENAI_API_KEY environment variable.")

def init_pinecone():
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

    return pinecone.Index("reddit-topic-index")  # adjust name if needed

def embed_query(query: str) -> list:
    """
    Calls OpenAI to embed the user query (text-embedding-ada-002).
    Return a float list vector.
    """
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    return response.data[0].embedding

def query_chatbot(query: str, index, top_k=5, model_name="gpt-4o") -> str:
    """
    1) Embed the query using OpenAI Embeddings.
    2) Retrieve top-k docs from Pinecone.
    3) Summarize/truncate doc content as needed.
    4) Build a structured system + user prompt for GPT-4.
    5) Return GPT-4's answer.
    """
    # --- STEP 1: EMBED THE QUERY ---
    query_embedding = embed_query(query)

    # --- STEP 2: RETRIEVE MATCHES FROM PINECONE ---
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)

    # --- STEP 3: Summarize or truncate the doc content ---
    # We'll build a "context" string that GPT can reference
    context_list = []
    for match in results["matches"]:
        meta = match.get("metadata", {})
        title = meta.get("title", "[No Title]")
        submission_id = meta.get("submission_id", "UnknownID")
        body_text = meta.get("body", "")
        snippet = summarize_text(body_text, max_chars=MAX_CHARS_RETRIEVED)

        context_list.append(f"SubmissionID: {submission_id}\nTitle: {title}\nSnippet: {snippet}")

    combined_context = "\n\n".join(context_list)

    # --- STEP 4: Build system and user messages ---
    # System message: Provide role/instructions
    system_message = {
        "role": "system",
        "content": (
            "You are an expert counselor specializing in helping people quit substances. "
            "Use the provided context to answer user questions. "
            "If the context is insufficient, say so. "
            "Keep responses empathetic and fact-based."
        )
    }

    # Additional instructions or persona can also be placed here or as an assistant message:
    # assistant_message = {
    #     "role": "assistant",
    #     "content": "I can help you with substance quitting advice..."
    # }

    user_message = {
        "role": "user",
        "content": (
            f"Context:\n{combined_context}\n\n"
            f"User Question: {query}\n\n"
            "Provide a helpful answer based on the above context."
        )
    }

    # We can optionally add more messages
    messages = [system_message, user_message]

    # --- STEP 5: Call GPT-4 (or 'gpt-4o') with retry logic ---
    response = openai_chat_completion(
        messages=messages,
        model=model_name,
        temperature=0.7,
        max_tokens=500,
    )

    # Extract GPT-4's text
    answer = response.choices[0].message.content
    return answer

def main():
    # 1. Initialize APIs
    init_openai()
    index = init_pinecone()

    # 2. Example usage
    user_query = "What are the common challenges in quitting cannabis?"
    answer = query_chatbot(user_query, index, top_k=3, model_name="gpt-4o")
    print("Chatbot Response:")
    print(answer)

if __name__ == "__main__":
    main()
