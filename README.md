# RAG Chatbot Workflow using LangChain and OpenAI Embeddings

## Overview

This repository contains a step-by-step workflow for creating a Retrieval-Augmented Generation (RAG) chatbot using LangChain and OpenAI embeddings. The scripts handle data extraction, preprocessing, embedding creation, and chatbot deployment for both testing and production environments.

---

## Features
- **Data Pipeline**:
  - Extract raw data from `.zst` files (`1_zst_extract.py`).
  - Preprocess extracted data for embedding generation (`2_preprocess_data.py`).
  - Generate OpenAI embeddings for preprocessed data (`3_create_embeddings.py`).
  - Convert and optimize embeddings for storage (`4_convert_embeddings.py`).
  - Store embeddings in a vector database (`5_store_embeddings.py`).

- **Chatbot Deployment**:
  - Test chatbot functionality in a development environment (`6_test_chat.py`).
  - Deploy the chatbot for production use (`7_prod_chat.py`).

---

## Requirements

### Prerequisites
- Python 3.8 or higher
- OpenAI API Key (for embedding generation)
- LangChain for RAG pipeline
- Vector database (e.g., Pinecone, Weaviate, or Chroma)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/rag-chatbot-workflow.git
   cd rag-chatbot-workflow
