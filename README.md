# Summarize Private Documents with RAG, LangChain, and Llama 3

Python project to query and summarize private documents using Retrieval-Augmented Generation (RAG) with LangChain, FAISS, sentence-transformers embeddings, Llama 3 via Hugging Face, and conversational memory. Includes Streamlit UI, FastAPI API, CLI, and a notebook. [web:31][web:39]

## Features

- Chat with your own private documents using a RAG pipeline.
- Uses FAISS as a vector store with `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- Llama 3 served via Hugging Face for answer generation.
- Conversational memory for multi-turn Q&A.
- Three frontends:
  - Streamlit web UI.
  - FastAPI JSON API.
  - Command-line chat loop.
- Jupyter notebook for ingestion and experimentation. [web:31][web:41]

## Project Files

- `RAG_Private_Documents.ipynb` – Notebook to build the FAISS index and test the pipeline.
- `streamlit_app.py` – Streamlit app for interactive Q&A.
- `fastapi_app.py` – FastAPI backend exposing a `/ask` endpoint.
- `qa_cli.py` – Simple command-line QA loop.
- `rag_private_documents.py` – Script to build the FAISS vector store.
- `requirements.txt` – Python dependencies. [web:31]

You create `data/` (documents) and `vectorstore/` (FAISS index) locally; these folders are not required in the repo itself.

## Installation

Create and activate a virtual environment (optional but recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

Set your Hugging Face token (for Llama 3 via Hugging Face Hub):

```bash
export HUGGINGFACEHUB_API_TOKEN="your_token_here"
```

On Windows PowerShell:

```powershell
$env:HUGGINGFACEHUB_API_TOKEN="your_token_here"
```

## Build the Vector Store

Place your private documents (TXT, MD, PDF converted to text, etc.) into a local `data` directory (next to the scripts), then run:

```bash
python rag_private_documents.py
```

Or execute the notebook cell inside `RAG_Private_Documents.ipynb` that loads from `data/`, splits, embeds, and saves FAISS to `vectorstore`. [web:39]

This will:

1. Load documents from `data/`.
2. Chunk them with a recursive text splitter.
3. Embed them using `sentence-transformers/all-MiniLM-L6-v2`.
4. Save the FAISS index into `vectorstore/`. [web:31][web:43]

## Run the Streamlit App

Start the web UI:

```bash
streamlit run streamlit_app.py
```

Open the provided local URL in your browser. Ask questions like:

- “What is the smoking policy?”
- “List all points of the smoking policy.”
- “Summarize the health and safety section.”

The app will retrieve relevant chunks from your documents and generate an answer with context. [web:31][web:42]

## Run the FastAPI Service

Start the API server:

```bash
uvicorn fastapi_app:app --reload
```

Send a POST request to `http://localhost:8000/ask`:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the smoking policy."}'
```

You will receive a JSON response:

```json
{ "answer": "..." }
```

This is useful for integrating the RAG backend into other tools or UIs. [web:10][web:31]

## Run the CLI Chat

Use the command-line QA loop:

```bash
python qa_cli.py
```

Then type questions:

```text
Question: What is the smoking policy?
Question: Can you summarize all disciplinary rules?
Question: bye
```

Type `quit`, `exit`, or `bye` to stop the loop. [web:39]

## How It Works (High Level)

1. **Ingestion** – Documents in `data/` are loaded and split into overlapping text chunks.
2. **Embedding** – Each chunk is embedded with `sentence-transformers/all-MiniLM-L6-v2`.
3. **Indexing** – Embeddings are stored in a FAISS vector index on disk.
4. **Retrieval** – At query time, the most relevant chunks are retrieved from FAISS.
5. **Generation** – Llama 3 (via Hugging Face) uses the retrieved context to generate grounded answers. [web:31][web:35][web:44]

This pattern follows standard RAG practices described in LangChain and RAG tutorials. [web:31][web:35][web:43]

## Privacy Note

Only documents you place in your local `data` directory are used. The FAISS index and all content stay on 
