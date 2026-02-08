# Writing Copilot — Step 1: Indexing Your Writing

## What This Does

This is the foundation of your writing copilot. It takes a folder of your markdown files,
splits them into meaningful chunks (sentences and short phrases), converts each chunk into
a numerical vector that represents its meaning, and stores everything in a local vector
database (ChromaDB). Once indexed, you can query it: give it a word or sentence, and it
returns the most similar things you've ever written.

This is the retrieval half of RAG. Step 2 (later) adds the generation half.

## Tech Stack

| Tool | What It Does | Why This One |
|---|---|---|
| **Python 3.10+** | Language for the backend | Best ML ecosystem, all libraries available |
| **sentence-transformers** | Converts text → vectors | Industry standard, runs locally, no API needed |
| **all-MiniLM-L6-v2** | The specific embedding model | 80MB, fast on CPU, good quality for semantic search |
| **ChromaDB** | Stores and searches vectors | Simple API, handles persistence, no server needed |
| **spaCy** | Splits text into sentences | More accurate than regex, handles abbreviations etc. |
| **rich** | Terminal output formatting | Makes the CLI tool pleasant to use |

## Setup

### 1. Install Python (if needed)

Check with `python3 --version`. You need 3.10 or higher.

### 2. Create a virtual environment

```bash
cd writing-copilot
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Index your writing

```bash
python index_writing.py /path/to/your/markdown/folder
```

This will:
- Recursively find all `.md` files in that folder
- Extract text (stripping markdown formatting)
- Split into sentences
- Generate embeddings for each sentence
- Store everything in a local ChromaDB database at `./chroma_db/`

### 5. Test retrieval

```bash
python query_writing.py "she said the results were unclear"
```

This will return the 10 most similar sentences from your own writing,
ranked by cosine similarity. This is how you verify the system is working
before building the suggestion engine or the Obsidian plugin.

### 6. Interactive mode

```bash
python query_writing.py --interactive
```

Type sentences or words and see what your past writing has that's similar.
Press Ctrl+C to exit.

## Project Structure

```
writing-copilot/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── index_writing.py          # Indexes your markdown files into ChromaDB
├── query_writing.py          # CLI tool to test retrieval
├── config.py                 # Configuration (paths, model names, etc.)
├── chroma_db/                # Created on first run — your vector database
└── sample_markdown/          # (Optional) Place test .md files here
```

## How the Pieces Fit Together

```
Your Markdown Files
       │
       ▼
┌─────────────────┐
│ index_writing.py │ ← Reads files, strips markdown, splits sentences
└────────┬────────┘
         │ sentences
         ▼
```
