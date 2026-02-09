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
├── config.py                 # Configuration (paths, models, server settings)
├── index_writing.py          # Step 1: Index your markdown files into ChromaDB
├── query_writing.py          # Step 1: CLI tool to test retrieval
├── server.py                 # Step 2: FastAPI server — the brain
├── llm_client.py             # Step 2: Abstraction over Ollama / OpenAI APIs
├── prompts.py                # Step 2: Prompt templates for each suggestion type
├── test_server.py            # Step 2: CLI tool to test the server
├── chroma_db/                # Created on first run — your vector database
└── sample_markdown/          # Test .md files
```

---

## Step 2: Suggestion Server

The suggestion server combines retrieval (your indexed writing) with generation
(a local or remote LLM) to produce personalized suggestions.

### How It Works

```
Your Editor (Obsidian, etc.)
       │
       │  POST /suggest/word  {"text": "said", "sentence": "She said..."}
       ▼
┌──────────────────────────────┐
│    Suggestion Server         │
│    (FastAPI on localhost)    │
│                              │
│  1. Retrieve: search ChromaDB│ ← finds sentences where you used
│     for similar past writing │   similar words in similar contexts
│                              │
│  2. Prompt: build a prompt   │ ← "given these examples from this
│     with your writing as     │   writer's past work, suggest
│     context                  │   alternatives for 'said'"
│                              │
│  3. Generate: send to LLM    │ ← local Ollama or remote API
│                              │
│  4. Parse: extract clean     │ ← returns: remarked, noted,
│     suggestions              │   observed, offered, stated
└──────────────────────────────┘
```

### Prerequisites

1. **Your writing must be indexed** (Step 1 — see above)

2. **An LLM must be available.** Pick one:

   **Option A: Ollama (recommended — free, private, local)**
   ```bash
   # Install Ollama: https://ollama.com
   # Then pull a model:
   ollama pull phi3:mini      # Fast, 3.8B params, ~2GB download
   # OR
   ollama pull mistral        # Better quality, 7B params, ~4GB
   # OR
   ollama pull deepseek-r1:7b # Strong reasoning, 7B params
   ```

   **Option B: DeepSeek API (or any OpenAI-compatible API)**
   ```python
   # In config.py, set:
   LLM_BACKEND = "openai_compatible"
   OPENAI_API_BASE = "https://api.deepseek.com/v1"
   OPENAI_API_KEY = "your-key-here"
   OPENAI_MODEL = "deepseek-chat"
   ```

### Running the Server

```bash
# Terminal 1: Start the server
source venv/bin/activate
python server.py

# Terminal 2: Test it
python test_server.py --examples          # Run all example types
python test_server.py --word "said"       # Single word synonym
python test_server.py --interactive       # Interactive REPL
python test_server.py --health            # Check status
```

### API Endpoints

**POST /suggest/word** — Synonyms for a single word
```bash
curl -X POST http://localhost:8111/suggest/word \
  -H "Content-Type: application/json" \
  -d '{"text": "said", "sentence": "She said the results were inconclusive."}'
```

**POST /suggest/phrase** — Rephrase a phrase
```bash
curl -X POST http://localhost:8111/suggest/phrase \
  -H "Content-Type: application/json" \
  -d '{"text": "idea generating", "sentence": "Why am I so idea generating these days."}'
```

**POST /suggest/sentence** — Rewrite a full sentence
```bash
curl -X POST http://localhost:8111/suggest/sentence \
  -H "Content-Type: application/json" \
  -d '{"text": "The results were not what we expected."}'
```

**POST /suggest/complete** — Complete an unfinished sentence
```bash
curl -X POST http://localhost:8111/suggest/complete \
  -H "Content-Type: application/json" \
  -d '{"text": "The fundamental problem with"}'
```

**POST /suggest** — Auto-detect type
```bash
curl -X POST http://localhost:8111/suggest \
  -H "Content-Type: application/json" \
  -d '{"text": "determination"}'
```

**GET /health** — Server status
```bash
curl http://localhost:8111/health
```

### Tuning Suggestions

Edit `config.py` to adjust:
- `LLM_TEMPERATURE` — Lower (0.2) = more predictable, Higher (0.8) = more creative
- `RETRIEVAL_TOP_K` — More retrieved examples = more context but slower
- `NUM_SUGGESTIONS` — How many alternatives to generate
- `LLM_TIMEOUT` — Increase if your model is slow to respond

Edit `prompts.py` to adjust the actual prompts sent to the LLM. This is where
you have the most control over suggestion quality.

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
┌─────────────────────────┐
│ sentence-transformers    │ ← Converts each sentence to a 384-dim vector
│ (all-MiniLM-L6-v2)      │
└────────┬────────────────┘
         │ vectors + metadata
         ▼
┌─────────────────┐
│    ChromaDB      │ ← Stores vectors, enables fast similarity search
│  (./chroma_db/)  │
└─────────────────┘
         │
         ▼ query time
┌─────────────────┐
│ query_writing.py │ ← Takes your current text, finds similar past writing
└─────────────────┘
```

## Next Steps (after this works)

- **Step 3**: Build the Obsidian plugin that calls this server and renders ghost text
- **Step 4**: Fine-tune a small model on your writing for even better personalization
