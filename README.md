# Writing Copilot

A personalized writing assistant that learns from your own writing style.

## What This Does

Three-step system:

1. **Index** your markdown writing into a vector database (ChromaDB)
2. **Suggest** alternatives for words, phrases, and sentences using retrieval + LLM
3. **Integrate** with Obsidian plugin for in-editor suggestions

## Architecture: Two Strategies

### Word Synonyms — Context-Aware + Personal

**LLM synonyms + Vocabulary Hashmap** (default: `SYNONYM_BACKEND = "llm"`)
```
"inclined" + sentence → LLM (Gemini/Ollama) → [prone, apt, willing, ...]
                     → Check which ones you've used before (O(1) lookup)
                     → Return: personal words first, ranked by frequency
```

- **Context-aware:** the full sentence disambiguates polysemous words
- **Resilient:** falls back to the free Datamuse API automatically if the
  LLM is unavailable (set `SYNONYM_BACKEND = "datamuse"` to always use it)

### Phrases & Sentences — Two-Stage RAG

**ChromaDB → Cross-Encoder Reranker → LLM**
```
"rephrase this phrase" → ChromaDB: bi-encoder retrieves 30 candidate sentences
                      → Reranker: cross-encoder scores (query, sentence) pairs
                                  jointly, keeps the best 8
                      → LLM: "Given these examples of my voice..."
                      → Generate: personalized suggestions
```

- Bi-encoder (sentence-transformers) is fast but lossy; the cross-encoder
  reranker reads each pair jointly and is much more accurate — this is the
  standard retrieve-then-rerank RAG architecture
- Personal vocabulary & style (reranked examples inform the LLM in-context)

## Tech Stack

| Tool | What It Does | Why |
|---|---|---|
| **Gemini API** | Suggestion generation (default LLM) | Fast, cheap, high quality (`gemini-2.5-flash`) |
| **Ollama** | Local LLM fallback | Free, private, no API key needed |
| **ChromaDB** | Vector database for semantic search | Fast similarity search of your past writing |
| **sentence-transformers** | Text → vectors + cross-encoder reranker | Industry standard, runs locally |
| **Datamuse API** | Synonym fallback when no LLM available | Free, no credentials |
| **spaCy** | Sentence segmentation | Accurate, handles abbreviations |
| **FastAPI** | Web server | Simple, fast, async |
| **httpx** | HTTP client | Clean API for external calls |

## Setup

### 1. Clone & install dependencies

```bash
cd /path/to/Linger/files
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Add your Gemini API key

```bash
cp .env.example .env
# then edit .env and set GEMINI_API_KEY=...
# (get a key at https://aistudio.google.com/apikey)
```

No key? Run everything locally instead: `LLM_BACKEND=ollama python server.py`
(requires Ollama with `ollama pull phi3:mini`).

### 3. Index your writing

```bash
python vocab_index.py /path/to/your/markdown/vault
python index_writing.py /path/to/your/markdown/vault
```

This creates:
- `vocab_index.json` — word frequency counter (~1MB)
- `chroma_db/` — ChromaDB vector database

### 4. Start the server

```bash
python server.py
```

Server runs at `http://127.0.0.1:8111`  
API docs at `http://127.0.0.1:8111/docs`

### 5. Test it

```bash
python test_server.py
```

Choose an endpoint, test with your own text.

## Configuration

Secrets live in `.env` (gitignored — never put keys in config.py):

```bash
GEMINI_API_KEY=...
# optional overrides:
# LLM_BACKEND=ollama
# GEMINI_MODEL=gemini-2.5-flash
```

Everything else is in `config.py`:

```python
# Phrase/sentence/completion: which LLM?
LLM_BACKEND = "gemini"   # default; also: "ollama", "openai_compatible"

# Word synonyms: LLM with automatic Datamuse fallback
SYNONYM_BACKEND = "llm"  # or "datamuse"

# Two-stage retrieval
USE_RERANKER = True
RETRIEVAL_CANDIDATES = 30  # stage 1: bi-encoder candidates
RETRIEVAL_TOP_K = 8        # stage 2: kept after cross-encoder rerank

# Ollama settings (local fallback)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi3:mini"  # avoid reasoning models like deepseek-r1
```

## Project Structure

```
files/
├── README.md                    # This file
├── CLAUDE.md                    # Context/instructions for AI agents
├── config.py                    # Configuration (no secrets)
├── .env.example                 # Template for .env (API keys)
│
├── index_writing.py             # Index markdown → ChromaDB
├── query_writing.py             # CLI tool to test retrieval
├── vocab_index.py               # Build word frequency index
│
├── server.py                    # FastAPI suggestion server
├── llm_client.py                # LLM abstraction (Gemini/Ollama/OpenAI-compatible)
├── reranker.py                  # Cross-encoder reranker (retrieval stage 2)
├── thesaurus.py                 # Word synonym lookup (LLM/Datamuse)
├── prompts.py                   # Prompt templates + output parsing
│
├── test_server.py               # CLI tool to test server
│
├── chroma_db/                   # Vector database (created on first run)
└── vocab_index.json             # Word frequencies (created by vocab_index.py)
```

## Step-by-Step Usage

### Step 1: Index Your Writing

```bash
# Build vocabulary index (one-time setup)
python vocab_index.py ~/Documents/MyVault/
# Shows: word frequencies, statistics

# Index into ChromaDB (one-time setup)
python index_writing.py ~/Documents/MyVault/
# Shows: sentences indexed, embedding time
```

Both create persistent files, re-run when you add new writing.

### Step 2: Test Retrieval

```bash
# Interactive retrieval testing
python query_writing.py --interactive

# Type: "she said the results"
# Returns: Most similar sentences from your past writing
```

### Step 3: Start the Server

**Terminal 1:**
```bash
python server.py
```

**Terminal 2:**
```bash
python test_server.py
```

Or use curl:
```bash
# Test word synonyms
curl -X POST http://localhost:8111/suggest/word \
  -H "Content-Type: application/json" \
  -d '{
    "word": "big",
    "sentence": "The problem is very big."
  }'

# Response:
# {
#   "suggestions": ["large", "huge", "substantial", "massive", "enormous"],
#   "retrieved_examples": ["big (10x in your writing)"],
#   "latency_ms": 45
# }
```

## Which Backend Should I Use?

### **Gemini (default)**
✅ Use when:
- You want the best quality and lowest latency
- You're OK with a (very cheap) API dependency
- Setup: put `GEMINI_API_KEY` in `.env`, done

### **Ollama (local fallback)**
✅ Use when:
- You want privacy / fully offline operation
- You don't want any API key
- Use `phi3:mini` or similar; avoid reasoning models (deepseek-r1, qwq) —
  they spend seconds emitting `<think>` text before answering

### **Datamuse for Synonyms**
✅ Use when:
- You want zero-dependency synonym lookup (<100ms, free)
- Note: it's the automatic fallback when the LLM is unreachable, so you
  usually don't need to set it explicitly

❌ Doesn't work well for:
- Polysemous words ("inclined" → physical meaning instead of figurative)

## Tuning

Edit `config.py`:

```python
NUM_SUGGESTIONS = 5           # How many alternatives to show
RETRIEVAL_TOP_K = 8           # How many examples to retrieve (more = slower)
LLM_TEMPERATURE = 0.4         # Lower = more predictable, Higher = creative
LLM_TIMEOUT = 120             # Seconds before request times out
```

Edit `prompts.py` to change how suggestions are generated.

## Troubleshooting

**"ModuleNotFoundError: No module named 'spacy'"**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**"ChromaDB error: No collection found"**
```bash
# Run indexing first
python index_writing.py ~/Documents/MyVault/
```

**"OpenAI API: insufficient_quota"**
- Check your OpenAI account at https://platform.openai.com/account/billing
- Switch back to Datamuse: `SYNONYM_BACKEND = "datamuse"`

**"Connection refused" (server won't start)**
- Port 8111 might be in use, or Ollama isn't running
- Check: `lsof -i :8111` and `ollama serve`

## Next Steps
