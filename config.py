"""
Configuration for Writing Copilot.

Edit these values to customize behavior. The defaults work well for most cases.

Secrets (API keys) live in a .env file next to this file, never in here:
    GEMINI_API_KEY=...
Any setting read via os.environ below can also be overridden from .env.
"""

import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines). Real env vars take precedence."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv(Path(__file__).parent / ".env")

# --- Paths ---
# Where ChromaDB stores your indexed writing
CHROMA_DB_PATH = str(Path(__file__).parent / "chroma_db")

# ChromaDB collection name (you could have multiple, e.g., "academic", "creative")
COLLECTION_NAME = "my_writing"

# --- Embedding Model ---
# This model converts text to 384-dimensional vectors.
# all-MiniLM-L6-v2 is small (80MB), fast, and good quality.
# Alternatives if you want to experiment:
#   - "all-mpnet-base-v2"       → better quality, slower, 420MB
#   - "paraphrase-MiniLM-L3-v2" → faster, slightly lower quality, 60MB
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Text Processing ---
# File extensions to index
SUPPORTED_EXTENSIONS = {".md", ".txt", ".markdown"}

# Minimum sentence length (characters) to index.
# Filters out headings, short fragments, metadata lines.
MIN_SENTENCE_LENGTH = 20

# Maximum sentence length (characters). Very long "sentences" are usually
# parsing artifacts or code blocks.
MAX_SENTENCE_LENGTH = 500

# --- Retrieval ---
# Number of results to return by default
DEFAULT_TOP_K = 10

# Minimum similarity score (0-1) to include in results.
# ChromaDB returns distances (lower = more similar).
# For cosine distance: 0.0 = identical, 2.0 = opposite.
MAX_DISTANCE = 1.2  # Filters out very dissimilar results

# --- Markdown Stripping ---
# Elements to remove from markdown before indexing
STRIP_CODE_BLOCKS = True       # Remove ```code blocks```
STRIP_FRONTMATTER = True       # Remove YAML frontmatter (---...---)
STRIP_LINKS = True             # Convert [text](url) to just text
STRIP_IMAGES = True            # Remove ![alt](url) entirely
STRIP_HTML = True              # Remove inline HTML tags
STRIP_TAGS = True              # Remove #tags

# ─── Step 2: Suggestion Server ─────────────────────────────────────────

# --- LLM Backend ---
# Which LLM provider to use for generating suggestions.
# Options: "gemini", "ollama", "openai_compatible"
#
# "gemini" — Google Gemini API (recommended: fast, cheap, high quality)
#   Put GEMINI_API_KEY=... in .env (get one at https://aistudio.google.com/apikey)
#
# "ollama" — runs locally via Ollama (free, private, no key needed)
#   Install: https://ollama.com → then `ollama pull phi3:mini`
#
# "openai_compatible" — any OpenAI-compatible API (DeepSeek, Together, etc.)
#   Set OPENAI_API_KEY in .env and OPENAI_API_BASE below
LLM_BACKEND = os.environ.get("LLM_BACKEND", "gemini")

# Gemini settings
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Ollama settings (local fallback)
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "phi3:mini"  # Fast, no reasoning tags in output
# Alternatives:
#   "mistral"                    → better quality, slower, needs 8GB+ VRAM
#   "qwen2.5:1.5b"              → fastest, lower quality
#   "llama3.1:8b"               → good all-rounder
# Avoid reasoning models (deepseek-r1, qwq) — they waste seconds "thinking"
# before emitting suggestions.

# OpenAI-compatible API settings (for DeepSeek, Together, etc.)
OPENAI_API_BASE = "https://api.openai.com/v1"  # Change for other providers
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")  # Set in .env
OPENAI_MODEL = "gpt-4o-mini"

# --- Generation Settings ---
# Temperature controls randomness. Lower = more predictable suggestions.
# For synonym/rephrase tasks, 0.3–0.6 works well.
LLM_TEMPERATURE = 0.4

# Max tokens for LLM response
LLM_MAX_TOKENS = 500

# Timeout in seconds for LLM requests
LLM_TIMEOUT = 120

# --- Suggestion Settings ---
# Which backend to use for word synonym suggestions.
# Options:
#   "llm"      — context-aware synonyms via the configured LLM backend,
#                falls back to Datamuse if the LLM is unavailable
#   "datamuse" — free dictionary API, fast but no context awareness
SYNONYM_BACKEND = "llm"

# Number of alternative suggestions to generate per request
NUM_SUGGESTIONS = 5

# --- Retrieval (two-stage) ---
# Stage 1: ChromaDB bi-encoder fetches RETRIEVAL_CANDIDATES nearest sentences.
# Stage 2: a cross-encoder reranks them and keeps RETRIEVAL_TOP_K.
# The cross-encoder scores (query, sentence) pairs jointly, which is much more
# accurate than embedding distance alone — that's the reranker in the RAG stack.
USE_RERANKER = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RETRIEVAL_CANDIDATES = 30
RETRIEVAL_TOP_K = 8

# --- Server Settings ---
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8111
