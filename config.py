"""
Configuration for Writing Copilot.

Edit these values to customize behavior. The defaults work well for most cases.
"""

from pathlib import Path

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
# Options: "ollama", "openai_compatible"
#
# "ollama" — runs locally via Ollama (recommended, free, private)
#   Install: https://ollama.com → then `ollama pull phi3:mini`
#
# "openai_compatible" — any OpenAI-compatible API (DeepSeek, Together, etc.)
#   Set OPENAI_API_BASE and OPENAI_API_KEY below
LLM_BACKEND = "ollama"

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "deepseek-r1:7b"  # Good balance of speed + quality
# Alternatives:
#   "mistral"                    → better quality, slower, needs 8GB+ VRAM
#   "deepseek-r1:7b"            → strong reasoning, slower
#   "qwen2.5:1.5b"              → fastest, lower quality
#   "llama3.1:8b"               → good all-rounder

# OpenAI-compatible API settings (for DeepSeek API, Together, etc.)
OPENAI_API_BASE = "https://api.deepseek.com/v1"  # Change for other providers
OPENAI_API_KEY = ""  # Set this or use OPENAI_API_KEY env var
OPENAI_MODEL = "deepseek-chat"

# --- Generation Settings ---
# Temperature controls randomness. Lower = more predictable suggestions.
# For synonym/rephrase tasks, 0.3–0.6 works well.
LLM_TEMPERATURE = 0.4

# Max tokens for LLM response
LLM_MAX_TOKENS = 500

# Timeout in seconds for LLM requests
LLM_TIMEOUT = 120

# --- Suggestion Settings ---
# Number of alternative suggestions to generate per request
NUM_SUGGESTIONS = 5

# Number of similar sentences to retrieve from your corpus as context
RETRIEVAL_TOP_K = 8

# --- Server Settings ---
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8111
