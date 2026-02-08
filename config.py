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
