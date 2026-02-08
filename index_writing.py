#!/usr/bin/env python3
"""
Index your markdown writing into a ChromaDB vector database.

Usage:
    python index_writing.py /path/to/your/markdown/folder
    python index_writing.py /path/to/folder --reindex    # Clear and rebuild
    python index_writing.py /path/to/folder --stats      # Show index stats only

What it does:
    1. Recursively finds all .md/.txt files in the given folder
    2. Strips markdown formatting to get clean prose
    3. Splits text into individual sentences using spaCy
    4. Embeds each sentence using a sentence transformer model
    5. Stores everything in a local ChromaDB database

The database persists at ./chroma_db/ so you only need to run this once,
then again when you've written new content.
"""

import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

import config

console = Console()


# ─── Markdown Stripping ────────────────────────────────────────────────

def strip_markdown(text: str) -> str:
    """
    Remove markdown formatting to get clean prose text.

    We want the actual words the writer chose, not markdown syntax.
    This is important because the embedding model should encode meaning,
    not formatting artifacts.
    """
    # Remove YAML frontmatter (--- ... ---)
    if config.STRIP_FRONTMATTER:
        text = re.sub(r'^---\s*\n.*?\n---\s*\n', '', text, flags=re.DOTALL)

    # Remove code blocks (``` ... ```) — these aren't prose
    if config.STRIP_CODE_BLOCKS:
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)  # inline code

    # Remove images ![alt](url)
    if config.STRIP_IMAGES:
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # Convert links [text](url) → text
    if config.STRIP_LINKS:
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove HTML tags
    if config.STRIP_HTML:
        text = re.sub(r'<[^>]+>', '', text)

    # Remove hashtags/tags like #topic
    if config.STRIP_TAGS:
        text = re.sub(r'(?<!\w)#\w+', '', text)

    # Remove markdown emphasis markers but keep the text
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)  # ***bold italic***
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)        # **bold**
    text = re.sub(r'\*(.+?)\*', r'\1', text)             # *italic*
    text = re.sub(r'__(.+?)__', r'\1', text)             # __bold__
    text = re.sub(r'_(.+?)_', r'\1', text)               # _italic_
    text = re.sub(r'~~(.+?)~~', r'\1', text)             # ~~strikethrough~~

    # Remove heading markers (# ## ### etc.) but keep the text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Remove blockquote markers
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # Remove list markers (-, *, numbered) but keep the text
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Remove task list markers
    text = re.sub(r'\[[ x]\]\s*', '', text)

    # Collapse multiple blank lines into one
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace per line
    text = '\n'.join(line.strip() for line in text.split('\n'))

    return text.strip()


# ─── Sentence Splitting ────────────────────────────────────────────────

def split_into_sentences(text: str, nlp) -> list[str]:
    """
    Split text into individual sentences using spaCy.

    Why spaCy instead of splitting on periods?
    - Handles abbreviations: "Dr. Smith said..." is one sentence, not two
    - Handles quotes: '"I agree," she said.' is one sentence
    - Handles ellipses, decimals, URLs, etc.

    Returns sentences that pass length filters defined in config.
    """
    doc = nlp(text)
    sentences = []

    for sent in doc.sents:
        s = sent.text.strip()

        # Filter by length
        if len(s) < config.MIN_SENTENCE_LENGTH:
            continue
        if len(s) > config.MAX_SENTENCE_LENGTH:
            continue

        # Skip sentences that are mostly non-alphabetic
        # (catches leftover formatting, URLs, file paths, etc.)
        alpha_ratio = sum(c.isalpha() for c in s) / max(len(s), 1)
        if alpha_ratio < 0.5:
            continue

        sentences.append(s)

    return sentences


# ─── File Discovery ────────────────────────────────────────────────────

def find_markdown_files(folder: Path) -> list[Path]:
    """Recursively find all supported text files in a folder."""
    files = []
    for ext in config.SUPPORTED_EXTENSIONS:
        files.extend(folder.rglob(f"*{ext}"))
    # Sort for deterministic ordering
    return sorted(files)


def make_sentence_id(filepath: str, sentence: str) -> str:
    """
    Create a deterministic unique ID for a sentence.

    Uses a hash of filepath + sentence text so the same sentence
    from the same file always gets the same ID. This lets us
    detect duplicates and avoid re-indexing unchanged content.
    """
    content = f"{filepath}::{sentence}"
    return hashlib.sha256(content.encode()).hexdigest()


# ─── Main Indexing Logic ───────────────────────────────────────────────

def index_folder(folder: Path, reindex: bool = False):
    """
    Index all markdown files in a folder into ChromaDB.

    This is the main function. It:
    1. Loads the spaCy model for sentence splitting
    2. Connects to (or creates) the ChromaDB database
    3. Reads each file, strips markdown, splits into sentences
    4. Adds new sentences to the database (skipping duplicates)
    """

    # --- Load NLP model ---
    console.print("\n[bold]Loading NLP models...[/bold]")

    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        console.print("[yellow]Downloading spaCy model (one-time)...[/yellow]")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # Increase max length for large documents
    nlp.max_length = 2_000_000

    console.print("[green]✓[/green] spaCy loaded")

    # --- Connect to ChromaDB ---
    import chromadb

    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

    if reindex:
        # Delete existing collection and start fresh
        try:
            client.delete_collection(config.COLLECTION_NAME)
            console.print("[yellow]Cleared existing index[/yellow]")
        except ValueError:
            pass

    # ChromaDB will use sentence-transformers for embedding automatically
    # when we specify the embedding function
    from chromadb.utils import embedding_functions

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )

    existing_count = collection.count()
    console.print(f"[green]✓[/green] ChromaDB connected ({existing_count} existing entries)")

    # --- Find files ---
    files = find_markdown_files(folder)
    if not files:
        console.print(f"[red]No markdown files found in {folder}[/red]")
        return

    console.print(f"[green]✓[/green] Found {len(files)} files to index\n")

    # --- Process files ---
    total_sentences = 0
    new_sentences = 0
    skipped_duplicates = 0
    file_stats = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing files...", total=len(files))

        for filepath in files:
            try:
                raw_text = filepath.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                console.print(f"[red]Error reading {filepath}: {e}[/red]")
                progress.advance(task)
                continue

            # Strip markdown
            clean_text = strip_markdown(raw_text)

            if not clean_text.strip():
                progress.advance(task)
                continue

            # Split into sentences
            sentences = split_into_sentences(clean_text, nlp)

            if not sentences:
                progress.advance(task)
                continue

            # Prepare batch for ChromaDB
            ids = []
            documents = []
            metadatas = []
            seen_ids = set()  # Track IDs in this batch to avoid duplicates

            relative_path = str(filepath.relative_to(folder))

            for sentence in sentences:
                total_sentences += 1
                sid = make_sentence_id(relative_path, sentence)

                # Skip if already in this batch
                if sid in seen_ids:
                    skipped_duplicates += 1
                    continue
                seen_ids.add(sid)

                # Check if already indexed (for incremental updates)
                try:
                    existing = collection.get(ids=[sid])
                    if existing and existing['ids']:
                        skipped_duplicates += 1
                        continue
                except Exception:
                    pass

                ids.append(sid)
                documents.append(sentence)
                metadatas.append({
                    "source_file": relative_path,
                    "char_length": len(sentence),
                    "word_count": len(sentence.split()),
                })
                new_sentences += 1

            # Add batch to ChromaDB
            if ids:
                # ChromaDB handles embedding automatically via the embedding function
                # Batch in groups of 500 to avoid memory issues
                batch_size = 500
                for i in range(0, len(ids), batch_size):
                    collection.add(
                        ids=ids[i:i + batch_size],
                        documents=documents[i:i + batch_size],
                        metadatas=metadatas[i:i + batch_size],
                    )

            file_stats.append((relative_path, len(sentences), len(ids)))
            progress.advance(task)

    # --- Print Summary ---
    console.print()

    summary = Table(title="Indexing Summary")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")

    summary.add_row("Files processed", str(len(files)))
    summary.add_row("Total sentences found", str(total_sentences))
    summary.add_row("New sentences indexed", str(new_sentences))
    summary.add_row("Duplicates skipped", str(skipped_duplicates))
    summary.add_row("Total in database", str(collection.count()))

    console.print(summary)

    # Show per-file breakdown if not too many files
    if len(file_stats) <= 30:
        console.print()
        file_table = Table(title="Per-File Breakdown")
        file_table.add_column("File", style="cyan")
        file_table.add_column("Sentences", justify="right")
        file_table.add_column("New", justify="right", style="green")

        for fname, total, new in sorted(file_stats, key=lambda x: x[1], reverse=True):
            file_table.add_row(fname, str(total), str(new))

        console.print(file_table)

    console.print(f"\n[bold green]Done![/bold green] Database saved to {config.CHROMA_DB_PATH}/")
    console.print("Next: run [bold]python query_writing.py \"your sentence here\"[/bold] to test retrieval.\n")


def show_stats():
    """Show statistics about the current index."""
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

    try:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )
        collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    except Exception:
        console.print("[red]No index found. Run indexing first.[/red]")
        return

    count = collection.count()
    if count == 0:
        console.print("[yellow]Index is empty.[/yellow]")
        return

    # Get a sample to show source files
    sample = collection.get(limit=min(count, 1000), include=["metadatas"])
    source_files = set()
    total_words = 0
    for meta in sample["metadatas"]:
        source_files.add(meta.get("source_file", "unknown"))
        total_words += meta.get("word_count", 0)

    table = Table(title="Index Statistics")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total sentences", str(count))
    table.add_row("Source files (sampled)", str(len(source_files)))
    table.add_row("Avg words/sentence", f"{total_words / len(sample['metadatas']):.1f}")
    table.add_row("Embedding model", config.EMBEDDING_MODEL)
    table.add_row("Database path", config.CHROMA_DB_PATH)

    console.print(table)

    # Show some source files
    console.print("\n[bold]Source files:[/bold]")
    for f in sorted(source_files)[:20]:
        console.print(f"  • {f}")
    if len(source_files) > 20:
        console.print(f"  ... and {len(source_files) - 20} more")


# ─── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Index your writing into a searchable vector database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python index_writing.py ~/Documents/my-vault
    python index_writing.py ~/Documents/my-vault --reindex
    python index_writing.py --stats
        """,
    )
    parser.add_argument(
        "folder",
        nargs="?",
        help="Path to folder containing your markdown files",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Clear existing index and rebuild from scratch",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show index statistics",
    )

    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if not args.folder:
        parser.print_help()
        console.print("\n[red]Error: Please provide a folder path.[/red]")
        sys.exit(1)

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        console.print(f"[red]Error: {folder} is not a directory.[/red]")
        sys.exit(1)

    console.print(f"[bold]Writing Copilot — Indexer[/bold]")
    console.print(f"Source: {folder}")
    console.print(f"Database: {config.CHROMA_DB_PATH}")

    start = time.time()
    index_folder(folder, reindex=args.reindex)
    elapsed = time.time() - start
    console.print(f"Completed in {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()
