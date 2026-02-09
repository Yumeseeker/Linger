"""
Personal vocabulary index — a simple word frequency counter.

Replaces ChromaDB for the synonym ranking feature. Instead of embedding
sentences into vectors and doing similarity search, we just count how
many times you've used each word across your writing.

This is:
    - Faster to build (seconds, not minutes)
    - Faster to query (dict lookup, not vector search)
    - More accurate for "have I used this word before?" questions
    - Tiny on disk (~1MB for 100k+ unique words)

The index is a JSON file: {"word": count, "another": count, ...}

Usage:
    python vocab_index.py /path/to/your/markdown/folder        # Build index
    python vocab_index.py --stats                               # Show stats
    python vocab_index.py --lookup "remarked"                   # Check a word
    python vocab_index.py --top 50                              # Most used words
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

import config

console = Console()

INDEX_PATH = Path(config.CHROMA_DB_PATH).parent / "vocab_index.json"


# ─── Markdown Stripping (reused from index_writing.py) ────────────────

def strip_markdown(text: str) -> str:
    """Remove markdown formatting to get clean prose."""
    if config.STRIP_FRONTMATTER:
        text = re.sub(r'^---\s*\n.*?\n---\s*\n', '', text, flags=re.DOTALL)
    if config.STRIP_CODE_BLOCKS:
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)
    if config.STRIP_IMAGES:
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    if config.STRIP_LINKS:
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    if config.STRIP_HTML:
        text = re.sub(r'<[^>]+>', '', text)
    if config.STRIP_TAGS:
        text = re.sub(r'(?<!\w)#\w+', '', text)
    # Remove markdown emphasis
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,2}(.+?)_{1,2}', r'\1', text)
    text = re.sub(r'~~(.+?)~~', r'\1', text)
    # Remove headings, blockquotes, list markers, rules
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[[ x]\]\s*', '', text)
    return text.strip()


# ─── Tokenization ─────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Simple word tokenization. Returns lowercase words only.
    Strips punctuation, numbers, and short tokens.
    """
    # Split on non-alpha characters
    words = re.findall(r"[a-zA-Z']+", text.lower())
    # Filter: min 2 chars, not just apostrophes
    return [w for w in words if len(w) >= 2 and w.strip("'")]


# ─── Build Index ──────────────────────────────────────────────────────

def build_vocab_index(folder: Path) -> Counter:
    """
    Scan all markdown files and count word frequencies.

    Returns a Counter: {"word": count, ...}
    """
    word_counts = Counter()
    files = []
    for ext in config.SUPPORTED_EXTENSIONS:
        files.extend(folder.rglob(f"*{ext}"))

    for filepath in sorted(files):
        try:
            raw = filepath.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        clean = strip_markdown(raw)
        words = tokenize(clean)
        word_counts.update(words)

    return word_counts


def save_index(word_counts: Counter, path: Path = None):
    """Save word frequency index to JSON."""
    path = path or INDEX_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(word_counts.most_common()), f)


def load_index(path: Path = None) -> dict:
    """Load word frequency index from JSON."""
    path = path or INDEX_PATH
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ─── Query Functions (used by thesaurus.py) ───────────────────────────

_vocab_cache = None


def get_vocab() -> dict:
    """Load vocab index, cached in memory after first call."""
    global _vocab_cache
    if _vocab_cache is None:
        _vocab_cache = load_index()
    return _vocab_cache


def reload_vocab():
    """Force reload from disk (after rebuilding index)."""
    global _vocab_cache
    _vocab_cache = None
    return get_vocab()


def word_count(word: str) -> int:
    """How many times has this word appeared in the writer's corpus?"""
    return get_vocab().get(word.lower().strip(), 0)


def has_used(word: str) -> bool:
    """Has the writer ever used this word?"""
    return word_count(word) > 0


def rank_words(words: list[str]) -> list[tuple[str, int]]:
    """
    Given a list of words, return them sorted by personal usage.
    Words the writer has used come first (sorted by frequency),
    followed by words they haven't used.
    """
    vocab = get_vocab()
    personal = []
    other = []

    for w in words:
        count = vocab.get(w.lower().strip(), 0)
        if count > 0:
            personal.append((w, count))
        else:
            other.append((w, 0))

    personal.sort(key=lambda x: x[1], reverse=True)
    return personal + other


# ─── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build and query your personal vocabulary index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python vocab_index.py ~/Documents/MyVault         # Build index
    python vocab_index.py --stats                     # Show stats
    python vocab_index.py --lookup "remarked"         # Check a word
    python vocab_index.py --top 50                    # Most used words
        """,
    )
    parser.add_argument("folder", nargs="?", help="Path to markdown folder")
    parser.add_argument("--stats", action="store_true", help="Show index stats")
    parser.add_argument("--lookup", "-l", help="Look up a word's frequency")
    parser.add_argument("--top", type=int, help="Show top N most frequent words")
    parser.add_argument("--output", "-o", help="Custom output path for index")

    args = parser.parse_args()
    output_path = Path(args.output) if args.output else INDEX_PATH

    if args.lookup:
        vocab = load_index(output_path)
        word = args.lookup.lower().strip()
        count = vocab.get(word, 0)
        if count > 0:
            console.print(f"[green]✓[/green] You've used [bold]\"{word}\"[/bold] {count} time(s)")
        else:
            console.print(f"[dim]✗[/dim] [bold]\"{word}\"[/bold] not found in your writing")
        return

    if args.stats:
        vocab = load_index(output_path)
        if not vocab:
            console.print("[yellow]No index found. Build one first.[/yellow]")
            return
        table = Table(title="Vocabulary Index Stats")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Unique words", f"{len(vocab):,}")
        table.add_row("Total word count", f"{sum(vocab.values()):,}")
        table.add_row("Index file", str(output_path))
        table.add_row("File size", f"{output_path.stat().st_size / 1024:.1f} KB")
        console.print(table)
        return

    if args.top:
        vocab = load_index(output_path)
        if not vocab:
            console.print("[yellow]No index found.[/yellow]")
            return
        table = Table(title=f"Top {args.top} Words")
        table.add_column("#", style="dim", width=4)
        table.add_column("Word", style="bold")
        table.add_column("Count", justify="right")
        sorted_words = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        for i, (word, count) in enumerate(sorted_words[:args.top], 1):
            table.add_row(str(i), word, f"{count:,}")
        console.print(table)
        return

    if not args.folder:
        parser.print_help()
        return

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        console.print(f"[red]{folder} is not a directory[/red]")
        sys.exit(1)

    console.print(f"[bold]Building vocabulary index...[/bold]")
    console.print(f"Source: {folder}")

    word_counts = build_vocab_index(folder)

    save_index(word_counts, output_path)

    console.print(f"[green]✓[/green] Indexed [bold]{len(word_counts):,}[/bold] unique words "
                  f"({sum(word_counts.values()):,} total)")
    console.print(f"[green]✓[/green] Saved to {output_path}")

    # Show a preview
    console.print(f"\nTop 20 words:")
    for word, count in word_counts.most_common(20):
        bar = "█" * min(count // 5, 40)
        console.print(f"  {word:>15} {count:>5}  [cyan]{bar}[/cyan]")


if __name__ == "__main__":
    main()
