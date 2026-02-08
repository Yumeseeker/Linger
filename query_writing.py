#!/usr/bin/env python3
"""
Query your indexed writing to find similar sentences.

Usage:
    python query_writing.py "she said the results were unclear"
    python query_writing.py "determination" --top 20
    python query_writing.py --interactive
    python query_writing.py --word "said"

This is your testing tool. Use it to verify that retrieval is working
before building the suggestion server or Obsidian plugin.
"""

import argparse
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import config

console = Console()


def get_collection():
    """Connect to the existing ChromaDB collection."""
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )

    try:
        collection = client.get_collection(
            name=config.COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    except Exception:
        console.print("[red]No index found. Run index_writing.py first.[/red]")
        sys.exit(1)

    if collection.count() == 0:
        console.print("[red]Index is empty. Run index_writing.py first.[/red]")
        sys.exit(1)

    return collection


def query(collection, text: str, top_k: int = None):
    """
    Query the collection and return similar sentences.

    ChromaDB returns results sorted by distance (lower = more similar).
    For cosine distance: 0.0 = identical, 2.0 = completely opposite.
    We convert to a similarity score (1 - distance/2) for readability.
    """
    top_k = top_k or config.DEFAULT_TOP_K

    results = collection.query(
        query_texts=[text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    entries = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # Filter by max distance
        if dist > config.MAX_DISTANCE:
            continue

        # Convert cosine distance to similarity percentage
        similarity = max(0, (1 - dist / 2)) * 100

        entries.append({
            "text": doc,
            "source": meta.get("source_file", "unknown"),
            "distance": dist,
            "similarity": similarity,
        })

    return entries


def display_results(text: str, results: list[dict]):
    """Pretty-print query results."""
    console.print(Panel(f"[bold]{text}[/bold]", title="Query", border_style="blue"))

    if not results:
        console.print("[yellow]No similar sentences found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("Similarity", justify="right", width=10)
    table.add_column("Sentence", style="white", ratio=3)
    table.add_column("Source", style="cyan", ratio=1)

    for i, r in enumerate(results, 1):
        # Color code similarity
        sim = r["similarity"]
        if sim >= 80:
            sim_str = f"[bold green]{sim:.1f}%[/bold green]"
        elif sim >= 60:
            sim_str = f"[yellow]{sim:.1f}%[/yellow]"
        else:
            sim_str = f"[dim]{sim:.1f}%[/dim]"

        table.add_row(str(i), sim_str, r["text"], r["source"])

    console.print(table)
    console.print()


def word_search(collection, word: str, top_k: int = None):
    """
    Search specifically for sentences containing or related to a word.

    This is more targeted than full sentence search — it finds sentences
    where you used a specific word or its synonyms, which is exactly
    what you need for the tip-of-the-tongue feature.
    """
    top_k = top_k or config.DEFAULT_TOP_K

    # Strategy: query with just the word to find semantically similar usage,
    # then also do a metadata/document search for literal occurrences
    results = query(collection, word, top_k=top_k * 2)

    # Separate into: contains the word literally vs. semantically similar
    literal_matches = []
    semantic_matches = []

    for r in results:
        if word.lower() in r["text"].lower():
            literal_matches.append(r)
        else:
            semantic_matches.append(r)

    console.print(Panel(f"[bold]Word: {word}[/bold]", title="Word Search", border_style="green"))

    if literal_matches:
        console.print(f"\n[bold green]Sentences where you used \"{word}\":[/bold green]")
        table = Table(show_header=False)
        table.add_column("Index", style="dim", width=3)
        table.add_column("Similarity")
        table.add_column("Sentence")
        table.add_column("Source", style="cyan")

        for i, r in enumerate(literal_matches, 1):
            sim = r["similarity"]
            if sim >= 80:
                sim_str = f"[bold green]{sim:.1f}%[/bold green]"
            elif sim >= 60:
                sim_str = f"[yellow]{sim:.1f}%[/yellow]"
            else:
                sim_str = f"[dim]{sim:.1f}%[/dim]"
            table.add_row(str(i), sim_str, r["text"], r["source"])

        console.print(table)

    if semantic_matches:
        console.print(
            f"\n[bold cyan]Semantically similar (you might mean one of these):[/bold cyan]"
        )
        table = Table(show_header=False)
        table.add_column("Index", style="dim", width=3)
        table.add_column("Similarity")
        table.add_column("Sentence")
        table.add_column("Source", style="cyan")

        for i, r in enumerate(semantic_matches, 1):
            sim = r["similarity"]
            if sim >= 80:
                sim_str = f"[bold green]{sim:.1f}%[/bold green]"
            elif sim >= 60:
                sim_str = f"[yellow]{sim:.1f}%[/yellow]"
            else:
                sim_str = f"[dim]{sim:.1f}%[/dim]"
            table.add_row(str(i), sim_str, r["text"], r["source"])

        console.print(table)

    console.print()


def interactive_mode(collection):
    """Interactive query mode."""
    console.print("[bold cyan]Interactive Query Mode[/bold cyan]")
    console.print("Type a sentence or word, press Enter. Ctrl+C to exit.\n")

    while True:
        try:
            query_text = input("[cyan]>[/cyan] ").strip()
        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query_text:
            continue

        results = query(collection, query_text)
        display_results(query_text, results)


# ─── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Query your indexed writing for similar sentences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python query_writing.py "she said the results were unclear"
    python query_writing.py "determination" --top 20
    python query_writing.py --word "said"
    python query_writing.py --interactive
        """,
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Sentence or words to search for",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=config.DEFAULT_TOP_K,
        help=f"Number of results to return (default: {config.DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--word",
        help="Search for sentences containing or related to a specific word",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive query mode",
    )

    args = parser.parse_args()

    collection = get_collection()

    if args.interactive:
        interactive_mode(collection)
    elif args.word:
        word_search(collection, args.word, top_k=args.top)
    elif args.query:
        results = query(collection, args.query, top_k=args.top)
        display_results(args.query, results)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
