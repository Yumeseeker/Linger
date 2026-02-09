#!/usr/bin/env python3
"""
Test the suggestion server from the command line.

Usage:
    python test_server.py                           # Interactive mode
    python test_server.py --word "said" --sentence "She said the results were unclear."
    python test_server.py --phrase "idea generating" --sentence "Why am I so idea generating these days."
    python test_server.py --sentence "The findings were inconclusive."
    python test_server.py --complete "I think the problem is"
    python test_server.py --health

Start the server first:  python server.py
"""

import argparse
import json
import sys

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config

console = Console()
BASE_URL = f"http://{config.SERVER_HOST}:{config.SERVER_PORT}"


def call_endpoint(endpoint: str, payload: dict) -> dict:
    """Make a POST request to the suggestion server."""
    try:
        resp = httpx.post(
            f"{BASE_URL}{endpoint}",
            json=payload,
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        console.print("[red]Cannot connect to server. Is it running?[/red]")
        console.print(f"[dim]Start it with: python server.py[/dim]")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Server error: {e.response.status_code}[/red]")
        console.print(e.response.text)
        sys.exit(1)


def display_result(data: dict):
    """Pretty-print a suggestion response."""
    req_type = data.get("request_type", "unknown")
    latency = data.get("latency_ms", 0)

    # Show raw response first
    raw = data.get("raw_response", "")
    if raw:
        console.print(f"\n[dim]Raw LLM output:[/dim]")
        console.print(f"[dim]{raw}[/dim]\n")

    # Show suggestions
    suggestions = data.get("suggestions", [])
    if suggestions:
        table = Table(
            title=f"Suggestions ({req_type})",
            caption=f"Generated in {latency}ms",
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Suggestion", style="bold white")

        for i, s in enumerate(suggestions, 1):
            style = "bold green" if i == 1 else "white"
            table.add_row(str(i), f"[{style}]{s}[/{style}]")

        console.print(table)
    else:
        console.print("[yellow]No suggestions generated.[/yellow]")

    # Show retrieved examples
    examples = data.get("retrieved_examples", [])
    if examples:
        console.print(f"\n[dim]Based on {len(examples)} examples from your writing:[/dim]")
        for ex in examples[:3]:
            console.print(f"  [dim]→ {ex[:100]}{'...' if len(ex) > 100 else ''}[/dim]")

    console.print()


def check_health():
    """Check server and backend health."""
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=10.0)
        data = resp.json()

        table = Table(title="Health Check")
        table.add_column("Component", style="bold")
        table.add_column("Status")
        table.add_column("Details")

        status_color = "green" if data["status"] == "ok" else "yellow"
        table.add_row("Overall", f"[{status_color}]{data['status']}[/{status_color}]", "")

        db = data.get("database", {})
        db_color = "green" if db.get("status") == "ok" else "red"
        db_detail = f"{db.get('sentences_indexed', '?')} sentences" if db.get("status") == "ok" else db.get("message", "")
        table.add_row("Database", f"[{db_color}]{db.get('status', '?')}[/{db_color}]", db_detail)

        llm = data.get("llm", {})
        llm_color = "green" if llm.get("status") == "ok" else "red"
        table.add_row("LLM", f"[{llm_color}]{llm.get('status', '?')}[/{llm_color}]", llm.get("message", ""))

        console.print(table)

    except httpx.ConnectError:
        console.print("[red]Cannot connect to server.[/red]")
        console.print(f"Start it with: [bold]python server.py[/bold]")


def interactive_mode():
    """Interactive test loop."""
    console.print(Panel(
        "[bold]Writing Copilot — Interactive Test[/bold]\n\n"
        "Commands:\n"
        "  [bold]w:[/bold] word search     → [dim]w:said | She said the results.[/dim]\n"
        "  [bold]p:[/bold] phrase rephrase  → [dim]p:idea generating | Why am I so idea generating.[/dim]\n"
        "  [bold]s:[/bold] sentence rewrite → [dim]s:The findings were inconclusive.[/dim]\n"
        "  [bold]c:[/bold] complete         → [dim]c:I think the problem is[/dim]\n"
        "  [bold]h[/bold]  health check\n"
        "  [bold]q[/bold]  quit\n\n"
        "For word and phrase modes, separate the target from the sentence with |",
        border_style="blue",
    ))

    while True:
        try:
            raw = input("\n→ ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not raw:
            continue
        if raw.lower() in ("q", "quit", "exit"):
            break
        if raw.lower() in ("h", "health"):
            check_health()
            continue

        if raw.startswith("w:"):
            parts = raw[2:].split("|", 1)
            word = parts[0].strip()
            sentence = parts[1].strip() if len(parts) > 1 else ""
            if not sentence:
                console.print("[yellow]Usage: w:word | Full sentence containing the word[/yellow]")
                continue
            console.print(f'[dim]Finding alternatives for "{word}" in: {sentence}[/dim]')
            data = call_endpoint("/suggest/word", {"word": word, "sentence": sentence})
            display_result(data)

        elif raw.startswith("p:"):
            parts = raw[2:].split("|", 1)
            phrase = parts[0].strip()
            sentence = parts[1].strip() if len(parts) > 1 else ""
            if not sentence:
                console.print("[yellow]Usage: p:phrase | Full sentence containing the phrase[/yellow]")
                continue
            console.print(f'[dim]Rephrasing "{phrase}" in: {sentence}[/dim]')
            data = call_endpoint("/suggest/phrase", {"phrase": phrase, "sentence": sentence})
            display_result(data)

        elif raw.startswith("s:"):
            sentence = raw[2:].strip()
            console.print(f'[dim]Rewriting: {sentence}[/dim]')
            data = call_endpoint("/suggest/sentence", {"sentence": sentence})
            display_result(data)

        elif raw.startswith("c:"):
            partial = raw[2:].strip()
            console.print(f'[dim]Completing: {partial}...[/dim]')
            data = call_endpoint("/suggest/complete", {"sentence_so_far": partial})
            display_result(data)

        else:
            console.print(f'[dim]Rewriting: {raw}[/dim]')
            data = call_endpoint("/suggest/sentence", {"sentence": raw})
            display_result(data)


def main():
    parser = argparse.ArgumentParser(
        description="Test the Writing Copilot suggestion server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_server.py --word "said" --sentence "She said the results were unclear."
    python test_server.py --phrase "idea generating" --sentence "Why am I so idea generating."
    python test_server.py --sentence "The findings were inconclusive."
    python test_server.py --complete "I think the problem is"
    python test_server.py   # interactive mode
        """,
    )
    parser.add_argument("--word", "-w", help="Word to find alternatives for")
    parser.add_argument("--phrase", "-p", help="Phrase to rephrase")
    parser.add_argument("--sentence", "-s", help="Sentence (context for word/phrase, or sentence to rewrite)")
    parser.add_argument("--complete", "-c", help="Partial sentence to complete")
    parser.add_argument("--paragraph", help="Additional paragraph context")
    parser.add_argument("--health", action="store_true", help="Check server health")

    args = parser.parse_args()

    if args.health:
        check_health()
        return

    if args.word:
        if not args.sentence:
            console.print("[red]--sentence is required with --word[/red]")
            sys.exit(1)
        data = call_endpoint("/suggest/word", {
            "word": args.word, "sentence": args.sentence, "paragraph": args.paragraph or "",
        })
        display_result(data)
    elif args.phrase:
        if not args.sentence:
            console.print("[red]--sentence is required with --phrase[/red]")
            sys.exit(1)
        data = call_endpoint("/suggest/phrase", {
            "phrase": args.phrase, "sentence": args.sentence, "paragraph": args.paragraph or "",
        })
        display_result(data)
    elif args.complete:
        data = call_endpoint("/suggest/complete", {
            "sentence_so_far": args.complete, "paragraph": args.paragraph or "",
        })
        display_result(data)
    elif args.sentence:
        data = call_endpoint("/suggest/sentence", {
            "sentence": args.sentence, "paragraph": args.paragraph or "",
        })
        display_result(data)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
