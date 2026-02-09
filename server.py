#!/usr/bin/env python3
"""
Writing Copilot — Suggestion Server (Step 2)

A local FastAPI server that combines retrieval (ChromaDB) with generation (LLM)
to produce personalized writing suggestions.

Usage:
    python server.py                    # Start the server
    python server.py --check            # Check LLM and database health

The server exposes these endpoints:
    POST /suggest/word      — Synonym suggestions for a single word
    POST /suggest/phrase    — Rephrase a phrase within a sentence
    POST /suggest/sentence  — Alternative ways to write a full sentence
    POST /suggest/complete  — Complete a partial sentence (Copilot-style)
    GET  /health            — Check if everything is connected

The Obsidian plugin (Step 3) will call these endpoints.
You can also test them directly with the test_server.py CLI tool.

Architecture:
    Obsidian Plugin → HTTP POST → this server
                                     │
                            ┌────────┴────────┐
                            │                  │
                      ChromaDB             Ollama/LLM
                   (your writing)        (generates suggestions
                   (retrieves similar     informed by your style)
                    examples)
"""

import asyncio
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from llm_client import LLMClient
from prompts import (
    SYSTEM_PROMPT,
    build_word_prompt,
    build_phrase_prompt,
    build_sentence_prompt,
    build_continuation_prompt,
    parse_suggestions,
)


# ─── Database Connection ───────────────────────────────────────────────

_collection = None


def get_collection():
    """Lazily connect to ChromaDB collection."""
    global _collection
    if _collection is None:
        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )
        _collection = client.get_collection(
            name=config.COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    return _collection


def retrieve_similar(text: str, top_k: int = None) -> list[str]:
    """
    Retrieve the most similar sentences from your writing corpus.

    This is the "retrieval" in RAG — it finds examples from YOUR past writing
    that are contextually similar to what you're currently writing.
    The LLM then uses these examples to generate suggestions in your voice.
    """
    top_k = top_k or config.RETRIEVAL_TOP_K
    collection = get_collection()

    results = collection.query(
        query_texts=[text],
        n_results=top_k,
        include=["documents", "distances"],
    )

    # Filter by max distance and deduplicate
    seen = set()
    examples = []
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        if dist > config.MAX_DISTANCE:
            continue
        # Simple dedup — skip near-identical sentences
        normalized = doc.lower().strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        examples.append(doc)

    return examples


# ─── Request/Response Models ───────────────────────────────────────────

class WordRequest(BaseModel):
    """Request for word-level synonym suggestions."""
    word: str = Field(..., description="The word to find alternatives for")
    sentence: str = Field(..., description="The full sentence containing the word")
    paragraph: str = Field("", description="Surrounding paragraph for context")

    model_config = {"json_schema_extra": {
        "examples": [{
            "word": "said",
            "sentence": "She said the results were unclear.",
            "paragraph": "The researcher presented her findings. She said the results were unclear. The committee asked for more data.",
        }]
    }}


class PhraseRequest(BaseModel):
    """Request for phrase-level rephrase suggestions."""
    phrase: str = Field(..., description="The phrase to rephrase")
    sentence: str = Field(..., description="The full sentence containing the phrase")
    paragraph: str = Field("", description="Surrounding paragraph for context")


class SentenceRequest(BaseModel):
    """Request for full sentence rephrase suggestions."""
    sentence: str = Field(..., description="The sentence to rephrase")
    paragraph: str = Field("", description="Surrounding paragraph for context")


class CompletionRequest(BaseModel):
    """Request for sentence completion (Copilot-style)."""
    sentence_so_far: str = Field(..., description="The partial sentence to complete")
    paragraph: str = Field("", description="Surrounding paragraph for context")


class SuggestionResponse(BaseModel):
    """Response containing ranked suggestions."""
    suggestions: list[str] = Field(..., description="Ranked list of alternatives")
    retrieved_examples: list[str] = Field(
        default_factory=list,
        description="Examples from your writing that informed the suggestions",
    )
    request_type: str = Field(..., description="Type of suggestion (word/phrase/sentence/completion)")
    latency_ms: int = Field(..., description="Total request time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database: dict
    llm: dict


# ─── App Setup ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown logic."""
    # Warm up on startup
    try:
        collection = get_collection()
        count = collection.count()
        print(f"✓ ChromaDB connected: {count} sentences indexed")
    except Exception as e:
        print(f"⚠ ChromaDB not available: {e}")
        print("  Run index_writing.py first to create the index.")

    llm = LLMClient()
    health = await llm.health_check()
    if health["status"] == "ok":
        print(f"✓ LLM connected: {health['backend']} / {health.get('model', 'unknown')}")
    else:
        print(f"⚠ LLM: {health['message']}")

    print(f"\n  Server running at http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    print(f"  API docs at http://{config.SERVER_HOST}:{config.SERVER_PORT}/docs")
    print(f"  Test with: python test_server.py\n")

    yield  # Server runs
    print("Server shutting down.")


app = FastAPI(
    title="Writing Copilot",
    description="Personalized writing suggestions powered by your own writing style.",
    version="0.2.0",
    lifespan=lifespan,
)

# Allow CORS for the Obsidian plugin (and any other local client)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Obsidian uses app:// protocol
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_client = LLMClient()


# ─── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the database and LLM are connected and ready."""
    # Check database
    try:
        collection = get_collection()
        count = collection.count()
        db_status = {"status": "ok", "sentences_indexed": count}
    except Exception as e:
        db_status = {"status": "error", "message": str(e)}

    # Check LLM
    llm_status = await llm_client.health_check()

    overall = "ok" if db_status.get("status") == "ok" and llm_status.get("status") == "ok" else "degraded"

    return HealthResponse(status=overall, database=db_status, llm=llm_status)


@app.post("/suggest/word", response_model=SuggestionResponse)
async def suggest_word(req: WordRequest):
    """
    Get synonym suggestions for a single word.

    Uses Datamuse API for synonyms, ranked by your personal vocabulary.
    Words you've used before appear first. No LLM, no vector DB — just
    a dictionary API + a hashmap lookup.
    """
    start = time.time()

    from thesaurus import suggest_synonyms_detailed

    detailed = suggest_synonyms_detailed(
        word=req.word,
        sentence=req.sentence,
        max_results=config.NUM_SUGGESTIONS,
    )

    suggestions = [s["word"] for s in detailed]

    # Show which ones are from personal vocab
    examples = [
        f"{s['word']} ({s['personal_usage']}x in your writing)"
        for s in detailed
        if s.get("personal_usage", 0) > 0
    ]

    latency = int((time.time() - start) * 1000)

    return SuggestionResponse(
        suggestions=suggestions,
        retrieved_examples=examples,
        request_type="word",
        latency_ms=latency,
    )


@app.post("/suggest/phrase", response_model=SuggestionResponse)
async def suggest_phrase(req: PhraseRequest):
    """
    Get rephrase suggestions for a phrase within a sentence.

    Handles cases like "idea generating" → "inclined toward ideation".
    """
    start = time.time()

    query_text = req.sentence if req.sentence else req.phrase
    examples = retrieve_similar(query_text)

    prompt = build_phrase_prompt(
        phrase=req.phrase,
        sentence=req.sentence,
        paragraph=req.paragraph or req.sentence,
        retrieved_examples=examples,
    )

    try:
        raw_response = await llm_client.generate(prompt, system=SYSTEM_PROMPT)
        suggestions = parse_suggestions(raw_response)
    except Exception as e:
        suggestions = [f"[LLM error: {e}]"]

    latency = int((time.time() - start) * 1000)

    return SuggestionResponse(
        suggestions=suggestions,
        retrieved_examples=examples[:5],
        request_type="phrase",
        latency_ms=latency,
    )


@app.post("/suggest/sentence", response_model=SuggestionResponse)
async def suggest_sentence(req: SentenceRequest):
    """
    Get alternative ways to write a full sentence.

    The writer has a complete sentence but wants to see how else they
    might express the same thought, informed by their past writing style.
    """
    start = time.time()

    examples = retrieve_similar(req.sentence)

    prompt = build_sentence_prompt(
        sentence=req.sentence,
        paragraph=req.paragraph or req.sentence,
        retrieved_examples=examples,
    )

    try:
        raw_response = await llm_client.generate(prompt, system=SYSTEM_PROMPT)
        suggestions = parse_suggestions(raw_response)
    except Exception as e:
        suggestions = [f"[LLM error: {e}]"]

    latency = int((time.time() - start) * 1000)

    return SuggestionResponse(
        suggestions=suggestions,
        retrieved_examples=examples[:5],
        request_type="sentence",
        latency_ms=latency,
    )


@app.post("/suggest/complete", response_model=SuggestionResponse)
async def suggest_completion(req: CompletionRequest):
    """
    Complete a partial sentence (Copilot-style).

    Given the start of a sentence, predict how the writer would finish it,
    based on how they've finished similar sentences in the past.
    """
    start = time.time()

    examples = retrieve_similar(req.sentence_so_far)

    prompt = build_continuation_prompt(
        sentence_so_far=req.sentence_so_far,
        paragraph=req.paragraph or req.sentence_so_far,
        retrieved_examples=examples,
    )

    try:
        raw_response = await llm_client.generate(prompt, system=SYSTEM_PROMPT)
        suggestions = parse_suggestions(raw_response)
    except Exception as e:
        suggestions = [f"[LLM error: {e}]"]

    latency = int((time.time() - start) * 1000)

    return SuggestionResponse(
        suggestions=suggestions,
        retrieved_examples=examples[:5],
        request_type="completion",
        latency_ms=latency,
    )


# ─── Run Server ────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Writing Copilot suggestion server")
    parser.add_argument("--check", action="store_true", help="Check health and exit")
    parser.add_argument("--host", default=config.SERVER_HOST)
    parser.add_argument("--port", type=int, default=config.SERVER_PORT)
    args = parser.parse_args()

    if args.check:
        async def run_check():
            llm = LLMClient()
            health = await llm.health_check()
            print(f"LLM: {health['status']} — {health['message']}")
            try:
                collection = get_collection()
                print(f"Database: ok — {collection.count()} sentences indexed")
            except Exception as e:
                print(f"Database: error — {e}")
        asyncio.run(run_check())
        return

    import uvicorn
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
