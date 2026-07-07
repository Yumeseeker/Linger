"""
Synonym lookup using the configured LLM or Datamuse + personal vocabulary ranking.

Supports two backends (SYNONYM_BACKEND in config.py):
    "llm"      — context-aware synonyms via the configured LLM backend
                 (Gemini/Ollama/OpenAI-compatible); falls back to Datamuse
                 if the LLM is unavailable or returns nothing
    "datamuse" — free, no-auth dictionary API; fast but no context awareness

Pipeline:
    1. Query backend for synonyms
    2. Check which synonyms you've used before (vocab_index hashmap)
    3. Personal words → top of list. Others ranked by backend score.

API docs:
    - Datamuse: https://www.datamuse.com/api/
"""

import re

import httpx

import config

DATAMUSE_BASE = "https://api.datamuse.com/words"


# ─── Datamuse Queries ─────────────────────────────────────────────────

def _query_datamuse(params: dict, max_results: int = 30) -> list[dict]:
    """
    Query the Datamuse API.

    Returns list of {"word": str, "score": int, "tags": list[str]}
    """
    params["max"] = max_results
    try:
        resp = httpx.get(DATAMUSE_BASE, params=params, timeout=5.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def get_synonyms_datamuse(
    word: str,
    sentence: str = "",
    max_results: int = 20,
) -> list[dict]:
    """
    Get synonyms from Datamuse, combining multiple query strategies.

    Strategy:
        1. rel_syn — strict synonyms (best quality)
        2. ml — "means like" — broader semantic matches
        3. If sentence provided, use lc/rc (left/right context) for contextual fit

    Results are merged and deduplicated, with strict synonyms scored higher.
    """
    word_lower = word.lower().strip()
    seen = {word_lower}
    results = []

    # Special handling for speech verbs Datamuse doesn't handle well
    # (it treats "said" as adjective "aforesaid")
    speech_verb_map = {
        "said": ["speak", "tell"],
        "says": ["speak", "tell"],
        "saying": ["speak", "tell"],
    }
    
    query_words = speech_verb_map.get(word_lower, [word_lower])

    # 1. Strict synonyms
    syn_results = _query_datamuse({"rel_syn": word_lower}, max_results=max_results)
    for item in syn_results:
        w = item["word"].lower()
        if w not in seen:
            seen.add(w)
            results.append({
                "word": item["word"],
                "score": item.get("score", 0) + 10000,  # Boost strict synonyms
                "source": "datamuse:synonym",
                "tags": item.get("tags", []),
            })

    # 2. "Means like" — broader matches
    ml_results = _query_datamuse({"ml": word_lower}, max_results=max_results)
    for item in ml_results:
        w = item["word"].lower()
        if w not in seen:
            seen.add(w)
            results.append({
                "word": item["word"],
                "score": item.get("score", 0),
                "source": "datamuse:means_like",
                "tags": item.get("tags", []),
            })

    # 3. Context-aware: if we have a sentence, use left-context hint
    #    e.g., for "eating" in "After eating dinner", use lc=After
    if sentence:
        words_in_sentence = sentence.split()
        try:
            idx = next(
                i for i, w in enumerate(words_in_sentence)
                if w.lower().strip(".,!?;:") == word_lower
            )
            # Left context: word before
            if idx > 0:
                left_context = words_in_sentence[idx - 1].lower().strip(".,!?;:")
                ctx_results = _query_datamuse(
                    {"ml": word_lower, "lc": left_context},
                    max_results=10,
                )
                for item in ctx_results:
                    w = item["word"].lower()
                    if w not in seen:
                        seen.add(w)
                        results.append({
                            "word": item["word"],
                            "score": item.get("score", 0) + 5000,  # Boost contextual
                            "source": "datamuse:contextual",
                            "tags": item.get("tags", []),
                        })
        except StopIteration:
            pass

    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:max_results]


# ─── LLM Queries ──────────────────────────────────────────────────────

async def get_synonyms_llm(
    word: str,
    sentence: str = "",
    max_results: int = 20,
) -> list[dict]:
    """
    Get context-aware synonyms from the configured LLM backend.

    Returns [] on any failure so the caller can fall back to Datamuse.
    """
    from llm_client import LLMClient

    context_hint = f' as used in this sentence: "{sentence}"' if sentence else ""
    prompt = f"""Give exactly {max_results} synonyms for the word "{word}"{context_hint}.
Each synonym must work as a drop-in replacement with the same grammatical form
(same tense, number, and capitalization).
Output ONLY a comma-separated list of words, nothing else.
Example: large, huge, enormous, vast, gigantic"""

    try:
        text = await LLMClient().generate(
            prompt,
            system="You are a thesaurus. Provide only synonyms, nothing else.",
        )
    except Exception as e:
        print(f"LLM synonym lookup failed, falling back to Datamuse: {e}")
        return []

    # Split on commas/newlines, strip any stray numbering or quotes
    candidates = re.split(r"[,\n]", text)
    results = []
    seen = {word.lower()}
    for i, raw in enumerate(candidates):
        syn = re.sub(r"^\s*\d+[\.\)]\s*", "", raw).strip().strip("\"'")
        if not syn or syn.lower() in seen:
            continue
        seen.add(syn.lower())
        results.append({
            "word": syn,
            "score": 10000 - i * 100,  # Earlier suggestions rank higher
            "source": "llm",
        })
    return results


# ─── Personal Vocabulary Ranking ──────────────────────────────────────

def rank_by_personal_usage(
    synonyms: list[dict],
) -> list[dict]:
    """
    Re-rank synonyms: words you've used before go to the top.

    Uses the vocab_index hashmap — a simple dict lookup, no database needed.
    """
    from vocab_index import word_count

    personal = []
    general = []

    for syn in synonyms:
        usage = word_count(syn["word"])
        syn["personal_usage"] = usage

        if usage > 0:
            syn["source"] = f"{syn['source']} + your writing ({usage}x)"
            personal.append(syn)
        else:
            general.append(syn)

    # Sort personal by usage count (most used first)
    personal.sort(key=lambda x: x["personal_usage"], reverse=True)

    return personal + general


# ─── Main Functions ───────────────────────────────────────────────────

async def suggest_synonyms_detailed(
    word: str,
    sentence: str = "",
    max_results: int = None,
) -> list[dict]:
    """
    Get synonyms with full metadata (score, usage count, source).

    Uses config.SYNONYM_BACKEND: "llm" tries the configured LLM first and
    falls back to Datamuse, "datamuse" goes straight to Datamuse.
    Personal vocabulary always ranks first.
    """
    max_results = max_results or config.NUM_SUGGESTIONS

    synonyms = []
    if getattr(config, "SYNONYM_BACKEND", "datamuse") == "llm":
        synonyms = await get_synonyms_llm(word, sentence, max_results=max_results * 2)

    if not synonyms:
        synonyms = get_synonyms_datamuse(word, sentence, max_results=max_results * 2)

    if not synonyms:
        return []

    synonyms = rank_by_personal_usage(synonyms)

    return synonyms[:max_results]


async def suggest_synonyms(
    word: str,
    sentence: str = "",
    max_results: int = None,
) -> list[str]:
    """Get synonyms for a word. Personal vocabulary first, then backend ranked."""
    detailed = await suggest_synonyms_detailed(word, sentence, max_results)
    return [s["word"] for s in detailed]
