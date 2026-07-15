# Linger - Writing Copilot

Read this file fully before doing any work in this repo.

## What this project is

A personalized writing assistant that suggests words, phrases, sentence rewrites, and
sentence completions in the user's own voice. It does this with RAG over the user's
private writing corpus:

1. **Index**: markdown vault -> strip formatting -> spaCy sentence split -> embed with
   sentence-transformers (`all-MiniLM-L6-v2`) -> store in ChromaDB. A separate word
   frequency index (`vocab_index.json`) ranks synonyms by personal usage.
2. **Serve**: FastAPI server on `127.0.0.1:8111` with four endpoints:
   `POST /suggest/word`, `/suggest/phrase`, `/suggest/sentence`, `/suggest/complete`,
   plus `GET /health`. Each retrieves stylistically similar sentences from ChromaDB and
   feeds them to an LLM as in-context style examples.
3. **Integrate**: an Obsidian plugin calls these endpoints from the editor.

Success criteria: suggestions sound like the user (not generic AI), arrive fast enough
for in-editor use (target under ~2s), and the whole chain works end to end from
Obsidian keystroke to inserted suggestion.

## Repo layout and related paths

This repo is `/Users/brianzhang/Documents/Linger/files/`. Key files:

- `README.md` - how to run it; `HOW_IT_WORKS.md` - architecture deep dive and
  resume material (keep both in sync with code changes)
- `config.py` - all settings; loads secrets from `.env` (gitignored, see `.env.example`)
- `index_writing.py` - builds the ChromaDB index; `--reindex` to rebuild, `--stats`
- `vocab_index.py` - builds/queries the word frequency index
- `server.py` - the FastAPI suggestion server
- `llm_client.py` - LLM abstraction (backends: gemini, ollama, openai_compatible)
- `thesaurus.py` - word synonym lookup (Datamuse + personal-usage ranking)
- `prompts.py` - prompt templates and `parse_suggestions()` output parser
- `query_writing.py`, `test_server.py` - CLI tools for testing retrieval / endpoints
- `reranker.py` - cross-encoder reranker (retrieval stage 2)
- `setup.sh` - one-shot environment bootstrap script

Outside the repo:

- `../Test Vault/` - sample Obsidian vault (gitignored). The Obsidian plugin lives at
  `../Test Vault/.obsidian/plugins/hello-world/` (TypeScript, esbuild). It is
  implemented (4 commands calling the server, suggestion modal, serverUrl setting);
  the plugin id stays `hello-world` because it must match the folder name.
- `../files (1)/` and `../files (2)/` - stale copies of this repo. Ignore them.

## Environment facts (verified 2026-07-07)

- Python venv at `./venv`, Python 3.9.6. Always use `./venv/bin/python`. Do not use
  3.10+ syntax like `X | Y` unions.
- ChromaDB index exists: collection `my_writing`, 27,643 sentences, cosine space.
  Changing `EMBEDDING_MODEL` requires `python index_writing.py <vault> --reindex`.
- Ollama is running locally with `deepseek-r1:7b` and `phi3:mini` pulled.
  IMPORTANT: deepseek-r1 is a reasoning model that emits `<think>...</think>` blocks;
  this broke suggestion parsing and was likely the main "it doesn't work" cause.
  Prefer `phi3:mini` locally, and strip think-blocks defensively when parsing.
- `GEMINI_API_KEY` is set in `./.env` (gitignored). The default backend is Gemini;
  use `LLM_BACKEND=ollama` for offline/local runs.

## Security

- An OpenAI API key was previously hardcoded in `config.py`. It was never committed
  (verified against full git history) and has been removed from the working tree, but
  Brian should still rotate it.
- Never write secrets into tracked files. Secrets go in `.env` only; `config.py`
  reads them via `os.environ`.

## Current state (as of 2026-07-07: implemented and E2E-verified)

The target architecture below is fully implemented, as uncommitted working-tree
changes (do not commit unless Brian asks):

- `llm_client.py`: `gemini` backend (REST via httpx, `x-goog-api-key` header,
  `systemInstruction`, `thinkingBudget=0` for 2.5-flash models), think-block
  stripping applied centrally in `generate()`, health checks for all backends.
- `reranker.py`: lazily loaded CrossEncoder; wired into `server.retrieve_similar()`
  (30 candidates -> top 8) and warmed in the FastAPI lifespan.
- `prompts.py`: `parse_suggestions()` handles think-blocks, bullets, bold, quotes,
  dedup, and a `reject=` arg for input echoes; `/suggest/complete` trims prefix echoes.
- `thesaurus.py`: `SYNONYM_BACKEND="llm"` (async, uses `LLMClient`) with automatic
  Datamuse fallback; `suggest_synonyms*` functions are now async.
- Obsidian plugin implemented (4 commands + suggestion modal + serverUrl setting);
  builds cleanly with `npm run build`.
- `README.md`, `requirements.txt`, `.env.example`, `.gitignore` updated.

Verification status:
- VERIFIED live on Gemini (2026-07-07, `gemini-2.5-flash`): `/health` ok
  (27,643 sentences, reranker loaded), all four `/suggest/*` endpoints return
  clean personalized suggestions; latency 0.7-1.5s.
- Also verified on the Ollama fallback (phi3:mini): same endpoints, 4.4-5.8s.
- NOT yet verified: the plugin inside a running Obsidian (compile-verified only).

## Target architecture

Two-stage retrieval, then generation:

```
query text -> ChromaDB (bi-encoder, top 30 candidates, cosine + MAX_DISTANCE filter,
              dedup) -> cross-encoder reranker (ms-marco-MiniLM-L-6-v2, keep top 8)
           -> prompt with style examples -> Gemini 2.5 Flash (thinking disabled)
           -> parse numbered list -> suggestions
```

Rationale already decided, do not relitigate: a cross-encoder reranker is the right
quality upgrade at this corpus size; HyDE/query-expansion add latency for little gain
on style-example retrieval; Gemini Flash over local Ollama for quality and speed, with
Ollama kept as the free/private fallback.

## Remaining tasks

1. **Plugin live verification**: open the Test Vault in Obsidian, enable the plugin
   (id `hello-world`, shows as "Linger Writing Copilot"), and exercise all four
   commands against the running server.

## Future work (after the above)

- Incremental reindexing: detect changed/deleted vault files and update ChromaDB
  instead of only appending (current code skips existing IDs but never deletes).
- A small retrieval evaluation harness (a handful of query -> expected-sentence pairs)
  so reranker/embedding changes can be compared objectively.
- Streaming completions to the plugin for perceived latency.
- Latency instrumentation per stage (retrieve / rerank / LLM) in the response payload.

## Working rules for agents in this repo

- Brian granted standing permission (2026-07-07) to commit when a meaningful unit
  of work completes. Never push unless he asks. Never add an agent co-author line
  to commit messages.
- Verify E2E before claiming done; be picky about output quality (a suggestion list
  containing think-text, echoes of the input, or numbering artifacts is a failure).
- This is a personal, single-user, localhost project: prefer simplicity; no auth
  layers, job queues, or microservices.
- The user-facing latency budget matters more than marginal quality: measure before
  adding pipeline stages.
