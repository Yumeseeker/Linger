# Linger - Writing Copilot

A personalized writing assistant that suggests words, phrases, sentence rewrites,
and sentence completions in your own voice, powered by RAG over your private
writing corpus.

Want to understand the internals? Read [HOW_IT_WORKS.md](HOW_IT_WORKS.md).

## Run it (everything is already set up on this machine)

The index (27k+ sentences), the venv, and the `.env` with the Gemini key already
exist, so running it is two steps:

**1. Start the server**

```bash
cd ~/Documents/Linger/files
./venv/bin/python server.py
```

You should see:

```
✓ ChromaDB connected: 27643 sentences indexed
✓ Reranker loaded: cross-encoder/ms-marco-MiniLM-L-6-v2
✓ LLM connected: gemini / gemini-2.5-flash
  Server running at http://127.0.0.1:8111
```

Sanity check: `curl http://127.0.0.1:8111/health` or open
http://127.0.0.1:8111/docs for interactive API docs.

**2. Use it from Obsidian**

Open the Test Vault in Obsidian, enable **Linger Writing Copilot** under
Settings -> Community plugins, then in any note open the command palette
(Cmd+P) and run:

- **Suggest synonyms for word** - word at cursor (or selection)
- **Rephrase selection** - select a phrase first
- **Rephrase current sentence** - cursor anywhere in the sentence
- **Complete sentence** - cursor at the end of your partial sentence

Pick a suggestion from the modal and it replaces the text in place. The plugin's
settings tab has the server URL if you ever change the port.

**No Obsidian handy?** Test from the terminal instead:

```bash
./venv/bin/python test_server.py          # interactive CLI

# or raw curl:
curl -X POST http://127.0.0.1:8111/suggest/sentence \
  -H "Content-Type: application/json" \
  -d '{"sentence": "The meeting went longer than I expected.", "paragraph": ""}'
```

## Run it offline (no API, fully local)

```bash
LLM_BACKEND=ollama ./venv/bin/python server.py
```

Requires Ollama running with `phi3:mini` pulled (`ollama pull phi3:mini`).
Quality and latency are worse than Gemini (~5s vs ~1s) but nothing leaves your
machine. Avoid reasoning models like `deepseek-r1` - they burn seconds emitting
`<think>` text before answering.

## Switching LLM providers

Gemini is the default, not a requirement. `LLM_BACKEND` picks one of three
backends in `llm_client.py`:

| `LLM_BACKEND` | Covers | Setup |
|---|---|---|
| `gemini` (default) | Google Gemini API | `GEMINI_API_KEY` in `.env`; pick a model with `GEMINI_MODEL` (default `gemini-2.5-flash`) |
| `ollama` | Any local Ollama model | Ollama running and a model pulled; no API key |
| `openai_compatible` | OpenAI, DeepSeek, Together, local vLLM - anything that speaks the OpenAI Chat Completions API | `OPENAI_API_KEY` in `.env`; point `OPENAI_API_BASE` and `OPENAI_MODEL` in `config.py` at your provider |

Switch for a single run with an environment variable:

```bash
LLM_BACKEND=openai_compatible ./venv/bin/python server.py
```

or persistently by setting `LLM_BACKEND=...` in `.env`.

Two caveats:

- The Ollama and OpenAI-compatible model/URL settings currently live in
  `config.py` (only the Gemini model and the keys are read from `.env`), so
  pointing at a different provider means editing `config.py` once.
- Providers whose native API is not OpenAI-compatible (e.g. Anthropic's
  Messages API) are not supported yet; adding one means a new `_generate_*`
  method in `llm_client.py`.

## Re-index when you've written new things

The indexes don't update themselves. After adding writing to your vault:

```bash
./venv/bin/python vocab_index.py "/path/to/your/vault"     # word frequencies
./venv/bin/python index_writing.py "/path/to/your/vault"   # sentence embeddings
```

Indexing is incremental (already-indexed sentences are skipped). Use
`--reindex` to wipe and rebuild, `--stats` to inspect what's in there.

## First-time setup (new machine)

```bash
cd files
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

cp .env.example .env        # then put GEMINI_API_KEY=... in .env
                            # (get a key at https://aistudio.google.com/apikey),
                            # or configure another provider - see
                            # "Switching LLM providers" above

python vocab_index.py  ~/path/to/vault
python index_writing.py ~/path/to/vault
python server.py
```

For the plugin on a new vault: copy
`Test Vault/.obsidian/plugins/hello-world/` into the new vault's
`.obsidian/plugins/` and enable it. To modify the plugin, edit `src/` and run
`npm run build` inside the plugin folder.

## The API

| Endpoint | Body | Does |
|---|---|---|
| `POST /suggest/word` | `{word, sentence, paragraph}` | Context-aware synonyms, your vocabulary first |
| `POST /suggest/phrase` | `{phrase, sentence, paragraph}` | Rephrase a span within a sentence |
| `POST /suggest/sentence` | `{sentence, paragraph}` | Five rewrites of the sentence |
| `POST /suggest/complete` | `{sentence_so_far, paragraph}` | Copilot-style endings |
| `GET /health` | - | Database + LLM status |

All suggestion responses: `{suggestions, retrieved_examples, request_type, latency_ms}`.

## Configuration

Secrets go in `.env` (gitignored), everything else in `config.py`:

```python
LLM_BACKEND = "gemini"       # or "ollama", "openai_compatible"
SYNONYM_BACKEND = "llm"      # or "datamuse" (free, no context awareness)

USE_RERANKER = True          # two-stage retrieval on/off
RETRIEVAL_CANDIDATES = 30    # stage 1: bi-encoder candidates from ChromaDB
RETRIEVAL_TOP_K = 8          # stage 2: kept after cross-encoder rerank

NUM_SUGGESTIONS = 5
LLM_TEMPERATURE = 0.4        # lower = safer, higher = more creative
```

`LLM_BACKEND`, `GEMINI_API_KEY`, and `GEMINI_MODEL` can also be set per-run as
environment variables, which is how you flip providers without editing anything
(see "Switching LLM providers" above).

## Troubleshooting

**"ChromaDB error: No collection found"** - run `index_writing.py` first.

**LLM status `no_api_key` in /health** - `.env` is missing or has no
`GEMINI_API_KEY`. Copy `.env.example` and fill it in.

**Plugin says "cannot reach server"** - the server isn't running; start it with
`./venv/bin/python server.py`. If you changed the port, update the plugin's
server URL setting.

**Port 8111 already in use** - `lsof -i :8111` to find the old process, or run
with `--port 8112` (and update the plugin setting).

**Suggestions are slow locally** - you're on the Ollama fallback. Check
`/health` shows `gemini`, and that you didn't leave `LLM_BACKEND=ollama` set.

## Project structure

```
files/
├── README.md            # This file: how to run it
├── HOW_IT_WORKS.md      # Deep dive into the architecture + resume material
├── config.py            # All settings (no secrets)
├── .env                 # Secrets (gitignored); .env.example is the template
│
├── index_writing.py     # Vault -> ChromaDB sentence index
├── vocab_index.py       # Vault -> word frequency index
├── query_writing.py     # CLI: test retrieval directly
│
├── server.py            # FastAPI suggestion server
├── llm_client.py        # Gemini / Ollama / OpenAI-compatible clients
├── reranker.py          # Cross-encoder reranker (retrieval stage 2)
├── thesaurus.py         # Synonyms: LLM w/ Datamuse fallback + vocab ranking
├── prompts.py           # Prompt templates + LLM output parsing
├── test_server.py       # CLI: test the endpoints
│
├── chroma_db/           # Vector database (generated)
└── vocab_index.json     # Word frequencies (generated)
```
