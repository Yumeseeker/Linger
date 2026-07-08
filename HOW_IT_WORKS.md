# How Linger Works

A study guide for the architecture: what each piece does, why it exists, and how
to talk about it. Read [README.md](README.md) for how to run it.

## The one-paragraph version

Linger is a retrieval-augmented generation (RAG) system. Your writing vault is
indexed twice: every sentence becomes a vector in ChromaDB, and every word gets
a frequency count in a JSON hashmap. When you ask for a suggestion in Obsidian,
a FastAPI server retrieves the most stylistically relevant sentences from your
past writing using two-stage retrieval (fast bi-encoder search, then accurate
cross-encoder reranking) and feeds them to Gemini 2.5 Flash as in-context style
examples, so the model imitates *your* voice instead of generating generic AI
prose. Parsed, cleaned suggestions come back to the editor in about a second.

## Stage by stage

### 1. Indexing (offline, run once per batch of new writing)

Two indexes are built from the same vault, because two different questions need
two different data structures:

- **Semantic index** (`index_writing.py` -> ChromaDB). Markdown is stripped to
  clean prose (frontmatter, code blocks, links, list markers removed), split
  into sentences with spaCy (which handles abbreviations, quotes, and decimals
  that naive period-splitting breaks on), and each sentence is embedded into a
  384-dimensional vector by the `all-MiniLM-L6-v2` sentence-transformer running
  locally. Vectors are stored in ChromaDB with cosine similarity. Each sentence
  gets a deterministic ID (hash of file path + text), so re-indexing skips
  what's already there instead of duplicating.
- **Vocabulary index** (`vocab_index.py` -> `vocab_index.json`). A plain word
  frequency counter. "Have I used this word, and how often?" is a dictionary
  lookup, not a semantic search problem: a hashmap answers it in O(1) and more
  accurately than any embedding could.

### 2. Retrieval (per request): the two-stage core

When a phrase/sentence/completion request arrives, `retrieve_similar()` finds
past writing that resembles what you're writing now. Two stages:

- **Stage 1 - bi-encoder, fast but approximate.** ChromaDB embeds the query
  with the same MiniLM model and returns the 30 nearest sentences by cosine
  distance. Fast because all 27k sentences were embedded ahead of time. Lossy
  because query and sentence were encoded *separately*: the model never read
  them together, so subtle relevance is missed.
- **Stage 2 - cross-encoder reranker, slow but accurate.** `reranker.py` runs
  each (query, candidate) pair *jointly* through
  `cross-encoder/ms-marco-MiniLM-L-6-v2`, which reads both texts at once and
  scores true relevance. The best 8 survive.

Why both? Running the cross-encoder over all 27k sentences would take far too
long; running only the bi-encoder gives mediocre relevance. The funnel gives
you nearly cross-encoder quality at bi-encoder speed. This retrieve-then-rerank
pattern is the standard production RAG architecture.

Near-duplicates and anything beyond a cosine-distance cutoff are filtered along
the way.

### 3. Generation: in-context style transfer

The 8 surviving sentences are formatted into a prompt (`prompts.py`) that
essentially says: "here is the text to rework, here is its surrounding
sentence, and here are examples of this writer's past work - match this style
and output a numbered list of 5 alternatives."

The personalization is **in-context learning**: the model was never trained on
your writing; it imitates the examples it sees in the prompt. That means no
fine-tuning cost, instant "retraining" (just re-index), and your corpus never
leaves the prompt.

`llm_client.py` abstracts three backends behind one `generate()` call:

- **Gemini 2.5 Flash** (default): direct REST call with `httpx`, key from
  `.env`, internal "thinking" disabled (`thinkingBudget: 0`) because a
  rephrasing task doesn't need chain-of-thought and you want sub-second
  latency.
- **Ollama** (`phi3:mini`): local fallback, free and private, ~5s per request.
- **Any OpenAI-compatible API**: third option for DeepSeek, Together, etc.

### 4. Parsing: never trust LLM output

`parse_suggestions()` is deliberately paranoid, because every model deviates
from "output a numbered list" eventually:

- strips `<think>...</think>` reasoning blocks (deepseek-r1 wraps everything in
  them - this single issue silently broke the original pipeline)
- accepts numbered items or bullets; strips quotes, markdown bold, numbering
- deduplicates case-insensitively
- rejects "suggestions" that just echo the original text
- completions get a trim of any repeated sentence prefix

### 5. The four endpoints

| Endpoint | Retrieval used | Notes |
|---|---|---|
| `/suggest/word` | vocab hashmap only | LLM proposes synonyms using the full sentence for disambiguation ("inclined to disagree" gets figurative synonyms, not geometric); your personally-used words rank first with usage counts; automatic Datamuse fallback if the LLM is down |
| `/suggest/phrase` | ChromaDB + reranker | Rephrases a span inside a sentence |
| `/suggest/sentence` | ChromaDB + reranker | Five full rewrites, style-anchored |
| `/suggest/complete` | ChromaDB + reranker | Copilot-style endings, informed by how you've finished similar sentences |

Every response includes the retrieved examples, so you can inspect *why* the
system suggested what it did, plus latency for each request.

### 6. The Obsidian plugin

Four editor commands mirror the four endpoints. Each grabs the relevant text
(word at cursor via `editor.wordAt`, the selection, or the current sentence
found by splitting the line at punctuation), POSTs it with Obsidian's
`requestUrl` (which bypasses CORS), and shows results in a searchable picker
modal. Choosing one replaces the original text using ranges captured *before*
the request, so a cursor that moved during the round-trip can't misplace the
edit. Server URL is a plugin setting.

## The whole flow

```
Your vault ──(index once)──> ChromaDB (27,643 sentence vectors, MiniLM, cosine)
                             vocab_index.json (word frequency hashmap)

Obsidian command ──> FastAPI server (127.0.0.1:8111)
                       ├─ word:   LLM synonyms (Datamuse fallback) + vocab ranking
                       └─ others: ChromaDB top-30 (bi-encoder)
                                    ──> cross-encoder rerank, keep top-8
                                    ──> style prompt with your examples
                                    ──> Gemini 2.5 Flash (thinking off)
                                    ──> parse/clean/dedupe
                                    ──> picker modal ──> text replaced in editor
```

## Design principles (good interview talking points)

1. **Cheap, broad steps first; expensive, precise steps last.** Hashmap before
   API call, bi-encoder before cross-encoder, cross-encoder before LLM. Each
   stage narrows the field for the costlier one after it.
2. **Right data structure for the question.** Semantic similarity -> vector DB.
   "Have I used this word?" -> frequency hashmap. Not everything needs
   embeddings.
3. **Personalization without fine-tuning.** In-context examples give style
   transfer with zero training cost and instant updates (just re-index).
4. **Graceful degradation.** Gemini -> Ollama -> Datamuse: every external
   dependency has a fallback, so the tool works offline and without keys.
5. **Never trust model output.** Defensive parsing turned an unreliable demo
   into a dependable tool; the biggest bug in the project's history was
   unparsed `<think>` output, not retrieval or prompting.
6. **Measure the user-facing budget.** In-editor suggestions have a latency
   budget (~1-2s). Switching from a local 7B reasoning model to Gemini Flash
   with thinking disabled took requests from ~5s to 0.7-1.5s, verified.

## Resume-ready descriptions

**One-liner:**

> Built Linger, a personalized AI writing assistant that grounds LLM
> suggestions in a 27k-sentence private writing corpus via a two-stage RAG
> pipeline (ChromaDB retrieval + cross-encoder reranking), served through
> FastAPI to a custom Obsidian plugin.

**Bullet form (pick 2-3):**

- Built a personalized semantic search and suggestion system over a private
  27k-sentence writing corpus using ChromaDB, sentence-transformers, and
  FastAPI, integrated into the Obsidian editor through a custom TypeScript
  plugin with in-place text replacement.
- Designed a two-stage retrieval-augmented generation pipeline (bi-encoder
  vector search + cross-encoder reranking) that grounds Gemini 2.5 Flash
  generations in the user's own writing style via in-context examples,
  delivering personalized rephrasing, sentence completion, and context-aware
  synonym generation.
- Cut end-to-end suggestion latency from ~5s to under 1.5s by profiling the
  pipeline, replacing a local reasoning LLM with thinking-disabled Gemini
  Flash, and adding defensive output parsing; built automatic fallbacks
  (local Ollama inference, Datamuse API) for fully-offline operation.

**Interview follow-ups to be ready for:**

- *Why a reranker instead of just more retrieval?* Bi-encoders encode query and
  document separately, so they miss interaction between the two texts;
  cross-encoders read the pair jointly and score much more accurately but are
  ~100x slower, so you funnel: retrieve wide and cheap, rerank narrow and
  precise.
- *Why not fine-tune on the writing corpus?* Cost, iteration speed, and data
  volume: in-context examples achieve style transfer with zero training, update
  instantly on re-index, and work with corpora far too small to fine-tune on.
- *Why sentence-level chunks?* The retrieval targets are style exemplars for
  rephrasing single sentences, so the chunk should match the unit of
  generation; larger chunks would dilute the embedding with unrelated content.
- *How would you evaluate it?* A held-out set of (query, expected-sentence)
  pairs scored by recall@k for retrieval changes, plus side-by-side human
  preference on suggestions for prompt/model changes. (Not built yet; listed
  as future work.)
