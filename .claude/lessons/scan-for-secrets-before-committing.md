Scan tracked files for key material before every commit; keys get pasted into whatever file is open.

What happened: on 2026-07-07 Brian added his Gemini API key by pasting it into
`.env.example` (the tracked template that happened to be open in the IDE) rather
than the gitignored `.env`. Committing without checking would have published the
key to git history, which is very hard to scrub.

Why it mattered: this repo previously had an OpenAI key hardcoded in `config.py`
too, so "secret lands in a tracked file" is a recurring failure mode here, not a
one-off.

How to apply: before any commit, grep the staged files for the known key prefixes
(`AIza`, `AQ.`, `sk-`) or the literal key value, and confirm `.env` is the only
file containing secrets (`git check-ignore .env` must succeed). If a key shows up
in a tracked file, move it to `.env`, strip the tracked file, and tell Brian to
rotate the key.
