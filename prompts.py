"""
Prompt templates for the suggestion engine.

These prompts are the most important part of the system — they determine
whether suggestions feel like YOUR words or generic AI output.

The key insight: we feed the LLM examples from YOUR past writing as context,
so it learns your style in-context rather than generating from its general training.

IMPORTANT: Small models (phi3:mini, qwen2.5:1.5b) need very explicit, constrained
prompts. The more specific the instruction, the better the output. We use concrete
examples in the prompt itself so the model understands the expected format.
"""

import re
import config


SYSTEM_PROMPT = """You are a thesaurus and writing aid. Your ONLY job is to suggest synonyms and alternative phrasings.

Rules:
- ONLY output the numbered list. No explanations, no commentary, no preamble.
"""

# - Every suggestion must be a direct replacement that fits grammatically in the original sentence.
#- Suggestions must preserve the original meaning.
#- Draw on the writer's past vocabulary when possible, but always prioritize correct synonyms.
#- Never repeat the original word/phrase as a suggestion.


def build_word_prompt(
    word: str,
    sentence: str,
    paragraph: str,
    retrieved_examples: list[str],
) -> str:
    """
    Build a prompt for single-word synonym suggestions.
    Must maintain grammatical correctness in context.
    """
    examples_block = "\n".join(f"  - \"{ex}\"" for ex in retrieved_examples[:4])
    
    return f"""TASK: Generate 5 exact word substitutes for "{word}".

CRITICAL REQUIREMENT: Each substitute must work EXACTLY as written in this sentence:
"{sentence}"

This means:
- If "{word}" is singular, suggestions must be singular
- If "{word}" is preceded by "a", suggestions cannot require "an" and vice versa
- If "{word}" is capitalized, suggestions must match that too
- Suggestions must have identical grammatical behavior (articles, tense, etc.)

Context from this writer's past work:
{examples_block}

ONLY OUTPUT the numbered list (no explanation):
1. 
2. 
3. 
4. 
5. """


def build_phrase_prompt(
    phrase: str,
    sentence: str,
    paragraph: str,
    retrieved_examples: list[str],
) -> str:
    """
    Build a prompt for phrase-level rephrasings.

    This handles cases like "why am I so idea generating these days" →
    "why am I so inclined toward ideation these days"
    """
    examples_block = "\n".join(f"  - \"{ex}\"" for ex in retrieved_examples[:6])

    return f"""Task: Give 5 alternative phrasings for "{phrase}" as used in this sentence.

Sentence: "{sentence}"

Each alternative must:
- Replace "{phrase}" in the sentence so it still reads naturally
- Keep the same meaning
- Be a different way of saying the same thing

Example:
- Sentence: "Why am I so idea generating these days."
- Phrase: "idea generating"
- Alternatives: inclined toward ideation, full of ideas, creatively productive, generating so many ideas, brimming with concepts

Here are sentences from this writer's past work for style reference:
{examples_block}

Give 5 alternative phrasings for "{phrase}" in: "{sentence}"

1.
2.
3.
4.
5."""


def build_sentence_prompt(
    sentence: str,
    paragraph: str,
    retrieved_examples: list[str],
) -> str:
    """
    Build a prompt for full sentence rephrasings.

    The writer has a complete sentence but wants to see alternative
    ways they might express the same thought.
    """
    examples_block = "\n".join(f"  - \"{ex}\"" for ex in retrieved_examples[:6])

    return f"""Task: Rewrite this sentence 5 different ways, preserving the meaning.

Sentence: "{sentence}"

Each rewrite must:
- Express the exact same idea
- Be a complete sentence
- Use different word choices or sentence structure
- Sound natural, not robotic

Here are sentences from this writer's past work — match this style:
{examples_block}

5 rewrites of "{sentence}":

1.
2.
3.
4.
5."""


def build_continuation_prompt(
    sentence_so_far: str,
    paragraph: str,
    retrieved_examples: list[str],
) -> str:
    """
    Build a prompt for sentence continuation / completion.

    This is the Copilot-style feature: the writer has started a sentence
    and the system predicts how they would finish it.
    """
    examples_block = "\n".join(f"  - \"{ex}\"" for ex in retrieved_examples[:6])

    return f"""Task: Complete this unfinished sentence 5 different ways.

Unfinished sentence: "{sentence_so_far}"

Each completion must:
- Continue directly from where the sentence stops
- Form a complete, natural sentence when combined with the beginning
- Offer a meaningfully different ending

Here are sentences from this writer's past work — match this style:
{examples_block}

5 completions for "{sentence_so_far}...":

1. {sentence_so_far}
2. {sentence_so_far}
3. {sentence_so_far}
4. {sentence_so_far}
5. {sentence_so_far}"""


def parse_suggestions(raw_response: str) -> list[str]:
    """
    Parse the LLM's response into a clean list of suggestions.
    
    Simply extracts numbered list items (1., 2., etc.)
    """
    lines = raw_response.strip().split("\n")
    suggestions = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match numbered items: "1. word", "1) word", etc.
        match = re.match(r'^\d+[\.\)]\s*(.+)$', line)
        if match:
            suggestion = match.group(1).strip()
            # Basic sanity check: must be at least 2 chars
            if len(suggestion) >= 2:
                suggestions.append(suggestion)

    return suggestions[:config.NUM_SUGGESTIONS]
