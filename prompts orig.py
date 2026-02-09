"""
Prompt templates for the suggestion engine.

These prompts are the most important part of the system — they determine
whether suggestions feel like YOUR words or generic AI output.

The key insight: we feed the LLM examples from YOUR past writing as context,
so it learns your style in-context rather than generating from its general training.
"""


SYSTEM_PROMPT = """You are a writing assistant that helps a specific writer find the right words.
You are NOT generating new content. You are helping the writer RETRIEVE words and phrases
from their own vocabulary — words they know and have used before but can't recall right now.

Your suggestions must:
- Match the tone and register of the writer's examples
- Be real alternatives the writer would plausibly choose
- Preserve the meaning and intent of the original
- Vary in style (some more formal, some more casual, some more vivid)

Your suggestions must NOT:
- Sound generic or AI-generated
- Change the fundamental meaning
- Be overly literary or flowery unless the writer's examples are
- Include explanations or commentary — just the alternatives"""


def build_word_prompt(
    word: str,
    sentence: str,
    paragraph: str,
    retrieved_examples: list[str],
) -> str:
    """
    Build a prompt for single-word synonym suggestions.

    This is the "tip of the tongue" case: the writer typed a word but
    suspects there's a better one they can't recall.
    """
    examples_block = "\n".join(f"  - {ex}" for ex in retrieved_examples)

    return f"""The writer is looking for a better word to replace "{word}" in this sentence:

Sentence: "{sentence}"

Broader context (paragraph):
"{paragraph}"

Here are sentences from this writer's past work that are contextually similar:
{examples_block}

Based on the writer's style and vocabulary shown above, suggest {5} alternative words
to replace "{word}" in the given sentence. Each alternative should fit grammatically
and match the writer's voice.

Return ONLY the alternatives, one per line, numbered. No explanations.
Format:
1. [word]
2. [word]
3. [word]
4. [word]
5. [word]"""


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
    examples_block = "\n".join(f"  - {ex}" for ex in retrieved_examples)

    return f"""The writer wants to rephrase this part of their sentence:

Phrase to rephrase: "{phrase}"
Full sentence: "{sentence}"

Broader context (paragraph):
"{paragraph}"

Here are sentences from this writer's past work that are contextually similar:
{examples_block}

Based on the writer's style shown above, suggest {5} alternative ways to express
the same idea as "{phrase}" within the given sentence. Each suggestion should:
- Replace the phrase naturally within the sentence
- Preserve the writer's intent and tone
- Match the vocabulary level shown in the examples

Return ONLY the alternative phrases, one per line, numbered. No explanations.
Format:
1. [phrase]
2. [phrase]
3. [phrase]
4. [phrase]
5. [phrase]"""


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
    examples_block = "\n".join(f"  - {ex}" for ex in retrieved_examples)

    return f"""The writer wants to see alternative ways to express this sentence:

Sentence: "{sentence}"

Broader context (paragraph):
"{paragraph}"

Here are sentences from this writer's past work in similar contexts:
{examples_block}

Based on the writer's style shown above, suggest {5} alternative ways to write
this sentence. Each alternative should:
- Express the same core idea
- Sound like this specific writer, not like generic AI
- Vary in approach (some more concise, some more elaborate, some restructured)

Return ONLY the alternative sentences, one per line, numbered. No explanations.
Format:
1. [sentence]
2. [sentence]
3. [sentence]
4. [sentence]
5. [sentence]"""


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
    examples_block = "\n".join(f"  - {ex}" for ex in retrieved_examples)

    return f"""The writer has started a sentence and needs help completing it.

Sentence so far: "{sentence_so_far}"

Broader context (paragraph):
"{paragraph}"

Here are complete sentences from this writer's past work in similar contexts:
{examples_block}

Based on the writer's style and how they typically construct sentences,
suggest {5} ways to complete the sentence. Each completion should:
- Continue naturally from where the writer left off
- Sound like this writer's voice
- Complete the thought in a way the writer would plausibly choose

Return ONLY the completions (the part that comes after what's already written),
one per line, numbered. No explanations.
Format:
1. [completion]
2. [completion]
3. [completion]
4. [completion]
5. [completion]"""


def parse_suggestions(raw_response: str) -> list[str]:
    """
    Parse the LLM's response into a clean list of suggestions.

    Handles various formatting quirks:
    - Numbered lists (1. suggestion, 1) suggestion)
    - Bullet points
    - Quoted suggestions
    - Extra whitespace or blank lines
    """
    lines = raw_response.strip().split("\n")
    suggestions = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove numbering: "1. ", "1) ", "- ", "* "
        import re
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        line = re.sub(r'^[-*]\s*', '', line)

        # Remove surrounding quotes
        line = line.strip('"\'""''')

        # Remove any trailing explanation after a dash or parenthetical
        # e.g., "remarked — more formal tone" → "remarked"
        line = re.sub(r'\s*[-—]\s+.*$', '', line)
        line = re.sub(r'\s*\(.*\)\s*$', '', line)

        line = line.strip()
        if line and len(line) > 0:
            suggestions.append(line)

    return suggestions[:config.NUM_SUGGESTIONS]


# Import config for NUM_SUGGESTIONS reference in prompts
import config
