#!/usr/bin/env python3
"""
Debug script to see the exact prompt being sent to the LLM.
"""

from prompts import build_word_prompt

# Test with your actual query
word = "eating"
sentence = "After eating dinner, he said"
paragraph = ""
retrieved_examples = ["Eventually, I returned to the dinner table.", "Anyway, I went home right after dinner."]

prompt = build_word_prompt(word, sentence, paragraph, retrieved_examples)

print("=" * 80)
print("EXACT PROMPT BEING SENT TO LLM:")
print("=" * 80)
print(prompt)
print("=" * 80)
