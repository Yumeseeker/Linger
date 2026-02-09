#!/usr/bin/env python3
"""
Test the actual synonym prompt with Ollama.
"""

import httpx

BASE_URL = "http://localhost:11434"
MODEL = "deepseek-r1:7b"

# The actual prompt we're sending
prompt = """Generate 5 synonyms for the word "dinner".

CRITICAL: All suggestions must:
- Be the SAME PART OF SPEECH as "dinner"
- Be able to directly replace "dinner" in this sentence: "After eating dinner, he said"
- Maintain grammatical correctness when substituted
- Have the same or very similar meaning

For example:
- If the word is a VERB (like "eating"), suggest other verbs: consuming, having, finishing, devouring
- If the word is a NOUN (like "dinner"), suggest other nouns: meal, supper, feast, banquet
- If the word is an ADJECTIVE (like "happy"), suggest other adjectives: pleased, delighted, cheerful, content

Now generate 5 synonyms for "dinner" (must maintain syntax and grammar):
1.
2.
3.
4.
5."""

print("Testing actual synonym prompt...\n")
print(f"Prompt:\n{prompt}\n")
print("="*80)

try:
    response = httpx.post(
        f"{BASE_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "system": "You are a helpful assistant.",
        },
        timeout=60.0,
    )
    
    if response.status_code == 200:
        data = response.json()
        result = data.get("response", "")
        print(f"\nRAW RESPONSE:\n{repr(result)}\n")
        print(f"FORMATTED RESPONSE:\n{result}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"Error: {e}")
