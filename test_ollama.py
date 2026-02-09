#!/usr/bin/env python3
"""
Test Ollama connection directly without the server.
"""

import httpx
import json

BASE_URL = "http://localhost:11434"
MODEL = "deepseek-r1:7b"

print(f"Testing Ollama connection to {BASE_URL}")
print(f"Model: {MODEL}\n")

try:
    # Test 1: Check if Ollama is running
    print("1. Checking if Ollama is running...")
    response = httpx.get(f"{BASE_URL}/api/tags", timeout=5.0)
    models = response.json()
    print(f"   ✓ Ollama is running")
    print(f"   Available models: {[m['name'] for m in models.get('models', [])]}\n")
    
    # Test 2: Try to generate a response
    print(f"2. Sending test prompt to {MODEL}...")
    response = httpx.post(
        f"{BASE_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": "Say 'hello'",
            "stream": False,
        },
        timeout=60.0,
    )
    
    if response.status_code == 200:
        data = response.json()
        result = data.get("response", "")
        print(f"   ✓ Got response: {result[:100]}")
    else:
        print(f"   ✗ Error: {response.status_code}")
        print(f"   Response: {response.text}")
        
except httpx.ConnectError:
    print(f"   ✗ Cannot connect to Ollama at {BASE_URL}")
    print(f"   Make sure Ollama is running: ollama serve")
except httpx.TimeoutException:
    print(f"   ✗ Ollama timed out (model may be loading)")
except Exception as e:
    print(f"   ✗ Error: {e}")
