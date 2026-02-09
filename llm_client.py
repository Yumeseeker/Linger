"""
LLM client that abstracts different backends (Ollama, OpenAI-compatible APIs).

This module handles all communication with the language model.
The suggestion server calls this — you shouldn't need to use it directly.
"""

import json
import os
from typing import Optional

import httpx

import config


class LLMClient:
    """
    Unified client for local (Ollama) and remote (OpenAI-compatible) LLMs.

    Usage:
        client = LLMClient()
        response = await client.generate("Your prompt here")
    """

    def __init__(self, backend: Optional[str] = None):
        self.backend = backend or config.LLM_BACKEND
        self.timeout = httpx.Timeout(config.LLM_TIMEOUT, connect=5.0)

    async def generate(self, prompt: str, system: str = "") -> str:
        """Send a prompt to the LLM and return the response text."""
        if self.backend == "ollama":
            return await self._generate_ollama(prompt, system)
        elif self.backend == "openai_compatible":
            return await self._generate_openai(prompt, system)
        else:
            raise ValueError(f"Unknown LLM backend: {self.backend}")

    async def _generate_ollama(self, prompt: str, system: str = "") -> str:
        """Call local Ollama instance."""
        url = f"{config.OLLAMA_BASE_URL}/api/generate"
        
        # Combine system and prompt for Ollama (it doesn't handle system separately well)
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        
        payload = {
            "model": config.OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": config.LLM_TEMPERATURE,
                "num_predict": config.LLM_MAX_TOKENS,
            },
        }

        print(f"[OLLAMA_DEBUG] Sending full prompt (len={len(full_prompt)})", flush=True)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            result = data.get("response", "").strip()
            print(f"[OLLAMA_DEBUG] Got response of length {len(result)}: {repr(result)}", flush=True)
            return result

    async def _generate_openai(self, prompt: str, system: str = "") -> str:
        """Call any OpenAI-compatible API (DeepSeek, Together, local vLLM, etc.)."""
        url = f"{config.OPENAI_API_BASE}/chat/completions"
        api_key = config.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": config.OPENAI_MODEL,
            "messages": messages,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

    async def health_check(self) -> dict:
        """Check if the LLM backend is reachable."""
        try:
            if self.backend == "ollama":
                async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
                    resp = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
                    resp.raise_for_status()
                    models = [m["name"] for m in resp.json().get("models", [])]
                    has_model = any(config.OLLAMA_MODEL in m for m in models)
                    return {
                        "status": "ok" if has_model else "model_missing",
                        "backend": "ollama",
                        "model": config.OLLAMA_MODEL,
                        "available_models": models,
                        "message": (
                            f"Model '{config.OLLAMA_MODEL}' is ready"
                            if has_model
                            else f"Model '{config.OLLAMA_MODEL}' not found. "
                                 f"Run: ollama pull {config.OLLAMA_MODEL}"
                        ),
                    }
            else:
                # For OpenAI-compatible, just check if we have an API key
                api_key = config.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY", "")
                return {
                    "status": "ok" if api_key else "no_api_key",
                    "backend": "openai_compatible",
                    "model": config.OPENAI_MODEL,
                    "message": (
                        f"API key configured for {config.OPENAI_API_BASE}"
                        if api_key
                        else "No API key set. Set OPENAI_API_KEY in config.py or env var."
                    ),
                }
        except httpx.ConnectError:
            return {
                "status": "unreachable",
                "backend": self.backend,
                "message": (
                    f"Cannot connect to Ollama at {config.OLLAMA_BASE_URL}. "
                    "Is Ollama running? Start it with: ollama serve"
                    if self.backend == "ollama"
                    else f"Cannot connect to {config.OPENAI_API_BASE}"
                ),
            }
        except Exception as e:
            return {
                "status": "error",
                "backend": self.backend,
                "message": str(e),
            }
