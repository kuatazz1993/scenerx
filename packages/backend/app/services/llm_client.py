"""
Abstract LLM Client + Provider Implementations

Supports Gemini, OpenAI, Anthropic, and DeepSeek (via OpenAI SDK with custom base_url).
Each provider uses lazy imports so only the active provider's SDK needs to be installed.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

LLM_PROVIDERS = {
    "gemini": {"name": "Google Gemini", "default_model": "gemini-2.5-flash"},
    "openai": {"name": "OpenAI", "default_model": "gpt-4o"},
    "anthropic": {"name": "Anthropic Claude", "default_model": "claude-sonnet-4-20250514"},
    "deepseek": {"name": "DeepSeek", "default_model": "deepseek-chat"},
}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMClient(ABC):
    """Abstract LLM client for text generation."""

    provider: str
    model: str

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Send prompt, return text response."""
        ...

    @abstractmethod
    def check_connection(self) -> bool:
        """Check if API key is configured and client can connect."""
        ...

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield text chunks. Default fallback: single yield of full response."""
        text = await self.generate(prompt)
        yield text


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

class GeminiLLM(LLMClient):
    provider = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def generate(self, prompt: str) -> str:
        import asyncio

        client = self._get_client()
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=self.model,
            contents=prompt,
        )
        # Safely extract text — thinking models may have no text part
        try:
            return response.text or ""
        except (ValueError, AttributeError):
            for candidate in (response.candidates or []):
                parts = getattr(candidate, "content", None)
                if parts:
                    texts = [p.text for p in (parts.parts or []) if hasattr(p, "text") and p.text]
                    if texts:
                        return "\n".join(texts)
            return ""

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        client = self._get_client()
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _produce():
            try:
                for chunk in client.models.generate_content_stream(
                    model=self.model, contents=prompt,
                ):
                    text = ""
                    try:
                        text = chunk.text or ""
                    except (ValueError, AttributeError):
                        for cand in (chunk.candidates or []):
                            parts = getattr(cand, "content", None)
                            if parts:
                                texts = [p.text for p in (parts.parts or [])
                                         if hasattr(p, "text") and p.text]
                                if texts:
                                    text = "".join(texts)
                    if text:
                        loop.call_soon_threadsafe(queue.put_nowait, text)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, _produce)
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def check_connection(self) -> bool:
        if not self.api_key:
            logger.warning("Gemini: API key is empty")
            return False
        try:
            self._get_client()
            return True
        except Exception as e:
            logger.error("Gemini check_connection failed: %s", e)
            return False


# ---------------------------------------------------------------------------
# OpenAI (also used for DeepSeek via custom base_url)
# ---------------------------------------------------------------------------

class OpenAILLM(LLMClient):
    provider = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    async def generate(self, prompt: str) -> str:
        import asyncio

        def _call():
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""

        return await asyncio.to_thread(_call)

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _produce():
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
                for chunk in response:
                    text = chunk.choices[0].delta.content if chunk.choices else ""
                    if text:
                        loop.call_soon_threadsafe(queue.put_nowait, text)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, _produce)
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def check_connection(self) -> bool:
        if not self.api_key:
            logger.warning("OpenAI: API key is empty")
            return False
        try:
            self._get_client()
            return True
        except Exception as e:
            logger.error("OpenAI check_connection failed: %s", e)
            return False


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicLLM(LLMClient):
    provider = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    async def generate(self, prompt: str) -> str:
        import asyncio

        def _call():
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text if response.content else ""

        return await asyncio.to_thread(_call)

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _produce():
            try:
                client = self._get_client()
                with client.messages.stream(
                    model=self.model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    for text in stream.text_stream:
                        if text:
                            loop.call_soon_threadsafe(queue.put_nowait, text)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        loop.run_in_executor(None, _produce)
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def check_connection(self) -> bool:
        if not self.api_key:
            logger.warning("Anthropic: API key is empty")
            return False
        try:
            self._get_client()
            return True
        except Exception as e:
            logger.error("Anthropic check_connection failed: %s", e)
            return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_llm_client(
    provider: str, api_key: str, model: str, **kwargs
) -> LLMClient:
    """Create an LLM client for the given provider."""
    if provider == "gemini":
        return GeminiLLM(api_key, model)
    if provider == "openai":
        return OpenAILLM(api_key, model)
    if provider == "anthropic":
        return AnthropicLLM(api_key, model)
    if provider == "deepseek":
        return OpenAILLM(api_key, model, base_url="https://api.deepseek.com")
    raise ValueError(f"Unknown LLM provider: {provider}")
