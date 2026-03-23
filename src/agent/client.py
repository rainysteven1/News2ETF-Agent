"""LLM client — OpenAI-compatible with tool calling and structured output.

Key design:
- tool_call id is preserved in return dict (required for LangGraph ToolMessage)
- chat_with_messages: multi-turn core method — takes full message history
- chat_with_tools: single-turn convenience (used in ReAct loop)
- chat_structured: provider-aware fallback for JSON output
"""

from __future__ import annotations

import json
import os

from openai import OpenAI


def resolve_provider(model: str) -> tuple[str, str, str]:
    PROVIDERS = {
        "zhipu": ("ZHIPU_API_KEY", "ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        "minimax": ("MINIMAX_API_KEY", "MINIMAX_BASE_URL", "https://api.minimaxi.chat/v1"),
        "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL", "https://api.openai.com/v1"),
    }
    MODEL_REGISTRY = {
        "glm-4-flash": "zhipu",
        "glm-4.7": "zhipu",
        "glm-4.5-airx": "zhipu",
        "MiniMax-M2.7": "minimax",
        "MiniMax-M2.7-highspeed": "minimax",
        "gpt-4o": "openai",
        "gpt-4o-mini": "openai",
    }
    provider_key = MODEL_REGISTRY.get(model, "zhipu")
    key_env, url_env, default_url = PROVIDERS[provider_key]
    api_key = os.environ.get(key_env, "")
    base_url = os.environ.get(url_env, default_url)
    return provider_key, api_key, base_url


class LLMClient:
    """OpenAI-compatible LLM client with tool calling and structured output."""

    def __init__(self, model: str = "glm-4-flash", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.provider, api_key, base_url = resolve_provider(model)
        if not api_key:
            raise ValueError(f"API key not set for model '{model}'")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    # ── Core multi-turn method ────────────────────────────────────────────────

    def chat_with_messages(
        self,
        messages: list[dict],
        tools: list | None = None,
        structured: bool = False,
        response_model: dict | None = None,
    ) -> dict:
        """Core method: multi-turn chat with full message history.

        Args:
            messages: List of OpenAI-format message dicts
                e.g. [{"role": "user", "content": "..."}, {"role": "tool", ...}]
            tools: LangChain @tool decorated functions (or dicts)
            structured: If True, request JSON object output
            response_model: OpenAI response_format dict (for structured output)

        Returns:
            {"content": str, "tool_calls": [{"id", "name", "args"}, ...]}
            tool_call id is PRESERVED — required for LangGraph ToolMessage attribution
        """
        openai_tools = self._convert_tools(tools) if tools else None

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools
        if structured and response_model:
            # Only use response_format for providers that support it well (OpenAI, Minimax)
            # Zhipu's older API may not support it — chat_structured handles the fallback
            kwargs["response_format"] = response_model

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        result: dict = {"content": message.content or "", "tool_calls": []}

        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                if not hasattr(tc, "function"):
                    continue
                result["tool_calls"].append(
                    {
                        "id": tc.id,  # ← preserved, critical for LangGraph ToolMessage
                        "name": tc.function.name,
                        "args": (
                            json.loads(tc.function.arguments)
                            if isinstance(tc.function.arguments, str)
                            else dict(tc.function.arguments)
                        ),
                    }
                )

        return result

    # ── Convenience wrappers ─────────────────────────────────────────────────

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Single-turn chat with system + user prompt."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def chat_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: list,
    ) -> dict:
        """Single-turn tool call — preserves tool_call id for LangGraph."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return self.chat_with_messages(messages, tools=tools)

    def chat_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: dict,
    ) -> str:
        """Structured JSON output with provider-aware fallback.

        OpenAI family: uses response_format=json_object
        Zhipu/others: fallback to forcing JSON in system prompt
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # OpenAI-style response_format works for OpenAI and Minimax
        use_response_format = self.provider in ("openai", "minimax")

        if use_response_format:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self.temperature,
            )
            return response.choices[0].message.content or ""

        # Zhipu fallback: rely on system prompt + response_model hint
        # Strip the structured schema requirements from system prompt
        fallback_system = (
            system_prompt
            + "\n\nIMPORTANT: You must respond with ONLY a valid JSON object. No explanation before or after."
            if system_prompt
            else "Respond with ONLY a valid JSON object."
        )
        fallback_messages = [{"role": "system", "content": fallback_system}] + [
            m for m in messages if m.get("role") != "system"
        ]
        response = self._client.chat.completions.create(
            model=self.model,
            messages=fallback_messages,
            temperature=self.temperature,
        )
        raw = response.choices[0].message.content or ""
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        return text

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _convert_tools(tools: list) -> list[dict]:
        """Convert LangChain @tool functions (or dicts) to OpenAI tool format.

        Handles:
        - LangChain BaseTool with .name, .description, .parameters
        - Plain dicts with name/description/parameters
        """
        openai_tools = []
        for t in tools:
            if isinstance(t, dict):
                name = t.get("name") or ""
                desc = t.get("description") or ""
                params = t.get("parameters") or {}
            else:
                name = getattr(t, "name", "") or ""
                desc = getattr(t, "description", "") or ""
                params = getattr(t, "parameters", {}) or {}

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": params,
                    },
                }
            )
        return openai_tools
