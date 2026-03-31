"""Memos-based historical memory retrieval via /search/memory API.

Uses the MemOS vector search endpoint instead of TF-IDF.
Stores investment decisions as conversation memories for future retrieval.
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests


class MemosRetrieval:
    """Vector search over historical investment decisions via Memos API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        user_id: str = "etf_agent",
    ):
        self.api_key = api_key or os.environ.get("MEMOS_API_KEY", "")
        self.base_url = base_url or os.environ.get("MEMOS_BASE_URL", "https://memos.memtensor.cn/api/openmem/v1")
        self.user_id = user_id
        self._session = requests.Session()
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}",
        }

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict:
        """Make a POST request to Memos API."""
        url = f"{self.base_url}{endpoint}"
        resp = self._session.post(
            url=url,
            headers=self._headers,
            data=json.dumps(payload),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def retrieve(
        self,
        query: str,
        conversation_id: str | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Search relevant historical memories via Memos vector search.

        Args:
            query: Natural language query describing the current situation
            conversation_id: Optional conversation ID for filtering
            top_k: Number of results to return

        Returns:
            List of relevant historical cases with similarity scores
        """
        payload: dict[str, Any] = {
            "query": query,
            "user_id": self.user_id,
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id

        try:
            result = self._post("/search/memory", payload)
            memories = result.get("data", {}).get("memory_detail_list", [])

            # Parse and format results
            formatted = []
            for mem in memories[:top_k]:
                created_ts = mem.get("create_time", 0)
                from datetime import datetime
                date_str = ""
                if created_ts:
                    date_str = datetime.fromtimestamp(created_ts / 1000).strftime("%Y-%m-%d")
                formatted.append(
                    {
                        "similarity": mem.get("relativity", 0.0),
                        "content": mem.get("memory_value", ""),
                        "date": date_str,
                        "conversation_id": mem.get("conversation_id", ""),
                    }
                )
            return formatted

        except Exception as e:
            # Fallback: return empty on error (don't crash the agent)
            return [{"error": str(e), "similarity": 0.0, "content": ""}]

    def add_decision(
        self,
        conversation_id: str,
        decision: str,
        context: str = "",
        date: str | None = None,
    ) -> bool:
        """Store an investment decision as a memory in Memos.

        Args:
            conversation_id: Unique ID for this week's decision
            decision: The decision made (e.g., "buy 银行 25%")
            context: Additional context (signals, news summary, etc.)
            date: Optional date string

        Returns:
            True if stored successfully
        """
        content = f"Decision: {decision}"
        if context:
            content += f"\n\nContext: {context}"

        messages = [
            {"role": "user", "content": f"Investment decision for {conversation_id}"},
            {"role": "assistant", "content": content},
        ]

        if date:
            messages[0]["content"] += f" (date: {date})"

        payload = {
            "user_id": self.user_id,
            "conversation_id": conversation_id,
            "messages": messages,
        }

        try:
            self._post("/add/message", payload)
            return True
        except Exception:
            return False
