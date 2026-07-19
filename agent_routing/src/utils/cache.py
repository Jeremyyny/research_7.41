"""Disk-backed cache for teacher API calls.

Keyed by (provider, model, prompt_hash, temperature, system_hash). Avoids
re-spending API credits on identical generations across re-runs.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any, Dict, List, Optional


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]


class TeacherCallCache:
    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def make_key(
        provider: str,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int] = None,
    ) -> str:
        obj: Dict[str, Any] = {
            "provider": provider,
            "model": model,
            "messages": messages,
            "temperature": round(float(temperature), 4),
        }
        # Included so a raised token budget re-generates instead of returning a
        # stale truncated response. None keeps legacy keys valid.
        if max_tokens is not None:
            obj["max_tokens"] = int(max_tokens)
        payload = json.dumps(obj, ensure_ascii=False, sort_keys=True)
        return _hash(payload)

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def put(self, key: str, value: Dict[str, Any]) -> None:
        path = self._path(key)
        with self._lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False, indent=2)