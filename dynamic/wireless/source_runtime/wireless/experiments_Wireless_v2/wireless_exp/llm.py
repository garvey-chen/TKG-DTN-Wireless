"""One logged API transport shared by both original LLM methods."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import requests


class LoggedLLM:
    def __init__(self, settings: dict, log_path: Path):
        self.settings = settings
        self.log_path = log_path
        self.calls = 0
        self.api_attempts = 0
        self.failures = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.user_id = None
        self.session = requests.Session()

    def _write(self, record: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def invoke(self, messages):
        roles = {"human": "user", "ai": "assistant", "system": "system"}
        converted = [
            m if isinstance(m, dict) else {"role": roles[m.type], "content": m.content}
            for m in messages
        ]
        return SimpleNamespace(content=self.complete(converted))

    def complete(self, messages: list[dict]) -> str:
        if self.calls >= self.settings["max_calls_per_run"]:
            self.failures += 1
            raise RuntimeError("LLM call limit reached; run is incomplete.")
        self.calls += 1
        payload = {
            "model": self.settings["model"],
            "messages": messages,
            "temperature": self.settings["temperature"],
            "max_tokens": self.settings["max_tokens"],
        }
        # Only strip trailing slash: config explicitly supplies the /v1 prefix.
        url = self.settings["base_url"].rstrip("/") + "/chat/completions"
        started = time.perf_counter()
        last_error = "unknown"
        for attempt in range(self.settings["max_retries"] + 1):
            self.api_attempts += 1
            retryable = True
            try:
                response = self.session.post(
                    url, json=payload,
                    headers={"Authorization": "Bearer " + self.settings["api_key"]},
                    timeout=self.settings["timeout_seconds"],
                )
                if response.status_code != 200:
                    retryable = response.status_code in (429, 500, 502, 503, 504)
                    raise RuntimeError(f"LLM HTTP {response.status_code}; response body omitted.")
                data = response.json()
                choice = data["choices"][0]
                usage = data.get("usage") or {}
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
                content = choice["message"].get("content")
                self._write({
                    "call": self.calls, "attempt": attempt + 1, "user_id": self.user_id,
                    "model_requested": self.settings["model"], "model_returned": data.get("model"),
                    "messages": messages, "response": content,
                    "finish_reason": choice.get("finish_reason"), "usage": usage,
                    "elapsed_seconds": time.perf_counter() - started,
                })
                if choice.get("finish_reason") == "length" or not content:
                    retryable = False
                    raise RuntimeError("LLM returned truncated/empty output; not a valid reproduction.")
                return content
            except (requests.RequestException, ValueError, KeyError, IndexError, RuntimeError) as exc:
                # Avoid serializing request headers or credentials in exception messages.
                last_error = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
                last_error = last_error.replace(self.settings["api_key"], "[REDACTED]")
                self._write({"call": self.calls, "attempt": attempt + 1, "user_id": self.user_id, "error": last_error})
                if not retryable:
                    break
                if attempt < self.settings["max_retries"]:
                    time.sleep(min(2 ** attempt, 4))
        self.failures += 1
        raise RuntimeError(last_error)

    def statistics(self) -> dict:
        return {
            "llm_calls": self.calls, "api_attempts": self.api_attempts,
            "llm_failures": self.failures, "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }
