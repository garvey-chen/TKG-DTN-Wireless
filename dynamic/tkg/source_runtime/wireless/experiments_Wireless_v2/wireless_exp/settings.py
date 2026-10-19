"""Explicit configuration; no credentials are copied into the experiment."""

import json
import os
from pathlib import Path

from . import ROOT


def load_config(path: Path | None = None) -> dict:
    return json.loads((path or ROOT / "configs/default.json").read_text(encoding="utf-8"))


def load_llm_settings(config: dict, credential_config: Path | None = None) -> dict:
    settings = dict(config["llm"])
    key = ""
    if credential_config is not None:
        import yaml
        value = yaml.safe_load(credential_config.read_text(encoding="utf-8")) or {}
        llm = value.get("llm", {})
        key = llm.get("api_key", "")
        # The model/URL in the explicitly chosen credential configuration win.
        for field in ("base_url", "model"):
            if llm.get(field):
                settings[field] = llm[field]
    key = os.environ.get("WIRELESS_LLM_API_KEY") or os.environ.get("NETWORK_LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or key
    if not key or str(key).startswith(("${", "REDACTED", "your_", "YOUR_")):
        raise ValueError("No usable LLM credential. Set WIRELESS_LLM_API_KEY or --credential-config.")
    settings["api_key"] = str(key)
    for field in ("base_url", "model"):
        settings[field] = os.environ.get(f"WIRELESS_LLM_{field.upper()}", settings[field])
    return settings


def public_settings(settings: dict) -> dict:
    return {k: v for k, v in settings.items() if k != "api_key"}
