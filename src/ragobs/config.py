"""Load observability settings and build the vendored RAG settings for this deployment."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "vendor" / "dronedocs-rag"


def _override(cfg: dict, prefix: str) -> dict:
    for key, raw in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = [p.lower() for p in key[len(prefix) :].split("__") if p]
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


@lru_cache(maxsize=1)
def obs_settings() -> dict:
    path = Path(os.environ.get("RAGOBS_CONFIG", ROOT / "configs" / "observability.yaml"))
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _override(cfg, "RAGOBS__")


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def rag_settings(prompt_file: str | None = None, model: str | None = None) -> dict:
    """Vendored DroneDocs settings, pointed at this repo's prompt + model + index location."""
    os.environ.setdefault("DRONEDOCS_ROOT", str(VENDOR))
    from dronedocs.config import load_settings

    obs = obs_settings()
    cfg = load_settings(resolve(obs["rag"]["settings_file"]))
    cfg["app"]["prompt_file"] = str(resolve(prompt_file or obs["rag"]["prompt_file"]))
    cfg["llm"]["model"] = model or obs["rag"]["model"]
    cfg["judge"]["model"] = obs["rag"]["judge_model"]
    for section, values in (obs["rag"].get("overrides") or {}).items():
        cfg.setdefault(section, {}).update(values)
    return cfg
