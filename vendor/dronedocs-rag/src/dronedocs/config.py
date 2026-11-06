"""Configuration loading: YAML settings + versioned prompt bundles + env overrides."""

from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ENV_PREFIX = "DRONEDOCS__"


def project_root() -> Path:
    """Repo root (directory that contains ``configs/``)."""
    env = os.environ.get("DRONEDOCS_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "settings.yaml").exists():
            return parent
    return Path.cwd()


def _coerce(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_env_overrides(cfg: dict, environ: dict[str, str] | None = None) -> dict:
    """Override nested keys from env vars like ``DRONEDOCS__RERANK__TOP_N=8``."""
    environ = os.environ if environ is None else environ
    out = copy.deepcopy(cfg)
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        path = [p.lower() for p in key[len(ENV_PREFIX) :].split("__") if p]
        node = out
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = _coerce(raw)
    return out


def load_settings(path: str | Path | None = None, environ: dict[str, str] | None = None) -> dict:
    root = project_root()
    path = Path(path) if path else Path(os.environ.get("DRONEDOCS_CONFIG", root / "configs" / "settings.yaml"))
    if not path.is_absolute():
        path = root / path
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg = apply_env_overrides(cfg, environ)
    # resolve relative paths against the repo root
    for k, v in cfg.get("paths", {}).items():
        p = Path(v)
        cfg["paths"][k] = str(p if p.is_absolute() else root / p)
    return cfg


@lru_cache(maxsize=1)
def get_settings() -> dict:
    return load_settings()


def load_prompt_bundle(path: str | Path) -> dict:
    """Load a versioned prompt YAML. Every bundle must carry a ``version`` key."""
    p = Path(path)
    if not p.is_absolute():
        p = project_root() / p
    with open(p, encoding="utf-8") as fh:
        bundle = yaml.safe_load(fh)
    if "version" not in bundle:
        raise ValueError(f"prompt bundle {p} has no 'version' key")
    bundle["_path"] = str(p)
    return bundle
