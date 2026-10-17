"""Configuration loading and repository paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repo root: $DEFACO_ROOT, or the first parent containing configs/default.yaml."""
    env = os.environ.get("DEFACO_ROOT")
    if env:
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "configs" / "default.yaml").exists():
            return p
    raise FileNotFoundError("Cannot locate repo root (configs/default.yaml); set DEFACO_ROOT.")


@dataclass
class Settings:
    root: Path
    raw: dict[str, Any]

    @property
    def data_dir(self) -> Path:
        """$DEFACO_DATA overrides the configured root (e.g. DEFACO_DATA=data_smoke)."""
        root = os.environ.get("DEFACO_DATA") or self.raw["data"]["root"]
        p = Path(root)
        return (p if p.is_absolute() else self.root / p).resolve()

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


def load_settings(path: str | Path | None = None) -> Settings:
    root = find_repo_root()
    cfg_path = Path(path) if path else root / "configs" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Settings(root=root, raw=raw)
