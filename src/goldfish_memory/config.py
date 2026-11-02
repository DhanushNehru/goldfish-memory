"""Engine configuration: JSON file next to the database plus env overrides."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONFIG_NAME = "config.json"
ENV_DB = "GOLDFISH_DB"
ENV_MODEL_DIR = "GOLDFISH_MODEL_DIR"


@dataclass
class Config:
    """Tunable engine behavior.

    Attributes:
        model_dir: Directory holding ``model.onnx`` + ``tokenizer.json`` for the
            default embedder. ``None`` means use the weights bundled with the
            package. The engine never downloads; point this at weights you
            provisioned yourself to use a different model.
        auto_compact: Run compaction automatically during capture when enough
            old memories have accumulated.
        compact_min_age_days: Only memories older than this are compacted.
        compact_batch: Auto-compaction triggers once at least this many
            compactable memories exist.
        compact_interval_hours: Minimum hours between automatic compactions.
    """

    model_dir: str | None = None
    auto_compact: bool = True
    compact_min_age_days: float = 14.0
    compact_batch: int = 50
    compact_interval_hours: float = 6.0

    @classmethod
    def load(cls, db_path: str | Path) -> Config:
        """Load ``config.json`` sitting next to the database, if present."""
        if str(db_path) == ":memory:":
            return cls()
        cfg_file = Path(db_path).parent / CONFIG_NAME
        data: dict[str, Any] = {}
        if cfg_file.exists():
            data = json.loads(cfg_file.read_text())
        known = {f for f in asdict(cls())}
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        env_model = os.environ.get(ENV_MODEL_DIR)
        if env_model:
            cfg.model_dir = env_model
        return cfg

    def save(self, db_path: str | Path) -> None:
        """Persist the config next to the database file."""
        if str(db_path) == ":memory:":
            return
        cfg_file = Path(db_path).parent / CONFIG_NAME
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(json.dumps(asdict(self), indent=2) + "\n")
