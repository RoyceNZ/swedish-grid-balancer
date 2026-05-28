from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from .pipeline_logging import get_pipeline_logger

logger = get_pipeline_logger("utils.bronze")

_COMPRESSION_SUFFIXES = {".gz", ".zst", ".bz2", ".xz"}


def _manifest_path(bronze_path: Path) -> Path:
    p = bronze_path
    while p.suffix in _COMPRESSION_SUFFIXES:
        p = p.with_suffix("")
    return p.with_suffix(".manifest.json")


class BronzeWriterBase:
    def __init__(self, bronze_dir: Path) -> None:
        self.bronze_dir = Path(bronze_dir)
        self.bronze_dir.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, manifest: BaseModel, bronze_path: Path) -> Path:
        mpath = _manifest_path(bronze_path)
        with open(mpath, "w", encoding="utf-8") as fh:
            json.dump(manifest.model_dump(), fh, indent=2, ensure_ascii=False)
        logger.debug("Manifest written → %s", mpath)
        return mpath
