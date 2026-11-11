"""Run context: configuration, paths and logging shared by every stage."""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Context:
    cfg: Dict[str, Any]
    root: Path
    log: logging.Logger = field(repr=False)

    # ---- paths -----------------------------------------------------------
    def path(self, key: str) -> Path:
        return self.root / self.cfg["paths"][key]

    @property
    def raw(self) -> Path:
        p = self.path("data_raw")
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def external(self) -> Path:
        p = self.path("data_external")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def stage_dir(self, name: str) -> Path:
        p = self.path("outputs") / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---- convenience -----------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.cfg["project"]["seed"])

    @property
    def n_jobs(self) -> int:
        return int(self.cfg["project"].get("n_jobs", 1))

    def datasets(self, dtype: Optional[str] = None):
        for acc, d in self.cfg["datasets"].items():
            if dtype is None or d["type"] == dtype:
                yield acc, d


def make_logger(log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("sjogren")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def load_context(config_path: Optional[str] = None, root: Optional[str] = None) -> Context:
    cfg_path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    base = Path(root).resolve() if root else cfg_path.resolve().parent
    log = make_logger(base / cfg["paths"]["outputs"] / "pipeline.log")
    return Context(cfg=cfg, root=base, log=log)
