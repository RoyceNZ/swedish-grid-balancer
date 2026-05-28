from __future__ import annotations

import logging

_FMT = logging.Formatter(
    fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def get_pipeline_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(_FMT)
        log.addHandler(_h)
    return log
