"""Logging helper used by the Python port.

Provides:
- `Logger`: lightweight wrapper around `logging` that emits compact JSON-like
    event objects (or writes to stdout/stderr/file) controlled by `LOG_<NAME>`
    environment variables.
- `log(logger, msg, **kwargs)`: convenience function mirroring Rust's
    `log!` macro used throughout the codebase.

Usage:
- Configure `LOG_MAIN=stdout` (or file path) to enable a named logger.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any


def now() -> int:
    return int(datetime.now().timestamp() * 1e9)


class Logger:
    def __init__(self, name: str):
        self.name = name
        env_key = f"LOG_{name}"
        target = os.environ.get(env_key, "").strip()
        self._enabled = bool(target)
        if not self._enabled:
            self._logger = None
            self._buf = None
            return

        logger_name = f"vrpr.{name}"
        self._logger = logging.getLogger(logger_name)
        # avoid duplicated handlers
        for h in list(self._logger.handlers):
            self._logger.removeHandler(h)

        if target == "stdout":
            handler = logging.StreamHandler(sys.stdout)
        elif target == "stderr":
            handler = logging.StreamHandler(sys.stderr)
        else:
            handler = logging.FileHandler(target, encoding="utf-8")

        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._buf = None

    def begin(self) -> None:
        if not self._enabled:
            return
        self._buf = {"__": self.name, "_t": now()}

    def end(self) -> None:
        if not self._enabled or self._buf is None:
            return
        try:
            self._logger.info(json.dumps(self._buf, default=str))
        finally:
            self._buf = None

    def log_message(self, msg: Any) -> None:
        if not self._enabled or self._buf is None:
            return
        self._buf["_"] = str(msg)

    def log_key_value(self, key: str, value: Any, comma: bool) -> None:
        if not self._enabled or self._buf is None:
            return
        # store value; serialization done in end()
        try:
            json.dumps(value)
            self._buf[key] = value
        except TypeError:
            self._buf[key] = str(value)

    def log(self, value: Any) -> None:
        if not self._enabled:
            return
        v = f"{datetime.now().strftime('%H:%M:%S.%f')},{self.name},{value}"
        self._logger.info(v)

    def enabled(self) -> bool:
        return self._enabled


def log(logger: Logger, msg: Any, **kwargs) -> None:
    """Convenience function that mirrors the Rust `log!` macro usage.

    Usage: log(logger, "event", key=value, ...)
    """
    if not logger.enabled():
        return
    logger.begin()
    logger.log_message(msg)
    for k, v in kwargs.items():
        logger.log_key_value(k, v, comma=True)
    logger.end()


__all__ = ["Logger", "log"]
