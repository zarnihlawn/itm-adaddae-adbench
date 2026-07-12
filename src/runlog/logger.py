"""Structured JSONL + console progress logging."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm


class RunLogger:
    def __init__(self, log_path: Path, run_id: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._fh = open(self.log_path, "a", encoding="utf-8")

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()

    def log(self, event: str, **fields: Any) -> None:
        payload = {
            "time": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event,
            **fields,
        }
        self._fh.write(json.dumps(payload, default=str) + "\n")
        self._fh.flush()

    def info(self, msg: str, **fields: Any) -> None:
        self.log("info", message=msg, **fields)
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        print(f"[{self.run_id}] {msg}" + (f" | {extra}" if extra else ""))


def epoch_progress(total: int, desc: str = "epochs"):
    return tqdm(range(total), desc=desc, leave=True)


def job_progress(total: int, desc: str = "protocol"):
    return tqdm(total=total, desc=desc, leave=True)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0
