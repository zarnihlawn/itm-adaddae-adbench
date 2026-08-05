#!/usr/bin/env python3
"""Loop 8: confirm ADBench semi train is normals-only (paper-faithful).

Writes thesis note; exits 0 if split_data matches AnoDDAE (train normals only).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import split_data  # noqa: E402


def main() -> int:
    rng = np.random.RandomState(0)
    X = rng.randn(200, 4).astype(np.float32)
    y = np.zeros(200, dtype=np.int64)
    y[180:] = 1
    Xtr, Xte, ytr, yte = split_data(X, y, train_setting="semi-supervised", random_state=111)
    train_anomaly_rate = float((ytr == 1).mean()) if len(ytr) else 0.0
    ok = train_anomaly_rate == 0.0 and int((yte == 1).sum()) == int((y == 1).sum())
    report = {
        "pass": ok,
        "train_n": int(len(ytr)),
        "test_n": int(len(yte)),
        "train_anomaly_rate": train_anomaly_rate,
        "test_anomalies": int((yte == 1).sum()),
        "all_anomalies_in_test": bool(int((yte == 1).sum()) == int((y == 1).sum())),
        "conclusion": (
            "Semi train is normals-only (paper-faithful). No labeled anomalies in train; "
            "do not invent semi-supervised losses on train labels. Improve via better "
            "normal modeling (horizon, robust scaler, RDT, selective A6)."
        ),
    }
    out = PROJECT_ROOT / "results/adadae_per/thesis/loop8_semi_labels.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
