#!/usr/bin/env python3
"""ADBench regime audit → CSV for thesis / AdaDDAE-4 meta gates."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data.datasets import build_registry, load_npz
from src.models.danc import estimate_meta_features


def _cluster_sep(X: np.ndarray, max_n: int = 2000) -> float:
    from sklearn.cluster import MiniBatchKMeans

    n = X.shape[0]
    rng = np.random.RandomState(0)
    idx = rng.choice(n, min(max_n, n), replace=False)
    Xs = X[idx].astype(np.float64)
    Xs = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-8)
    if Xs.shape[0] < 8:
        return 0.0
    km = MiniBatchKMeans(n_clusters=2, random_state=0, n_init=3, batch_size=256)
    lab = km.fit_predict(Xs)
    c0 = Xs[lab == 0].mean(0)
    c1 = Xs[lab == 1].mean(0)
    sep = float(np.linalg.norm(c0 - c1))
    w = float(((Xs[lab == 0] - c0) ** 2).sum() + ((Xs[lab == 1] - c1) ** 2).sum())
    return sep / (np.sqrt(w / len(Xs)) + 1e-8)


def regime_tags(n: int, d: int, c: float, nnz: float, cluster_sep: float, skew: float) -> str:
    tags = []
    if n < 300:
        tags.append("tiny")
    if n >= 50_000:
        tags.append("huge")
    if d >= 128:
        tags.append("highd")
    if c >= 0.2:
        tags.append("heavy")
    if c <= 0.02:
        tags.append("rare")
    if nnz < 0.3:
        tags.append("sparse")
    if cluster_sep > 1.5:
        tags.append("multi")
    if skew >= 5.0:
        tags.append("skew")
    return ",".join(tags) if tags else "mid"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/adadae_final.yaml")
    p.add_argument("--out", default="results/thesis/adbench_regimes.csv")
    args = p.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["paths"]["adbench_root"])
    specs = build_registry(root)

    rows = []
    for spec in specs:
        rel = spec.relative_paths[0]
        path = root / rel
        if not path.exists():
            continue
        X, y = load_npz(path)
        n, d = X.shape
        c_true = float(y.mean())
        nnz = float(np.count_nonzero(X) / max(X.size, 1))
        meta = estimate_meta_features(X, y, contamination_mode="oracle")
        meta_lf = estimate_meta_features(X, contamination_mode="label_free")
        try:
            csep = _cluster_sep(X)
        except Exception:
            csep = 0.0
        tags = regime_tags(n, d, c_true, nnz, csep, float(meta["skewness"]))
        rows.append(
            {
                "name": spec.name,
                "category": spec.category,
                "n": n,
                "d": d,
                "contam_true": f"{c_true:.6f}",
                "contam_lf": f"{float(meta_lf['contamination']):.6f}",
                "idim": f"{float(meta['intrinsic_dim']):.4f}",
                "skew": f"{float(meta['skewness']):.4f}",
                "nnz": f"{nnz:.4f}",
                "cluster_sep": f"{csep:.4f}",
                "n_files": len(spec.relative_paths),
                "regime_tags": tags,
            }
        )

    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows → {out}")
    # quick counts
    from collections import Counter

    cnt = Counter()
    for r in rows:
        for t in r["regime_tags"].split(","):
            cnt[t] += 1
    print("tag counts:", dict(cnt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
