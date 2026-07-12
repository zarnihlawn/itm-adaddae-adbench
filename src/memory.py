"""Memory helpers for CPU-safe and GPU VRAM-safe AdaDDAE training."""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Optional, Union

import psutil
import torch


@dataclass
class MemoryGuard:
    soft_limit_mb: float = 8192.0
    check_every_n_batches: int = 20
    _batch_counter: int = 0
    last_rss_mb: float = 0.0

    def rss_mb(self) -> float:
        proc = psutil.Process(os.getpid())
        self.last_rss_mb = proc.memory_info().rss / (1024 * 1024)
        return self.last_rss_mb

    def maybe_check(self) -> tuple[float, bool]:
        self._batch_counter += 1
        if self._batch_counter % max(1, self.check_every_n_batches) != 0:
            return self.last_rss_mb, False
        rss = self.rss_mb()
        return rss, rss > self.soft_limit_mb

    def force_check(self) -> tuple[float, bool]:
        rss = self.rss_mb()
        return rss, rss > self.soft_limit_mb


@dataclass
class VRAMGuard:
    soft_limit_mb: float = 6800.0
    check_every_n_batches: int = 10
    _batch_counter: int = 0
    last_vram_mb: float = 0.0

    def reset_peak(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def vram_mb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        self.last_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return self.last_vram_mb

    def maybe_check(self) -> tuple[float, bool]:
        if not torch.cuda.is_available():
            return 0.0, False
        self._batch_counter += 1
        if self._batch_counter % max(1, self.check_every_n_batches) != 0:
            return self.last_vram_mb, False
        vram = self.vram_mb()
        return vram, vram > self.soft_limit_mb

    def force_check(self) -> tuple[float, bool]:
        if not torch.cuda.is_available():
            return 0.0, False
        vram = self.vram_mb()
        return vram, vram > self.soft_limit_mb


GuardType = Union[MemoryGuard, VRAMGuard]


def create_guard(hw: dict, device: torch.device) -> GuardType:
    if device.type == "cuda":
        return VRAMGuard(
            soft_limit_mb=float(hw.get("vram_soft_limit_mb", 6800)),
            check_every_n_batches=int(hw.get("rss_check_every_n_batches", 10)),
        )
    return MemoryGuard(
        soft_limit_mb=float(hw.get("rss_soft_limit_mb", 8192)),
        check_every_n_batches=int(hw.get("rss_check_every_n_batches", 20)),
    )


def guard_memory_mb(guard: GuardType) -> float:
    if isinstance(guard, VRAMGuard):
        return guard.vram_mb()
    return guard.rss_mb()


def guard_over_limit(guard: GuardType) -> tuple[float, bool]:
    return guard.maybe_check()


def apply_thread_limits(num_threads: int = 8) -> None:
    num_threads = max(1, int(num_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_threads))
    torch.set_num_threads(num_threads)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(min(4, num_threads))
        except RuntimeError:
            pass


def setup_cuda(hw: dict) -> None:
    if not torch.cuda.is_available():
        return
    if hw.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True


def resolve_amp_dtype(name: str) -> torch.dtype:
    if name == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def choose_train_batch_size(
    n_samples: int,
    max_batch: int = 512,
    large_n_threshold: int = 100_000,
) -> int:
    if n_samples <= 0:
        return 32
    target = max(8, n_samples // 10)
    powers = [2**i for i in range(3, 14)]
    batch = min(powers, key=lambda x: abs(x - target))
    batch = min(batch, max_batch)
    if n_samples > large_n_threshold:
        batch = min(batch, max_batch // 2)
    return int(batch)


def choose_score_batch_size(
    n_samples: int,
    max_batch: int = 1024,
    large_n_threshold: int = 100_000,
    large_batch: int = 256,
) -> int:
    if n_samples > large_n_threshold:
        return min(max_batch, large_batch)
    return min(max_batch, 8192)


def shrink_batch(batch_size: int, min_batch: int = 8) -> int:
    return max(min_batch, batch_size // 2)


def cleanup_memory(device: Optional[torch.device] = None) -> None:
    gc.collect()
    if device is not None and device.type == "cuda":
        torch.cuda.empty_cache()
