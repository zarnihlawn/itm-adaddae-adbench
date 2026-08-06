"""AXION — Adaptive cross-feature Interaction Observation Network.

Components: MCB + FX-Enc + HPD + LATCH + MCS (+ SCALE).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from axion.models.axion_net import AxionNet, gaussian_nll
from axion.models.mcb import sample_masks, scale_hparams


class AxionModel:
    """Masked heteroscedastic recon + latent Mahalanobis anomaly scorer."""

    name = "axion"

    def __init__(
        self,
        hidden: Optional[int] = None,
        latent: Optional[int] = None,
        depth: Optional[int] = None,
        mask_rates: Optional[Sequence[float]] = None,
        score_k: Optional[int] = None,
        dropout: Optional[float] = None,
        epochs: int = 80,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 12,
        val_fraction: float = 0.15,
        latch_alpha: float = 0.4,
        latch_alpha_semi: Optional[float] = 0.25,
        mae_weight: float = 0.6,
        nll_weight: float = 0.4,
        device: Optional[str] = None,
        seed: int = 111,
        max_train_samples: int = 0,
    ):
        self.hidden = hidden
        self.latent = latent
        self.depth = depth
        self.mask_rates = tuple(mask_rates) if mask_rates is not None else None
        self.score_k = score_k
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_fraction = val_fraction
        self.latch_alpha = latch_alpha
        self.latch_alpha_semi = latch_alpha_semi
        self.mae_weight = mae_weight
        self.nll_weight = nll_weight
        self.seed = seed
        self.max_train_samples = max_train_samples

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.net: Optional[AxionNet] = None
        self.z_mean_: Optional[np.ndarray] = None
        self.z_inv_var_: Optional[np.ndarray] = None
        self.resolved_: Dict[str, Any] = {}
        self.best_val_loss_: float = float("inf")
        self.active_latch_alpha_: float = float(latch_alpha)
        self.mcs_mean_: Optional[float] = None
        self.mcs_std_: Optional[float] = None
        self.latch_score_mean_: Optional[float] = None
        self.latch_score_std_: Optional[float] = None

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "resolved": self.resolved_,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "latch_alpha": self.latch_alpha,
            "latch_alpha_semi": self.latch_alpha_semi,
            "active_latch_alpha": self.active_latch_alpha_,
            "mae_weight": self.mae_weight,
            "nll_weight": self.nll_weight,
            "device": str(self.device),
            "best_val_loss": self.best_val_loss_,
            "train_anchored": self.mcs_mean_ is not None,
        }

    def _resolve_scale(self, n: int, d: int) -> Dict[str, Any]:
        base = scale_hparams(n, d)
        return {
            "hidden": int(self.hidden if self.hidden is not None else base["hidden"]),
            "latent": int(self.latent if self.latent is not None else base["latent"]),
            "depth": int(self.depth if self.depth is not None else base["depth"]),
            "mask_rates": tuple(self.mask_rates if self.mask_rates is not None else base["mask_rates"]),
            "score_k": int(self.score_k if self.score_k is not None else base["score_k"]),
            "dropout": float(self.dropout if self.dropout is not None else base["dropout"]),
            "high_mask_delta": float(base.get("high_mask_delta", 0.25)),
            "high_mask_cap": float(base.get("high_mask_cap", 0.85)),
        }

    def _torch_gen(self) -> torch.Generator:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(self.seed))
        return g

    def _rate_banks(self) -> List[Tuple[float, ...]]:
        rates = tuple(self.resolved_.get("mask_rates", (0.2, 0.35, 0.5)))
        delta = float(self.resolved_.get("high_mask_delta", 0.25))
        cap = float(self.resolved_.get("high_mask_cap", 0.85))
        high = float(min(cap, max(rates) + delta))
        return [rates, (high,)]

    def _resolve_active_latch(self, y_train: Optional[np.ndarray]) -> None:
        """Use lower LATCH weight when train is all-normal (paper semi split)."""
        self.active_latch_alpha_ = float(self.latch_alpha)
        if self.latch_alpha_semi is None or y_train is None:
            return
        y = np.asarray(y_train).reshape(-1)
        if y.size > 0 and np.all(y == 0):
            self.active_latch_alpha_ = float(self.latch_alpha_semi)

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "AxionModel":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = np.asarray(X_train, dtype=np.float32)
        if self.max_train_samples > 0 and X.shape[0] > self.max_train_samples:
            rng = np.random.RandomState(self.seed)
            idx = rng.choice(X.shape[0], size=self.max_train_samples, replace=False)
            X = X[idx]
            if y_train is not None:
                y_train = np.asarray(y_train)[idx]

        n, d = X.shape
        hp = self._resolve_scale(n, d)
        self.resolved_ = dict(hp)
        self.resolved_["n"] = n
        self.resolved_["d"] = d

        # Train/val carve for early stop (never uses test)
        rng = np.random.RandomState(self.seed)
        if self.val_fraction > 0 and n >= 40:
            n_val = max(8, int(round(n * self.val_fraction)))
            n_val = min(n_val, n // 5 if n >= 50 else max(4, n // 4))
            perm = rng.permutation(n)
            val_idx, fit_idx = perm[:n_val], perm[n_val:]
            X_fit, X_val = X[fit_idx], X[val_idx]
        else:
            X_fit, X_val = X, X[:0]

        self.net = AxionNet(
            d_in=d,
            hidden=hp["hidden"],
            latent=hp["latent"],
            depth=hp["depth"],
            dropout=hp["dropout"],
        ).to(self.device)

        opt = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        fit_ds = TensorDataset(torch.from_numpy(X_fit))
        # GPU: allow larger batches for throughput
        req_bs = self.batch_size
        if self.device.type == "cuda":
            req_bs = max(req_bs, 512)
        bs = min(req_bs, max(8, len(X_fit)))
        loader = DataLoader(
            fit_ds,
            batch_size=bs,
            shuffle=True,
            drop_last=False,
            pin_memory=(self.device.type == "cuda"),
            num_workers=0,
        )

        best_state = None
        best_val = float("inf")
        stale = 0
        gen = self._torch_gen()
        use_amp = self.device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        except (TypeError, AttributeError):
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        for epoch in range(self.epochs):
            self.net.train()
            last_loss = 0.0
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                mask = sample_masks(
                    xb.shape[0],
                    d,
                    rates=hp["mask_rates"],
                    generator=gen,
                    device=self.device,
                )
                opt.zero_grad(set_to_none=True)
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        mu, log_var, _z = self.net(xb, mask)
                    loss = gaussian_nll(xb, mu.float(), log_var.float(), mask).mean()
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    mu, log_var, _z = self.net(xb, mask)
                    loss = gaussian_nll(xb, mu, log_var, mask).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                    opt.step()
                last_loss = float(loss.detach().cpu().item())

            # Val loss on fixed random masks (reproducible via seed+epoch)
            if len(X_val) > 0:
                val_loss = self._eval_nll(X_val, hp["mask_rates"], seed_offset=epoch)
            else:
                val_loss = last_loss

            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val_loss_ = float(best_val)

        # LATCH: fit diagonal Mahalanobis on unmasked (mask=0) latents of train
        self._fit_latch(X_fit)
        self._resolve_active_latch(y_train)
        self._fit_score_anchors(X_fit)
        return self

    @torch.no_grad()
    def _eval_nll(self, X: np.ndarray, rates: Sequence[float], seed_offset: int = 0) -> float:
        assert self.net is not None
        self.net.eval()
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + 10_000 + seed_offset)
        xb = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        losses = []
        bs = min(512, len(X))
        for i in range(0, len(X), bs):
            chunk = xb[i : i + bs]
            mask = sample_masks(chunk.shape[0], chunk.shape[1], rates=rates, generator=g, device=self.device)
            mu, log_var, _ = self.net(chunk, mask)
            losses.append(gaussian_nll(chunk, mu, log_var, mask).mean().item())
        return float(np.mean(losses))

    @torch.no_grad()
    def _encode_visible(self, X: np.ndarray) -> np.ndarray:
        assert self.net is not None
        self.net.eval()
        xb = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        # Fully visible: mask = 0
        mask = torch.zeros_like(xb)
        zs = []
        bs = min(512, len(X))
        for i in range(0, len(X), bs):
            _, z = self.net.encoder(xb[i : i + bs], mask[i : i + bs])
            zs.append(z.cpu().numpy())
        return np.concatenate(zs, axis=0)

    def _fit_latch(self, X: np.ndarray) -> None:
        z = self._encode_visible(X)
        self.z_mean_ = z.mean(axis=0)
        var = z.var(axis=0)
        self.z_inv_var_ = 1.0 / np.maximum(var, 1e-6)

    def _latch_score(self, X: np.ndarray) -> np.ndarray:
        if self.z_mean_ is None or self.z_inv_var_ is None:
            return np.zeros(X.shape[0], dtype=np.float64)
        z = self._encode_visible(X)
        diff = z - self.z_mean_
        return np.sum((diff ** 2) * self.z_inv_var_, axis=1)

    @torch.no_grad()
    def _mcs_scores(self, X: np.ndarray) -> np.ndarray:
        """Raw MCS: hybrid MAE+NLL over K masks × dual rate banks."""
        assert self.net is not None
        self.net.eval()
        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape
        k = int(self.resolved_.get("score_k", 8))
        rate_banks = self._rate_banks()

        xb = torch.from_numpy(X).to(self.device)
        acc = np.zeros(n, dtype=np.float64)
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + 12345)

        bs = min(512, n)
        n_passes = 0
        for bank in rate_banks:
            for _ in range(k):
                n_passes += 1
                for i in range(0, n, bs):
                    chunk = xb[i : i + bs]
                    mask = sample_masks(
                        chunk.shape[0], d, rates=bank, generator=g, device=self.device
                    )
                    mu, log_var, _ = self.net(chunk, mask)
                    nll = gaussian_nll(chunk, mu, log_var, mask)
                    mae = ((chunk - mu).abs() * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
                    hybrid = self.mae_weight * mae + self.nll_weight * nll
                    acc[i : i + chunk.shape[0]] += hybrid.cpu().numpy()

        return acc / float(max(1, n_passes))

    def _fit_score_anchors(self, X: np.ndarray) -> None:
        """Train-only mean/std for MCS and LATCH (avoid test-batch z-norm)."""
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        # Cap anchor compute on huge tables
        if n > 8000:
            rng = np.random.RandomState(self.seed + 7)
            idx = rng.choice(n, size=8000, replace=False)
            X = X[idx]
        mcs = self._mcs_scores(X)
        latch = self._latch_score(X)
        self.mcs_mean_ = float(mcs.mean())
        self.mcs_std_ = float(max(float(mcs.std()), 1e-8))
        self.latch_score_mean_ = float(latch.mean())
        self.latch_score_std_ = float(max(float(latch.std()), 1e-8))

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        """MCS + α · LATCH with train-anchored normalization when available."""
        if self.net is None:
            raise RuntimeError("AxionModel not fitted")
        X = np.asarray(X, dtype=np.float32)
        mcs = self._mcs_scores(X)
        latch = self._latch_score(X)
        alpha = float(self.active_latch_alpha_)

        if self.mcs_mean_ is not None and self.mcs_std_ is not None:
            mcs_n = (mcs - self.mcs_mean_) / (self.mcs_std_ + 1e-8)
            latch_n = (latch - float(self.latch_score_mean_ or 0.0)) / (
                float(self.latch_score_std_ or 1.0) + 1e-8
            )
            return (mcs_n + alpha * latch_n).astype(np.float64)

        # Fallback: batch z-norm (legacy; should not run after fit)
        if latch.std() > 1e-8 and mcs.std() > 1e-8:
            latch_n = (latch - latch.mean()) / (latch.std() + 1e-8)
            mcs_n = (mcs - mcs.mean()) / (mcs.std() + 1e-8)
            return (mcs_n + alpha * latch_n).astype(np.float64)
        return (mcs + alpha * latch).astype(np.float64)
