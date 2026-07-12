"""AdaDDAE: Adaptive Diffusion-Scheduled Denoising Autoencoder (v3)."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..memory import (
    GuardType,
    MemoryGuard,
    VRAMGuard,
    choose_score_batch_size,
    guard_memory_mb,
    guard_over_limit,
    resolve_amp_dtype,
    shrink_batch,
)
from ..runlog.logger import RunLogger
from .danc import NoiseConfig
from .network import DiffusionBottleneckAE
from .scheduler import DiffusionScheduler
from .scs import select_timesteps, timestep_weights, vectorized_q_sample
from .dte import (
    build_latent_memory,
    fuse_dte_scores,
    knn_dte_score,
    posterior_mean_from_recon,
)


class AdaDDAE:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 32,
        activation: str = "lrelu",
        noise_config: Optional[NoiseConfig] = None,
        time_emb_type: str = "sinusoidal",
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        device: Optional[torch.device] = None,
        eval_every: int = 10,
        contrastive: bool = True,
        contrastive_alpha: float = 0.2,
        contrastive_gamma: float = 1.0,
        contrastive_adaptive_alpha: bool = True,
        contrastive_pairing: str = "taps",
        hard_negative_mining: bool = True,
        use_scs: bool = True,
        scs_max_timesteps: int = 32,
        scs_mode: str = "snr_weighted",
        scs_selection: str = "snr_stratified",
        use_multiview: bool = True,
        use_uncertainty_view: bool = False,
        uncertainty_draws: int = 3,
        use_dte_view: bool = True,
        dte_knn: int = 5,
        dte_memory_size: int = 4096,
        dte_knn_weight: float = 0.5,
        use_rejection_training: bool = True,
        rejection_quantile: float = 0.95,
        rejection_min_weight: float = 0.1,
        rejection_warmup_epochs: int = 1,
        fusion_mode: str = "fixed",
        fusion_weights: Optional[Dict[str, float]] = None,
        setting: str = "semi-supervised",
        score_batch_size: int = 1024,
        memory_guard: Optional[GuardType] = None,
        early_stop_patience: int = 20,
        use_amp: bool = False,
        amp_dtype: str = "bfloat16",
        pin_memory: bool = False,
        num_workers: int = 0,
        vectorized_scoring: bool = False,
        preupload_test_threshold: int = 50000,
        y_train: Optional[torch.Tensor] = None,
    ):
        self.input_dim = input_dim
        self.setting = setting
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.eval_every = eval_every
        self.contrastive = contrastive
        self.contrastive_alpha = contrastive_alpha
        self.contrastive_gamma = contrastive_gamma
        self.contrastive_adaptive_alpha = contrastive_adaptive_alpha
        self.contrastive_pairing = contrastive_pairing
        self.hard_negative_mining = hard_negative_mining
        self.use_scs = use_scs
        self.scs_max_timesteps = scs_max_timesteps
        self.scs_mode = scs_mode
        self.scs_selection = scs_selection
        self.use_multiview = use_multiview
        self.use_uncertainty_view = use_uncertainty_view and (device or torch.device("cpu")).type == "cuda"
        self.uncertainty_draws = max(2, int(uncertainty_draws))
        self.use_dte_view = use_dte_view
        self.dte_knn = max(1, int(dte_knn))
        self.dte_memory_size = int(dte_memory_size)
        self.dte_knn_weight = float(dte_knn_weight)
        self.use_rejection_training = use_rejection_training
        self.rejection_quantile = float(rejection_quantile)
        self.rejection_min_weight = float(rejection_min_weight)
        self.rejection_warmup_epochs = max(1, int(rejection_warmup_epochs))
        self.fusion_mode = fusion_mode
        self.fusion_weights = fusion_weights or {
            "reconstruction": 0.6,
            "latent": 0.3,
            "residual": 0.1,
            "uncertainty": 0.1,
            "diffusion_time": 0.1,
        }
        self.device = device or torch.device("cpu")
        self.score_batch_size = score_batch_size
        self.memory_guard = memory_guard or MemoryGuard()
        self.early_stop_patience = early_stop_patience
        self.use_amp = use_amp and self.device.type == "cuda"
        self.amp_dtype = resolve_amp_dtype(amp_dtype) if self.use_amp else torch.float32
        self.pin_memory = pin_memory and self.device.type == "cuda"
        self.num_workers = num_workers
        self.vectorized_scoring = vectorized_scoring and self.device.type == "cuda"
        self.preupload_test_threshold = preupload_test_threshold
        self._calibrated_fusion: Optional[Dict[str, float]] = None
        self._normal_pool: Optional[torch.Tensor] = None
        self._train_z_cache: Optional[torch.Tensor] = None
        self._cached_t_grid: Optional[torch.Tensor] = None
        self._cached_score_weights: Optional[torch.Tensor] = None
        self._dte_memory: Optional[torch.Tensor] = None
        self._sample_weights: Optional[torch.Tensor] = None
        self.timing: Dict[str, float] = {}

        if y_train is not None and setting == "semi-supervised":
            normal_mask = (y_train == 0)
            if normal_mask.any():
                self._normal_pool = torch.where(normal_mask)[0]

        nc = noise_config or NoiseConfig(50, "linear", 1e-4, 0.02, 8)
        self.noise_config = nc
        self.T = nc.num_timesteps

        self.model = DiffusionBottleneckAE(
            input_dim=input_dim,
            hidden_dims=hidden_dims or [512, 512],
            latent_dim=latent_dim,
            time_emb_dim=nc.time_emb_dim,
            activation=activation,
            time_emb_type=time_emb_type,
        ).to(self.device)

        self.scheduler = DiffusionScheduler(
            num_timesteps=nc.num_timesteps,
            device=self.device,
            beta_start=nc.beta_start,
            beta_end=nc.beta_end,
            scheduler=nc.scheduler,
        )

        selection = scs_selection if use_scs and scs_mode != "full_sum" else "linspace"
        if use_scs and scs_mode != "full_sum":
            self.score_timesteps = select_timesteps(
                self.T,
                setting,
                scs_max_timesteps,
                mode=scs_mode,
                selection=selection,
                alpha_bar=self.scheduler.alpha_bar,
            )
        elif scs_mode == "full_sum":
            self.score_timesteps = list(range(1, self.T))
        else:
            self.score_timesteps = list(range(1, self.T))

        self.score_weights = timestep_weights(
            self.score_timesteps,
            setting,
            self.T,
            mode=scs_mode,
            alpha_bar=self.scheduler.alpha_bar,
        )
        self._cache_score_tensors()

        self._scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and self.amp_dtype == torch.float16,
        )

    def _cache_score_tensors(self) -> None:
        if self.device.type == "cuda":
            self._cached_t_grid = torch.tensor(
                self.score_timesteps, device=self.device, dtype=torch.long
            )
            self._cached_score_weights = torch.tensor(
                self.score_weights, device=self.device, dtype=torch.float32
            )

    def _snr_weight_at_t(self, t: torch.Tensor) -> torch.Tensor:
        """Per-sample SNR weight for TAPS contrastive scaling."""
        ab = self.scheduler.alpha_bar
        idx = t.clamp(0, ab.numel() - 1)
        ab_t = ab[idx]
        if self.setting == "unsupervised":
            w = 1.0 - ab_t
        else:
            w = ab_t
        return w.clamp(min=1e-6)

    def _sample_positive_z(self, z0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """TAPS: semi normal-only positives from cache; unsupervised same-batch pairing."""
        bsz = z0.size(0)
        if self.contrastive_pairing == "random":
            return z0[torch.randperm(bsz, device=self.device)]

        if (
            self.setting == "semi-supervised"
            and self._train_z_cache is not None
            and self._train_z_cache.numel() > 0
        ):
            pool_pos = torch.randint(0, self._train_z_cache.size(0), (bsz,), device=self.device)
            return self._train_z_cache[pool_pos]

        return z0

    def _contrastive_loss_ddae_c(
        self,
        z0: torch.Tensor,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_pos: torch.Tensor,
    ) -> torch.Tensor:
        """DDAE-C Eq. 7: margin m = 1 + gamma * t/T."""
        delta_neg = torch.norm(z0 - zt, dim=1)
        m = 1.0 + self.contrastive_gamma * (t.float() / max(self.T, 1))
        loss_neg = torch.clamp(m - delta_neg, min=0.0) ** 2

        delta_pos = torch.norm(z0 - z_pos, dim=1)
        loss_pos = delta_pos**2

        if self.hard_negative_mining:
            hard_w = torch.softmax(-delta_neg.detach(), dim=0) * z0.size(0)
            loss_neg = loss_neg * hard_w

        return (loss_pos + loss_neg).mean()

    def _effective_contrastive_alpha(self, t: torch.Tensor) -> float:
        if not self.contrastive_adaptive_alpha:
            base = self.contrastive_alpha
        else:
            base = self.contrastive_alpha * (t.float().mean() / max(self.T, 1))
        snr_w = self._snr_weight_at_t(t).mean()
        alpha_t = base * snr_w
        return float(alpha_t.clamp(0.0, 0.5).item())

    def _train_step(
        self,
        x_0: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        sample_idx: Optional[torch.Tensor] = None,
    ) -> float:
        bsz = x_0.size(0)
        t = torch.randint(1, self.T, (bsz,), device=self.device).long()
        t0 = torch.ones(bsz, device=self.device, dtype=torch.long)

        batch_w = None
        if self._sample_weights is not None and sample_idx is not None:
            # _sample_weights stays on CPU; indices must match indexed tensor device.
            batch_w = self._sample_weights[sample_idx.cpu()].to(self.device)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            x_t, _ = self.scheduler.q_sample(x_0, t)
            x0_hat, zt = self.model(x_t, t)
            _, z0 = self.model(x_0, t0)

            rec_ps = nn.functional.mse_loss(x0_hat, x_0, reduction="none").mean(dim=1)

            if self.contrastive and self.contrastive_alpha > 0:
                z_pos = self._sample_positive_z(z0, t)
                cont_ps = self._contrastive_loss_per_sample(z0, zt, t, z_pos)
                alpha_eff = self._effective_contrastive_alpha(t)
                loss_ps = (1 - alpha_eff) * rec_ps + alpha_eff * cont_ps
            else:
                loss_ps = rec_ps

            if batch_w is not None:
                loss = (loss_ps * batch_w).sum() / batch_w.sum().clamp(min=1e-8)
            else:
                loss = loss_ps.mean()

        optimizer.zero_grad(set_to_none=True)
        if self._scaler.is_enabled():
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

        return float(loss.item())

    def _contrastive_loss_per_sample(
        self,
        z0: torch.Tensor,
        zt: torch.Tensor,
        t: torch.Tensor,
        z_pos: torch.Tensor,
    ) -> torch.Tensor:
        delta_neg = torch.norm(z0 - zt, dim=1)
        m = 1.0 + self.contrastive_gamma * (t.float() / max(self.T, 1))
        loss_neg = torch.clamp(m - delta_neg, min=0.0) ** 2
        delta_pos = torch.norm(z0 - z_pos, dim=1)
        loss_pos = delta_pos**2
        if self.hard_negative_mining:
            hard_w = torch.softmax(-delta_neg.detach(), dim=0) * z0.size(0)
            loss_neg = loss_neg * hard_w
        return loss_pos + loss_neg

    def _build_normal_z_cache(self, x_train: torch.Tensor) -> None:
        if self._normal_pool is None or self._normal_pool.numel() == 0:
            return
        self.model.eval()
        with torch.no_grad():
            x_norm = x_train[self._normal_pool].to(self.device)
            t0 = torch.ones(x_norm.size(0), device=self.device, dtype=torch.long)
            _, z = self.model(x_norm, t0)
            self._train_z_cache = z.detach()
        self.model.train()

    @torch.inference_mode()
    def _compute_rdt_weights(self, x_train: torch.Tensor) -> torch.Tensor:
        """RDT: down-weight train points with suspiciously high clean reconstruction loss."""
        self.model.eval()
        n = x_train.size(0)
        losses = []
        bs = min(self.batch_size, 512)
        t1 = torch.ones(1, device=self.device, dtype=torch.long)
        for i in range(0, n, bs):
            xb = x_train[i : i + bs].to(self.device)
            x0_hat, _ = self.model(xb, t1.expand(xb.size(0)))
            rec = nn.functional.mse_loss(x0_hat, xb, reduction="none").mean(dim=1)
            losses.append(rec.cpu())
        all_rec = torch.cat(losses, dim=0)
        tau = torch.quantile(all_rec, self.rejection_quantile)
        weights = torch.where(
            all_rec <= tau,
            torch.ones_like(all_rec),
            torch.full_like(all_rec, self.rejection_min_weight),
        )
        self.model.train()
        return weights

    @torch.inference_mode()
    def _build_dte_memory(self, x_train: torch.Tensor) -> None:
        """Build latent memory bank for kNN DTE scoring."""
        self.model.eval()
        zs = []
        bs = min(self.batch_size, 512)
        t0 = torch.ones(1, device=self.device, dtype=torch.long)
        if self._normal_pool is not None and self._normal_pool.numel() > 0:
            src = x_train[self._normal_pool]
        else:
            src = x_train
        for i in range(0, src.size(0), bs):
            xb = src[i : i + bs].to(self.device)
            _, z = self.model(xb, t0.expand(xb.size(0)))
            zs.append(z)
        z_all = torch.cat(zs, dim=0)
        self._dte_memory = build_latent_memory(z_all, max_samples=self.dte_memory_size)
        self.model.train()

    def fit(
        self,
        x_train: torch.Tensor,
        x_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
        logger: Optional[RunLogger] = None,
        eval_fn=None,
    ) -> Dict[str, Any]:
        t_train_start = time.perf_counter()
        if isinstance(self.memory_guard, VRAMGuard):
            self.memory_guard.reset_peak()

        if self.setting == "semi-supervised" and self.contrastive_pairing == "taps":
            self._build_normal_z_cache(x_train)

        if self.setting == "semi-supervised" and self.contrastive_pairing == "taps":
            self._build_normal_z_cache(x_train)

        use_rdt = self.use_rejection_training
        use_idx = use_rdt
        if use_idx:
            dataset = TensorDataset(x_train, torch.arange(x_train.size(0)))
        else:
            dataset = TensorDataset(x_train)

        optimizer = Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

        best_metric = -1.0
        best_state = None
        patience = 0
        history = []
        cur_batch = self.batch_size

        epoch_bar = tqdm(range(self.epochs), desc="train", leave=False)
        for epoch in epoch_bar:
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for batch in loader:
                if use_idx:
                    x_0, batch_idx = batch
                    # Keep indices on CPU for RDT weight lookup.
                else:
                    x_0 = batch[0]
                    batch_idx = None

                if self.pin_memory:
                    x_0 = x_0.to(self.device, non_blocking=True)
                else:
                    x_0 = x_0.to(self.device)

                total_loss += self._train_step(x_0, optimizer, sample_idx=batch_idx)
                n_batches += 1

                mem, over = guard_over_limit(self.memory_guard)
                if over:
                    new_bs = shrink_batch(cur_batch)
                    if new_bs < cur_batch and logger:
                        logger.log(
                            "memory_shrink",
                            mem_mb=mem,
                            vram_mb=mem if isinstance(self.memory_guard, VRAMGuard) else None,
                            rss_mb=mem if isinstance(self.memory_guard, MemoryGuard) else None,
                            old_batch=cur_batch,
                            new_batch=new_bs,
                        )
                    cur_batch = new_bs
                    self.batch_size = new_bs

            avg_loss = total_loss / max(n_batches, 1)
            mem = guard_memory_mb(self.memory_guard)
            rss = self.memory_guard.rss_mb() if isinstance(self.memory_guard, MemoryGuard) else None
            vram = self.memory_guard.vram_mb() if isinstance(self.memory_guard, VRAMGuard) else None
            postfix = {"loss": f"{avg_loss:.4f}"}
            if vram is not None:
                postfix["vram_mb"] = f"{vram:.0f}"
            if rss is not None:
                postfix["rss_mb"] = f"{rss:.0f}"
            epoch_bar.set_postfix(**postfix)

            metrics = {}
            if (
                x_test is not None
                and y_test is not None
                and eval_fn is not None
                and (epoch + 1) % self.eval_every == 0
            ):
                if self.fusion_mode == "calibrated":
                    self._calibrate_fusion(x_train)
                scores = self.predict(x_test)
                metrics = eval_fn(scores.detach().cpu().numpy(), y_test.numpy())
                pr = metrics.get("PR-AUC", float("nan"))
                postfix["pr_auc"] = f"{pr:.4f}"
                epoch_bar.set_postfix(**postfix)
                if np.isfinite(pr) and pr > best_metric:
                    best_metric = pr
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1

            row = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "rss_mb": rss,
                "vram_mb": vram,
                "batch_size": self.batch_size,
                **metrics,
            }
            history.append(row)
            if logger:
                logger.log(
                    "epoch",
                    setting=self.setting,
                    T=self.T,
                    scheduler=self.noise_config.scheduler,
                    **{k: v for k, v in row.items() if v is not None},
                )

            if patience >= max(1, self.early_stop_patience // max(1, self.eval_every)):
                if logger:
                    logger.log("early_stop", epoch=epoch + 1, best_pr_auc=best_metric)
                break

            if use_rdt and (epoch + 1) == self.rejection_warmup_epochs:
                self._sample_weights = self._compute_rdt_weights(x_train)
                if logger:
                    n_rej = int((self._sample_weights < 1.0).sum().item())
                    logger.log(
                        "rdt_update",
                        epoch=epoch + 1,
                        n_rejected=n_rej,
                        quantile=self.rejection_quantile,
                    )

            if self.use_dte_view and self._dte_memory is None and (epoch + 1) >= self.rejection_warmup_epochs:
                self._build_dte_memory(x_train)

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if self.use_dte_view:
            self._build_dte_memory(x_train)

        self.timing["train_sec"] = time.perf_counter() - t_train_start
        return {"history": history, "best_pr_auc": best_metric, "timing": self.timing}

    def _calibrate_fusion(self, x_train: torch.Tensor, n_cal: int = 256) -> None:
        """Calibrate lambda_v from training data (semi: normals implied by train split)."""
        self.model.eval()
        n = min(n_cal, x_train.size(0))
        idx = torch.randperm(x_train.size(0), device=x_train.device)[:n]
        xb = x_train[idx].to(self.device)
        with torch.inference_mode():
            rec, lat, res, var, dte = self._score_views(xb, vectorized=self.vectorized_scoring)
        eps = 1e-8
        lr = 1.0 / (rec.mean().item() + eps)
        lz = 1.0 / (lat.mean().item() + eps)
        le = 1.0 / (res.mean().item() + eps)
        parts = {"reconstruction": lr, "latent": lz, "residual": le}
        if self.use_uncertainty_view:
            lu = 1.0 / (var.mean().item() + eps)
            parts["uncertainty"] = lu
        if self.use_dte_view:
            ld = 1.0 / (dte.mean().item() + eps)
            parts["diffusion_time"] = ld
        s = sum(parts.values())
        self._calibrated_fusion = {k: v / s for k, v in parts.items()}

    def _score_views(
        self,
        xb: torch.Tensor,
        vectorized: bool = True,
        score_seed: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-sample rec, lat, res, var, dte accumulators."""
        b = xb.size(0)
        rec = torch.zeros(b, device=xb.device, dtype=torch.float32)
        lat = torch.zeros(b, device=xb.device, dtype=torch.float32)
        res = torch.zeros(b, device=xb.device, dtype=torch.float32)
        var = torch.zeros(b, device=xb.device, dtype=torch.float32)
        dte = torch.zeros(b, device=xb.device, dtype=torch.float32)

        ts = self.score_timesteps
        ws = self.score_weights
        t0 = torch.ones(b, device=self.device, dtype=torch.long)
        _, z0 = self.model(xb, t0)

        M = self.uncertainty_draws if self.use_uncertainty_view else 1
        t_grid = self._cached_t_grid if self._cached_t_grid is not None else torch.tensor(
            ts, device=self.device, dtype=torch.long
        )
        w_tensor = self._cached_score_weights if self._cached_score_weights is not None else torch.tensor(
            ws, device=self.device, dtype=torch.float32
        )

        if vectorized and len(ts) > 1:
            K = len(ts)
            rec_draws = []
            rec_k_last = None
            for m in range(M):
                rng = torch.Generator(device=self.device)
                rng.manual_seed(int(score_seed) + 17 + m * 997)
                noise = torch.randn(b, K, xb.size(1), generator=rng, device=self.device, dtype=xb.dtype)
                x_t = vectorized_q_sample(xb, t_grid, self.scheduler.alpha_bar, noise)

                flat_x = x_t.reshape(b * K, -1)
                flat_t = t_grid.unsqueeze(0).expand(b, K).reshape(-1)
                x0_hat, zt = self.model(flat_x, flat_t)
                x0_hat = x0_hat.view(b, K, -1)
                zt = zt.view(b, K, -1)

                rec_k = torch.norm(xb.unsqueeze(1) - x0_hat, dim=2)
                rec_k_last = rec_k
                lat_k = torch.norm(z0.unsqueeze(1) - zt, dim=2)
                ab = self.scheduler.alpha_bar[t_grid - 1].view(1, K, 1)
                implied = (x_t - torch.sqrt(ab) * x0_hat) / torch.sqrt(1.0 - ab + 1e-8)
                res_k = torch.norm(noise - implied, dim=2)

                w = w_tensor.view(1, K)
                rec_draws.append(rec_k)
                rec += (rec_k * w).sum(dim=1) / M
                lat += (lat_k * w).sum(dim=1) / M
                res += (res_k * w).sum(dim=1) / M

            if self.use_uncertainty_view and M > 1:
                stacked = torch.stack(rec_draws, dim=0)
                var_per_t = stacked.var(dim=0, unbiased=False)
                var = (var_per_t * w_tensor.view(1, K)).sum(dim=1)
            if self.use_dte_view and rec_k_last is not None:
                post = posterior_mean_from_recon(rec_k_last, t_grid)
                if self._dte_memory is not None:
                    knn = knn_dte_score(z0, self._dte_memory, k=self.dte_knn, T=self.T)
                    dte = fuse_dte_scores(knn, post, knn_weight=self.dte_knn_weight)
                else:
                    dte = post
        else:
            rng = torch.Generator(device=self.device)
            rec_draws = []
            rec_k_collect = []
            for m in range(M):
                rng.manual_seed(int(score_seed) + 17 + m * 997)
                rec_m = torch.zeros(b, device=xb.device, dtype=torch.float32)
                rec_t_vec = []
                for ti, t_val in enumerate(ts):
                    w = float(ws[ti])
                    t = torch.full((b,), int(t_val), device=self.device, dtype=torch.long)
                    noise = torch.randn(xb.shape, generator=rng, device=self.device, dtype=xb.dtype)
                    ab = self.scheduler.alpha_bar[t].view(-1, 1)
                    x_t = torch.sqrt(ab) * xb + torch.sqrt(1.0 - ab) * noise
                    x0_hat, zt = self.model(x_t, t)
                    rec_k = torch.norm(xb - x0_hat, dim=1)
                    rec_t_vec.append(rec_k)
                    rec_m += w * rec_k
                    if m == 0:
                        lat += w * torch.norm(z0 - zt, dim=1)
                        implied = (x_t - torch.sqrt(ab) * x0_hat) / torch.sqrt(1.0 - ab + 1e-8)
                        res += w * torch.norm(noise - implied, dim=1)
                rec_draws.append(rec_m)
                rec += rec_m / M
                if m == 0:
                    rec_k_collect = rec_t_vec
            if self.use_uncertainty_view and M > 1:
                stacked = torch.stack(rec_draws, dim=0)
                var = stacked.var(dim=0, unbiased=False)
            if self.use_dte_view and rec_k_collect:
                rec_k_stack = torch.stack(rec_k_collect, dim=1)
                t_vec = torch.tensor(ts, device=self.device, dtype=torch.long)
                post = posterior_mean_from_recon(rec_k_stack, t_vec)
                if self._dte_memory is not None:
                    knn = knn_dte_score(z0, self._dte_memory, k=self.dte_knn, T=self.T)
                    dte = fuse_dte_scores(knn, post, knn_weight=self.dte_knn_weight)
                else:
                    dte = post

        return rec, lat, res, var, dte

    def predict(self, x_test: torch.Tensor, score_seed: int = 0) -> torch.Tensor:
        t_score_start = time.perf_counter()
        with torch.inference_mode():
            self.model.eval()
            n = x_test.size(0)
            score_bs = choose_score_batch_size(n, max_batch=self.score_batch_size)
            rec_acc = torch.zeros(n, dtype=torch.float64)
            lat_acc = torch.zeros(n, dtype=torch.float64)
            res_acc = torch.zeros(n, dtype=torch.float64)
            var_acc = torch.zeros(n, dtype=torch.float64)
            dte_acc = torch.zeros(n, dtype=torch.float64)

            preupload = self.device.type == "cuda" and n <= self.preupload_test_threshold
            x_gpu = x_test.to(self.device) if preupload else None

            K = len(self.score_timesteps)
            while K >= 1:
                try:
                    for i in range(0, n, score_bs):
                        xb = x_gpu[i : i + score_bs] if preupload else x_test[i : i + score_bs].to(self.device)
                        rec, lat, res, var, dte = self._score_views(
                            xb,
                            vectorized=self.vectorized_scoring and K == len(self.score_timesteps),
                            score_seed=score_seed,
                        )
                        rec_acc[i : i + xb.size(0)] += rec.double().cpu()
                        lat_acc[i : i + xb.size(0)] += lat.double().cpu()
                        res_acc[i : i + xb.size(0)] += res.double().cpu()
                        var_acc[i : i + xb.size(0)] += var.double().cpu()
                        dte_acc[i : i + xb.size(0)] += dte.double().cpu()

                        mem, over = guard_over_limit(self.memory_guard)
                        if over:
                            score_bs = shrink_batch(score_bs, min_batch=32)
                            self.score_batch_size = score_bs
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and K > 4:
                        K = max(4, K // 2)
                        self.score_timesteps = self.score_timesteps[:K]
                        self.score_weights = self.score_weights[:K]
                        self._cache_score_tensors()
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        raise e

            if not self.use_multiview:
                self.timing["score_sec"] = time.perf_counter() - t_score_start
                return rec_acc.float()

            def _norm(v: torch.Tensor) -> torch.Tensor:
                m = v.mean().clamp_min(1e-12)
                return v / m

            if self._calibrated_fusion is not None:
                fw = self._calibrated_fusion
            else:
                fw = self.fusion_weights

            scores = (
                fw.get("reconstruction", 0.6) * _norm(rec_acc)
                + fw.get("latent", 0.3) * _norm(lat_acc)
                + fw.get("residual", 0.1) * _norm(res_acc)
            )
            if self.use_uncertainty_view:
                scores = scores + fw.get("uncertainty", 0.1) * _norm(var_acc)
            if self.use_dte_view:
                scores = scores + fw.get("diffusion_time", 0.1) * _norm(dte_acc)

        self.timing["score_sec"] = time.perf_counter() - t_score_start
        return scores.float()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "noise_config": self.noise_config.__dict__,
            "score_timesteps": self.score_timesteps,
            "setting": self.setting,
            "input_dim": self.input_dim,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
