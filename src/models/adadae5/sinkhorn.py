"""SINKHORN — cheap 1D Wasserstein / sorted-mass geometry between x and x̂."""
from __future__ import annotations

import torch


@torch.inference_mode()
def sinkhorn_ot_score(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    epsilon: float = 0.05,
    n_iters: int = 10,
) -> torch.Tensor:
    """
    Per-sample 1D Wasserstein-1 between sorted |x| and |x̂| feature masses.
    O(d log d); safe for high-d (caller should still gate d>64).
    epsilon/n_iters kept for API compatibility (unused).
    """
    del epsilon, n_iters
    b, d = x.shape
    if d < 2:
        return torch.norm(x - x_hat, dim=1)

    a = torch.abs(x.float())
    b_mass = torch.abs(x_hat.float())
    a_s, _ = torch.sort(a, dim=1)
    b_s, _ = torch.sort(b_mass, dim=1)
    # Normalize to probability masses then W1 on sorted supports {1..d}
    a_s = a_s / a_s.sum(dim=1, keepdim=True).clamp(min=1e-8)
    b_s = b_s / b_s.sum(dim=1, keepdim=True).clamp(min=1e-8)
    # Cumulative mass difference × support spacing
    cdf_diff = torch.cumsum(a_s - b_s, dim=1).abs()
    w1 = cdf_diff.mean(dim=1)
    return w1 + 0.1 * torch.norm(x - x_hat, dim=1)
