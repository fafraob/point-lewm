"""Farthest-point sampling for packed point-cloud batches (torch_cluster).

One FPS implementation shared by every encoder, on the compute device (CUDA or
CPU) via torch_cluster's kernels, with this repo's conventions baked in:

* **Deterministic.** ``random_start=False`` seeds FPS at each cloud's first
  packed row, so training is reproducible under the run seed and closed-loop
  eval sees no per-call sampling noise (the same guarantee the previous
  fpsample ``start_idx=0`` path gave).
* **Exact per-cloud counts.** torch_cluster keeps ``ceil(ratio * n)`` points
  per cloud; the ratio is chosen so that is exactly ``k`` (or all ``n`` for
  clouds already at/below the budget).
* **Packed in, packed out.** Takes the collate layout (``coord`` + ascending
  ``batch``) and returns row indices grouped by cloud in that same order.

This replaced fpsample's CPU kernel: running FPS on the GPU removes the host
round-trip and the per-cloud Python loops. At the training shape (1024 clouds
x ~7k points -> 2048) it is ~90 ms/batch vs ~4.8 s single-core on CPU.
"""

import torch


def fps_index(
    coord: torch.Tensor, batch: torch.Tensor, k: int, n_clouds: int | None = None
) -> torch.Tensor:
    """Row indices keeping at most ``k`` farthest-point samples per cloud.

    Clouds already at/below ``k`` points keep all their points; larger clouds
    are thinned to exactly ``k``. Indices come back grouped by cloud in
    ascending ``batch`` order, each cloud starting at its first packed row
    (the deterministic FPS seed).

    Args:
        coord: ``(M, C)`` packed float coords.
        batch: ``(M,)`` cloud index per point, ascending -- the contiguous
            grouping the collate_fn guarantees (torch_cluster requires it).
        k: per-cloud point budget (> 0).
        n_clouds: number of clouds; inferred from ``batch`` when omitted.

    Returns:
        ``(M',)`` long tensor of row indices into the packed arrays.
    """
    import torch_cluster

    assert k > 0, k
    if batch.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=coord.device)
    if n_clouds is None:
        n_clouds = int(batch.max().item()) + 1
    counts = torch.bincount(batch, minlength=n_clouds)
    if int(counts.max()) <= k:  # nothing to thin -- keep every row
        return torch.arange(batch.numel(), device=coord.device)
    # torch_cluster keeps ceil(ratio * n) per cloud. (k - 0.5) / n survives
    # float rounding to give exactly k (a bare k / n can ceil to k + 1), and
    # the clamp makes small clouds keep exactly their n points.
    ratio = ((k - 0.5) / counts.float()).clamp(max=1.0)
    return torch_cluster.fps(coord, batch, ratio=ratio, random_start=False)
