# Vendored: Utonia (Pointcept)

Copy of the `utonia/` package from
https://github.com/Pointcept/Utonia at rev
`da776a0bd3a48c6df83ac2ae0e27b26141cc7e31` (paper: *Utonia: Toward One Encoder
for All Point Clouds*, ICML'26, arXiv:2603.03283). Apache-2.0, see `LICENSE`
(the demo scripts and setup.py were dropped).

Two local patches, both in `model.py` and both marked `[le-wm vendored patch]`:

**#1 — memory (`SerializedAttention.forward`).** The upstream non-flash attention
materializes the dense `(patches, heads, 1024, 1024)` matrix — tens of GB on
batched clouds — so the `enable_rpe=False` branch (the one the released
checkpoint takes) now routes through
`torch.nn.functional.scaled_dot_product_attention` instead. Identical math,
memory-efficient kernel, no flash-attn dependency.

**#2 — batch-invariant patch size (`get_padding_and_inverse` +
`SerializedAttention.forward`).** Upstream pads a cloud only when it is *larger*
than `patch_size`, so the non-flash path shrinks `patch_size` to the smallest
cloud in the batch to keep every cloud patch-aligned. That makes a cloud's
attention window — and therefore its features — depend on which other clouds
share the batch: measured on tabletop lidar, a cloud beside 63 same-size
neighbours moved 1.4% (rel L2), and beside a 300-point cloud 56%. Now every
cloud is padded up to a whole number of patches, and the padding of a cloud that
fits inside a single patch is masked out via `patch_valid` — which is what the
flash path already does with its per-cloud `cu_seqlens` (a ragged final segment
for a sub-patch cloud, duplicate-padded full segments otherwise). `patch_size`
therefore stays at the checkpoint's configured 1024 regardless of batching.
Verified: unbatched features are bit-identical to before the patch, and after it
a cloud varies by at most half an fp16 step across batch sizes 1 to 64. Note the
`enable_rpe=True` branch cannot mask, so it asserts no sub-patch cloud is
present; the released checkpoint uses `enable_rpe=False`.

Vendored rather than pip-installed because upstream's `setup.py` has a dead
`import pkg_resources` that breaks under setuptools >= 81, and because a
checked-in copy removes the network/git dependency from environment setup
entirely. All internal imports are relative, so the package works
unmodified from the repo root (`import utonia` in `pc_encoders/utonia_encoder.py`).

Runtime deps (declared in `pixi.toml`): `spconv-cu126`, `timm`, `addict`, and
`torch-scatter` (compiled in-env by `pixi run build-kernels`). `flash_attn` is
optional upstream and deliberately NOT installed here — the encoder always
loads the checkpoint with `enable_flash=False` (the upstream demos' own
fallback) so results are reproducible across machines.

If you update this copy, bump the rev above and re-check
`pc_encoders/utonia_encoder.py` against upstream API changes.
