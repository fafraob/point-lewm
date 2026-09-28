# Point-cloud encoders

Encoders here are the point-cloud counterpart of the image ViT used by the
pixel JEPA. Each one turns a batch of point clouds into one embedding per cloud,
which the JEPA model then predicts forward in time exactly as it does for image
embeddings.

> **Encoders shipped:**
> - **`pointvit_encoder.PointViTEncoder`** — Point-BERT/Uni3D-style tokenizer
>   (FPS centers + radius groups + mini-PointNet) feeding the *same* HF ViT the
>   image model uses, CLS token read out as the cloud latent (Point-LeWM,
>   Point-Delta-JEPA).
> - **`utonia_encoder.UtoniaEncoder`** — frozen Utonia backbone (vendored under
>   `utonia/`) pooled onto a fixed token grid (Utonia-WM).
> - **`voxstats_encoder.VoxelStatsEncoder`** — per-cell point statistics on the
>   same token grid, the training-free geometric baseline (Vox-WM).
>
> Shared piece: `sampling.fps_index` (deterministic torch_cluster FPS on the
> compute device, exact per-cloud budgets).

## The interface

Subclass [`PointCloudEncoder`](base.py) and implement `forward`:

```python
from pc_encoders.base import PointCloudEncoder

class MyEncoder(PointCloudEncoder):
    def __init__(self, embed_dim, in_channels=3, **hparams):
        super().__init__(embed_dim=embed_dim, in_channels=in_channels)
        ...  # build layers

    def forward(self, data):          # data: packed batch (see below)
        ...                           # -> Tensor (num_clouds, embed_dim)
```

`embed_dim` must match the JEPA `embed_dim` so the projector/predictor line up.

## Packed batch format

A batch of `num_clouds` point clouds is passed as a plain dict of flat tensors
(the Pointcept / PyTorch-Geometric convention), so clouds may have different
sizes without any padding:

| key     | shape / type            | meaning                                                     |
| ------- | ----------------------- | ----------------------------------------------------------- |
| `coord` | `(M, C)` float          | point coords, all clouds concatenated (`C` = `in_channels`; 3 = xyz for lidar) |
| `batch` | `(M,)` long             | cloud index in `[0, num_clouds)` for each point             |
| `feat`  | `(M, F)` float or `None`| optional per-point features (`None` if unused)              |

`M = sum_i N_i` is the total number of points. `forward` returns
`(num_clouds, embed_dim)` — one embedding per cloud, ordered by batch index.
JEPA reshapes that to `(B, T, D)`, so `num_clouds == B * T` (one cloud per
`(sample, timestep)` frame).

`PointCloudEncoder` provides `batch_to_offset(batch)` for backbones that want a
Pointcept-style `offset` instead of the `batch` index.

**Normalization is intentionally *not* done in the data pipeline.** The
`collate_fn` only packs; the reshape transform only reshapes. Centering,
scaling, voxelization and point sampling belong inside the encoder so each
backbone can use its own strategy.

The `collate_fn` *can* optionally drop sentinel points (coords all equal to a
value, e.g. `(-1,-1,-1)` for missing lidar returns) via `invalid_value` in the
data config's `obs` block. That is format-level cleaning, not normalization, and
it yields ragged clouds — handled natively by the packed layout.

## Adding an encoder

1. Add `pc_encoders/my_encoder.py` with a `PointCloudEncoder` subclass.
2. Point the config at it, e.g. in `config/train/model/point_lewm_cube.yaml`:

   ```yaml
   encoder:
     _target_: pc_encoders.my_encoder.MyEncoder
     embed_dim: ${embed_dim}
     in_channels: 3
     # ... encoder-specific hyperparameters
   ```

3. Train with the lidar configs (`python train.py --config-name=point_lewm_cube ...`).
