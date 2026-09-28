import json
from pathlib import Path

import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ReshapePointCloud(dt.transforms.Transform):
    """Reshape a flat per-frame lidar column into ``(..., N, in_channels)``.

    The lidar column is stored/read flat as ``(T, N * in_channels)``; this splits
    the last dim into ``(N, in_channels)`` (point-major, xyz last). The point
    count ``N`` is inferred from the flat size, so it need not be fixed -- only
    ``in_channels`` is required. No normalization, so each encoder can apply its
    own point normalization strategy -- but it DOES cast to ``float32``: the
    lance columns are ``float64`` and everything downstream is ``float32``, so
    casting per-sample here (instead of at collate time) halves what a
    DataLoader worker holds while it assembles a batch -- at batch_size 256 x 4
    frames x ~7k points that is hundreds of MB per in-flight batch, the
    workers' dominant RAM cost. Written as a picklable ``Transform`` subclass
    (not a closure) because DataLoader workers are spawned for the Lance
    dataset, mirroring :class:`ZScoreNormalizer`.
    """

    def __init__(self, in_channels: int, source: str = "lidar", target: str = "points"):
        super().__init__()
        self.in_channels = int(in_channels)
        self.source = source
        self.target = target

    def __call__(self, x):
        v = self.nested_get(x, self.source)  # (..., N * in_channels)
        v = v.reshape(*v.shape[:-1], -1, self.in_channels)  # (..., N, in_channels); N inferred
        v = v.to(torch.float32)  # collate would cast anyway; do it before buffering
        self.nested_set(x, v, self.target)
        if self.target != self.source and self.source in x:
            del x[self.source]  # drop the raw flat column so it isn't collated
        return x


def get_pc_preprocessor(source: str, target: str, in_channels: int = 3):
    """Reshape transform for point clouds, mirroring :func:`get_img_preprocessor`.

    Splits the flat lidar column ``(..., N * in_channels)`` into ``(..., N,
    in_channels)`` (``N`` inferred, may vary across frames/datasets). Reshape only
    (no normalization); packing into the batched ``{coord, batch, feat}`` layout
    happens in the collate_fn, which supports variable point counts.
    """
    return ReshapePointCloud(in_channels, source=source, target=target)


class LazyMemmapColumn:
    """Read-only virtual dataset column backed by an on-disk ``.npy`` memmap.

    Inserted into ``LanceDataset._cache`` so window fetches slice it by global
    row index exactly like a real cached column (``_process_col`` reads
    ``self._cache[col][g_start:g_end]``). Opens the memmap lazily PER PROCESS
    and pickles only the path: a raw ndarray in ``_cache`` would be serialized
    wholesale into every spawned DataLoader worker (~GBs each for the frozen
    Utonia embedding cache).
    """

    def __init__(self, path):
        self.path = str(path)
        self._mm = None

    def _open(self):
        if self._mm is None:
            self._mm = np.load(self.path, mmap_mode="r")
        return self._mm

    def __getitem__(self, idx):
        return np.asarray(self._open()[idx])  # materialize the slice (copy)

    def __len__(self):
        return len(self._open())

    @property
    def shape(self):
        return self._open().shape

    def __getstate__(self):
        return {"path": self.path, "_mm": None}


def encoder_cache_signature(encoder_cfg) -> dict:
    """Resolved encoder config minus keys that cannot change the features.

    The signature stored in an embedding cache's ``.meta.json`` and compared
    against the training run's encoder config: any mismatch means the cached
    features were produced by a DIFFERENT frozen encoder, which would silently
    invalidate the experiment.
    """
    from omegaconf import OmegaConf

    if not isinstance(encoder_cfg, dict):
        encoder_cfg = OmegaConf.to_container(encoder_cfg, resolve=True)
    from pc_encoders.utonia_encoder import FEATURE_VERSION

    sig = dict(encoder_cfg)
    sig.pop("_target_", None)
    # The backbone batch size is the ONE key that does not change the features:
    # since vendored patch #2 the attention patch size no longer depends on batch
    # composition, and a cloud's tokens shift by at most half an fp16 cache step
    # between batch sizes 1 and 64 (measured) -- inside the tolerance
    # precompute_utonia.py --verify already allows. Excluding it means a cache
    # built with a different chunk size is still accepted.
    sig.pop("max_clouds_per_forward", None)
    # Everything else remaining DOES change the features.
    # The code stamp catches what the config cannot: an encoder IMPLEMENTATION
    # change that silently makes an existing cache stale.
    sig["_feature_version"] = FEATURE_VERSION
    return sig


def obs_cache_signature(obs_cfg) -> dict:
    """The DATA-side settings that also decide an embedding cache's contents.

    The encoder config is not the whole story: which raw column is read and
    which points are discarded as missed rays happen in the data config, and
    they change every cached feature (keeping the (-1,-1,-1) sentinels rebases
    the voxel grid, which measurably moves ~60 of 256 tokens on every frame).
    Recorded next to the encoder signature so a cache built under a different
    drop rule is refused rather than silently trained on. ``live_invalid_value``
    must be the value precompute actually USED, including a CLI override.
    """
    from omegaconf import OmegaConf

    if not isinstance(obs_cfg, dict):
        obs_cfg = OmegaConf.to_container(obs_cfg, resolve=True)
    return {
        "live_source_key": obs_cfg.get("live_source_key", "lidar"),
        "live_invalid_value": obs_cfg.get("live_invalid_value", -1.0),
    }


def attach_embedding_cache(dataset, column: str, cache_path, encoder_cfg, repo_root,
                           obs_cfg=None):
    """Serve precomputed frozen-encoder embeddings as a dataset column.

    ``cache_path`` (relative paths resolve against ``repo_root``) must be a
    ``(total_rows, num_tokens * embed_dim)`` float16 ``.npy`` written by
    ``precompute_utonia.py``, with its ``.meta.json`` sidecar. Verifies the
    row count against the dataset and the encoder signature against the run's
    config, then registers a :class:`LazyMemmapColumn` in the dataset's
    column cache (the same private hook ``LanceDataset.merge_col`` uses). The
    data pipeline's reshape transform splits each row back into
    ``(num_tokens, embed_dim)`` -- so the flat row width, not ``embed_dim``
    alone, is what must match.
    """
    path = Path(cache_path)
    if not path.is_absolute():
        path = Path(repo_root) / path
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"embedding cache missing: {path} (+ .meta.json). Precompute it once with\n"
            "  pixi run python precompute_utonia.py --config-name <this run's config>"
        )
    meta = json.loads(meta_path.read_text())
    total_rows = int(np.sum(dataset.lengths))
    assert int(meta["rows"]) == total_rows, (
        f"embedding cache holds {meta['rows']} rows, dataset has {total_rows} -- "
        "the cache was built for a different dataset"
    )
    assert int(meta["episodes_done"]) == len(dataset.lengths), (
        f"embedding cache is incomplete ({meta['episodes_done']}/"
        f"{len(dataset.lengths)} episodes) -- finish precompute_utonia.py first"
    )
    sig = encoder_cache_signature(encoder_cfg)
    assert meta["encoder"] == sig, (
        "embedding cache was built with a DIFFERENT encoder config -- rebuild it "
        f"(cache: {meta['encoder']}\n run: {sig})"
    )
    if obs_cfg is not None:
        obs_sig = obs_cache_signature(obs_cfg)
        assert meta.get("obs") == obs_sig, (
            "embedding cache was built with a DIFFERENT raw-observation rule "
            "(source column / missed-ray sentinel), which changes every cached "
            f"feature -- rebuild it (cache: {meta.get('obs')}\n run: {obs_sig})"
        )
    col = LazyMemmapColumn(path)
    width = int(meta["embed_dim"]) * int(meta.get("num_tokens", 1))
    assert col.shape == (total_rows, width), (col.shape, (total_rows, width))
    dataset._cache[column] = col
    if column not in dataset._keys:
        dataset._keys.append(column)
    dataset._update_fetch_columns()
    print(f"[train] serving frozen embeddings from {path} as column '{column}'")


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained.

    ``cache_dir`` is forwarded to ``save_pretrained`` so the export lands in
    ``<cache_dir>/checkpoints/<run_name>/``; pass the current run folder to keep
    each run's weights isolated. When ``None`` it falls back to ``get_cache_dir()``.
    """

    # Validation metric that picks weights_best.pt (see _save). Lightning runs
    # the epoch's validation INSIDE the training epoch loop, before the
    # on_train_epoch_end hooks fire (fit_loop.on_advance_end), so the reduced
    # epoch value is already in trainer.callback_metrics at that point.
    BEST_METRIC = 'validate/loss_epoch'

    def __init__(self, run_name, cfg, epoch_interval: int = 1, cache_dir: str = None):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval
        self.cache_dir = cache_dir
        self.best_val = float('inf')
        self.best_epoch = None

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            val = trainer.callback_metrics.get(self.BEST_METRIC)
            val = float(val) if val is not None else None
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1, val)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1, val)

    def _save(self, model, epoch, val_loss=None):
        import shutil
        from stable_worldmodel.wm.utils import get_cache_dir, save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
            cache_dir=self.cache_dir,
        )
        # weights_final.pt always mirrors the LATEST export (+ final_epoch.txt
        # naming it), so a dependent eval sweep can reference one filename
        # whatever epoch a time-limited run reaches -- training may stop at a
        # wall-clock budget (max_train_hours) and the evals still find the
        # newest export. Cheap: one file copy per epoch.
        ckpt_dir = get_cache_dir(self.cache_dir, sub_folder='checkpoints') / self.run_name
        shutil.copyfile(ckpt_dir / f'weights_epoch_{epoch}.pt', ckpt_dir / 'weights_final.pt')
        (ckpt_dir / 'final_epoch.txt').write_text(f'{epoch}\n')
        # weights_best.pt mirrors the export with the LOWEST validation loss so
        # far (+ best_epoch.txt: "<epoch> <loss>"), as insurance against a late
        # training divergence that would leave weights_final.pt a poor
        # predictor. Skipped when the run logs no validation (the metric is
        # then absent). The best is tracked in memory only, so a Lightning
        # resume restarts the comparison from the resumed epoch.
        if val_loss is not None and val_loss < self.best_val:
            self.best_val, self.best_epoch = val_loss, epoch
            shutil.copyfile(ckpt_dir / f'weights_epoch_{epoch}.pt', ckpt_dir / 'weights_best.pt')
            (ckpt_dir / 'best_epoch.txt').write_text(f'{epoch} {val_loss:.6e}\n')