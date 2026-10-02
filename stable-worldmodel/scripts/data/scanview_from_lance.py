"""Replace a table's ``pixels`` with renders from the sensor's camera, in place.

The point-cloud tables carry the *original* camera's image next to each scan.
This pass restores every recorded state in the scene the scan was cast in and
renders it from the sensor's own camera (``iso``), so image and scan share
pose and field of view. The new images overwrite the ``pixels`` column of the
same table; with ``--lidar`` the scan is recast too (used to repair the Push-T
table, whose first scans restored the block with the wrong setter order).

Lance makes the replacement safe and shardable: every step is a new table
version (the previous one stays readable until a cleanup), and new columns are
produced fragment by fragment, so many workers can render in parallel and one
commit assembles them.

    plan      how many fragments / rows the table has
    shard     render the fragments of one shard; writes fragment metadata to
              --stage (run N of these, e.g. a SLURM array)
    commit    one Merge commit adding the staged columns (pixels_scanview,
              lidar_scanview) to the table
    finalize  drop the old columns and rename the new ones into their place
    verify    re-render a few episodes and compare with what the table holds

Typical:

    S=scripts/data/scanview_from_lance.py; C=scripts/data/config/lidar_from_h5_pusht.yaml
    python $S plan     --table pusht.lance
    python $S shard    --table pusht.lance --config $C --lidar --stage-dir st/ --shard 0 --num-shards 16
    ...
    python $S commit   --table pusht.lance --stage-dir st/
    python $S finalize --table pusht.lance --lidar
    python $S verify   --table pusht.lance --config $C --lidar --episodes 3 4 5
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')

import gymnasium as gym
import lance
import numpy as np
import pyarrow as pa
from lance.fragment import FragmentMetadata
from loguru import logger as logging
from omegaconf import OmegaConf
from PIL import Image

import stable_worldmodel  # noqa: F401  (registers swm/ envs)
from stable_worldmodel.data.formats.lance import _DEFAULT_JPEG_QUALITY, _encode_frame
from stable_worldmodel.envs import replay
from stable_worldmodel.wrapper import AddRaycastLidarWrapper


PIXELS_NEW = 'pixels_scanview'
LIDAR_NEW = 'lidar_scanview'
# Columns the restore may read, per table flavour (only those present are read).
STATE_COLS = ['pos_agent', 'pos_target', 'state', 'goal_pose', 'block_pose', 'qpos', 'qvel']


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


class Scene:
    """The env the table was cast in, plus the sensor to recast with."""

    def __init__(self, config: str, lidar: bool):
        cfg = OmegaConf.load(config)
        env_kwargs = OmegaConf.to_container(cfg.get('env', {}) or {}, resolve=True)
        self.env = gym.make(cfg.env_name, render_mode='rgb_array', **env_kwargs)
        self.env.reset(seed=0)
        self.base = self.env.unwrapped
        self.sensor = None
        if lidar:
            kw = OmegaConf.to_container(cfg.lidar, resolve=True)
            if replay.is_mirror_env(self.env) and kw.get('backend') == 'warp':
                kw['backend'] = 'mujoco'  # the mirror is rebuilt per episode
            self.sensor = AddRaycastLidarWrapper(self.env, add_goal_lidar=False, **kw)
            self.sensor._detect_backend()
        self._episode = None

    def restore(self, row: dict) -> None:
        """Put the scene into the state of one table row."""
        ep = {k: np.asarray([v]) for k, v in row.items() if k in STATE_COLS}
        if row['episode_idx'] != self._episode:
            replay.begin_episode(self.env, ep)
            self._episode = row['episode_idx']
        replay.set_step(self.env, ep, 0)

    def render(self) -> bytes:
        return _encode_frame(np.asarray(self.base.render(camera='iso')), _DEFAULT_JPEG_QUALITY)

    def cast(self) -> np.ndarray:
        pts, _ = self.sensor._cast(self.base.model, self.base.data)
        return pts.reshape(-1).astype(np.float32)

    def close(self):
        self.env.close()


def new_fields(lidar: bool, lidar_len: int | None) -> list[pa.Field]:
    fields = [pa.field(PIXELS_NEW, pa.binary())]
    if lidar:
        fields.append(pa.field(LIDAR_NEW, pa.list_(pa.float32(), lidar_len)))
    return fields


def make_udf(scene: Scene, lidar: bool, lidar_len: int | None):
    """RecordBatch -> RecordBatch of the new columns, row for row."""

    def udf(batch: pa.RecordBatch) -> pa.RecordBatch:
        rows = batch.to_pylist()
        images, scans = [], []
        for row in rows:
            scene.restore(row)
            images.append(scene.render())
            if lidar:
                scans.append(scene.cast())
        arrays = [pa.array(images, type=pa.binary())]
        if lidar:
            flat = pa.array(np.concatenate(scans), type=pa.float32())
            arrays.append(pa.FixedSizeListArray.from_arrays(flat, lidar_len))
        return pa.RecordBatch.from_arrays(arrays, schema=pa.schema(new_fields(lidar, lidar_len)))

    return udf


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def read_columns(ds: lance.LanceDataset) -> list[str]:
    names = set(ds.schema.names)
    return ['episode_idx', 'step_idx'] + [c for c in STATE_COLS if c in names]


def lidar_length(ds: lance.LanceDataset) -> int:
    t = ds.schema.field('lidar').type
    return int(t.list_size)


def stage_plan(args):
    ds = lance.dataset(args.table)
    frags = ds.get_fragments()
    logging.info(
        f'{args.table}: version {ds.version}, {ds.count_rows()} rows, '
        f'{len(frags)} fragments, read columns {read_columns(ds)}'
    )
    for n in (8, 16, 32):
        per = int(np.ceil(len(frags) / n))
        logging.info(f'  {n} shards -> up to {per} fragments each')


def stage_shard(args):
    ds = lance.dataset(args.table)
    frags = ds.get_fragments()
    mine = frags[args.shard :: args.num_shards]
    if not mine:
        logging.warning(f'shard {args.shard}/{args.num_shards}: no fragments')
        return
    os.makedirs(args.stage_dir, exist_ok=True)
    lidar_len = lidar_length(ds) if args.lidar else None
    scene = Scene(args.config, args.lidar)
    udf = make_udf(scene, args.lidar, lidar_len)
    cols = read_columns(ds)
    for frag in mine:
        fid = frag.fragment_id
        out = os.path.join(args.stage_dir, f'frag_{fid:06d}.json')
        if os.path.exists(out) and not args.redo:
            logging.info(f'fragment {fid}: staged already, skipping')
            continue
        meta, schema = frag.merge_columns(udf, columns=cols, batch_size=args.batch_size)
        with open(out + '.tmp', 'w') as f:
            json.dump(
                {
                    'read_version': ds.version,
                    'fragment': meta.to_json(),
                    'schema': schema.to_pyarrow().serialize().to_pybytes().hex(),
                },
                f,
            )
        os.replace(out + '.tmp', out)
        logging.info(f'fragment {fid}: {meta.num_rows} rows staged')
    scene.close()
    logging.success(f'shard {args.shard}/{args.num_shards}: {len(mine)} fragments done')


def stage_commit(args):
    ds = lance.dataset(args.table)
    staged = sorted(glob.glob(os.path.join(args.stage_dir, 'frag_*.json')))
    want = {f.fragment_id for f in ds.get_fragments()}
    frags, versions, schemas = [], set(), set()
    for path in staged:
        with open(path) as f:
            d = json.load(f)
        versions.add(d['read_version'])
        schemas.add(d['schema'])
        frags.append(FragmentMetadata.from_json(json.dumps(d['fragment'])))
    have = {int(os.path.basename(p)[5:11]) for p in staged}
    if have != want:
        sys.exit(f'staged fragments {sorted(have ^ want)} do not match the table; run the missing shards')
    if versions != {ds.version}:
        sys.exit(f'staged against version(s) {sorted(versions)} but the table is at {ds.version}; re-run the shards')
    if len(schemas) != 1:
        sys.exit('staged fragments disagree on the schema')
    schema = pa.ipc.read_schema(pa.py_buffer(bytes.fromhex(schemas.pop())))
    op = lance.LanceOperation.Merge(frags, schema)
    new = lance.LanceDataset.commit(args.table, op, read_version=ds.version)
    logging.success(f'committed version {new.version}: columns {new.schema.names}')


def stage_finalize(args):
    ds = lance.dataset(args.table)
    names = ds.schema.names
    if PIXELS_NEW not in names or (args.lidar and LIDAR_NEW not in names):
        sys.exit(f'nothing to finalize: columns are {names}')
    old = ['pixels'] + (['lidar'] if args.lidar else [])
    ds.drop_columns(old)
    renames = [{'path': PIXELS_NEW, 'name': 'pixels'}]
    if args.lidar:
        renames.append({'path': LIDAR_NEW, 'name': 'lidar'})
    ds.alter_columns(*renames)
    ds = lance.dataset(args.table)
    logging.success(f'version {ds.version}: columns {ds.schema.names}')


def stage_verify(args):
    ds = lance.dataset(args.table)
    scene = Scene(args.config, args.lidar)
    cols = read_columns(ds) + ['pixels'] + (['lidar'] if args.lidar else [])
    worst_img, bad_scan, n = 0.0, 0, 0
    for ep in args.episodes:
        tbl = ds.scanner(columns=cols, filter=f'episode_idx = {ep}').to_table()
        for row in tbl.to_pylist():
            scene.restore(row)
            rendered = np.asarray(Image.open(io.BytesIO(scene.render())).convert('RGB'), dtype=np.float32)
            stored = np.asarray(Image.open(io.BytesIO(row['pixels'])).convert('RGB'), dtype=np.float32)
            worst_img = max(worst_img, float(np.abs(rendered - stored).mean()))
            if args.lidar:
                bad_scan += int(np.abs(scene.cast() - np.asarray(row['lidar'], dtype=np.float32)).max() > 1e-3)
            n += 1
    scene.close()
    logging.info(f'{n} rows: worst mean |render - stored pixels| = {worst_img:.3f}' + (f', rows with scan mismatch = {bad_scan}' if args.lidar else ''))
    if worst_img > 1.0 or bad_scan:
        sys.exit('verify FAILED')
    logging.success('verify passed')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp, config=False, lidar=False):
        sp.add_argument('--table', required=True, help='path of the .lance table (modified in place)')
        if config:
            sp.add_argument('--config', required=True, help='the lidar_from_h5_*.yaml the table was generated with')
        if lidar:
            sp.add_argument('--lidar', action='store_true', help='also recast the scan into the lidar column')

    common(sub.add_parser('plan'))
    s = sub.add_parser('shard'); common(s, config=True, lidar=True)
    s.add_argument('--stage-dir', required=True, help='directory for staged fragment metadata')
    s.add_argument('--shard', type=int, required=True)
    s.add_argument('--num-shards', type=int, required=True)
    s.add_argument('--batch-size', type=int, default=256)
    s.add_argument('--redo', action='store_true', help='re-render fragments already staged')
    s = sub.add_parser('commit'); common(s)
    s.add_argument('--stage-dir', required=True)
    s = sub.add_parser('finalize'); common(s, lidar=True)
    s = sub.add_parser('verify'); common(s, config=True, lidar=True)
    s.add_argument('--episodes', type=int, nargs='+', default=[0, 1, 2])

    args = p.parse_args()
    {'plan': stage_plan, 'shard': stage_shard, 'commit': stage_commit,
     'finalize': stage_finalize, 'verify': stage_verify}[args.cmd](args)


if __name__ == '__main__':
    main()
