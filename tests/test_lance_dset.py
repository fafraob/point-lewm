"""DINO-WM's Lance-backed dataset serves the h5 contract on the shipped sample."""

import json
import os

import numpy as np
import pytest

SAMPLE = os.path.join(os.path.dirname(__file__), '..', 'data_sample', 'cube_sample.lance')


@pytest.fixture(scope='module')
def loaders(tmp_path_factory):
    pytest.importorskip('lance')
    from dinowm_swm import add_upstream_to_path

    add_upstream_to_path()
    from datasets.img_transforms import default_transform
    from dinowm_swm.h5_dset import load_swm_h5_slice_train_val

    stats_path = tmp_path_factory.mktemp('run') / 'swm_h5_norm_stats.json'
    # the .lance suffix dispatches to dinowm_swm.lance_dset
    datasets, traj = load_swm_h5_slice_train_val(
        default_transform(img_size=224),
        SAMPLE,
        num_hist=3,
        num_pred=1,
        frameskip=5,
        state_key='observation',
        stats_path=str(stats_path),
    )
    return datasets, traj, stats_path


def test_windows_have_dinowm_shapes(loaders):
    datasets, traj, _ = loaders
    assert len(traj['train']) + len(traj['valid']) == 3  # the sample's episodes
    obs, act, state = datasets['train'][0]
    assert tuple(obs['visual'].shape) == (4, 3, 224, 224)
    assert tuple(obs['proprio'].shape) == (4, 1)  # noprop: constant zero
    assert act.shape == (4, 5 * traj['train'].action_dim)  # frameskip actions concatenated
    assert state.shape[0] == 4
    assert -1.0 <= float(obs['visual'].min()) and float(obs['visual'].max()) <= 1.0


def test_stats_file_matches_h5_convention(loaders):
    datasets, traj, stats_path = loaders
    stats = json.loads(stats_path.read_text())
    assert {'action_mean', 'action_std', 'action_key', 'state_mean', 'state_std', 'source'} <= set(stats)
    assert len(stats['action_mean']) == traj['train'].action_dim
    # z-scored actions: whole-table mean ~ 0 over all (non-terminal) rows
    all_actions = np.concatenate([traj[k].get_all_actions().numpy() for k in ('train', 'valid')])
    finite = all_actions[np.isfinite(all_actions).all(axis=1)]
    assert np.abs(finite.mean(0)).max() < 0.2


def test_frames_are_the_tables_rows(loaders):
    datasets, traj, _ = loaders
    import io

    import lance
    from PIL import Image

    ds = lance.dataset(SAMPLE)
    dset = traj['train']
    ep = int(dset.episodes[0])
    obs, _, _, info = dset.get_frames(0, np.array([0, 5]))
    assert info['episode'] == ep
    row = int(dset.ep_offset[ep]) + 5
    stored = np.asarray(Image.open(io.BytesIO(ds.take([row], columns=['pixels'])['pixels'][0].as_py())).convert('RGB'))
    # default_transform only normalises (resize/crop are no-ops at 224)
    got = ((obs['visual'][1].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).round().astype(np.uint8)
    assert np.abs(got.astype(int) - stored.astype(int)).mean() < 1.0
