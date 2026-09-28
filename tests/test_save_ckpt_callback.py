"""utils.SaveCkptCallback: per-epoch export + weights_final.pt + weights_best.pt.

save_pretrained is stubbed (it only torch.saves a state_dict and a config.json),
so the test checks the callback's own bookkeeping: which export the two
mirror files point at after a sequence of epochs with a validation loss that
dips and then diverges (the late-training divergence weights_best.pt exists for).
"""
import types

import pytest
import torch

import utils


class _Trainer:
    def __init__(self, epoch, max_epochs, val):
        self.current_epoch = epoch
        self.max_epochs = max_epochs
        self.is_global_zero = True
        self.callback_metrics = {} if val is None else {"validate/loss_epoch": torch.tensor(val)}


def _run(tmp_path, monkeypatch, vals, max_epochs=None):
    import stable_worldmodel.wm.utils as swm_utils

    def fake_save_pretrained(model, run_name, config=None, filename="weights.pt", cache_dir=None):
        d = swm_utils.get_cache_dir(cache_dir, sub_folder="checkpoints") / run_name
        d.mkdir(parents=True, exist_ok=True)
        (d / filename).write_text(filename)  # content identifies the export

    monkeypatch.setattr(swm_utils, "save_pretrained", fake_save_pretrained)
    cb = utils.SaveCkptCallback(run_name="m", cfg={}, epoch_interval=1, cache_dir=str(tmp_path))
    module = types.SimpleNamespace(model=torch.nn.Linear(1, 1))
    max_epochs = max_epochs or len(vals)
    for ep, v in enumerate(vals):
        cb.on_train_epoch_end(_Trainer(ep, max_epochs, v), module)
    return cb, tmp_path / "checkpoints" / "m"


def test_best_tracks_validation_minimum_not_last(tmp_path, monkeypatch):
    # val loss: improves to epoch 3 (index 2), then diverges and stays high
    cb, d = _run(tmp_path, monkeypatch, [3e-4, 1.5e-4, 7e-5, 2.5e-4, 1e-2])
    assert sorted(p.name for p in d.glob("weights_epoch_*.pt")) == sorted(
        f"weights_epoch_{k}.pt" for k in range(1, 6))
    assert (d / "weights_final.pt").read_text() == "weights_epoch_5.pt"
    assert (d / "final_epoch.txt").read_text() == "5\n"
    assert (d / "weights_best.pt").read_text() == "weights_epoch_3.pt"
    epoch, loss = (d / "best_epoch.txt").read_text().split()
    assert epoch == "3" and float(loss) == pytest.approx(7e-5)
    assert cb.best_epoch == 3


def test_best_follows_a_monotone_run_to_the_end(tmp_path, monkeypatch):
    _, d = _run(tmp_path, monkeypatch, [3e-4, 2e-4, 1e-4])
    assert (d / "weights_best.pt").read_text() == "weights_epoch_3.pt"
    assert (d / "weights_final.pt").read_text() == "weights_epoch_3.pt"


def test_no_validation_metric_means_no_best_file(tmp_path, monkeypatch):
    _, d = _run(tmp_path, monkeypatch, [None, None])
    assert (d / "weights_final.pt").read_text() == "weights_epoch_2.pt"
    assert not (d / "weights_best.pt").exists()
    assert not (d / "best_epoch.txt").exists()
