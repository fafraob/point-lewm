"""Unit tests for the Delta-JEPA adaptation (LDAD, from the Delta-JEPA paper).

Run with:  pixi run python -m pytest tests/ -q
The integration test composes the real train config and runs the real
``lejepa_forward`` on a fake packed point-cloud batch (CPU).
"""

import pytest
import torch
import torch.nn.functional as F

from module import Attention, LatentDifferenceActionDecoder


def tiny_decoder(horizon=5, latent_dim=16, action_dim=6, seed=0):
    torch.manual_seed(seed)
    return LatentDifferenceActionDecoder(
        latent_dim=latent_dim,
        action_dim=action_dim,
        horizon=horizon,
        depth=2,
        heads=2,
        dim_head=8,
        mlp_dim=32,
    )


# ------------------------------------------------------------------- LDAD --


def test_ldad_output_shape():
    dec = tiny_decoder()
    out = dec(torch.randn(7, 16))
    assert out.shape == (7, 5, 6)


def test_attention_causal_flag():
    torch.manual_seed(0)
    attn = Attention(dim=16, heads=2, dim_head=8).eval()
    x = torch.randn(1, 4, 16)
    x_mod = x.clone()
    # perturb only the last token (non-constant: the input LayerNorm would
    # absorb a uniform shift)
    x_mod[:, -1] += torch.randn(16)
    # causal: earlier tokens cannot see the perturbation
    assert torch.allclose(attn(x, causal=True)[:, :-1], attn(x_mod, causal=True)[:, :-1])
    # non-causal (LDAD's mode): every query attends to every other one
    assert not torch.allclose(attn(x, causal=False)[:, 0], attn(x_mod, causal=False)[:, 0])


def test_ldad_loss_single_span():
    """T = N + 1 frames: exactly one span, delta = z_N - z_0 (paper's Eq. 6)."""
    dec = tiny_decoder(horizon=5)
    emb = torch.randn(3, 6, 16)
    actions = torch.randn(3, 6, 6)
    expected = F.mse_loss(dec(emb[:, 5] - emb[:, 0]), actions[:, 0:5])
    assert torch.allclose(dec.loss(emb, actions), expected)


def test_ldad_loss_all_spans():
    """T > N + 1: every start offset is supervised (stacked into the batch)."""
    dec = tiny_decoder(horizon=2)
    emb = torch.randn(3, 4, 16)
    actions = torch.randn(3, 4, 6)
    delta = torch.cat([emb[:, 2] - emb[:, 0], emb[:, 3] - emb[:, 1]], dim=0)
    target = torch.cat([actions[:, 0:2], actions[:, 1:3]], dim=0)
    expected = F.mse_loss(dec(delta), target)
    assert torch.allclose(dec.loss(emb, actions), expected)


def test_ldad_gradients_reach_conditioning_and_queries():
    """AdaLN-zero starts gated shut, but its projection (the path that lets the
    displacement steer the decode) must receive gradient from step one, as must
    the action queries and output head."""
    dec = tiny_decoder()
    emb = torch.randn(2, 6, 16, requires_grad=True)
    dec.loss(emb, torch.randn(2, 6, 6)).backward()
    assert dec.action_queries.grad.abs().sum() > 0
    assert dec.transformer.output_proj.weight.grad.abs().sum() > 0
    for block in dec.transformer.layers:
        assert block.adaLN_modulation[-1].weight.grad.abs().sum() > 0
    # the encoder embeddings get gradient through the (ungated) query pathway
    assert emb.grad is not None and torch.isfinite(emb.grad).all()


def test_ldad_learns_displacement_conditioning():
    """A few optimizer steps must make the output depend on the displacement
    (gate opens) and fit a fixed (delta -> actions) mapping."""
    dec = tiny_decoder()
    delta = torch.randn(32, 16)
    actions = torch.randn(32, 5, 6)
    opt = torch.optim.Adam(dec.parameters(), lr=1e-2)
    first = None
    for _ in range(150):
        loss = F.mse_loss(dec(delta), actions)
        first = loss.item() if first is None else first
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first
    # conditioning is now live: different displacements give different actions
    out = dec(torch.zeros(2, 16) + torch.tensor([[0.0], [1.0]]))
    assert not torch.allclose(out[0], out[1])


# ------------------------------------------------- train-forward integration --


def test_ldad_solver_executes_decoded_plan():
    """planner=ldad: LDADSolver returns the adapter's decoded plan verbatim
    (full horizon, no CEM), and receding-horizon prefixes are honored."""
    eval_lidar = pytest.importorskip("eval_lidar")
    gym = pytest.importorskip("gymnasium")
    from types import SimpleNamespace

    class FakePlanModel:
        """Actionable stand-in: step index baked into each decoded action."""

        def get_action(self, info, horizon=1, prefix_actions=None):
            b = len(next(iter(info.values())))
            n_prev = 0 if prefix_actions is None else prefix_actions.shape[1]
            steps = torch.arange(n_prev, n_prev + horizon, dtype=torch.float32)
            return steps.view(1, horizon, 1).expand(b, horizon, 2).clone()

    solver = eval_lidar.LDADSolver(model=FakePlanModel(), device="cpu")
    solver.configure(
        action_space=gym.spaces.Box(-1, 1, shape=(3, 2)),
        n_envs=3,
        config=SimpleNamespace(horizon=5, action_block=1),
    )
    assert solver.action_dim == 2 and solver.horizon == 5

    out = solver.solve({"obs": torch.zeros(3, 4)})
    assert out["actions"].shape == (3, 5, 2)
    # the decoded plan comes through unrefined: step k -> action value k
    assert torch.equal(out["actions"][0, :, 0], torch.arange(5, dtype=torch.float32))

    # receding-horizon leftover plan: prefix is kept, tail decoded after it
    prefix = torch.full((3, 2, 2), 9.0)
    out = solver.solve({"obs": torch.zeros(3, 4)}, init_action=prefix)
    assert out["actions"].shape == (3, 5, 2)
    assert torch.equal(out["actions"][:, :2], prefix)
    assert torch.equal(out["actions"][0, 2:, 0], torch.tensor([2.0, 3.0, 4.0]))


@pytest.mark.parametrize(
    "config_name",
    [
        "point_delta_jepa_cube",
        "point_delta_jepa_tworoom",
        "point_delta_jepa_pusht",
        "point_delta_jepa_reacher",
    ],
)
def test_point_delta_jepa_forward_integration(config_name):
    """Compose each shipped point_delta_jepa_<env> config (config/train/), build the
    real model, and run the real training forward on a fake packed
    point-cloud batch."""
    hydra = pytest.importorskip("hydra")
    from omegaconf import open_dict

    import train as train_mod
    from pc_encoders.collate import collate_point_cloud

    with hydra.initialize(config_path="../config/train", version_base=None):
        cfg = hydra.compose(config_name=config_name)

    n_frames = cfg.data.dataset.num_steps
    assert n_frames == cfg.ldad_horizon + 1 == 6
    # full-window teacher forcing: prediction loss covers every adjacent pair,
    # so no frame is gradient-dead (predictor pos-embedding follows suit)
    assert cfg.history_size + cfg.num_preds == n_frames
    assert cfg.model.predictor.num_frames == cfg.history_size == 5

    act_dim = 10  # stands in for frameskip * action_dim, set by train.py
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = act_dim
        cfg.model.action_decoder.action_dim = act_dim
        # shrink the encoder so the CPU forward stays fast
        cfg.model.encoder.num_tokens = 16
        cfg.model.encoder.group_size = 8
    # faithful Delta-JEPA: no SIGReg (LDAD is the anti-collapse mechanism)
    assert cfg.loss.get("sigreg") is None

    torch.manual_seed(0)
    model = hydra.utils.instantiate(cfg.model)

    samples = [
        {"points": torch.rand(n_frames, 200, 3) * 0.2 + torch.tensor([1.27, 0.0, 0.25]),
         "action": torch.randn(n_frames, act_dim)}
        for _ in range(2)
    ]
    batch = collate_point_cloud(samples, point_key="points")

    class Shim:
        def __init__(self, model):
            self.model = model
            self.logged = {}

        def log_dict(self, d, **kw):
            self.logged.update(d)

    shim = Shim(model)
    out = train_mod.lejepa_forward(shim, batch, "fit", cfg)

    total = out["pred_loss"] + cfg.loss.action.weight * out["action_loss"]
    assert set(k for k in out if k.endswith("loss")) == {"pred_loss", "action_loss", "loss"}
    assert torch.isfinite(out["loss"])
    # in the fit stage the total is divided by accumulate_grad_batches (micro-
    # batch gradients SUM over the window; see lejepa_forward) -- the logged
    # component losses are pre-division
    accumulate = cfg.get("accumulate_grad_batches", 1)
    assert torch.allclose(out["loss"] * accumulate, total)
    assert "fit/action_loss" in shim.logged

    # one backward: the encoder must receive gradient from BOTH objectives
    out["loss"].backward()
    enc_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert enc_grads and any(g.abs().sum() > 0 for g in enc_grads)
    assert model.action_decoder.action_queries.grad is not None
