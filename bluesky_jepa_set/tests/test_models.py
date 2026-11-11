"""Unit tests (CPU, no BlueSky). Run: python -m pytest tests -q, or python -m tests.run_tests."""
import copy
import os
import tempfile
import torch
from jepa_set.obs import OWN_KEYS, AC_KEYS, ZONE_KEYS, EDGE_KEYS, ALL_KEYS
from jepa_set.encoders import build_encoder, n_params
from jepa_set.policy import Actor, SetCritic, Jepa, save_checkpoint, load_actor
from jepa_set.data import history_index
from jepa_set.common import pass_criterion

GROUPS = {"ac": AC_KEYS, "zone": ZONE_KEYS, "edge": EDGE_KEYS}
TOL = 1e-5


def make_obs(b=4, n_ac=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    sizes = {**{k: 1 for k in OWN_KEYS}, **{k: n_ac for k in AC_KEYS}, **{k: 5 for k in ZONE_KEYS}, **{k: 12 for k in EDGE_KEYS}}
    return {k: torch.randn(b, sizes[k], generator=g) for k in ALL_KEYS}


def permute(obs, keys, perm):
    out = dict(obs)
    for k in keys:
        out[k] = obs[k][:, perm]
    return out


def test_tf_permutation_invariance():
    enc = build_encoder("set_tf").eval()
    obs = make_obs(n_ac=6)
    ref = enc(obs)
    for name, keys in GROUPS.items():
        n = obs[keys[0]].shape[1]
        out = enc(permute(obs, keys, torch.randperm(n, generator=torch.Generator().manual_seed(1))))
        assert torch.allclose(ref, out, atol=TOL), f"output changed after a permutation of {name}"


def test_tf_masked_entries_do_not_change_output():
    enc = build_encoder("set_tf").eval()
    obs = make_obs(n_ac=6)
    masks = {"ac": torch.zeros(4, 6, dtype=torch.bool), "zone": torch.zeros(4, 5, dtype=torch.bool),
             "edge": torch.zeros(4, 12, dtype=torch.bool)}
    masks["ac"][:, 4:] = True
    masks["zone"][:, 3:] = True
    masks["edge"][:, 10:] = True
    ref = enc(obs, masks)
    noisy = {k: v.clone() for k, v in obs.items()}
    for k in AC_KEYS:
        noisy[k][:, 4:] = 99.0
    for k in ZONE_KEYS:
        noisy[k][:, 3:] = -7.0
    for k in EDGE_KEYS:
        noisy[k][:, 10:] = 5.0
    assert torch.allclose(ref, enc(noisy, masks), atol=TOL)
    # zero rows are absent objects: same output as the shorter input
    short = {k: (v[:, :4] if k in AC_KEYS else v) for k, v in obs.items()}
    padded = {k: (torch.cat([v[:, :4], torch.zeros(4, 2)], 1) if k in AC_KEYS else v) for k, v in obs.items()}
    assert torch.allclose(enc(short), enc(padded), atol=TOL)


def test_variable_aircraft_count_one_set_of_weights():
    for kind in ["set_tf", "cnn"]:
        actor = Actor(kind).eval()
        critic = SetCritic(kind).eval()
        for n_ac in [1, 2, 9, 20]:
            d, z = actor.dist(make_obs(n_ac=n_ac))
            assert d.mean.shape == (4, 2) and torch.isfinite(d.mean).all()
        for n_agents in [1, 2, 5]:
            obs = {k: v.reshape(2, n_agents, -1) for k, v in make_obs(b=2 * n_agents, n_ac=max(n_agents - 1, 1)).items()}
            v = critic(obs, torch.ones(2, n_agents, dtype=torch.bool))
            assert v.shape == (2, n_agents) and torch.isfinite(v).all()


def test_no_nan_with_all_other_objects_absent():
    for kind in ["set_tf", "cnn"]:
        enc = build_encoder(kind).eval()
        obs = make_obs()
        for k in AC_KEYS + ZONE_KEYS + EDGE_KEYS:
            obs[k] = torch.zeros_like(obs[k])
        out = enc(obs)
        assert torch.isfinite(out).all(), kind
        out.sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in enc.parameters() if p.grad is not None), kind


def test_critic_masks_dead_aircraft():
    critic = SetCritic("set_tf").eval()
    obs = {k: v.reshape(2, 3, -1) for k, v in make_obs(b=6, n_ac=2).items()}
    alive = torch.tensor([[True, True, False], [True, False, False]])
    ref = critic(obs, alive)
    noisy = {k: v.clone() for k, v in obs.items()}
    for k in noisy:
        noisy[k][0, 2] = 50.0
        noisy[k][1, 1:] = -50.0
    out = critic(noisy, alive)
    assert torch.allclose(ref, out, atol=TOL)
    assert (out[~alive] == 0).all()
    assert torch.isfinite(critic(obs, torch.zeros(2, 3, dtype=torch.bool))).all()


def test_checkpoint_round_trip():
    for kind in ["set_tf", "cnn"]:
        actor = Actor(kind).eval()
        obs = make_obs()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            save_checkpoint(path, actor, SetCritic(kind), Jepa(actor.encoder), {"encoder": kind, "emb_dim": 64})
            loaded, ck = load_actor(path)
        assert torch.equal(actor.dist(obs)[0].mean, loaded.eval().dist(obs)[0].mean)
        assert ck["n_params_actor"] == n_params(actor)


def _jepa_batch(k=3):
    hist = {key: torch.stack([v] * k, 1) + 0.1 * torch.randn(8, k, v.shape[1]) for key, v in make_obs(b=8).items()}
    return hist, torch.rand(8, 2) * 2 - 1, make_obs(b=8, seed=5)


def test_jepa_target_gets_no_gradient():
    for kind in ["set_tf", "cnn"]:
        jepa = Jepa(build_encoder(kind))
        loss, stats = jepa.loss(*_jepa_batch())
        loss.backward()
        assert all(p.grad is None and not p.requires_grad for p in jepa.target.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in jepa.online.parameters())
        assert stats["spread"] > 0


def test_ema_update_changes_target():
    jepa = Jepa(build_encoder("set_tf"))
    before = copy.deepcopy(jepa.target.state_dict())
    with torch.no_grad():
        for p in jepa.online.parameters():
            p.add_(1.0)
    jepa.ema_update(tau=0.9)
    changed = [not torch.equal(before[k], v) for k, v in jepa.target.state_dict().items()]
    assert all(changed)
    k0 = next(iter(before))
    assert torch.allclose(jepa.target.state_dict()[k0], before[k0] + 0.1, atol=1e-6)


def test_parameter_counts_within_20_percent():
    a, b = n_params(Actor("cnn")), n_params(Actor("set_tf"))
    assert abs(a - b) / max(a, b) <= 0.20, (a, b)


def test_history_index():
    agent = [0, 1, 0, 1, 0, 0, 1]
    episode = [0, 0, 0, 0, 0, 1, 1]
    idx = history_index(agent, episode, 3)
    assert idx[4].tolist() == [0, 2, 4]      # aircraft 0, episode 0, third frame
    assert idx[3].tolist() == [1, 1, 3]      # short history: first frame repeated
    assert idx[5].tolist() == [5, 5, 5]      # new episode: no frame of the old episode


def test_pass_criterion():
    bad = {"waypoint_reached": 0.0, "intrusion_events": 1.0, "restricted_area_events": 0.0,
           "sector_exit_events": 1.0, "flight_time": 3000.0, "ep_return": -15000.0}
    good = {"waypoint_reached": 1.0, "intrusion_events": 0.0, "restricted_area_events": 0.0,
            "sector_exit_events": 0.0, "flight_time": 1000.0, "ep_return": -5000.0}
    eps = [{"env_steps": 100 * (i + 1), **(bad if i < 30 else good)} for i in range(60)]
    res = pass_criterion(eps)
    # window of episodes 29..48 holds 1 bad episode in 20: 0.95 and 0.05 meet the limits
    assert res["pass"] and res["steps_to_criterion"] == 4900
    assert not pass_criterion(eps[:45])["pass"]
    assert not pass_criterion(eps[:10])["pass"]
