"""Unit tests without pytest and without BlueSky: python -m tests.run_tests (about 2 seconds)."""
import sys
import time
import traceback
import numpy as np
import torch
from rl_tf.lidar import Lidar, NM2KM, GEOM_KEY, to_own_frame
from rl_tf.envs import ObsBuilder, obs_dims
from rl_tf.networks import Policy, LidarTransformerEncoder
from rl_tf.common import compute_gae


def square(cx, cy, half):
    return np.array([[cx + half, cy + half], [cx + half, cy - half], [cx - half, cy - half], [cx - half, cy + half]])


def fake_obs(zone=None, ac=None, goal=(500.0, 500.0), half=50.0):
    """Own aircraft at the centre of a square sector (side 2 * half km), own frame (x = nose, y = starboard).
    zone = (x, y, half side) of a square zone in km. ac = (x, y) km. goal = (x, y) km, capture radius 5 km."""
    o = {k: np.zeros(1) for k in ["airspeed", "cos_drift", "sin_drift", "waypoint_distance"]}
    o["inside_sector"] = np.ones(1)
    o[GEOM_KEY] = dict(zones=[square(*zone)] if zone else [], sector=square(0.0, 0.0, half),
                       aircraft=np.array([ac]) if ac else np.zeros((0, 2)), goal=np.array(goal), goal_radius=5.0)
    return o


def test_lidar_zone_ahead_and_starboard():
    lid = Lidar(n_rays=120, max_range_km=100.0)
    d = lid.distances(fake_obs(zone=(50.0, 0.0, 10.0)))
    assert abs(d[0, 60] - 40.0) < 1e-6 and np.isinf(d[0, 0])        # ray 60 = nose; ray 0 = tail
    d = lid.distances(fake_obs(zone=(0.0, 30.0, 10.0)))
    assert abs(d[0, 90] - 20.0) < 1e-6 and np.isinf(d[0, 30])       # ray 90 = starboard; ray 30 = port


def test_lidar_one_hit_per_ray_and_sector():
    lid = Lidar(n_rays=120, max_range_km=100.0)
    d = lid.distances(fake_obs(ac=(30.0, 0.0)))
    assert abs(d[2, 60] - (30.0 - 5.0 * NM2KM)) < 1e-6
    assert np.isinf(d[0]).all()                                     # no zone
    assert abs(d[1, 60] - 50.0) < 1e-6 and abs(d[1, 90] - 50.0) < 1e-6 and abs(d[1, 0] - 50.0) < 1e-6
    assert np.isfinite(d[1]).all()                                  # inside a closed sector every ray hits
    s = lid.scan(fake_obs(ac=(30.0, 0.0)))
    assert s.shape == (3, 120) and s.dtype == np.float32 and 0.0 <= s.min() and s.max() <= 1.0
    assert ((s > 0).sum(0) <= 1).all()                              # reference: one channel per ray at most
    assert abs(s[2, 60] - (1.0 - (30.0 - 5.0 * NM2KM) / 100.0)) < 1e-6 and s[1, 60] == 0.0   # aircraft hides the edge
    assert abs(s[1, 90] - 0.5) < 1e-6 and s[2, 90] == 0.0


def test_lidar_goal_inside_and_fov():
    s = Lidar().scan(fake_obs(goal=(20.0, 0.0)))
    assert (s[:, 60] == 0.0).all()                                  # nearest hit = own goal: all channels 0
    d = Lidar().distances(fake_obs(zone=(5.0, 0.0, 10.0)))         # inside the zone: distance to the exit edge
    assert abs(d[0, 60] - 15.0) < 1e-6 and abs(d[0, 0] - 5.0) < 1e-6   # zone spans x = -5 to 15 km
    d = Lidar().distances(fake_obs(ac=(3.0, 0.0)))                  # inside a disc: exit point (reference rule)
    assert abs(d[2, 60] - (3.0 + 5.0 * NM2KM)) < 1e-6
    lid = Lidar(n_rays=8, fov_deg=210.0)
    assert lid.angles_deg[0] == -105.0 and lid.angles_deg[-1] == 105.0


def test_own_frame():
    east = to_own_frame(0.0, 0.1, 0.0, 0.0, 90.0)                   # heading east, point to the east
    south = to_own_frame(-0.1, 0.0, 0.0, 0.0, 90.0)                 # heading east, point to the south = starboard
    assert east[0] > 11.0 and abs(east[1]) < 1e-9 and south[1] > 11.0 and abs(south[0]) < 1e-9


def test_inside_zone_flag():
    from rl_tf.lidar import inside_zone, own_state
    assert inside_zone(fake_obs(zone=(5.0, 0.0, 10.0))) == 1.0      # own aircraft inside the square zone
    assert inside_zone(fake_obs(zone=(50.0, 0.0, 10.0))) == 0.0     # zone ahead
    assert inside_zone(fake_obs(zone=(-50.0, 3.0, 10.0))) == 0.0    # zone behind
    assert inside_zone(fake_obs()) == 0.0
    s = own_state(fake_obs(zone=(5.0, 0.0, 10.0)), (0.5, -0.5), zone_flag=True)
    assert s.shape == (8,) and s[5] == 1.0 and s[6] == 0.5 and s[7] == -0.5
    assert own_state(fake_obs(), (0.0, 0.0)).shape == (7,)


def test_lidar_speed_channel_and_radius():
    o = fake_obs(ac=(30.0, 0.0), half=500.0)                            # aircraft 30 km ahead, sector far away
    o[GEOM_KEY]["aircraft_vel"] = np.array([[-175.0, 40.0]])            # comes nearer at 175 m/s (the side part has no effect)
    s = Lidar(n_rays=120, max_range_km=50.0, speed=True).scan(o)
    assert s.shape == (4, 120) and s[2, 60] > 0 and abs(s[3, 60] - 0.5) < 1e-6      # 175 / 350
    assert np.all(s[3][s[2] == 0] == 0) and s[1].max() == 0            # speed only on aircraft rays; boundary out of range
    o[GEOM_KEY]["aircraft_vel"] = np.array([[175.0, 0.0]])
    assert abs(Lidar(max_range_km=50.0, speed=True).scan(o)[3, 60] + 0.5) < 1e-6    # it moves away: negative
    assert Lidar(max_range_km=20.0, speed=True).scan(o).max() == 0     # 30 km - 9.26 km radius > 20 km: outside the lidar radius
    assert Lidar(max_range_km=50.0).scan(o).shape == (3, 120)          # no speed channel: 3 channels, as before
    assert obs_dims(dict(n_rays=120, speed=True), 4)["lidar_ch"] == 16 and obs_dims(dict(n_rays=120), 4)["lidar_ch"] == 12


def test_aircraft_tokens():
    o = fake_obs(ac=(30.0, 0.0), half=500.0)
    o[GEOM_KEY]["aircraft"] = np.array([[30.0, 0.0], [0.0, -10.0], [200.0, 0.0]])     # the third is out of the lidar radius
    o[GEOM_KEY]["aircraft_vel"] = np.array([[-175.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    lid = Lidar(max_range_km=50.0, speed=True, ac_tokens=4)
    t = lid.aircraft_tokens(o); f, flag = t[:24].reshape(4, 6), t[24:]
    reach = 50.0 + lid.ac_radius
    assert list(flag) == [1.0, 1.0, 0.0, 0.0] and np.all(f[2:] == 0)                   # 2 aircraft in reach, nearest first
    assert abs(f[0, 1] + 10.0 / reach) < 1e-6 and abs(f[0, 4] - 10.0 / reach) < 1e-6   # row 0: 10 km to port
    assert abs(f[1, 0] - 30.0 / reach) < 1e-6 and abs(f[1, 5] - 0.5) < 1e-6            # row 1: ahead, closing 175 / 350
    d = obs_dims(dict(n_rays=120, speed=True, ac_tokens=4), 4)
    b = ObsBuilder(lid, 4); b.reset(o)
    assert d["state_dim"] == 32 + 28 and b.get()[0].shape == (60,) and d["ac_tokens"] == 4
    torch.manual_seed(0)
    pol = Policy(state_dim=60, lidar_ch=16, n_rays=120, ac_tokens=4)
    s, l = torch.as_tensor(b.get()[0])[None, None], torch.as_tensor(b.get()[1])[None, None]
    m0 = pol(s, l)[0].mean
    s2 = s.clone(); s2[..., 32 + 12:32 + 24] = 7.0                                     # values in the two empty rows
    assert torch.allclose(m0, pol(s2, l)[0].mean, atol=1e-6)                           # masked: no effect
    s3 = s.clone(); s3[..., 32 + 5] = -0.5                                             # closing speed of a real aircraft
    assert not torch.allclose(m0, pol(s3, l)[0].mean, atol=1e-6)                       # used
    s4 = s.clone(); s4[..., 32 + 24:] = 0.0                                            # no aircraft at all: no NaN
    assert torch.isfinite(pol(s4, l)[0].mean).all()


def test_zone_goal_and_action_tokens():
    from rl_tf.lidar import ZONE_FEATURES as ZF, GOAL_FEATURES as GF, DIST_NORM_KM as N, extra_dim
    o = fake_obs(zone=(40.0, 0.0, 10.0), goal=(100.0, 0.0), half=500.0)                # square area ahead, goal behind it
    o[GEOM_KEY]["zones"].append(square(0.0, -80.0, 10.0))                              # second area to port, farther
    o[GEOM_KEY]["time_left"] = 0.25
    lid = Lidar(max_range_km=50.0, speed=True, zone_tokens=3, act_tokens=2)
    t = lid.zone_tokens_vec(o)
    f, flag, goal = t[:3 * ZF].reshape(3, ZF), t[3 * ZF:3 * ZF + 3], t[3 * ZF + 3:]
    assert list(flag) == [1.0, 1.0, 0.0] and np.all(f[2] == 0) and len(goal) == GF
    assert abs(f[0, 0] - 30.0 / N) < 1e-6 and abs(f[0, 1]) < 1e-6 and abs(f[0, 2] - 30.0 / N) < 1e-6   # nearest edge point: 30 km ahead
    assert f[0, 4] < 0 < f[0, 7] and abs(f[0, 4] + f[0, 7]) < 1e-6                     # port end to port, starboard end to starboard
    assert abs(f[0, 4] + 10.0 / np.hypot(30.0, 10.0)) < 1e-6                           # port end = the corner (30, -10)
    assert f[0, 9] == 1.0 and f[0, 10] == 0.0                                          # it blocks the line to the goal; not inside
    assert f[1, 9] == 0.0 and abs(f[1, 2] - 70.0 / N) < 1e-6                           # the second area does not block
    assert abs(goal[2] - 100.0 / N) < 1e-6 and goal[3] == 1.0 and abs(goal[5] - 0.25) < 1e-6
    o_in = fake_obs(zone=(0.0, 0.0, 10.0), half=500.0)
    assert lid.zone_tokens_vec(o_in)[10] == 1.0 and lid.zone_tokens_vec(o_in)[9] == 1.0   # inside the area
    d = obs_dims(dict(n_rays=120, speed=True, zone_tokens=3, act_tokens=2), 4)
    extra = extra_dim(0, 3, 2)
    assert extra == 3 * (ZF + 1) + GF + 2 * 3 and d["state_dim"] == 32 + extra
    b = ObsBuilder(lid, 4); b.reset(o)
    assert b.get()[0].shape == (d["state_dim"],) and np.all(b.get()[0][-6:] == 0)       # no action yet: empty rows
    b.push(o, (0.5, -0.25)); st = b.get()[0]
    assert list(st[-6:]) == [0.0, 0.0, 0.5, -0.25, 0.0, 1.0]                            # oldest first; flags last
    torch.manual_seed(0)
    pol = Policy(state_dim=d["state_dim"], lidar_ch=16, n_rays=120, zone_tokens=3, act_tokens=2)
    s, l = torch.as_tensor(st)[None, None], torch.as_tensor(b.get()[1])[None, None]
    m0 = pol(s, l)[0].mean
    z0 = 32
    s2 = s.clone(); s2[..., z0 + 2 * ZF:z0 + 3 * ZF] = 7.0                              # values in the empty area row
    s2[..., -6:-4] = 7.0                                                                # values in the empty action row
    assert torch.allclose(m0, pol(s2, l)[0].mean, atol=1e-6)                            # masked: no effect
    for i in [z0 + 9, z0 + 3 * ZF + 3 + 5, -3]:                                         # blocks flag, time left, last turn
        s3 = s.clone(); s3[..., i] += 0.5
        assert not torch.allclose(m0, pol(s3, l)[0].mean, atol=1e-6)                    # used
    torch.manual_seed(0); base = Policy(state_dim=32, lidar_ch=16, n_rays=120)
    torch.manual_seed(0); new = Policy(state_dim=d["state_dim"], lidar_ch=16, n_rays=120, zone_tokens=3, act_tokens=2)
    sb, sn = base.state_dict(), new.state_dict()
    assert all(torch.equal(sb[k], sn[k]) for k in sb if k.startswith("encoder."))       # the shared encoder layers start as the baseline


def test_edge_cpa_fourier_and_readout_tokens():
    from rl_tf.lidar import EDGE_FEATURES as EF, AC_FEATURES as AF, DIST_NORM_KM as N, CPA_NORM_S, NM2KM, extra_dim
    o = fake_obs(half=500.0)
    o[GEOM_KEY]["aircraft"] = np.array([[20.0, 0.0], [0.0, 30.0]])                     # one ahead, one to starboard
    o[GEOM_KEY]["aircraft_vel"] = np.array([[-100.0, 0.0], [0.0, 50.0]])               # the first comes, the second goes
    lid = Lidar(max_range_km=50.0, speed=True, ac_tokens=3, ac_cpa=1, edge_tokens=8)
    reach = 50.0 + 5.0 * NM2KM
    t = lid.aircraft_tokens(o); f = t[:3 * (AF + 2)].reshape(3, AF + 2)
    assert abs(f[0, 6] - 200.0 / CPA_NORM_S) < 1e-6 and abs(f[0, 7]) < 1e-6            # 20 km at 100 m/s: 200 s, distance 0
    assert f[1, 6] == 0.0 and abs(f[1, 7] - 30.0 / reach) < 1e-6                       # goes away: time 0, distance of now
    e = lid.edge_tokens_vec(o); ef, flag = e[:8 * EF].reshape(8, EF), e[8 * EF:]
    assert list(flag) == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]                      # square sector: ahead, starboard, behind, port
    assert abs(ef[0, 0] - 500.0 / N) < 1e-6 and abs(ef[0, 2] - 500.0 / N) < 1e-6 and ef[0, 3] == 1.0 and ef[0, 5] == 0.0
    assert abs(ef[2, 1] - 500.0 / N) < 1e-6 and abs(ef[6, 1] + 500.0 / N) < 1e-6       # bin 2 = starboard, bin 6 = port
    o["inside_sector"] = np.zeros(1)
    assert lid.edge_tokens_vec(o)[5] == 1.0                                            # outside flag
    o["inside_sector"] = np.ones(1)
    cfg = dict(n_rays=120, speed=True, ac_tokens=3, ac_cpa=1, edge_tokens=8)
    d = obs_dims(cfg, 4)
    assert d["state_dim"] == 32 + extra_dim(3, 0, 0, 8, 1) == 32 + 3 * 9 + 8 * 7
    b = ObsBuilder(lid, 4); b.reset(o); st = b.get()[0]
    assert st.shape == (d["state_dim"],)
    s, l = torch.as_tensor(st)[None, None], torch.as_tensor(b.get()[1])[None, None]
    for kw in [dict(), dict(fourier=4), dict(cls_token=1), dict(fourier=4, cls_token=1, d_model=128, n_layers=3)]:
        torch.manual_seed(0)
        pol = Policy(state_dim=d["state_dim"], lidar_ch=16, n_rays=120, ac_tokens=3, ac_cpa=1, edge_tokens=8, **kw)
        m0 = pol(s, l)[0].mean
        assert torch.isfinite(m0).all()
        s2 = s.clone(); s2[..., 32 + 2 * 8:32 + 3 * 8] = 7.0                            # empty aircraft row
        s2[..., 32 + 27 + EF:32 + 27 + 2 * EF] = 7.0                                    # empty edge row (bin 1)
        assert torch.allclose(m0, pol(s2, l)[0].mean, atol=1e-6)                        # masked: no effect
        for i in [32 + 6, 32 + 27 + 2]:                                                 # time to closest point; edge distance
            s3 = s.clone(); s3[..., i] += 0.5
            assert not torch.allclose(m0, pol(s3, l)[0].mean, atol=1e-6)                # used
    torch.manual_seed(0); old = Policy(state_dim=32 + 3 * 7, lidar_ch=16, n_rays=120, ac_tokens=3)
    torch.manual_seed(0); new = Policy(state_dim=32 + 3 * 7, lidar_ch=16, n_rays=120, ac_tokens=3, edge_tokens=0,
                                       ac_cpa=0, fourier=0, cls_token=0)
    so, sn = old.state_dict(), new.state_dict()
    assert so.keys() == sn.keys() and all(torch.equal(so[k], sn[k]) for k in so)       # options at 0: the same model


def test_frame_stack():
    lid = Lidar(n_rays=16)
    b = ObsBuilder(lid, history=4)
    b.reset(fake_obs())
    s, l = b.get()
    d = obs_dims(dict(n_rays=16), 4)
    assert s.shape == (d["state_dim"],) and l.shape == (d["lidar_ch"], 16)
    assert (l[:9] == 0).all() and (s[:21] == 0).all()               # older frames are zeros (supersuit frame_stack)
    assert (l[9:] == lid.scan(fake_obs())).all()
    b.push(fake_obs(zone=(50.0, 0.0, 10.0)), (0.5, -0.5))
    s2, l2 = b.get()
    assert (l2[:6] == 0).all() and (l2[6:9] == l[9:]).all() and (l2[9] != l[9]).any()   # shift by one frame
    assert s2[-2] == 0.5 and s2[-1] == -0.5                         # previous action is in the newest frame


def _policy():
    torch.manual_seed(0)
    return Policy(state_dim=28, lidar_ch=12, n_rays=120)


def test_policy_shapes_and_shared_value():
    pol = _policy()
    state, lidar = torch.randn(5, 3, 28), torch.rand(5, 3, 12, 120)
    d, v = pol(state, lidar)
    assert d.mean.shape == (5, 3, 2) and v.shape == (5,) and torch.isfinite(v).all()
    assert torch.allclose(d.stddev, torch.ones_like(d.stddev))      # log_std init 0 (reference)
    _, _, va = pol.act(state, lidar)
    assert va.shape == (5, 3) and (va == v[:, None]).all()          # one environment value, repeated per aircraft
    perm = torch.tensor([2, 0, 1])
    assert torch.allclose(pol(state[:, perm], lidar[:, perm])[1], v, atol=1e-5)   # mean over aircraft: order-free


def test_actor_is_decentralized_critic_is_centralized():
    pol = _policy()
    state, lidar = torch.randn(1, 2, 28), torch.rand(1, 2, 12, 120)
    d, v = pol(state, lidar)
    lidar2 = lidar.clone()
    lidar2[0, 1] = 0.0
    d2, v2 = pol(state, lidar2)
    assert torch.allclose(d.mean[0, 0], d2.mean[0, 0], atol=1e-6)   # actor 0 does not see aircraft 1
    assert not torch.allclose(v, v2, atol=1e-6)                     # the critic does


def test_encoder_reads_ray_position():
    torch.manual_seed(0)
    enc = LidarTransformerEncoder(28, 12, 120)
    with torch.no_grad():
        enc.pos.normal_(0.0, 0.5)
    state, lidar = torch.zeros(1, 28), torch.zeros(1, 12, 120)
    a, b = lidar.clone(), lidar.clone()
    a[0, :, 60] = 1.0
    b[0, :, 20] = 1.0
    assert not torch.allclose(enc(state, a), enc(state, b), atol=1e-5)


def test_gae_reference_rule():
    rew = np.ones((3, 1), np.float32)
    val, last = np.zeros((3, 1), np.float32), np.full(1, 10.0, np.float32)
    done = np.zeros((3, 1), bool)
    adv, _ = compute_gae(rew, val, done, last, 1.0, 1.0)
    assert np.allclose(adv[:, 0], [13, 12, 11])                     # end of rollout: bootstrap
    done[2] = True
    adv, _ = compute_gae(rew, val, done, last, 1.0, 1.0)
    assert np.allclose(adv[:, 0], [3, 2, 1])                        # environment end: no bootstrap
    done[:] = False
    done[0] = True
    val[1] = 5.0
    adv, _ = compute_gae(rew, val, done, last, 1.0, 1.0)
    assert np.allclose(adv[:, 0], [1, 7, 11])                       # done at step 0: no carry, no bootstrap


def test_reward_has_only_negative_weights():
    from rl_tf.envs import REWARD_DEFAULTS, check_reward
    assert "reach_reward" not in REWARD_DEFAULTS and all(v <= 0.0 for v in REWARD_DEFAULTS.values())
    check_reward(REWARD_DEFAULTS)
    try:
        check_reward({**REWARD_DEFAULTS, "time_penalty": 1.0})
    except ValueError:
        return
    raise AssertionError("a positive weight was accepted")


def test_ppo_step_learns_a_target_action():
    pol = _policy()
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3)
    state, lidar = torch.randn(64, 1, 28), torch.rand(64, 1, 12, 120)
    target = torch.tensor([0.5, -0.5])
    first = None
    for _ in range(60):
        d, _ = pol(state, lidar)
        loss = (d.mean - target).pow(2).mean()
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * first


def test_jepa_loss_start_and_target_update():
    from rl_tf.jepa import TemporalJepa
    pol = _policy()
    jepa = TemporalJepa(pol.encoder, hidden=32)
    state, lidar, act = torch.randn(8, 3, 28), torch.rand(8, 3, 12, 120), torch.rand(8, 2, 2) * 2 - 1
    z = pol.encoder(state[:, 0], lidar[:, 0])
    assert torch.allclose(jepa.step(z, act[:, 0]), z)               # start: the predictor gives "no change"
    loss, info = jepa.loss(state, lidar, act)
    assert torch.isfinite(loss) and len(info["pred_per_k"]) == 2
    loss.backward()
    assert all(p.grad is None for p in jepa.target.parameters())    # no gradient into the slow copy
    with torch.no_grad():
        pol.encoder.dense[0].weight.add_(1.0)
    jepa.update_target(0.0)                                         # tau 0: the slow copy becomes the encoder
    assert all(torch.equal(a, b) for a, b in zip(jepa.online.parameters(), jepa.target.parameters()))


def test_jepa_windows_labels_probes_schedule():
    from rl_tf.jepa import window_starts, future_labels, auc, ridge_fit, ridge_predict, ema_tau
    has_next = np.array([1, 1, 0, 1, 1, 1, 0], np.uint8)            # flight A: rows 0-2, flight B: rows 3-6
    assert window_starts(has_next, 2).tolist() == [0, 3, 4]
    flag = np.array([0, 0, 0, 0, 0, 1, 0], np.uint8)
    label, ok = future_labels(flag, has_next, 3)
    assert label.tolist() == [False, False, False, True, True, True, False]
    assert ok.tolist() == [True, True, True, True, True, True, False]
    assert auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    assert auc(np.zeros(4), np.array([0, 1, 0, 1])) == 0.5
    x = np.random.default_rng(0).normal(size=(200, 3))
    w = ridge_fit(x, x @ np.array([1.0, -2.0, 0.5]) + 3.0, lam=1e-8)
    assert np.allclose(ridge_predict(w, x[:5]), x[:5] @ np.array([1.0, -2.0, 0.5]) + 3.0, atol=1e-4)
    assert abs(ema_tau(0, 100) - 0.996) < 1e-12 and ema_tau(100, 100) == 1.0


def test_curriculum_stages():
    from rl_tf.common import parse_curriculum, curriculum_stage
    st = parse_curriculum("5:3@300000, 2:1@0,10:5@800000")
    assert st == [(0, 2, 1), (300000, 5, 3), (800000, 10, 5)] and parse_curriculum("") == []
    assert curriculum_stage(st, 0) == (2, 1) and curriculum_stage(st, 299999) == (2, 1)
    assert curriculum_stage(st, 300000) == (5, 3) and curriculum_stage(st, 5_000_000) == (10, 5)
    try:
        parse_curriculum("2:1@100")
    except ValueError:
        return
    raise AssertionError("a curriculum without a stage at step 0 was accepted")


def test_encoder_save_and_load():
    import tempfile
    from pathlib import Path
    from rl_tf.networks import save_encoder, load_encoder
    args = dict(patch=4, d_model=64, n_heads=4, n_layers=2, features_dim=256, fourier=0, cls_token=0)
    dims = dict(state_dim=28, lidar_ch=12, n_rays=120)
    a = _policy()
    torch.manual_seed(1)
    b = Policy(state_dim=28, lidar_ch=12, n_rays=120)
    state, lidar = torch.randn(4, 28), torch.rand(4, 12, 120)
    assert not torch.allclose(a.encoder(state, lidar), b.encoder(state, lidar))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "enc.pt"
        save_encoder(path, a.encoder, args, dims)
        load_encoder(b, path, dims, args)
        assert torch.equal(a.encoder(state, lidar), b.encoder(state, lidar))
        try:
            load_encoder(b, path, dims, {**args, "d_model": 128})
        except ValueError:
            return
    raise AssertionError("an encoder with other settings was accepted")


def _map_setup():
    """Own aircraft in a square sector (half side 100 km), zone x 40..80 km, y -20..20 km, goal 75 km to starboard.
    Map: 8 x 8 px over +-100 km (25 km per pixel); 2 aircraft tokens (both empty)."""
    lid = Lidar(max_range_km=50.0, speed=True, ac_tokens=2, map_px=8, map_range_km=100.0)
    o = fake_obs(zone=(60.0, 0.0, 20.0), goal=(0.0, 75.0), half=100.0)
    b = ObsBuilder(lid, 4)
    b.reset(o)
    s, l = b.get()
    return lid, o, torch.as_tensor(s)[None], torch.as_tensor(l)[None]


def test_map_image_and_tokens():
    from rl_tf.lidar import extra_dim, MAP_CHANNELS
    lid, o, s, l = _map_setup()
    m = lid.map_vec(o)
    img = m[1:].reshape(MAP_CHANNELS, 8, 8)
    assert m[0] == 1.0                                                  # no "time_left" key: 1
    assert img[0, 1, 3] == 1.0 and img[0, 1, 4] == 1.0 and img[0, 2, 3] == 0.5 and img[0, 0, 3] == 0.0
    assert abs(img[0].sum() - 3.0) < 1e-6                               # row 1 full, row 2 half (2 x 2 samples)
    assert img[1].sum() == 0.0                                          # the sector fills the image
    assert all(abs(img[2, i, j] - 0.25) < 1e-6 for i in (3, 4) for j in (6, 7)) and abs(img[2].sum() - 1.0) < 1e-6
    o2 = fake_obs(goal=(500.0, 500.0), half=50.0)
    img2 = lid.map_vec(o2)[1:].reshape(MAP_CHANNELS, 8, 8)
    assert img2[2].sum() == 0.0                                         # goal far outside the image: nothing
    assert img2[1, 0, 0] == 1.0 and img2[1, 3, 3] == 0.0 and img2[1].sum() == 64 - 16    # +-50 km sector: 4 x 4 inside
    d = obs_dims(dict(n_rays=120, speed=True, ac_tokens=2, map_px=8, map_range_km=100.0), 4)
    assert d["state_dim"] == 32 + extra_dim(2) + 1 + 3 * 64 == s.shape[1] and d["map_px"] == 8
    assert torch.equal(s[0, -192:], torch.as_tensor(img.reshape(-1))) and s[0, -193] == 1.0
    torch.manual_seed(0)
    pol = Policy(state_dim=d["state_dim"], lidar_ch=16, n_rays=120, ac_tokens=2, map_px=8, map_patch=4)
    enc = pol.encoder
    assert enc.n_tokens == 1 + 30 + 2 + 4 and enc.map_start == 33 and enc.lidar_start == 1
    m0 = pol(s[None], l[None])[0].mean
    assert torch.isfinite(m0).all()
    for i in [-192 + 2 * 8 + 6, -193]:                                  # one pixel of channel 0; the time left
        s3 = s.clone(); s3[0, i] += 0.5
        assert not torch.allclose(m0, pol(s3[None], l[None])[0].mean, atol=1e-6)
    out, pad = enc.token_outputs(s, l)
    assert out.shape == (1, enc.n_tokens, 64) and pad[0, 31:33].all() and not pad[0, 33:].any()
    torch.manual_seed(0); a = Policy(state_dim=28, lidar_ch=12, n_rays=120)
    torch.manual_seed(0); b = Policy(state_dim=28, lidar_ch=12, n_rays=120, map_px=0, map_patch=8)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)     # map off: the same model


def test_masked_jepa():
    from rl_tf.jepa import TemporalJepa
    lid, o, s, l = _map_setup()
    torch.manual_seed(0)
    pol = Policy(state_dim=s.shape[1], lidar_ch=16, n_rays=120, ac_tokens=2, map_px=8, map_patch=4)
    enc = pol.encoder
    plain = TemporalJepa(enc, hidden=32)
    assert not hasattr(plain, "mpred") and plain.predictor_params() == sum(
        q.numel() for q in list(plain.predictor.parameters()) + list(plain.act_embed.parameters()))
    jepa = TemporalJepa(enc, hidden=32, masked_coef=1.0, map_block=1, lidar_arc=3, mpred_dim=32, mpred_layers=1)
    idx = jepa.sample_targets(64, "cpu")
    assert idx.shape == (64, 4)
    assert ((idx[:, 0] >= enc.map_start) & (idx[:, 0] < enc.n_tokens)).all()
    lid_idx = idx[:, 1:] - enc.lidar_start
    assert ((lid_idx >= 0) & (lid_idx < 30)).all() and (((lid_idx[:, 1:] - lid_idx[:, :-1]) % 30) == 1).all()
    hide = torch.zeros(1, enc.n_tokens, dtype=torch.bool); hide[0, enc.lidar_start] = True     # rays 0-3
    l2 = l.clone(); l2[..., 0:4] = 0.9
    a, _ = enc.token_outputs(s, l, hide)
    b, _ = enc.token_outputs(s, l2, hide)
    vis = ~hide[0]
    assert torch.allclose(a[0, vis], b[0, vis], atol=1e-6)              # a hidden token gives no information
    assert not torch.allclose(enc.token_outputs(s, l)[0], enc.token_outputs(s, l2)[0], atol=1e-6)
    st = (s[:, None] + 0.01 * torch.randn(8, 3, s.shape[1])).clone()
    li = (l[:, None] + 0.01 * torch.rand(8, 3, *l.shape[1:])).clone()
    act = torch.rand(8, 2, 2) * 2 - 1
    loss, info = jepa.loss(st, li, act)
    assert torch.isfinite(loss) and info["masked"] > 0 and len(info["pred_per_k"]) == 2
    loss.backward()
    assert all(q.grad is None for q in jepa.target.parameters())
    assert jepa.mpred_in.weight.grad is not None and enc.map_embed.weight.grad is not None
    only = TemporalJepa(enc, hidden=32, temporal_coef=0.0, masked_coef=1.0, map_block=2, lidar_arc=0,
                        mpred_dim=32, mpred_layers=1)
    loss2, info2 = only.loss(st, li, act)
    assert info2["pred"] == 0.0 and info2["pred_per_k"] == [] and only.sample_targets(3, "cpu").shape == (3, 4)


def test_no_lidar_map_jepa():
    """n_rays = 0: own state, aircraft tokens and map image only; masked JEPA on map blocks only."""
    from rl_tf.jepa import TemporalJepa
    lid = Lidar(n_rays=0, max_range_km=50.0, speed=True, ac_tokens=2, map_px=8, map_range_km=100.0)
    o = fake_obs(zone=(60.0, 0.0, 20.0), goal=(0.0, 75.0), half=100.0)
    b = ObsBuilder(lid, 4); b.reset(o); b.push(o, (0.1, 0.0))
    s, l = b.get()
    d = obs_dims(dict(n_rays=0, speed=True, ac_tokens=2, map_px=8, map_range_km=100.0), 4)
    assert l.shape == (d["lidar_ch"], 0) and s.shape == (d["state_dim"],)
    s, l = torch.as_tensor(s)[None], torch.as_tensor(l)[None]
    torch.manual_seed(0)
    pol = Policy(state_dim=d["state_dim"], lidar_ch=16, n_rays=0, ac_tokens=2, map_px=8, map_patch=4)
    enc = pol.encoder
    assert enc.n_tokens == 1 + 2 + 4 and enc.map_start == 3 and not hasattr(enc, "patch_embed")
    dist, v = pol(s[None], l[None])
    assert dist.mean.shape == (1, 1, 2) and torch.isfinite(dist.mean).all() and torch.isfinite(v).all()
    jepa = TemporalJepa(enc, hidden=32, masked_coef=1.0, map_block=1, lidar_arc=6, mpred_dim=32, mpred_layers=1)
    assert jepa.lidar_arc == 0 and jepa.sample_targets(5, "cpu").shape == (5, 1)
    st, li = s[:, None].expand(4, 3, -1).clone(), l[:, None].expand(4, 3, -1, -1).clone()
    loss, info = jepa.loss(st + 0.01 * torch.randn_like(st), li, torch.zeros(4, 2, 2))
    assert torch.isfinite(loss) and info["masked"] > 0


def test_start_intrusions_left_out():
    """Fake competition environment: the official metric code (events at the rising edge, time per sample)."""
    from rl_tf.envs import track_start_intrusions

    class Fake:
        sim_dt = 1.0

        def __init__(self):
            self.seq, self.i = [], 0

        def _init_metrics(self, ac):
            self.m, self._intruders_in_intrusion, self.i = {"intrusion_events": 0, "intrusion_time": 0.0}, set(), 0

        def _update_metrics(self, ac, dt=None):
            dt = self.sim_dt if dt is None else dt
            now = set(self.seq[self.i]); self.i += 1
            self.m["intrusion_events"] += len(now - self._intruders_in_intrusion)
            self.m["intrusion_time"] += len(now) * dt
            self._intruders_in_intrusion = now

    env = Fake()
    out = track_start_intrusions(env)
    # A is in intrusion at reset, stays 2 s, goes out, comes back 1 s; B starts an intrusion at t = 3 for 2 s
    env.seq = [{"A"}, {"A"}, {"A"}, {"B"}, {"A", "B"}, set()]
    env._init_metrics("own"); env._update_metrics("own", dt=0.0)                # reset sample
    for _ in range(5):
        env._update_metrics("own")
    assert env.m == {"intrusion_events": 3, "intrusion_time": 5.0}             # official metrics unchanged
    assert out == {"start_intrusions": 1, "intrusion_events_no_start": 2, "intrusion_time_no_start": 3.0}
    env.seq = [set(), {"A"}]                                                    # next flight: no start intrusion
    env._init_metrics("own"); env._update_metrics("own", dt=0.0); env._update_metrics("own")
    assert out == {"start_intrusions": 0, "intrusion_events_no_start": 1, "intrusion_time_no_start": 1.0}


def _wm_batch(b=4, h=3):
    torch.manual_seed(2)
    st, li = torch.randn(b, h + 1, 28), torch.rand(b, h + 1, 12, 120)
    act = torch.rand(b, h, 2) * 2 - 1
    lab = torch.cat([torch.randint(0, 2, (b, h, 3)).float(), torch.randn(b, h, 3)], dim=-1)
    return st, li, act, lab


def test_world_model_start_z_and_loss():
    from rl_tf.world_model import WorldModel
    pol = _policy()
    wm = WorldModel(pol.encoder, n_layers=1, n_heads=4)
    st, li, act, lab = _wm_batch()
    h, s, pad = wm.encode(st[:, 0], li[:, 0])
    assert h.shape == (4, pol.encoder.n_tokens, 64) and s.shape == (4, 128) and not pad.any()
    assert torch.allclose(wm.z(h, s), pol.encoder(st[:, 0], li[:, 0]), atol=1e-5)   # z = the policy features
    h2, s2 = wm.step(h, s, pad, act[:, 0])
    assert torch.equal(h2, h) and torch.equal(s2, s)               # start: the predictor gives "no change"
    assert wm.heads(h, s, pad).shape == (4, 6)
    loss, info = wm.loss(st, li, act, lab, torch.ones(3))
    assert torch.isfinite(loss) and len(info["roll_per_k"]) == 3
    loss.backward()
    assert all(p.grad is None for p in wm.encoder.parameters())    # the encoder stays frozen
    assert wm.dh.weight.grad is not None and wm.head_mlp[-1].weight.grad is not None


def test_world_model_step_labels():
    from rl_tf.world_model import step_labels
    d = {"f_intr": np.array([0, 1, 0, 0], np.uint8), "f_area": np.array([1, 0, 0, 0], np.uint8),
         "f_out": np.zeros(4, np.uint8), "goal_km": np.array([10.0, 8.5, 7.0, 6.0], np.float32),
         "reward": np.array([-1.0, -2.0, -3.0, 0.0], np.float32), "ret": np.array([-6.0, -5.0, -3.0, 0.0], np.float32)}
    y = step_labels(d, np.array([0, 1]), 2)
    assert y.shape == (2, 2, 6)
    assert np.allclose(y[0], [[0, 1, 0, 1.5, -1.0, -5.0], [1, 0, 0, 1.5, -2.0, -3.0]])   # step k: row t+k-1 -> row t+k
    assert np.allclose(y[1, 1], [0, 0, 0, 1.0, -3.0, 0.0])


def test_planner_filter_keeps_policy_and_search_beats_prior():
    from rl_tf.world_model import WorldModel, Planner
    pol = _policy()
    wm = WorldModel(pol.encoder, n_layers=1, n_heads=4)
    st, li, _, _ = _wm_batch(b=3)
    st, li, prev = st[:, 0], li[:, 0], torch.zeros(3)
    off = Planner(wm, pol, level=1, threshold=2.0, horizon=3)      # risk <= 1 < 2: never searches
    a, info = off.act(st, li, prev)
    ref = pol.act(st[:, None], li[:, None], deterministic=True)[0][:, 0].clamp(-1, 1)
    assert not info["active"].any() and torch.allclose(a, ref, atol=1e-5)
    on = Planner(wm, pol, level=2, samples=8, rounds=2, horizon=3, elites=2, seed=1)
    a2, info2 = on.act(st, li, prev)
    assert a2.shape == (3, 2) and (a2.abs() <= 1).all() and info2["active"].all()
    assert (info2["best_J"] >= info2["prior_J"] - 1e-6).all()      # the policy rollout is a candidate


def test_world_model_save_and_load():
    import tempfile
    from pathlib import Path
    from rl_tf.world_model import WorldModel, save_world_model, load_world_model, same_weights
    args = dict(patch=4, d_model=64, n_heads=4, n_layers=2, features_dim=256, log_std_init=0.0, fourier=0, cls_token=0)
    dims = dict(state_dim=28, lidar_ch=12, n_rays=120)
    wm = WorldModel(_policy().encoder, n_layers=1, n_heads=4)
    with torch.no_grad():
        wm.dh.weight.add_(0.1)
        wm.norms.copy_(torch.tensor([-1.0, 2.0, 3.0, -4.0, 5.0]))
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "wm.pt"
        save_world_model(path, wm, args, dims)
        wm2, _ = load_world_model(path)
    assert same_weights(wm, wm2) and same_weights(wm.encoder, wm2.encoder)
    assert not same_weights(wm.encoder, Policy(state_dim=28, lidar_ch=12, n_rays=120).encoder)


def test_returns_and_neighbour_commands():
    from rl_tf.world_model import returns_to_go, action_lookup, neighbour_commands
    g = returns_to_go(np.array([-1.0, -2.0, 0.0, -4.0, 0.0], np.float32), np.array([1, 1, 0, 1, 0], np.uint8), 0.5, 2.0)
    assert np.allclose(g, [-4.0, -4.0, 0.0, -8.0, 0.0])            # two flights; 0 at the last row of each
    d = {"episode": np.zeros(6, np.int32), "agent": np.array([0, 0, 0, 1, 1, 1], np.int8),
         "step": np.array([0, 1, 2, 0, 1, 2], np.int16), "has_next": np.array([1, 1, 0, 1, 1, 0], np.uint8),
         "action": np.array([[.1, 0], [.2, 0], [0, 0], [.5, .1], [.6, .2], [0, 0]], np.float32),
         "nb_ids": np.array([[1, -1]] * 3 + [[0, -1]] * 3, np.int8)}     # one flight, aircraft 0 and 1
    nb = neighbour_commands(d, np.array([0]), 3, action_lookup(d))
    assert nb.shape == (1, 3, 2, 3)
    assert np.allclose(nb[0, :, 0], [[.5, .1, 1], [.6, .2, 1], [0, 0, 0]])   # step 2: aircraft 1 has no action left
    assert (nb[0, :, 1] == 0).all()                                 # row 1 holds no learning aircraft


def test_aircraft_row_ids_follow_token_order():
    from rl_tf.world_model import aircraft_row_ids
    o = fake_obs(half=500.0)
    o[GEOM_KEY]["aircraft"] = np.array([[30.0, 0.0], [0.0, -10.0], [200.0, 0.0], [5.0, 5.0]])
    o[GEOM_KEY]["aircraft_ids"] = ["A", "B", "C", "D"]
    assert aircraft_row_ids(Lidar(max_range_km=50.0, ac_tokens=3), o) == ["D", "B", "A"]   # nearest first; C too far


def test_planner_multi_agent_neighbours():
    from rl_tf.world_model import WorldModel, Planner
    torch.manual_seed(0)
    pol = Policy(state_dim=42, lidar_ch=12, n_rays=120, ac_tokens=2)       # 28 own + 2 rows x 6 + 2 flags
    wm = WorldModel(pol.encoder, n_layers=1, n_heads=4)
    with torch.no_grad():                                           # trained-like weights: neighbour input matters
        for p in list(wm.nb_in[-1].parameters()) + [wm.dh.weight, wm.ds.weight]:
            p.normal_(0.0, 0.3)
    st, li = torch.randn(3, 42), torch.rand(3, 12, 120)
    st[:, 40:] = 1.0                                                # both aircraft rows are real
    h, s, pad = wm.encode(st, li)
    a = torch.zeros(3, 2)
    h0, _ = wm.step(h, s, pad, a)
    h1, _ = wm.step(h, s, pad, a, torch.tensor([0.9, -0.3, 1.0]).expand(3, 2, 3))
    assert not torch.allclose(h0, h1)
    row_ids = torch.tensor([[1, 2], [0, 2], [0, 1]])
    for order in ("independent", "sequential"):
        pl = Planner(wm, pol, level=2, samples=8, rounds=2, horizon=3, elites=2, value="head", order=order, seed=1)
        out, info = pl.act(st, li, torch.zeros(3), row_ids, [2, 0, 1])
        assert out.shape == (3, 2) and torch.isfinite(out).all()
        assert (info["best_J"] >= info["prior_J"] - 1e-6).all()


def test_load_actor_keeps_new_critic():
    import tempfile
    from pathlib import Path
    from rl_tf.networks import save_checkpoint, load_actor
    from rl_tf.world_model import same_weights
    args = dict(patch=4, d_model=64, n_heads=4, n_layers=2, features_dim=256, log_std_init=0.0, fourier=0, cls_token=0)
    dims = dict(state_dim=28, lidar_ch=12, n_rays=120)
    a = _policy()
    with torch.no_grad():
        a.log_std.fill_(-1.0)
    torch.manual_seed(1)
    b = Policy(state_dim=28, lidar_ch=12, n_rays=120)
    crit = {k: v.clone() for k, v in b.critic.state_dict().items()}
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "p.pt"
        save_checkpoint(path, a, args, dims)
        load_actor(b, path, dims, args)
    assert same_weights(a.encoder, b.encoder) and same_weights(a.actor, b.actor) and torch.equal(a.log_std, b.log_std)
    assert all(torch.equal(v, b.critic.state_dict()[k]) for k, v in crit.items())   # the critic starts new


def test_lagrangian_parse_step_and_advantage():
    from rl_tf.common import parse_lagrangian, lagrange_step, lagrangian_advantage, step_costs
    cons = parse_lagrangian("intrusion_events_no_start:0.3, time_in_restricted_area:5")
    assert cons == [("intrusion_events_no_start", 0.3), ("time_in_restricted_area", 5.0)] and parse_lagrangian("") == []
    for bad in ["a:0", "a:1,a:2"]:
        try:
            parse_lagrangian(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    lam = lagrange_step([0.0, 1.0], [0.6, 2.5], [0.3, 5.0], 0.1, 100.0)
    assert np.allclose(lam, [0.1, 0.95])                             # violation 1.0 (clipped); inside: -0.5 * 0.1
    assert np.allclose(lagrange_step([0.02, 5.0], [0.0, 1000.0], [0.3, 5.0], 0.1, 5.05), [0.0, 5.05])   # 0 and max
    adv_r = np.random.default_rng(0).normal(size=(4, 3)).astype(np.float32)
    adv_c = np.random.default_rng(1).normal(size=(2, 4, 3)).astype(np.float32)
    assert lagrangian_advantage(adv_r, adv_c, [0.0, 0.0]) is adv_r   # lam = 0: the advantage of before, unchanged
    a = lagrangian_advantage(adv_r, adv_c, [1.0, 3.0])
    assert a.dtype == np.float32 and np.allclose(a, (adv_r - adv_c[0] - 3.0 * adv_c[1]) / 5.0, atol=1e-6)
    c, prev = step_costs({"x": 2.0, "y": 10.0}, np.array([1.0, 4.0]), ["x", "y"])
    assert np.allclose(c, [1.0, 6.0]) and np.allclose(prev, [2.0, 10.0])


def test_policy_act_features_do_not_change_actions():
    pol = _policy()
    state, lidar = torch.randn(2, 3, 28), torch.rand(2, 3, 12, 120)
    torch.manual_seed(5)
    a = pol.act(state, lidar)
    torch.manual_seed(5)
    b = pol.act(state, lidar, return_feats=True)
    assert len(a) == 3 and len(b) == 4 and all(torch.equal(x, y) for x, y in zip(a, b[:3]))
    assert torch.equal(b[3], pol.encoder(state.reshape(6, 28), lidar.reshape(6, 12, 120)).reshape(2, 3, -1))


def test_inflate_polygon_and_zone_distance():
    from rl_tf.lidar import inflate_polygon, zone_distance
    sq = square(10.0, 0.0, 2.0)                        # x from 8 to 12
    big = inflate_polygon(sq, 1.0)
    assert np.allclose(np.sort(big[:, 0]), [7, 7, 13, 13]) and np.allclose(np.sort(big[:, 1]), [-3, -3, 3, 3])
    assert inflate_polygon(sq, 0.0) is not None and np.array_equal(inflate_polygon(sq, 0.0), sq)
    assert abs(zone_distance([sq]) - 8.0) < 1e-9 and abs(zone_distance([big]) - 7.0) < 1e-9
    assert zone_distance([square(0.0, 0.0, 1.0)]) == 0.0 and zone_distance([]) == np.inf


def test_ensemble_two_copies_equal_one_model_and_zone_switch():
    from rl_tf.ensemble import EnsembleFn
    lid = Lidar(n_rays=12)
    f0 = lambda raw, built: {a: np.tanh(built[a][0][-4:-2] + built[a][1].mean()).astype(np.float32) for a in raw}
    f1 = lambda raw, built: {a: np.full(2, 0.5, np.float32) for a in raw}
    obs = [fake_obs(zone=(30.0 + 5 * t, 0.0, 5.0), ac=(10.0, 2.0 * t)) for t in range(4)]
    ens = EnsembleFn([(f0, lid, 3), (f0, lid, 3)], mode="mean")
    b, act = ObsBuilder(lid, 3), None
    for t, o in enumerate(obs):                        # reference: one model with its own frame stack
        if t == 0:
            b.reset(o)
        else:
            b.push(o, act)
        act = f0({"own": o}, {"own": b.get()})["own"]
        assert np.array_equal(ens({"own": o}, None)["own"], act)
    z = EnsembleFn([(f0, lid, 3), (f1, lid, 3)], mode="zone", switch_km=20.0)
    assert not np.array_equal(z({"own": fake_obs(zone=(40.0, 0.0, 5.0))}, None)["own"], np.full(2, 0.5))   # 35 km
    assert np.array_equal(z({"own": fake_obs(zone=(20.0, 0.0, 5.0))}, None)["own"], np.full(2, 0.5))       # 15 km
    assert z.stats() == {"ens_decisions": 2, "ens_second_model": 1}


def test_v2_margin_costs_and_kwargs():
    from rl_tf.envs import cpa_margin_cost, zone_margin_cost, v2_kwargs, V2_DEFAULTS
    g = dict(aircraft=np.array([[20.0, 0.0]]), aircraft_vel=np.array([[-100.0, 0.0]]), zones=[])
    assert abs(cpa_margin_cost(g, 10.0, 300.0) - 1.0) < 1e-9          # head-on: d_cpa = 0 at t = 200 s
    assert cpa_margin_cost(g, 10.0, 100.0) == 0.0                      # horizon 100 s: d = 10 km = margin
    g2 = dict(aircraft=np.array([[0.0, 8.0]]), aircraft_vel=np.array([[0.0, 50.0]]), zones=[])
    assert abs(cpa_margin_cost(g2, 10.0, 120.0) - 0.2) < 1e-9          # moving away: d_cpa = now = 8 km
    assert cpa_margin_cost(dict(aircraft=np.zeros((0, 2)), aircraft_vel=np.zeros((0, 2))), 10.0, 120.0) == 0.0
    assert abs(zone_margin_cost(dict(zones=[square(10.0, 0.0, 2.0)]), 10.0) - 0.2) < 1e-9   # 8 km away
    assert zone_margin_cost(dict(zones=[square(0.0, 0.0, 1.0)]), 10.0) == 1.0                # inside
    assert zone_margin_cost(dict(zones=[]), 10.0) == 0.0
    assert v2_kwargs({}) == V2_DEFAULTS and v2_kwargs({"action_map": None}) == V2_DEFAULTS
    for bad in ({"action_map": "x"}, {"cpa_margin_penalty": 1.0}):
        try:
            v2_kwargs(bad)
            raise AssertionError(f"no error for {bad}")
        except ValueError:
            pass


def test_failure_replay_pool():
    from rl_tf.envs import _replay_update, clean_flight
    ok = dict(waypoint_reached=1.0, intrusion_events=0.0, restricted_area_events=0.0, sector_exit_events=0.0)
    assert clean_flight(ok) and not clean_flight({**ok, "cat_end": 1.0}) and not clean_flight({**ok, "waypoint_reached": 0.0})
    cfg, pool = {"replay_cap": 2, "replay_max": 2}, {}
    _replay_update(pool, 5, {**ok, "intrusion_events": 1.0, "intrusion_events_no_start": 0.0}, False, cfg)
    assert pool == {}                                                  # start intrusion only: not replayed
    for sd in (1, 2, 3):
        _replay_update(pool, sd, {**ok, "restricted_area_events": 1.0}, False, cfg)
    assert list(pool) == [2, 3]                                        # cap 2: the oldest seed leaves
    pool[2] = 1
    _replay_update(pool, 2, {**ok, "restricted_area_events": 1.0}, True, cfg)
    assert 2 in pool                                                   # 1 replay < max 2: stays
    pool[2] = 2
    _replay_update(pool, 2, {**ok, "restricted_area_events": 1.0}, True, cfg)
    assert 2 not in pool                                               # max replays reached
    pool[3] = 1
    _replay_update(pool, 3, ok, True, cfg)
    assert 3 not in pool                                               # clean on replay: leaves


def test_random_cat_schedule():
    from rl_tf.envs import cat_prob_at
    assert cat_prob_at(0, 0.1, 1.0, 0) == 1.0                          # no ramp: the end value
    assert abs(cat_prob_at(0, 0.1, 1.0, 1000) - 0.1) < 1e-12
    assert abs(cat_prob_at(500, 0.1, 1.0, 1000) - 0.55) < 1e-12
    assert cat_prob_at(5000, 0.1, 1.0, 1000) == 1.0                    # after the ramp: the end value


def main():
    t0, failed = time.time(), 0
    tests = sorted(k for k in globals() if k.startswith("test_"))
    for name in tests:
        try:
            globals()[name]()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}\n{traceback.format_exc()}")
    print(f"{len(tests) - failed} passed, {failed} failed in {time.time() - t0:.1f} s")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
