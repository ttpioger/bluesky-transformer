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
