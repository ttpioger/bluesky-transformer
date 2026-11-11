"""Unit test of the GAE function (no BlueSky)."""
import numpy as np
from scripts.train import compute_gae


def test_gae_terminal_truncation_and_dead_rows():
    g, lam = 0.9, 0.8
    rew = np.array([[1.0], [1.0], [1.0], [1.0]], dtype=np.float32)
    val = np.array([[0.5], [0.4], [0.3], [0.2]], dtype=np.float32)
    nxt = np.array([[0.0], [0.0], [0.0], [0.7]], dtype=np.float32)
    alive = np.ones((4, 1), dtype=bool)
    term = np.array([[False], [True], [False], [False]])
    trunc = np.zeros((4, 1), dtype=bool)
    adv, ret = compute_gae(rew, val, nxt, alive, term, trunc, g, lam)
    d3 = 1.0 + g * 0.7 - 0.2              # last row: bootstrap from next_val
    d2 = 1.0 + g * 0.2 - 0.3
    d1 = 1.0 - 0.4                        # terminal: no bootstrap, no carry
    d0 = 1.0 + g * 0.4 - 0.5
    want = [d0 + g * lam * d1, d1, d2 + g * lam * d3, d3]
    assert np.allclose(adv[:, 0], want, atol=1e-6)
    assert np.allclose(ret, adv + val)
    trunc2 = np.array([[False], [True], [False], [False]])
    nxt2 = nxt.copy(); nxt2[1, 0] = 0.9
    adv2, _ = compute_gae(rew, val, nxt2, alive, np.zeros((4, 1), dtype=bool), trunc2, g, lam)
    assert np.isclose(adv2[1, 0], 1.0 + g * 0.9 - 0.4)   # time limit: bootstrap, no carry
    alive3 = alive.copy(); alive3[2, 0] = False
    adv3, _ = compute_gae(rew, val, nxt, alive3, term, trunc, g, lam)
    assert adv3[2, 0] == 0.0
