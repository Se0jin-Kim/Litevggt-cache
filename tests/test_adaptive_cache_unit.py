import torch
from vggt.merging.adaptive_cache import AdaptiveCacheScheduler

# ── Common fixtures (module-level, reused in every test) ─────────────────────
B, num_dst, N, C = 1, 16, 64, 64
b_idx = torch.arange(16).reshape(1, 16, 1)        # shape [1, 16, 1]
x = torch.randn(B, N, C)                           # shape [1, 64, 64]
dummy_m_u = (lambda t, **kw: t, lambda t, **kw: t, b_idx)

scheduler = AdaptiveCacheScheduler(num_layers=24, tau_base=0.15, k_max=6)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_cold_start():
    scheduler.reset()
    result, reason = scheduler.should_recompute(5, x, b_idx=None)
    assert result is True and reason == "cold_start"


def test_forced_layer0():
    scheduler.reset()
    scheduler.update_cache(0, dummy_m_u, x, b_idx)
    result, reason = scheduler.should_recompute(0, x, b_idx)
    assert result is True and reason == "forced"


def test_forced_age_exceeded():
    scheduler.reset()
    scheduler.update_cache(5, dummy_m_u, x, b_idx)
    for _ in range(scheduler.k_max):
        r, reason = scheduler.should_recompute(5, x, b_idx)
        assert r is False and reason == "cache_hit"
    result, reason = scheduler.should_recompute(5, x, b_idx)
    assert result is True and reason == "forced"


def test_cache_hit():
    scheduler.reset()
    scheduler.update_cache(5, dummy_m_u, x, b_idx)
    result, reason = scheduler.should_recompute(5, x, b_idx)
    assert result is False and reason == "cache_hit"


def test_drift_trigger():
    scheduler.reset()
    scheduler.update_cache(5, dummy_m_u, x, b_idx)
    x_diff = torch.randn(B, N, C) * 100
    result, reason = scheduler.should_recompute(5, x_diff, b_idx)
    assert result is True and reason == "drift"


def test_tau_layerwise():
    s = AdaptiveCacheScheduler(num_layers=24, use_layer_wise_tau=True)
    assert s.tau_l[0] > s.tau_l[23]


def test_reset_clears_state():
    scheduler.reset()
    scheduler.update_cache(5, dummy_m_u, x, b_idx)
    scheduler.reset()
    result, reason = scheduler.should_recompute(5, x, b_idx)
    assert result is True and reason == "cold_start"
