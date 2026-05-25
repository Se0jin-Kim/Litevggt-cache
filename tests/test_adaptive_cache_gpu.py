import pytest
import torch
from vggt.merging.adaptive_cache import AdaptiveCacheScheduler
from vggt.models.aggregator import Aggregator

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU not available"
)

def _setup_model():
    import os
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    model = Aggregator().cuda().eval()
    model.patch_embed = model.patch_embed.to(torch.bfloat16)
    for block in model.global_blocks:
        block.patch_width = 37
        block.patch_height = 37
    return model

def test_output_shape_consistency():
    model = _setup_model()
    dummy = torch.randn(1, 5, 3, 518, 518).cuda()
    with torch.no_grad():
        out_adaptive = model(dummy, use_adaptive_cache=True)
        out_original = model(dummy, use_adaptive_cache=False)
    ref_a = next(x for x in out_adaptive[0] if x is not None)
    ref_o = next(x for x in out_original[0] if x is not None)
    assert ref_a.shape == ref_o.shape, \
        f"Shape mismatch: {ref_a.shape} vs {ref_o.shape}"

def test_recompute_log():
    log = []
    original_fn = AdaptiveCacheScheduler.should_recompute

    def patched(self, layer_idx, x, b_idx):
        result, reason = original_fn(self, layer_idx, x, b_idx)
        drift = None
        if (self.cached_dst_feat[layer_idx] is not None
                and b_idx is not None):
            import torch.nn.functional as F
            dst_now = x[:, b_idx[0, :, 0], :]
            cos = F.cosine_similarity(
                dst_now, self.cached_dst_feat[layer_idx], dim=-1
            ).mean().item()
            drift = round(1.0 - cos, 4)
        log.append((layer_idx, reason, drift))
        return result, reason

    AdaptiveCacheScheduler.should_recompute = patched
    try:
        model = _setup_model()
        dummy = torch.randn(1, 5, 3, 518, 518).cuda()
        with torch.no_grad():
            model(dummy, use_adaptive_cache=True)
    finally:
        AdaptiveCacheScheduler.should_recompute = original_fn

    print("\nlayer | reason      | drift")
    print("------|-------------|-------")
    for layer_idx, reason, drift in log:
        drift_str = f"{drift:.4f}" if drift is not None else "  —  "
        print(f"  {layer_idx:<3} | {reason:<11} | {drift_str}")

    assert len(log) == 24, f"Expected 24 log entries, got {len(log)}"
