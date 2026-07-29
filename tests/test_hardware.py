"""Feature 01 — hardware profiles, KV math, budget derivation."""

import pytest

from hardware import (
    DEVICES,
    RUNTIME_OVERHEAD_GB,
    HardwareProfile,
    derive_budget,
    get_device,
    kv_cache_bytes,
    parse_size_gb,
    profile_from_flags,
)

DESCRIPTOR = {
    "family": "gemma3",
    "layer_count": 18,
    "embedding_length": 640,
    "context_length": 32768,
}


class TestParseSize:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("24GB", 24.0), ("24", 24.0), ("12.5g", 12.5), ("16GiB", 16.0), (8, 8.0), ("512MB", 0.5)],
    )
    def test_accepted(self, raw, expected):
        assert parse_size_gb(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["", "abc", "-4GB", "0"])
    def test_rejected(self, raw):
        with pytest.raises(ValueError):
            parse_size_gb(raw)


class TestProfiles:
    def test_named_device(self):
        p = get_device("rtx-3060-12gb")
        assert p.kind == "gpu" and p.memory_pool_gb() == 12

    def test_unknown_device_lists_known(self):
        with pytest.raises(ValueError, match="rtx-3060-12gb"):
            get_device("nope-9000")

    def test_gpu_flag(self):
        p = profile_from_flags(gpu="24GB")
        assert p.kind == "gpu" and p.vram_gb == 24 and p.source == "flags"

    def test_ram_flag_cpu_only(self):
        p = profile_from_flags(ram="32GB", cpu_only=True)
        assert p.kind == "cpu" and p.memory_pool_gb() == 32

    def test_no_flags_raises(self):
        with pytest.raises(ValueError, match="--device"):
            profile_from_flags()

    def test_device_db_profiles_valid(self):
        for p in DEVICES.values():
            assert p.memory_pool_gb() > 0
            assert 0 < p.effective_usable_fraction() <= 1


class TestKvCache:
    def test_mha_exact(self):
        # 2 × layers × ctx × embd × 2 bytes (f16)
        expected = 2 * 18 * 4096 * 640 * 2
        assert kv_cache_bytes(DESCRIPTOR, 4096, "f16") == expected

    def test_gqa_uses_kv_heads(self):
        desc = dict(DESCRIPTOR, head_count=8, head_count_kv=2, embedding_length=640)
        # head_dim = 640/8 = 80, kv_dim = 2×80 = 160 → quarter of MHA
        assert kv_cache_bytes(desc, 4096, "f16") == kv_cache_bytes(DESCRIPTOR, 4096, "f16") // 4

    def test_missing_dims_returns_none(self):
        assert kv_cache_bytes({}, 4096) is None

    def test_bad_dtype(self):
        with pytest.raises(ValueError, match="dtype"):
            kv_cache_bytes(DESCRIPTOR, 4096, "int4")


class TestDeriveBudget:
    def test_budget_is_auditable_math(self):
        p = get_device("rtx-3060-12gb")
        plan = derive_budget(p, DESCRIPTOR, ctx=4096)
        kv_gb = kv_cache_bytes(DESCRIPTOR, 4096) / (1024**3)
        expected_gb = 12 * 0.90 - kv_gb - RUNTIME_OVERHEAD_GB
        assert plan.weight_budget_bytes == pytest.approx(expected_gb * 1024**3, rel=1e-9)
        assert plan.kv_cache_exact is True
        assert plan.weight_budget_mb == pytest.approx(plan.weight_budget_bytes / 1024**2)

    def test_ctx_defaults_to_capped_model_ctx(self):
        plan = derive_budget(get_device("mac-16gb"), DESCRIPTOR)
        assert plan.ctx == 4096  # min(4096, 32768)

    def test_missing_dims_falls_back_to_reserve(self):
        plan = derive_budget(get_device("mac-16gb"), {})
        assert plan.kv_cache_exact is False
        assert plan.kv_cache_gb == pytest.approx(0.5)

    def test_impossible_fit_raises(self):
        tiny = HardwareProfile(id="tiny", kind="gpu", vram_gb=0.25, ram_gb=8)
        with pytest.raises(ValueError, match="No room for weights"):
            derive_budget(tiny, DESCRIPTOR, ctx=32768)

    def test_display_rows_show_every_subtraction(self):
        plan = derive_budget(get_device("rtx-4090-24gb"), DESCRIPTOR, ctx=8192)
        text = " ".join(f"{k} {v}" for k, v in plan.display_rows())
        for needle in ("memory pool", "usable", "kv cache", "runtime overhead", "--budget-mb"):
            assert needle in text
