"""Feature 02 — paired statistics, llama-bench parsing, benchresult schema."""

import json

import numpy as np
import pytest

from benchmark import (
    SCHEMA,
    SUITES,
    paired_bootstrap_delta,
    parse_llama_bench_json,
    run_benchmark,
    sha256_file,
)


class TestPairedStats:
    def test_ci_contains_true_delta(self):
        rng = np.random.default_rng(7)
        ref = rng.random(500)
        cand = ref - 0.05 + rng.normal(0, 0.02, 500)  # true delta ≈ −0.05
        r = paired_bootstrap_delta(cand, ref, seed=1)
        assert r["ci_low"] <= -0.05 <= r["ci_high"]
        assert r["paired_delta"] == pytest.approx(-0.05, abs=0.01)
        assert r["n"] == 500

    def test_identical_scores_give_zero_delta(self):
        scores = [0.0, 1.0, 1.0, 0.0, 1.0]
        r = paired_bootstrap_delta(scores, scores)
        assert r["paired_delta"] == 0.0
        assert r["ci_low"] == 0.0 and r["ci_high"] == 0.0

    def test_deterministic_given_seed(self):
        cand, ref = [0.4, 0.6, 0.5], [0.5, 0.5, 0.5]
        assert paired_bootstrap_delta(cand, ref, seed=3) == paired_bootstrap_delta(
            cand, ref, seed=3
        )

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError, match="equal-length"):
            paired_bootstrap_delta([1.0, 2.0], [1.0])


class TestLlamaBenchParse:
    def test_parses_pp_and_tg(self):
        out = json.dumps(
            [
                {"n_prompt": 512, "n_gen": 0, "avg_ts": 812.34, "backends": "Metal"},
                {"n_prompt": 0, "n_gen": 128, "avg_ts": 34.21},
            ]
        )
        r = parse_llama_bench_json("noise before " + out)
        assert r == {
            "measured": True,
            "tool": "llama-bench",
            "pp_tps": 812.34,
            "backend": "Metal",
            "tg_tps": 34.21,
        }

    @pytest.mark.parametrize("bad", ["", "not json", "[]", '[{"no": "tps"}]'])
    def test_garbage_returns_none(self, bad):
        assert parse_llama_bench_json(bad) is None


class TestRunBenchmark:
    def test_smoke_suite_writes_valid_benchresult(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        gguf.write_bytes(b"GGUF" + b"\x00" * 1024)
        out_dir = tmp_path / "bench"

        result = run_benchmark(
            gguf,
            suite_id="smoke",
            model_ref="test:model",
            device_profile={"id": "mac-16gb"},
            llama_bench=tmp_path / "missing-binary",  # forces honest skip
            out_dir=out_dir,
        )

        saved = json.loads((out_dir / "benchresult.json").read_text())
        assert saved["schema"] == SCHEMA
        assert saved["gguf_sha256"] == sha256_file(gguf)
        assert saved["suite"] == "smoke"
        assert saved["memory"]["weights_bytes"] == gguf.stat().st_size
        # No fake numbers: throughput and quality both recorded as unavailable
        assert saved["throughput"] is None
        assert any("Throughput not measured" in n for n in saved["notes"])
        assert result.result_path == str(out_dir / "benchresult.json")

    def test_unknown_suite_rejected(self, tmp_path):
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"x")
        with pytest.raises(ValueError, match="Unknown suite"):
            run_benchmark(gguf, suite_id="nope")

    def test_missing_gguf_rejected(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_benchmark(tmp_path / "absent.gguf")

    def test_suites_have_pinned_configs(self):
        for s in SUITES.values():
            assert s.tasks and s.num_fewshot >= 0
