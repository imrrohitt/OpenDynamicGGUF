"""Feature 03 — report data extraction and self-contained HTML rendering."""

import json

import pytest

from report import build_report_data, generate_report, render_html


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def run_dir(tmp_path):
    """Minimal but complete synthetic run mirroring the store layout."""
    root = tmp_path / "run"
    _write(
        root / "run.json",
        {
            "run_id": "20260101-test",
            "model_ref": "test:model",
            "status": "done",
            "quant_format": "q4_k_m",
            "quant_label": "Q4_K_M",
        },
    )
    _write(
        root / "steps/01_resolve/output.json",
        {
            "user_ref": "test:model",
            "hf_repo_id": "org/test-model",
            "source_sha256": "aa11",
            "descriptor": {
                "family": "gemma3",
                "layer_count": 2,
                "embedding_length": 64,
                "parameter_count": 1000,
                "context_length": 8192,
            },
        },
    )
    _write(
        root / "steps/05_catalog/tensor_catalog.json",
        {
            "tensors": {
                "blk.0.ffn_up.weight": {"n_elements": 1000, "group_id": "ffn_up@early"},
                "blk.0.attn_v.weight": {"n_elements": 500, "group_id": "attn_v@early"},
            },
            "groups": {
                "ffn_up@early": {"tensor_names": ["blk.0.ffn_up.weight"]},
                "attn_v@early": {"tensor_names": ["blk.0.attn_v.weight"]},
            },
        },
    )
    _write(
        root / "steps/12_sensitivity/sensitivity.json",
        {
            "method": "proxy_from_features",
            "baseline_type": "Q6_K",
            "probe_types": ["Q3_K", "Q4_K"],
            "rows": [
                {
                    "group_id": "ffn_up@early",
                    "probe": "Q3_K",
                    "delta_kld": 0.004,
                    "delta_bytes": 300,
                },
                {
                    "group_id": "attn_v@early",
                    "probe": "Q4_K",
                    "delta_kld": 0.037,
                    "delta_bytes": 45,
                },
            ],
        },
    )
    _write(
        root / "steps/13_optimize/output.json",
        {
            "method": "greedy_knapsack_v1",
            "budget_bytes": 5000,
            "estimated_bytes": 4800,
            "predicted_delta_kld": 0.01,
            "recipe_path": "recipe.yaml",
            "assignments": {"ffn_up@early": "Q3_K", "attn_v@early": "Q5_K"},
        },
    )
    _write(
        root / "steps/13_optimize/optimize_manifest.json",
        {
            "primary": {
                "estimated_bytes": 4800,
                "history": [
                    {
                        "group_id": "ffn_up@early",
                        "to": "Q3_K",
                        "delta_bytes": 300.0,
                        "delta_kld_inc": 0.004,
                        "efficiency": 75000.0,
                        "size_after": 4800,
                    }
                ],
            },
            "pareto": [
                {"estimated_bytes": 4800, "predicted_delta_kld": 0.01, "meets_budget": True},
                {"estimated_bytes": 6000, "predicted_delta_kld": 0.004, "meets_budget": False},
            ],
        },
    )
    _write(
        root / "steps/15_validate/output.json",
        {
            "verdict": "PROVISIONAL",
            "method": "proxy_gates",
            "tier1": {
                "gates": {"mean_kld_max": 0.5, "top1_agree_min": 0.5},
                "metrics": {"mean_kld": 0.01, "top1_agree": 0.98},
                "pass": True,
                "note": "proxy",
            },
        },
    )
    _write(
        root / "benchmarks/20260101-000000-smoke/benchresult.json",
        {
            "schema": "odg/benchresult/v1",
            "suite": "smoke",
            "created_at": "2026-01-01T00:00:00Z",
            "gguf_sha256": "bb22" * 16,
            "quality": {"skipped": True, "reason": "lm-eval not installed", "tasks": {}},
            "throughput": {"pp_tps": 800.0, "tg_tps": 30.0, "device": "mac-16gb"},
        },
    )
    return root


class TestExtraction:
    def test_all_sections_available(self, run_dir):
        data = build_report_data(run_dir)
        assert all(v["available"] for v in data.values()), {
            k: v["available"] for k, v in data.items()
        }

    def test_allocation_reasons_trace_to_evidence(self, run_dir):
        rows = {r["group"]: r for r in build_report_data(run_dir)["allocations"]["rows"]}
        # downgraded group cites the greedy decision
        assert "greedy downgrade" in rows["ffn_up@early"]["reason"]
        assert rows["ffn_up@early"]["delta_kld"] == 0.004
        # pinned role cites the pin, never a fabricated probe
        assert "pinned" in rows["attn_v@early"]["reason"]

    def test_gates_extracted_with_thresholds(self, run_dir):
        gates = {g["metric"]: g for g in build_report_data(run_dir)["gates"]["gates"]}
        assert gates["mean_kld"]["pass"] is True
        assert gates["top1_agree"]["higher_better"] is True

    def test_partial_run_degrades_not_crashes(self, tmp_path):
        root = tmp_path / "partial"
        _write(root / "run.json", {"run_id": "x", "model_ref": "m", "status": "running"})
        data = build_report_data(root)
        assert data["summary"]["available"]
        assert not data["allocations"]["available"]
        html = render_html(data)  # must render "not run" sections
        assert "not run" in html


class TestRendering:
    def test_generate_writes_self_contained_html(self, run_dir):
        result = generate_report(run_dir)
        html = (run_dir / "report.html").read_text()
        assert result.sections_rendered == 7 and not result.sections_missing
        # self-contained: no external fetches (SVG xmlns is a namespace id, not a fetch)
        for external in ('<script src', "<link rel", 'src="http', "@import", "url(http"):
            assert external not in html
        assert "<svg" in html and "<style>" in html
        # key content present
        for needle in ("test:model", "PROVISIONAL", "ffn_up@early", "Q3_K", "smoke"):
            assert needle in html
        # sections caption their source artifacts
        assert "steps/12_sensitivity/sensitivity.json" in html

    def test_custom_out_path(self, run_dir, tmp_path):
        out = tmp_path / "elsewhere" / "r.html"
        result = generate_report(run_dir, out_path=out)
        assert out.is_file() and result.report_path == str(out)

    def test_not_a_run_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="run.json"):
            generate_report(tmp_path)
