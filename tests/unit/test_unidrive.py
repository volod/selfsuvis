from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.vision.unidrive import _parse_unidrive_response
from pipeline.workflows.local.steps_report import write_multi_model_comparison_md


def test_parse_unidrive_response_normalizes_schema():
    raw = """```json
    {
      "understanding": {
        "scene_summary": "urban road with light traffic",
        "traffic_context": "vehicles moving steadily",
        "risk_level": "medium",
        "key_agents": ["car", "truck"]
      },
      "perception": {
        "objects": [{"label": "car", "count": 2, "salience": "high"}],
        "drivable_area": "clear",
        "lane_structure": "two visible lanes"
      },
      "planning": {
        "recommended_action": "keep lane",
        "trajectory_hint": "follow centre of drivable area",
        "hazards": ["truck ahead"]
      },
      "mixture_of_experts": {
        "consensus_summary": "continue with caution",
        "expert_agreement": "high",
        "disagreement_points": []
      }
    }
    ```"""
    parsed = _parse_unidrive_response(raw)
    assert parsed["understanding"]["risk_level"] == "medium"
    assert parsed["perception"]["objects"][0]["label"] == "car"
    assert parsed["planning"]["recommended_action"] == "keep lane"
    assert parsed["mixture_of_experts"]["expert_agreement"] == "high"


def test_write_multi_model_comparison_md_writes_expected_sections(tmp_path: Path):
    output = tmp_path / "multi_model_comparison.md"
    gemma_result = {
        "n_frames": 8,
        "task_results": {
            "scene_classification": {
                "category_distribution": {"urban_road": 0.8}
            }
        },
    }
    qwen_result = {
        "ok_count": 2,
        "results": [
            {"t_sec": 1.0, "scene_summary": "urban road with cars"},
            {"t_sec": 3.0, "scene_summary": "intersection with truck"},
        ],
    }
    unidrive_result = {
        "results": [
            {
                "t_sec": 1.1,
                "understanding": {"scene_summary": "urban road with moving cars", "risk_level": "low"},
                "planning": {"recommended_action": "keep lane"},
                "mixture_of_experts": {"consensus_summary": "continue forward", "expert_agreement": "high"},
            },
            {
                "t_sec": 3.2,
                "understanding": {"scene_summary": "intersection with truck ahead", "risk_level": "high"},
                "planning": {"recommended_action": "slow down"},
                "mixture_of_experts": {"consensus_summary": "yield to truck", "expert_agreement": "medium"},
            },
        ],
    }

    summary = write_multi_model_comparison_md(output, "demo_video", gemma_result, qwen_result, unidrive_result)
    text = output.read_text(encoding="utf-8")

    assert summary["matched_frames"] == 2
    assert "Multi-Model Comparison" in text
    assert "UniDriveVLA" in text
    assert "Qwen" in text
    assert "Gemma dominant scene category" in text
