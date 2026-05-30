import subprocess

from sslm.playground import orchestrator
from sslm.playground.catalog import MODEL_CATALOG
from sslm.playground.orchestrator import SequentialRunConfig


def test_detected_gpu_total_gb_parses_nvidia_smi(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="15916\n", stderr="")

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)

    assert round(orchestrator.detected_gpu_total_gb(), 1) == 15.5


def test_skipped_model_queues_fallback(monkeypatch, tmp_path, capsys) -> None:
    calls = []

    monkeypatch.setattr(orchestrator, "detected_gpu_total_gb", lambda: 16.0)
    def fake_write_compose(models, output):
        calls.append(("compose_models", tuple(model.key for model in models)))
        return output

    monkeypatch.setattr(orchestrator, "write_compose_file", fake_write_compose)

    class FakeSidecar:
        def __init__(self, compose_file, model) -> None:
            self.model = model

        def prefetch(self) -> None:
            calls.append((self.model.key, "prefetch"))

        def up(self, *, build=False) -> None:
            calls.append((self.model.key, "up"))

        def wait_ready(self) -> None:
            calls.append((self.model.key, "wait_ready"))

        def down(self) -> None:
            calls.append((self.model.key, "down"))

    monkeypatch.setattr(orchestrator, "DockerComposeSidecar", FakeSidecar)
    monkeypatch.setattr(orchestrator, "run_smoke", lambda **kwargs: calls.append(("smoke", kwargs["model_id"])))

    orchestrator.run_sequential(
        SequentialRunConfig(
            models=[MODEL_CATALOG["zaya1-8b"]],
            results_dir=tmp_path / "results",
            compose_file=tmp_path / "compose.yml",
            suite="smoke",
        )
    )

    output = capsys.readouterr().out
    assert "queued fallback model qwen3-4b-fp8" in output
    assert ("compose_models", ("zaya1-8b", "qwen3-4b-fp8")) in calls
    assert ("qwen3-4b-fp8", "up") in calls
    assert ("smoke", "Qwen/Qwen3-4B-FP8") in calls


def test_models_with_fallbacks_adds_fallback_once() -> None:
    models = orchestrator.models_with_fallbacks(
        [MODEL_CATALOG["zaya1-8b"], MODEL_CATALOG["qwen3-4b-fp8"]]
    )

    assert [model.key for model in models] == ["zaya1-8b", "qwen3-4b-fp8"]
