import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "collect_pedal_evidence.py"
SPEC = importlib.util.spec_from_file_location("collect_pedal_evidence", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sha256_and_labels(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    source = repo / "results_eval" / "edge-10ms" / "metric.json"
    source.parent.mkdir(parents=True)
    workspace.mkdir()
    source.write_bytes(b"pedal evidence\n")

    assert MODULE.sha256_file(source) == (
        "8f95e023085c74983f3b3fd9447c1972d1e691f6600d75fd0a861c3218a20990"
    )
    assert MODULE.path_label(source, repo, workspace) == (
        "repo/results_eval/edge-10ms/metric.json"
    )


def test_missing_dataset_is_explicit() -> None:
    assert MODULE.dataset_record(None, None) == {"available": False}
