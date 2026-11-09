import json
from pathlib import Path

import pytest

from scripts.closeout_v2_2 import ROOT, check_budget, release_paths


def test_release_allowlist_excludes_old_reports_and_runtime_artifacts():
    paths = release_paths()
    assert {"scripts/publish_v2_2.sh", "scripts/closeout_v2_2.py", "README.md"} <= set(paths)
    assert not any(p.startswith(("artifacts/", "reports/v2_1/", "reports/v2_2/remaining_assistant/")) for p in paths)
    assert len(paths) == len(set(paths))


def test_release_rejects_path_traversal_and_duplicates(tmp_path):
    config = tmp_path / "configs/v2_2/release_files.json"
    config.parent.mkdir(parents=True)
    for paths in (["../secret"], ["README.md", "README.md"], ["artifacts/log.txt"]):
        config.write_text(json.dumps(paths))
        with pytest.raises(ValueError):
            release_paths(tmp_path)


def test_release_requires_full_task_budget(tmp_path):
    frozen, smoke, full = [tmp_path / x for x in ("frozen", "smoke", "full")]
    for p in (frozen, smoke, full):
        p.mkdir()
    (frozen / "test_paired.json").write_text(json.dumps([{}] * 315))
    (smoke / "protocol.json").write_text(json.dumps({"tasks": [{}] * 71}))
    tasks = [{"stage": stage} for stage, count in (("logistic", 882), ("heavy_fixed", 81),
                                                  ("heavy_closed", 324), ("cost", 45)) for _ in range(count)]
    (full / "protocol.json").write_text(json.dumps({"tasks": tasks}))
    assert sum(check_budget(frozen, smoke, full).values()) == 1332
    (full / "protocol.json").write_text(json.dumps({"tasks": tasks[:-1]}))
    with pytest.raises(ValueError, match="budget"):
        check_budget(frozen, smoke, full)
