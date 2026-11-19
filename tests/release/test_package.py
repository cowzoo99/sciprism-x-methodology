from __future__ import annotations

import json
from pathlib import Path

from sciprism_x.release.package import (
    audit_package_tree,
    build_release_package,
)


def test_package_builder_is_allowlist_based_and_preserves_benchmark_ids(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    run = tmp_path / "run"
    target = tmp_path / "release-package"
    (project / "src").mkdir(parents=True)
    (project / ".venv").mkdir()
    (run / "computed").mkdir(parents=True)
    (project / "src" / "method.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / ".venv" / "secret.txt").write_text("excluded\n", encoding="utf-8")
    (run / "computed" / "sample_manifest.json").write_text(
        json.dumps({"instance_id": "SCICU-0417"}),
        encoding="utf-8",
    )
    config = {
        "schema_version": 1,
        "package_name": "sciprism_x_methodology_release",
        "package_version": "1.0.0-review",
        "review_availability": "immediate_on_request",
        "public_availability": "upon_publication",
        "project_include": ["src"],
        "run_include": ["computed"],
        "excluded_names": [".venv"],
        "forbidden_identity_tokens": ["C:" + "\\Users\\"],
        "allowed_absolute_paths": ["D:" + "\\SciPRISM-X-runs"],
    }

    build_release_package(
        project_root=project,
        run_root=run,
        target=target,
        config=config,
    )
    audit = audit_package_tree(target, config=config)

    assert audit.errors == ()
    assert (target / "project" / "src" / "method.py").is_file()
    assert not (target / "project" / ".venv").exists()
    assert "SCICU-0417" in (
        target / "results" / "computed" / "sample_manifest.json"
    ).read_text(encoding="utf-8")
    assert (target / "package_manifest.json").is_file()
