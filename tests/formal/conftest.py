from pathlib import Path

import pytest

from sciprism_x.formal.workflow import run_formal_experiment
from sciprism_x.results.schema import FormalRunReceipt

from .helpers import formal_config, run_config


@pytest.fixture(scope="session")
def formal_smoke_receipt(tmp_path_factory: pytest.TempPathFactory) -> FormalRunReceipt:
    return run_formal_experiment(
        run_config(),
        formal_config(),
        scale="smoke",
        output_root=Path(tmp_path_factory.mktemp("formal-smoke")),
    )


@pytest.fixture(scope="session")
def formal_smoke_run(formal_smoke_receipt: FormalRunReceipt) -> Path:
    return formal_smoke_receipt.final_path
