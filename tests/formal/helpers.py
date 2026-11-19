from pathlib import Path

from sciprism_x.config import load_run_config
from sciprism_x.formal.config import load_formal_config, resolve_scale_contract


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_config():
    return load_run_config(project_root() / "configs" / "track_b.yaml")


def formal_config():
    return load_formal_config(project_root() / "configs" / "formal_execution.yaml")


def smoke_contract():
    return resolve_scale_contract(run_config(), formal_config(), scale="smoke")


def paper_contract():
    return resolve_scale_contract(run_config(), formal_config(), scale="paper")
