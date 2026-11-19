import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_inventory_matches_verified_official_counts() -> None:
    path = PROJECT_ROOT / "data" / "manifests" / "dataset_inventory.json"
    inventory = json.loads(path.read_text(encoding="utf-8"))

    scicu = inventory["scicueval"]
    assert scicu["commit"] == "423484f99541beed575b28645661afed86dd4ff9"
    assert scicu["file_count"] == 40
    assert scicu["total_records"] == 11343
    assert scicu["dataset_totals"] == {
        "BioText": 968,
        "GoKG": 1180,
        "HipKG": 1165,
        "IaeaTab": 1130,
        "MatTab": 936,
        "MatText": 940,
        "MolTab": 1305,
        "PhaKG": 1217,
        "PriKG": 1250,
        "ProtTab": 1252,
    }

    dbp = inventory["dbp15k"]
    assert dbp["commit"] == "73729a2df0547d79ad07c2758d27c9873930adec"
    assert dbp["seed_fraction"] == 0.3
    for language_pair in ("ZH-EN", "JA-EN", "FR-EN"):
        assert dbp["language_pairs"][language_pair]["supervised_pairs"] == 4500
        assert dbp["language_pairs"][language_pair]["reference_pairs"] == 10500
