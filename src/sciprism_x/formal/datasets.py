from __future__ import annotations

import hashlib
import json
import math
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from sciprism_x.config import DBP15KConfig
from sciprism_x.data.schema import SciCUCase


LANGUAGE_DIRECTORIES = {
    "ZH-EN": "zh_en",
    "JA-EN": "ja_en",
    "FR-EN": "fr_en",
}

COMPETENCY_TO_MOTIF = {
    "Context-aware_Inference": "causal-mechanistic relation",
    "Information-absence_Detection": "uncertainty qualification",
    "Information_Integration": "evidence binding",
    "Relevant_Information_Identification": "modality grounding",
}

SOURCE_TO_SCIENTIFIC_DOMAIN = {
    "BioText": "biomedicine",
    "GoKG": "biology",
    "HipKG": "biomedicine",
    "IaeaTab": "physics",
    "MatTab": "materials_science",
    "MatText": "materials_science",
    "MolTab": "chemistry",
    "PhaKG": "biomedicine",
    "PriKG": "biomedicine",
    "ProtTab": "biology",
}


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value if str(item).strip()).strip()
    return str(value).strip()


def _reference_answer(row: Mapping[str, object]) -> str:
    raw_answer = row.get("answer")
    answer = "[]" if isinstance(raw_answer, list) and not raw_answer else _text(raw_answer)
    options_value = row.get("options")
    if not isinstance(options_value, Mapping):
        return answer
    options = {str(key).strip(): _text(value) for key, value in options_value.items()}
    keys = tuple(part.strip() for part in answer.split(",") if part.strip())
    selected = tuple(options[key] for key in keys if key in options)
    return " ".join(selected) if selected else answer


def load_scicueval_cases(root: Path) -> tuple[SciCUCase, ...]:
    source_root = Path(root).resolve()
    cases: list[SciCUCase] = []
    for path in sorted(source_root.rglob("*.json")):
        competency = path.stem
        try:
            motif_family = COMPETENCY_TO_MOTIF[competency]
        except KeyError as error:
            raise ValueError(f"unsupported SciCUEval competency: {competency}") from error
        try:
            scientific_domain = SOURCE_TO_SCIENTIFIC_DOMAIN[path.parent.name]
        except KeyError as error:
            raise ValueError(
                f"unsupported SciCUEval source domain: {path.parent.name}"
            ) from error
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"SciCUEval source must contain a list: {path}")
        relative = path.relative_to(source_root).as_posix()
        for index, raw in enumerate(payload):
            if not isinstance(raw, dict):
                raise ValueError(f"SciCUEval record must be a mapping: {relative}:{index}")
            record_id = _text(raw.get("id")) or str(index)
            context = "\n".join(
                part
                for part in (
                    _text(raw.get("Context")),
                    _text(raw.get("Correct Context")),
                )
                if part
            )
            cases.append(
                SciCUCase(
                    task_id=f"{relative}::{record_id}::{index}",
                    prompt=_text(raw.get("question")),
                    reference_context=context,
                    reference_answer=_reference_answer(raw),
                    domain=scientific_domain,
                    competency=competency,
                    motif_family=motif_family,
                    source_hash=_canonical_hash(raw),
                )
            )
    identifiers = tuple(case.task_id for case in cases)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("SciCUEval task identifiers must be unique")
    return tuple(cases)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_dbp15k(archive: Path, destination: Path) -> Path:
    archive_path = Path(archive).resolve()
    output_root = Path(destination).resolve()
    dataset_root = output_root / "dbp15k"
    marker = dataset_root / ".archive.sha256"
    digest = _sha256(archive_path)
    if marker.is_file() and marker.read_text(encoding="ascii").strip() == digest:
        return dataset_root
    output_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            target = (output_root / member.name).resolve()
            if not target.is_relative_to(output_root):
                raise ValueError(f"DBP15K archive member escapes extraction root: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"DBP15K archive links are not allowed: {member.name}")
        bundle.extractall(output_root, members=members)
    if not dataset_root.is_dir():
        raise ValueError("DBP15K archive did not contain the expected root")
    marker.write_text(digest + "\n", encoding="ascii")
    return dataset_root


def _read_int_pairs(path: Path) -> tuple[tuple[int, int], ...]:
    rows: list[tuple[int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        left, right = line.split("\t")
        rows.append((int(left), int(right)))
    return tuple(rows)


def _read_triples(path: Path) -> tuple[tuple[int, int, int], ...]:
    rows: list[tuple[int, int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        subject, relation, object_ = line.split("\t")
        rows.append((int(subject), int(relation), int(object_)))
    return tuple(rows)


def _read_entities(path: Path) -> tuple[tuple[int, str], ...]:
    rows: list[tuple[int, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        identifier, label = line.split("\t", 1)
        rows.append((int(identifier), label))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class DBPLanguageDataset:
    language_pair: str
    entity_ids_1: tuple[tuple[int, str], ...]
    entity_ids_2: tuple[tuple[int, str], ...]
    triples_1: tuple[tuple[int, int, int], ...]
    triples_2: tuple[tuple[int, int, int], ...]
    seed_pairs: tuple[tuple[int, int], ...]
    reference_pairs: tuple[tuple[int, int], ...]
    target_candidate_ids: tuple[int, ...]
    seed_fraction: float = 0.30
    candidate_pruning: bool = False


@dataclass(frozen=True, slots=True)
class DBP15KDataset:
    language_pairs: Mapping[str, DBPLanguageDataset]


@dataclass(frozen=True, slots=True)
class FormalDatasets:
    scicueval: tuple[SciCUCase, ...]
    dbp15k: DBP15KDataset


def load_dbp15k_dataset(
    root: Path,
    protocol_config: DBP15KConfig | None = None,
) -> DBP15KDataset:
    dataset_root = Path(root).resolve()
    configured_languages = (
        tuple(protocol_config.language_pairs)
        if protocol_config is not None
        else tuple(LANGUAGE_DIRECTORIES)
    )
    unknown_languages = set(configured_languages).difference(LANGUAGE_DIRECTORIES)
    if unknown_languages:
        raise ValueError(f"unsupported DBP15K language pairs: {sorted(unknown_languages)}")
    if protocol_config is not None and not protocol_config.full_candidate_pool:
        raise ValueError("the paper protocol requires the full DBP15K target-entity pool")
    language_pairs: dict[str, DBPLanguageDataset] = {}
    for language_pair in configured_languages:
        directory = LANGUAGE_DIRECTORIES[language_pair]
        protocol = dataset_root / directory / "0_3"
        seed_pairs = _read_int_pairs(protocol / "sup_ent_ids")
        reference_pairs = _read_int_pairs(protocol / "ref_ent_ids")
        source_entities = _read_entities(protocol / "ent_ids_1")
        target_entities = _read_entities(protocol / "ent_ids_2")
        candidates = tuple(sorted(identifier for identifier, _label in target_entities))
        expected_seed_fraction = (
            protocol_config.seed_fraction if protocol_config is not None else 0.30
        )
        all_pairs = seed_pairs + reference_pairs
        if not seed_pairs or not reference_pairs or len(set(all_pairs)) != len(all_pairs):
            raise ValueError(
                f"DBP15K {language_pair} 0_3 split is empty, duplicated, or overlapping"
            )
        observed_seed_fraction = len(seed_pairs) / len(all_pairs)
        if not math.isclose(
            observed_seed_fraction,
            expected_seed_fraction,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        ):
            raise ValueError(
                f"DBP15K {language_pair} seed fraction differs from the frozen 0_3 split"
            )
        source_ids = {identifier for identifier, _label in source_entities}
        target_ids = {identifier for identifier, _label in target_entities}
        if len(source_ids) != len(source_entities) or len(target_ids) != len(target_entities):
            raise ValueError(f"DBP15K {language_pair} entity IDs are duplicated")
        if any(left not in source_ids or right not in target_ids for left, right in all_pairs):
            raise ValueError(f"DBP15K {language_pair} split references unknown entity IDs")
        if len(candidates) != len(target_ids) or set(candidates) != target_ids:
            raise ValueError(f"DBP15K {language_pair} target candidate pool was pruned")
        language_pairs[language_pair] = DBPLanguageDataset(
            language_pair=language_pair,
            entity_ids_1=source_entities,
            entity_ids_2=target_entities,
            triples_1=_read_triples(protocol / "triples_1"),
            triples_2=_read_triples(protocol / "triples_2"),
            seed_pairs=seed_pairs,
            reference_pairs=reference_pairs,
            target_candidate_ids=candidates,
            seed_fraction=expected_seed_fraction,
            candidate_pruning=(
                protocol_config.candidate_pruning if protocol_config is not None else False
            ),
        )
    return DBP15KDataset(language_pairs=language_pairs)


def load_formal_datasets(
    data_root: Path,
    *,
    extraction_root: Path,
    dbp_protocol: DBP15KConfig | None = None,
) -> FormalDatasets:
    root = Path(data_root).resolve()
    return FormalDatasets(
        scicueval=load_scicueval_cases(root / "raw" / "scicueval"),
        dbp15k=load_dbp15k_dataset(
            extract_dbp15k(root / "raw" / "dbp15k" / "dbp15k.tar.gz", extraction_root),
            dbp_protocol,
        ),
    )
