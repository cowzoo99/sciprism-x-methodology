from __future__ import annotations

import re
from functools import lru_cache

import spacy

from .schema import TraceUnit

_EQUATION_CUE = re.compile(r"(?:=|≠|≈|≤|≥|∝|→|←|\+|\*|/|\^)" )
_VARIABLE_CUE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\s*(?:=|≠|≈|≤|≥|∝)")
_UNIT_CUE = re.compile(
    r"(?<![\w.])[+-]?(?:\d+(?:\.\d+)?|\.\d+)\s*"
    r"(?:m/s(?:²|\^2)?|km/h|V/m|kg|Pa|Hz|mol|K|N|T|J|W|V|s|A|m)\b"
)
_UNCERTAINTY_CUE = re.compile(
    r"\b(?:approximately|uncertain|uncertainty|confidence|may|might|could|"
    r"assuming|assume|ideal|caveat|limitation)\b",
    re.IGNORECASE,
)
_CAUSAL_CUE = re.compile(
    r"\b(?:because|cause[sd]?|causal|therefore|thus|consequently|lead[sd]? to|"
    r"result[sd]? in|mechanism|balance[sd]?|oppose[sd]?)\b",
    re.IGNORECASE,
)
_MODALITY_CUE = re.compile(
    r"\b(?:table|figure|fig\.?|graph|image|record|passage|source|reference)\b",
    re.IGNORECASE,
)
_STEP_CUE = re.compile(
    r"^(?:\s*(?:\d+[.)]|[-*•])\s*)?(?:compute|calculate|derive|solve|set|"
    r"substitute|compare|explain|state|identify)\b",
    re.IGNORECASE,
)
_LIST_MARKER = re.compile(r"(?<!\S)(?:\d+[.)]|[-*•])\s+")
_TRANSITION = re.compile(
    r"\b(?:Therefore|Thus|However|Consequently|Next|Then)\b[,]?\s*"
)


@lru_cache(maxsize=1)
def _nlp():
    pipeline = spacy.blank("en")
    pipeline.add_pipe("sentencizer", config={"punct_chars": [".", "!", "?"]})
    return pipeline


def _strip_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _split_secondary(text: str, start: int, end: int) -> list[tuple[int, int]]:
    raw = text[start:end]
    boundaries = {0, len(raw)}
    for pattern in (_LIST_MARKER, _TRANSITION):
        matches = list(pattern.finditer(raw))
        for match in matches:
            if match.start() > 0:
                boundaries.add(match.start())
    for match in re.finditer(r"[;\n]+", raw):
        boundaries.add(match.end())
    ordered = sorted(boundaries)
    spans: list[tuple[int, int]] = []
    for left, right in zip(ordered, ordered[1:]):
        stripped = _strip_span(text, start + left, start + right)
        if stripped is not None:
            spans.append(stripped)
    return spans


def _initial_spans(text: str) -> list[tuple[int, int]]:
    document = _nlp()(text)
    spans: list[tuple[int, int]] = []
    for sentence in document.sents:
        spans.extend(_split_secondary(text, sentence.start_char, sentence.end_char))

    combined: list[tuple[int, int]] = []
    index = 0
    while index < len(spans):
        start, end = spans[index]
        if re.fullmatch(r"\d+[.)]", text[start:end].strip()) and index + 1 < len(spans):
            combined.append((start, spans[index + 1][1]))
            index += 2
        else:
            combined.append((start, end))
            index += 1
    return combined


def _equations(fragment: str) -> tuple[str, ...]:
    return tuple(
        match.group(0).strip()
        for match in re.finditer(r"[^.;!?]*(?:=|≠|≈|≤|≥|∝)[^.;!?]*", fragment)
        if match.group(0).strip()
    )


def _variables(fragment: str) -> tuple[str, ...]:
    if not _EQUATION_CUE.search(fragment):
        return ()
    equation_text = " ".join(_equations(fragment)) or fragment
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", equation_text)
    excluded = {"and", "or", "the", "is", "from", "to", "in", "of"}
    symbols: set[str] = set()
    for token in tokens:
        if token.lower() in excluded:
            continue
        if len(token) == 1:
            symbols.add(token)
        elif not (token.islower() or token.isupper() or token.istitle()):
            symbols.update(token)
    return tuple(sorted(symbols))


def _units(fragment: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.group(0) for match in _UNIT_CUE.finditer(fragment)))


def _protected_evidence(fragment: str) -> tuple[str, ...]:
    evidence: list[str] = []
    for name, pattern in (
        ("variable", _VARIABLE_CUE),
        ("equation", _EQUATION_CUE),
        ("unit", _UNIT_CUE),
        ("uncertainty", _UNCERTAINTY_CUE),
        ("causal", _CAUSAL_CUE),
        ("modality", _MODALITY_CUE),
        ("step", _STEP_CUE),
    ):
        if pattern.search(fragment):
            evidence.append(name)
    return tuple(evidence)


def _token_count(fragment: str) -> int:
    return len(re.findall(r"[^\W_]+(?:[²³])?", fragment, flags=re.UNICODE))


def _merge_short_fragments(
    text: str,
    spans: list[tuple[int, int]],
    threshold: int,
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    index = 0
    while index < len(spans):
        start, end = spans[index]
        fragment = text[start:end].strip()
        is_short = _token_count(fragment) < threshold
        if is_short and not _protected_evidence(fragment):
            if index + 1 < len(spans):
                merged.append((start, spans[index + 1][1]))
                index += 2
                continue
            if merged:
                previous_start, _ = merged.pop()
                merged.append((previous_start, end))
                index += 1
                continue
        merged.append((start, end))
        index += 1
    return merged


def segment_trace(
    text: str,
    *,
    short_fragment_token_threshold: int,
) -> tuple[TraceUnit, ...]:
    if short_fragment_token_threshold < 1:
        raise ValueError("short_fragment_token_threshold must be positive")
    if not text.strip():
        return ()
    spans = _merge_short_fragments(
        text,
        _initial_spans(text),
        short_fragment_token_threshold,
    )
    return tuple(
        TraceUnit(
            unit_id=f"U{index + 1}",
            text=text[start:end].strip(),
            start=start,
            end=end,
            variables=_variables(text[start:end]),
            equations=_equations(text[start:end]),
            units=_units(text[start:end]),
            protected_evidence=_protected_evidence(text[start:end]),
        )
        for index, (start, end) in enumerate(spans)
    )
